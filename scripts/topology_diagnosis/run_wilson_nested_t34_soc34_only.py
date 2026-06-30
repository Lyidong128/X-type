#!/usr/bin/env python3
"""Wilson polarization and nested reliability for strict t34 + soc34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_soc34_only_common import (  # noqa: E402
    ModelParams,
    compute_bulk_gap,
    compute_wilson_centers_direction,
    middle_sector_gap,
    nested_polarization_from_tracked,
    output_dirs,
    parse_float_list,
    track_sector_frames,
    token,
    write_csv,
    write_model_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wilson+nested scan for strict t34+soc34-only model.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only")
    p.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    p.add_argument("--nkx", type=int, default=81)
    p.add_argument("--nky", type=int, default=81)
    p.add_argument("--n-occ", type=int, default=4)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def classify_p(p: float) -> str:
    p = float(np.mod(p, 1.0))
    if min(abs(p - 0.0), abs(p - 1.0)) < 0.05:
        return "trivial_0"
    if abs(p - 0.5) < 0.05:
        return "nontrivial_0p5"
    return "not_quantized"


def polarization_type(px: str, py: str) -> str:
    if px == "nontrivial_0p5" and py == "nontrivial_0p5":
        return "2D_SSH_like_polarized"
    if (px == "nontrivial_0p5") ^ (py == "nontrivial_0p5"):
        return "1D_SSH_like_polarized"
    if px == "trivial_0" and py == "trivial_0":
        return "trivial_polarization"
    return "not_quantized"


def plot_wilson(scan: np.ndarray, centers: np.ndarray, out_png: Path, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for b in range(centers.shape[1]):
        ax.scatter(scan, centers[:, b], s=7, alpha=0.75)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Wannier center")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_model_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_w = dirs["04_wilson_flow"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_list = parse_float_list(args.v_list)

    wrows = []
    nrows = []
    for v in v_list:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=81, n_occ=args.n_occ)
        ky, cx, eigx, occx, errx = compute_wilson_centers_direction(
            v=v, params=params, n_occ=args.n_occ, nkx=args.nkx, nky=args.nky, direction="x"
        )
        kx, cy, eigy, occy, erry = compute_wilson_centers_direction(
            v=v, params=params, n_occ=args.n_occ, nkx=args.nkx, nky=args.nky, direction="y"
        )
        px = float(np.mod(np.mean(cx), 1.0))
        py = float(np.mod(np.mean(cy), 1.0))
        px_cls = classify_p(px)
        py_cls = classify_p(py)
        ptype = polarization_type(px_cls, py_cls)
        gx = middle_sector_gap(cx)
        gy = middle_sector_gap(cy)
        nested_allowed = int(gx > 0.05 and gy > 0.05)
        wrows.append(
            {
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "p_x": float(px),
                "p_y": float(py),
                "p_x_class": px_cls,
                "p_y_class": py_cls,
                "polarization_type": ptype,
                "wannier_gap_x": float(gx),
                "wannier_gap_y": float(gy),
                "nested_allowed": int(nested_allowed),
                "unitarity_error_x": float(errx),
                "unitarity_error_y": float(erry),
            }
        )

        if nested_allowed:
            tx, ovx = track_sector_frames(eigx, n_occ=args.n_occ)
            ty, ovy = track_sector_frames(eigy, n_occ=args.n_occ)
            py_nux = nested_polarization_from_tracked(tx, occx)
            px_nuy = nested_polarization_from_tracked(ty, occy)
            qxy = float(np.mod(0.5 * (py_nux + px_nuy), 1.0))
            nested_reliable = int(gx > 0.05 and gy > 0.05 and ovx > 0.5 and ovy > 0.5)
            nested_hoti = int(nested_reliable and abs(qxy - 0.5) < 0.05)
            comment = "reliable_and_hoti_like" if nested_hoti else ("reliable_non_hoti" if nested_reliable else "nested_unreliable")
        else:
            py_nux = float("nan")
            px_nuy = float("nan")
            qxy = float("nan")
            ovx = 0.0
            ovy = 0.0
            nested_reliable = 0
            nested_hoti = 0
            comment = "nested_not_allowed_by_wannier_gap"

        nrows.append(
            {
                "v": float(v),
                "nested_allowed": int(nested_allowed),
                "nested_reliable": int(nested_reliable),
                "p_y_nu_x": float(py_nux),
                "p_x_nu_y": float(px_nuy),
                "qxy": float(qxy),
                "min_sector_overlap_x": float(ovx),
                "min_sector_overlap_y": float(ovy),
                "nested_supports_hoti": int(nested_hoti),
                "comment": comment,
            }
        )

        if abs(v - 0.6) < 1e-9 or abs(v - 0.8) < 1e-9 or abs(v - 1.0) < 1e-9:
            plot_wilson(
                ky,
                cx,
                out_w / f"wilson_x_v{token(v)}_t34_soc34_only.png",
                xlabel=r"$k_y$",
                title=f"Wilson x v={v:.1f}, p_x={px:.4f}",
            )
            plot_wilson(
                kx,
                cy,
                out_w / f"wilson_y_v{token(v)}_t34_soc34_only.png",
                xlabel=r"$k_x$",
                title=f"Wilson y v={v:.1f}, p_y={py:.4f}",
            )

        print(
            f"[wilson-soc34] v={v:.1f} p=({px:.3f},{py:.3f}) "
            f"gaps=({gx:.3f},{gy:.3f}) nested_allowed={bool(nested_allowed)}"
        )

    wf = [
        "v",
        "bulk_gap",
        "p_x",
        "p_y",
        "p_x_class",
        "p_y_class",
        "polarization_type",
        "wannier_gap_x",
        "wannier_gap_y",
        "nested_allowed",
        "unitarity_error_x",
        "unitarity_error_y",
    ]
    nf = [
        "v",
        "nested_allowed",
        "nested_reliable",
        "p_y_nu_x",
        "p_x_nu_y",
        "qxy",
        "min_sector_overlap_x",
        "min_sector_overlap_y",
        "nested_supports_hoti",
        "comment",
    ]
    write_csv(out_w / "wilson_polarization_t34_soc34_only.csv", wrows, wf)
    write_csv(base / "wilson_polarization_t34_soc34_only.csv", wrows, wf)
    write_csv(out_w / "nested_wilson_t34_soc34_only.csv", nrows, nf)
    write_csv(base / "nested_wilson_t34_soc34_only.csv", nrows, nf)

    v = np.array([float(r["v"]) for r in wrows], dtype=float)
    px = np.array([float(r["p_x"]) for r in wrows], dtype=float)
    py = np.array([float(r["p_y"]) for r in wrows], dtype=float)
    na = np.array([int(r["nested_allowed"]) for r in wrows], dtype=float)
    rel = np.array([int(r["nested_reliable"]) for r in nrows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    axes[0, 0].plot(v, px, marker="o", label="p_x")
    axes[0, 0].plot(v, py, marker="s", label="p_y")
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("polarization")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)
    axes[0, 1].plot(v, [float(r["wannier_gap_x"]) for r in wrows], marker="o", label="gap_x")
    axes[0, 1].plot(v, [float(r["wannier_gap_y"]) for r in wrows], marker="s", label="gap_y")
    axes[0, 1].axhline(0.05, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Wannier gaps")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)
    axes[1, 0].plot(v, na, marker="o")
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].set_title("nested_allowed")
    axes[1, 0].grid(alpha=0.2)
    qxy = np.array([float(r["qxy"]) if str(r["qxy"]).lower() != "nan" else np.nan for r in nrows], dtype=float)
    axes[1, 1].plot(v, qxy, marker="o", label="qxy")
    axes[1, 1].plot(v, rel, marker="s", label="nested_reliable")
    axes[1, 1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("nested Wilson")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)
    for ax in axes.reshape(-1):
        ax.set_xlabel("v")
    fig.tight_layout()
    fig.savefig(out_w / "wilson_nested_summary_t34_soc34_only.png", dpi=170)
    fig.savefig(base / "wilson_nested_summary_t34_soc34_only.png", dpi=170)
    plt.close(fig)
    print(f"[ok] wilson+nested summaries written: {out_w}")


if __name__ == "__main__":
    main()
