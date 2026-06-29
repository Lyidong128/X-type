#!/usr/bin/env python3
"""Wilson loop and polarization diagnostics for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.topology_diagnosis.t34_only_common import (
    ModelParams,
    compute_bulk_gap,
    compute_wilson_centers_direction,
    middle_sector_gap,
    output_dirs,
    parse_float_list,
    token,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wilson polarization check for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--nkx", type=int, default=101)
    parser.add_argument("--nky", type=int, default=101)
    parser.add_argument("--n-occ", type=int, default=4)
    return parser.parse_args()


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


def write_centers(path: Path, scan_name: str, scan: np.ndarray, centers: np.ndarray, prefix: str) -> None:
    rows = []
    for i, s in enumerate(scan):
        row = {scan_name: float(s)}
        for b in range(centers.shape[1]):
            row[f"{prefix}_{b+1}"] = float(centers[i, b])
        rows.append(row)
    fields = [scan_name] + [f"{prefix}_{i+1}" for i in range(centers.shape[1])]
    write_csv(path, rows, fields)


def plot_wilson(scan: np.ndarray, centers: np.ndarray, out_png: Path, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for b in range(centers.shape[1]):
        ax.scatter(scan, centers[:, b], s=7, alpha=0.75)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Wannier center")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["06_wilson_polarization"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)

    rows = []
    for v in parse_float_list(args.v_list):
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=81, n_occ=args.n_occ)
        ky, c_x, _, _, err_x = compute_wilson_centers_direction(
            v=v,
            params=params,
            n_occ=args.n_occ,
            nkx=args.nkx,
            nky=args.nky,
            direction="x",
        )
        kx, c_y, _, _, err_y = compute_wilson_centers_direction(
            v=v,
            params=params,
            n_occ=args.n_occ,
            nkx=args.nkx,
            nky=args.nky,
            direction="y",
        )
        px = float(np.mod(np.mean(c_x), 1.0))
        py = float(np.mod(np.mean(c_y), 1.0))
        px_cls = classify_p(px)
        py_cls = classify_p(py)
        ptype = polarization_type(px_cls, py_cls)
        gap_x = middle_sector_gap(c_x)
        gap_y = middle_sector_gap(c_y)
        nested_allowed = int(gap_x > 0.05 and gap_y > 0.05)
        rows.append(
            {
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "p_x": float(px),
                "p_y": float(py),
                "p_x_class": px_cls,
                "p_y_class": py_cls,
                "polarization_type": ptype,
                "wannier_gap_x": float(gap_x),
                "wannier_gap_y": float(gap_y),
                "nested_allowed": int(nested_allowed),
                "unitarity_error_x": float(err_x),
                "unitarity_error_y": float(err_y),
            }
        )

        wx_csv = out_dir / f"wilson_x_centers_v{token(v)}_t34_only.csv"
        wy_csv = out_dir / f"wilson_y_centers_v{token(v)}_t34_only.csv"
        write_centers(wx_csv, "ky", ky, c_x, "nu_x")
        write_centers(wy_csv, "kx", kx, c_y, "nu_y")

        wx_png = out_dir / f"wilson_x_v{token(v)}_t34_only.png"
        wy_png = out_dir / f"wilson_y_v{token(v)}_t34_only.png"
        plot_wilson(
            ky,
            c_x,
            wx_png,
            xlabel=r"$k_y$",
            title=f"Wilson x-loop t34-only v={v:.1f}, gap={bulk_gap:.4f}, p_x={px:.4f}",
        )
        plot_wilson(
            kx,
            c_y,
            wy_png,
            xlabel=r"$k_x$",
            title=f"Wilson y-loop t34-only v={v:.1f}, gap={bulk_gap:.4f}, p_y={py:.4f}",
        )
        if abs(v - 0.5) < 1e-9 or abs(v - 0.8) < 1e-9:
            (base / wx_png.name).write_bytes(wx_png.read_bytes())
            (base / wy_png.name).write_bytes(wy_png.read_bytes())

        print(
            f"[wilson-t34] v={v:.1f} p=({px:.4f},{py:.4f}) "
            f"gaps=({gap_x:.4f},{gap_y:.4f}) nested_allowed={bool(nested_allowed)}"
        )

    fields = [
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
    write_csv(out_dir / "wilson_polarization_summary_t34_only.csv", rows, fields)
    write_csv(base / "wilson_polarization_summary_t34_only.csv", rows, fields)

    # compact summary figure (requested name)
    v = np.array([float(r["v"]) for r in rows], dtype=float)
    bulk = np.array([float(r["bulk_gap"]) for r in rows], dtype=float)
    px = np.array([float(r["p_x"]) for r in rows], dtype=float)
    py = np.array([float(r["p_y"]) for r in rows], dtype=float)
    gx = np.array([float(r["wannier_gap_x"]) for r in rows], dtype=float)
    gy = np.array([float(r["wannier_gap_y"]) for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    axes[0, 0].plot(v, bulk, marker="o")
    axes[0, 0].set_title("bulk gap vs v")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(v, px, marker="o", label="p_x")
    axes[0, 1].plot(v, py, marker="s", label="p_y")
    axes[0, 1].axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    axes[0, 1].set_title("polarization vs v")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(v, gx, marker="o", label="wannier_gap_x")
    axes[1, 0].plot(v, gy, marker="s", label="wannier_gap_y")
    axes[1, 0].axhline(0.05, color="black", linestyle="--", linewidth=0.9)
    axes[1, 0].set_title("Wannier sector gaps")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    nested = np.array([int(r["nested_allowed"]) for r in rows], dtype=int)
    axes[1, 1].plot(v, nested, marker="o")
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].set_title("nested_allowed")
    axes[1, 1].grid(alpha=0.25)
    for ax in axes.reshape(-1):
        ax.set_xlabel("v")
    fig.tight_layout()
    fig.savefig(out_dir / "hoti_wilson_summary_t34_only.png", dpi=180)
    fig.savefig(base / "hoti_wilson_summary_t34_only.png", dpi=180)
    plt.close(fig)
    print(f"[ok] wilson polarization summary: {out_dir / 'wilson_polarization_summary_t34_only.csv'}")


if __name__ == "__main__":
    main()
