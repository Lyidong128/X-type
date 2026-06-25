#!/usr/bin/env python3
"""Compute Wilson loops and bulk polarization diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.common_topology import compute_bulk_gap, ensure_dir, h8_k, token, unitary_part


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wilson loop and polarization check.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs/wilson_polarization_check")
    parser.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0")
    parser.add_argument("--nkx", type=int, default=101)
    parser.add_argument("--nky", type=int, default=101)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def occ_subspace(kx: float, ky: float, v: float, t: float, w: float, lm: float, n_occ: int) -> np.ndarray:
    _, vecs = np.linalg.eigh(h8_k(float(kx), float(ky), v=v, t=t, w=w, lm=lm))
    return vecs[:, :n_occ]


def wilson_centers(
    v: float,
    t: float,
    w: float,
    lm: float,
    n_occ: int,
    nkx: int,
    nky: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nky, endpoint=False)

    if direction == "x":
        scan = ky_grid
        loop = kx_grid
    elif direction == "y":
        scan = kx_grid
        loop = ky_grid
    else:
        raise ValueError(f"Unknown direction: {direction}")

    centers = np.zeros((scan.size, n_occ), dtype=float)
    max_unit_err = 0.0

    for j, fixed in enumerate(scan):
        wmat = np.eye(n_occ, dtype=complex)
        for i, k in enumerate(loop):
            k_next = float(loop[(i + 1) % loop.size])
            if direction == "x":
                oa = occ_subspace(float(k), float(fixed), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
                ob = occ_subspace(k_next, float(fixed), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            else:
                oa = occ_subspace(float(fixed), float(k), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
                ob = occ_subspace(float(fixed), k_next, v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            q = unitary_part(oa.conj().T @ ob)
            wmat = q @ wmat
        unit_err = float(np.linalg.norm(wmat.conj().T @ wmat - np.eye(n_occ)))
        max_unit_err = max(max_unit_err, unit_err)
        eigvals = np.linalg.eigvals(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        centers[j, :] = np.sort(nu)

    return scan, centers, float(max_unit_err)


def classify_p(p: float) -> str:
    p = float(np.mod(p, 1.0))
    if min(abs(p - 0.0), abs(p - 1.0)) < 0.05:
        return "trivial_0"
    if abs(p - 0.5) < 0.05:
        return "nontrivial_0p5"
    return "not_quantized"


def polarization_type(px_cls: str, py_cls: str) -> str:
    if px_cls == "nontrivial_0p5" and py_cls == "nontrivial_0p5":
        return "2D_SSH_like_polarized"
    if (px_cls == "nontrivial_0p5") ^ (py_cls == "nontrivial_0p5"):
        return "1D_SSH_like_polarized"
    if px_cls == "trivial_0" and py_cls == "trivial_0":
        return "trivial_polarization"
    return "not_quantized"


def middle_sector_gap_row(row: np.ndarray) -> float:
    arr = np.sort(np.mod(np.array(row, dtype=float), 1.0))
    n = arr.size
    half = n // 2
    best_gap = -1.0
    for cut in range(n):
        start = (cut + 1) % n
        rot = np.r_[arr[start:], arr[:start]]
        rot = rot.copy()
        for i in range(1, n):
            if rot[i] < rot[i - 1]:
                rot[i:] += 1.0
        gap = float(rot[half] - rot[half - 1])
        best_gap = max(best_gap, gap)
    return float(best_gap)


def middle_sector_gap(centers: np.ndarray) -> float:
    gaps = [middle_sector_gap_row(centers[i, :]) for i in range(centers.shape[0])]
    return float(np.min(np.array(gaps, dtype=float)))


def write_centers_csv(path: Path, scan_name: str, scan: np.ndarray, centers: np.ndarray, prefix: str) -> None:
    fields = [scan_name] + [f"{prefix}_{i+1}" for i in range(centers.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, s in enumerate(scan):
            row = {scan_name: float(s)}
            for b in range(centers.shape[1]):
                row[f"{prefix}_{b+1}"] = float(centers[i, b])
            writer.writerow(row)


def plot_wilson_scatter(
    scan: np.ndarray,
    centers: np.ndarray,
    out_png: Path,
    xlabel: str,
    title: str,
) -> None:
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
    out_root = ensure_dir(Path(args.output_root))
    v_list = parse_float_list(args.v_list)

    rows: list[dict] = []

    for v in v_list:
        bulk_gap = float(compute_bulk_gap(v=v, t=args.t, w=args.w, lm=args.lm, nk=61, n_occ=args.n_occ))

        ky, centers_x, err_x = wilson_centers(
            v=v,
            t=args.t,
            w=args.w,
            lm=args.lm,
            n_occ=args.n_occ,
            nkx=args.nkx,
            nky=args.nky,
            direction="x",
        )
        kx, centers_y, err_y = wilson_centers(
            v=v,
            t=args.t,
            w=args.w,
            lm=args.lm,
            n_occ=args.n_occ,
            nkx=args.nkx,
            nky=args.nky,
            direction="y",
        )

        if err_x > 1e-6 or err_y > 1e-6:
            print(f"[warning] v={v:.1f} large Wilson unitarity error: err_x={err_x:.3e}, err_y={err_y:.3e}")

        px = float(np.mod(np.mean(centers_x), 1.0))
        py = float(np.mod(np.mean(centers_y), 1.0))
        px_cls = classify_p(px)
        py_cls = classify_p(py)
        p_type = polarization_type(px_cls, py_cls)

        gap_x = middle_sector_gap(centers_x)
        gap_y = middle_sector_gap(centers_y)
        nested_allowed = bool(gap_x > 0.05 and gap_y > 0.05)

        write_centers_csv(
            out_root / f"wilson_x_centers_v{token(v)}.csv",
            scan_name="ky",
            scan=ky,
            centers=centers_x,
            prefix="nu_x",
        )
        write_centers_csv(
            out_root / f"wilson_y_centers_v{token(v)}.csv",
            scan_name="kx",
            scan=kx,
            centers=centers_y,
            prefix="nu_y",
        )

        plot_wilson_scatter(
            ky,
            centers_x,
            out_root / f"wilson_x_v{token(v)}.png",
            xlabel=r"$k_y$",
            title=f"Wilson x-loop v={v:.1f}, bulk_gap={bulk_gap:.4f}, p_x={px:.4f}",
        )
        plot_wilson_scatter(
            kx,
            centers_y,
            out_root / f"wilson_y_v{token(v)}.png",
            xlabel=r"$k_x$",
            title=f"Wilson y-loop v={v:.1f}, bulk_gap={bulk_gap:.4f}, p_y={py:.4f}",
        )

        rows.append(
            {
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "p_x": float(px),
                "p_y": float(py),
                "p_x_class": px_cls,
                "p_y_class": py_cls,
                "unitarity_error_x": float(err_x),
                "unitarity_error_y": float(err_y),
                "polarization_type": p_type,
                "wannier_gap_x": float(gap_x),
                "wannier_gap_y": float(gap_y),
                "nested_allowed": int(nested_allowed),
            }
        )
        print(
            f"[wilson-pol] v={v:.1f} px={px:.4f} py={py:.4f} "
            f"gap=({gap_x:.4f},{gap_y:.4f}) nested_allowed={nested_allowed}"
        )

    with (out_root / "wilson_polarization_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "v",
            "bulk_gap",
            "p_x",
            "p_y",
            "p_x_class",
            "p_y_class",
            "unitarity_error_x",
            "unitarity_error_y",
            "polarization_type",
            "wannier_gap_x",
            "wannier_gap_y",
            "nested_allowed",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] saved {out_root / 'wilson_polarization_summary.csv'}")


if __name__ == "__main__":
    main()
