#!/usr/bin/env python3
"""Scan M-point and global direct gaps versus v for lm=0.1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.M_point_spin_chern_analysis.common import (
    build_scan_values,
    direct_gap_at_k,
    direct_gap_global,
    ensure_dir,
    load_model,
    m_point_cartesian,
    m_point_from_model,
    set_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M-point/global gap scan.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/M_point_spin_chern_analysis")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--dv", type=float, default=0.001)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    model = load_model(args.model_file)
    mpt = m_point_from_model(model)
    mkx, mky = m_point_cartesian()

    dv = 0.002 if args.quick else float(args.dv)
    nk_global = 31 if args.quick else 61
    v_values = sorted(
        set(build_scan_values(0.55, 0.65, dv) + build_scan_values(1.08, 1.10, dv))
    )

    m_rows = []
    g_rows = []
    for v in v_values:
        set_params(model, v=v, lm=args.lm, t=args.t, w=args.w, j=0.0)
        evals = np.linalg.eigvalsh(model.Hxtype(mpt))
        valence_index = 3
        conduction_index = 4
        eval_v = float(np.real(evals[valence_index]))
        eval_c = float(np.real(evals[conduction_index]))
        delta_m = float(eval_c - eval_v)
        m_rows.append(
            {
                "v": float(v),
                "lm": float(args.lm),
                "t": float(args.t),
                "w": float(args.w),
                "E_valence_M": eval_v,
                "E_conduction_M": eval_c,
                "Delta_M": delta_m,
                "valence_band_index": int(valence_index),
                "conduction_band_index": int(conduction_index),
            }
        )

        delta_g, kx_min, ky_min = direct_gap_global(model, nk=nk_global, n_occ=4)
        dist_m = float(np.sqrt((kx_min - mkx) ** 2 + (ky_min - mky) ** 2))
        g_rows.append(
            {
                "v": float(v),
                "lm": float(args.lm),
                "t": float(args.t),
                "w": float(args.w),
                "Delta_global": float(delta_g),
                "kx_min": float(kx_min),
                "ky_min": float(ky_min),
                "distance_to_M": dist_m,
            }
        )

    m_csv = out_root / "M_gap_scan.csv"
    with m_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "lm",
                "t",
                "w",
                "E_valence_M",
                "E_conduction_M",
                "Delta_M",
                "valence_band_index",
                "conduction_band_index",
            ],
        )
        writer.writeheader()
        for row in m_rows:
            writer.writerow(row)

    g_csv = out_root / "global_gap_scan.csv"
    with g_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "lm",
                "t",
                "w",
                "Delta_global",
                "kx_min",
                "ky_min",
                "distance_to_M",
            ],
        )
        writer.writeheader()
        for row in g_rows:
            writer.writerow(row)

    # Plot 1: M gap versus v with minimum marker.
    v_arr = np.array([float(r["v"]) for r in m_rows], dtype=float)
    dm_arr = np.array([float(r["Delta_M"]) for r in m_rows], dtype=float)
    idx_min = int(np.argmin(dm_arr))
    vc = float(v_arr[idx_min])
    dmin = float(dm_arr[idx_min])

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(v_arr, dm_arr, linewidth=1.2, label=r"$\Delta_M$")
    ax.scatter([vc], [dmin], color="red", s=30, label=f"min at v={vc:.6f}")
    ax.set_xlabel("v")
    ax.set_ylabel(r"$\Delta_M$")
    ax.set_title("M-point direct gap vs v")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "M_gap_vs_v.png", dpi=180)
    plt.close(fig)

    # Plot 2: Delta_M vs Delta_global for both scan windows.
    dg_arr = np.array([float(r["Delta_global"]) for r in g_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    scatter = ax.scatter(dm_arr, dg_arr, c=v_arr, s=18, cmap="viridis")
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("v")
    lo = float(min(np.min(dm_arr), np.min(dg_arr)))
    hi = float(max(np.max(dm_arr), np.max(dg_arr)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y=x")
    ax.set_xlabel(r"$\Delta_M$")
    ax.set_ylabel(r"$\Delta_{global}$")
    ax.set_title("M-point gap vs global minimum gap")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "M_gap_vs_global_gap.png", dpi=180)
    plt.close(fig)

    # Plot 3: distance from global gap-min k to M.
    dist_arr = np.array([float(r["distance_to_M"]) for r in g_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(v_arr, dist_arr, linewidth=1.2)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("v")
    ax.set_ylabel(r"$|k_{min}-M|$")
    ax.set_title("Distance from global gap-min momentum to M")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_root / "kmin_distance_to_M_vs_v.png", dpi=180)
    plt.close(fig)

    # Summary text.
    global_idx_min = int(np.argmin(dg_arr))
    summary_lines = [
        "M-point gap scan summary",
        f"v_c_from_Delta_M_min={vc:.6f}",
        f"Delta_M_min={dmin:.6e}",
        f"v_at_Delta_global_min={v_arr[global_idx_min]:.6f}",
        f"Delta_global_min={dg_arr[global_idx_min]:.6e}",
        f"distance_to_M_at_global_min={dist_arr[global_idx_min]:.6e}",
    ]
    if abs(vc - 0.6) <= 0.02 or dmin < 1e-3:
        summary_lines.append("Delta_M is near-closing around v≈0.6.")
    else:
        summary_lines.append("No strong near-zero Delta_M around v≈0.6 under current discretization.")
    (out_root / "M_gap_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[ok] m_csv={m_csv}")
    print(f"[ok] g_csv={g_csv}")
    print(f"[ok] vc={vc:.6f}")


if __name__ == "__main__":
    main()
