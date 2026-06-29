#!/usr/bin/env python3
"""Bulk-gap diagnostics for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_only_common import (
    ModelParams,
    compute_bulk_bands_path,
    compute_bulk_gap,
    ensure_dir,
    output_dirs,
    parse_float_list,
    token,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bulk-gap scan for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--nk", type=int, default=101)
    parser.add_argument("--path-nseg", type=int, default=220)
    return parser.parse_args()


def plot_bulk_band(out_png: Path, v: float, params: ModelParams, nseg: int) -> None:
    bands, xcoords, ticks, labels = compute_bulk_bands_path(v=v, params=params, nseg=nseg)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for b in range(bands.shape[1]):
        ax.plot(xcoords, bands[:, b], linewidth=0.9, color="tab:blue")
    for x in ticks:
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy")
    ax.set_title(f"Bulk band t34-only: v={v:.1f}, t34={params.t34:.1f}, w={params.w:.1f}, lm={params.lm:.1f}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["01_bulk_band"]
    ensure_dir(out_dir)

    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_list = parse_float_list(args.v_list)
    rows = []
    for v in v_list:
        gap, kx, ky = compute_bulk_gap(v=v, params=params, nk=args.nk, n_occ=args.n_occ)
        row = {
            "v": float(v),
            "bulk_gap": float(gap),
            "gap_kx": float(kx),
            "gap_ky": float(ky),
            "n_occ": int(args.n_occ),
            "gap_status": "gap_closed" if gap < 1e-4 else "gapped",
        }
        rows.append(row)
        plot_bulk_band(out_dir / f"bulk_band_v{token(v)}_t34_only.png", v=v, params=params, nseg=args.path_nseg)
        print(
            f"[bulk-t34] v={v:.1f} bulk_gap={gap:.6e} k=({kx:.4f},{ky:.4f}) status={row['gap_status']}"
        )

    summary = out_dir / "bulk_gap_t34_only_summary.csv"
    write_csv(
        summary,
        rows,
        ["v", "bulk_gap", "gap_kx", "gap_ky", "n_occ", "gap_status"],
    )
    write_csv(
        base / "bulk_gap_t34_only_summary.csv",
        rows,
        ["v", "bulk_gap", "gap_kx", "gap_ky", "n_occ", "gap_status"],
    )

    v = np.array([float(r["v"]) for r in rows], dtype=float)
    g = np.array([float(r["bulk_gap"]) for r in rows], dtype=float)
    closed = g < 1e-4
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(v, g, marker="o", linewidth=1.3)
    if np.any(closed):
        ax.scatter(v[closed], g[closed], s=55, color="red", label="gap closing")
    ax.axhline(1e-4, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel("v")
    ax.set_ylabel("bulk gap")
    ax.set_title("bulk gap vs v (t34-only)")
    ax.grid(alpha=0.25)
    if np.any(closed):
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bulk_gap_vs_v_t34_only.png", dpi=180)
    fig.savefig(base / "bulk_gap_vs_v_t34_only.png", dpi=180)
    plt.close(fig)
    print(f"[ok] bulk summary: {summary}")


if __name__ == "__main__":
    main()
