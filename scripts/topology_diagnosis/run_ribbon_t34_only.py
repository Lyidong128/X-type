#!/usr/bin/env python3
"""Ribbon spectrum scan for t34-only model."""

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
    compute_bulk_gap,
    output_dirs,
    parse_float_list,
    ribbon_spectrum_with_edge_weight,
    token,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ribbon diagnostics for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.2,0.4,0.5,0.6,0.8,1.0,1.2")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--bulk-nk", type=int, default=101)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--nk", type=int, default=161)
    parser.add_argument("--edge-cells", type=int, default=3)
    parser.add_argument("--edge-threshold", type=float, default=0.6)
    return parser.parse_args()


def plot_ribbon(out_png: Path, ky: np.ndarray, evals: np.ndarray, edge_w: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.9))
    x = np.repeat(ky / np.pi, evals.shape[1])
    y = evals.reshape(-1)
    c = edge_w.reshape(-1)
    sc = ax.scatter(x, y, c=c, s=5, cmap="viridis", edgecolors="none", vmin=0.0, vmax=1.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="edge weight")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["02_ribbon"]

    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_list = parse_float_list(args.v_list)
    rows = []
    panel_paths = []

    for v in v_list:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=args.bulk_nk, n_occ=args.n_occ)
        ky, evals, edge_w = ribbon_spectrum_with_edge_weight(
            v=v,
            params=params,
            nx=args.nx,
            nk=args.nk,
            edge_cells=args.edge_cells,
        )
        edge_mask = edge_w >= args.edge_threshold
        in_gap = np.abs(evals) <= max(1e-4, 0.5 * bulk_gap)
        in_gap_edge = edge_mask & in_gap
        if np.any(edge_mask):
            min_abs_edge = float(np.min(np.abs(evals[edge_mask])))
            max_edge_w = float(np.max(edge_w[edge_mask]))
        else:
            min_abs_edge = float("nan")
            max_edge_w = 0.0
        num_in_gap_edge = int(np.sum(in_gap_edge))
        row = {
            "v": float(v),
            "bulk_gap": float(bulk_gap),
            "min_abs_edge_energy": float(min_abs_edge),
            "num_in_gap_edge_states": int(num_in_gap_edge),
            "max_edge_weight": float(max_edge_w),
            "edge_state_present": int(num_in_gap_edge > 0),
        }
        rows.append(row)

        out_png = out_dir / f"ribbon_v{token(v)}_t34_only.png"
        plot_ribbon(
            out_png,
            ky,
            evals,
            edge_w,
            title=f"t34-only ribbon v={v:.1f}, bulk_gap={bulk_gap:.4f}",
        )
        panel_paths.append(out_png)
        (base / f"ribbon_v{token(v)}_t34_only.png").write_bytes(out_png.read_bytes())
        print(
            f"[ribbon-t34] v={v:.1f} bulk_gap={bulk_gap:.5f} "
            f"in_gap_edge={num_in_gap_edge} max_edge_w={max_edge_w:.3f}"
        )

    write_csv(
        out_dir / "ribbon_t34_only_summary.csv",
        rows,
        ["v", "bulk_gap", "min_abs_edge_energy", "num_in_gap_edge_states", "max_edge_weight", "edge_state_present"],
    )
    write_csv(
        base / "ribbon_t34_only_summary.csv",
        rows,
        ["v", "bulk_gap", "min_abs_edge_energy", "num_in_gap_edge_states", "max_edge_weight", "edge_state_present"],
    )

    # montage
    n = len(panel_paths)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.0 * cols, 3.8 * rows_n))
    axes = np.array(axes).reshape(rows_n, cols)
    for i in range(rows_n * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            img = plt.imread(panel_paths[i])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(panel_paths[i].stem.replace("_t34_only", ""))
        else:
            ax.axis("off")
    fig.suptitle("Ribbon v scan (t34-only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "ribbon_v_scan_montage_t34_only.png", dpi=180)
    fig.savefig(base / "ribbon_v_scan_montage_t34_only.png", dpi=180)
    plt.close(fig)
    print(f"[ok] ribbon summary: {out_dir / 'ribbon_t34_only_summary.csv'}")


if __name__ == "__main__":
    main()
