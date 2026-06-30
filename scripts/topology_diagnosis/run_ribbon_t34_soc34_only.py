#!/usr/bin/env python3
"""Ribbon spectra scan for strict t34 + soc34-only model."""

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

from scripts.topology_diagnosis.t34_soc34_only_common import (  # noqa: E402
    ModelParams,
    compute_bulk_gap,
    output_dirs,
    parse_float_list,
    ribbon_spectrum_with_edge_weight,
    token,
    write_csv,
    write_model_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ribbon scan for strict t34+soc34-only model.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only")
    p.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--nk", type=int, default=121)
    p.add_argument("--edge-cells", type=int, default=3)
    p.add_argument("--edge-threshold", type=float, default=0.6)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_model_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out = dirs["03_ribbon"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_list = parse_float_list(args.v_list)

    rows = []
    panels = []
    for v in v_list:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=81, n_occ=4)
        ky, evals, edge_w = ribbon_spectrum_with_edge_weight(
            v=v, params=params, nx=args.nx, nk=args.nk, edge_cells=args.edge_cells
        )
        edge_mask = edge_w >= args.edge_threshold
        in_gap = np.abs(evals) <= max(1e-4, 0.5 * bulk_gap)
        n_in_gap = int(np.sum(edge_mask & in_gap))
        max_edge_w = float(np.max(edge_w)) if edge_w.size else 0.0
        min_abs_edge = float(np.min(np.abs(evals[edge_mask]))) if np.any(edge_mask) else float("nan")
        comment = "finite_size_sensitive_small_gap" if bulk_gap < 0.05 else "ok"
        rows.append(
            {
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "num_in_gap_edge_states": int(n_in_gap),
                "max_edge_weight": float(max_edge_w),
                "min_abs_edge_energy": float(min_abs_edge),
                "edge_state_present": int(n_in_gap > 0),
                "comment": comment,
            }
        )

        fig, ax = plt.subplots(figsize=(6.9, 4.7))
        x = np.repeat(ky / np.pi, evals.shape[1])
        y = evals.reshape(-1)
        c = edge_w.reshape(-1)
        sc = ax.scatter(x, y, c=c, s=4.5, cmap="viridis", edgecolors="none", vmin=0.0, vmax=1.0)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$k_y/\pi$")
        ax.set_ylabel("Energy")
        ax.set_title(f"ribbon v={v:.1f}, gap={bulk_gap:.4f}, edge_in_gap={n_in_gap}")
        ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label="edge weight")
        fig.tight_layout()
        png = out / f"ribbon_v{token(v)}_t34_soc34_only.png"
        fig.savefig(png, dpi=170)
        plt.close(fig)
        panels.append(png)
        print(f"[ribbon-soc34] v={v:.1f} gap={bulk_gap:.4f} edge_in_gap={n_in_gap}")

    csv_path = out / "ribbon_t34_soc34_only.csv"
    fields = ["v", "bulk_gap", "num_in_gap_edge_states", "max_edge_weight", "min_abs_edge_energy", "edge_state_present", "comment"]
    write_csv(csv_path, rows, fields)
    write_csv(base / "ribbon_t34_soc34_only.csv", rows, fields)

    cols = 3
    n = len(panels)
    rr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rr, cols, figsize=(4.9 * cols, 3.7 * rr))
    axes = np.array(axes).reshape(rr, cols)
    for i in range(rr * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(plt.imread(panels[i]))
            ax.axis("off")
            ax.set_title(panels[i].stem)
        else:
            ax.axis("off")
    fig.suptitle("ribbon scan (t34+soc34-only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "ribbon_scan_montage_t34_soc34_only.png", dpi=170)
    fig.savefig(base / "ribbon_scan_montage_t34_soc34_only.png", dpi=170)
    plt.close(fig)
    print(f"[ok] ribbon summary: {csv_path}")


if __name__ == "__main__":
    main()
