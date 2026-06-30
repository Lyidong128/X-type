#!/usr/bin/env python3
"""Open-open corner/edge diagnostics for strict t34 + soc34-only model."""

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
    build_open_open_hamiltonian,
    cell_density,
    classify_corner_edge,
    compute_bulk_gap,
    diagonalize_open_open,
    output_dirs,
    parse_float_list,
    region_masks,
    state_weights_2d,
    token,
    write_csv,
    write_model_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-open scan for strict t34+soc34-only model.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only")
    p.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    p.add_argument("--lx", type=int, default=10)
    p.add_argument("--ly", type=int, default=10)
    p.add_argument("--fallback-l", type=int, default=8)
    p.add_argument("--near-k", type=int, default=120)
    p.add_argument("--corner-size", type=int, default=2)
    p.add_argument("--edge-width", type=int, default=2)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_model_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out = dirs["05_open_open"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_list = parse_float_list(args.v_list)

    lx, ly = args.lx, args.ly
    used_fallback = False
    try:
        _ = build_open_open_hamiltonian(lx=lx, ly=ly, v=v_list[0], params=params, termination="A")
    except Exception:
        lx = ly = args.fallback_l
        used_fallback = True

    corner, edge, bulk = region_masks(lx=lx, ly=ly, corner_size=args.corner_size, edge_width=args.edge_width)
    rows = []
    montage_paths = []
    for v in v_list:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=81, n_occ=4)
        in_gap_window = max(1e-4, 0.4 * max(bulk_gap, 1e-6))
        h = build_open_open_hamiltonian(lx=lx, ly=ly, v=v, params=params, termination="A")
        evals, vecs = diagonalize_open_open(h, near_k=args.near_k)
        idxs = np.where(np.abs(evals) <= in_gap_window)[0]
        if idxs.size == 0:
            idxs = np.argsort(np.abs(evals))[:12]
        for idx in idxs:
            rho = cell_density(vecs[:, idx], lx=lx, ly=ly)
            wc, we, wb = state_weights_2d(rho, corner=corner, edge=edge, bulk=bulk)
            rows.append(
                {
                    "v": float(v),
                    "Lx": int(lx),
                    "Ly": int(ly),
                    "state_index": int(idx),
                    "energy": float(evals[idx]),
                    "abs_energy": float(abs(evals[idx])),
                    "W_corner": float(wc),
                    "W_edge": float(we),
                    "W_bulk": float(wb),
                    "classification": classify_corner_edge(wc, we),
                }
            )
        fig, ax = plt.subplots(figsize=(6.7, 4.2))
        ax.plot(np.arange(evals.size), evals, marker="o", linestyle="none", markersize=2.8)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(in_gap_window, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(-in_gap_window, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("state index")
        ax.set_ylabel("energy")
        ax.set_title(f"open-open spectrum v={v:.1f}, L=({lx},{ly}), win={in_gap_window:.3f}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        png = out / f"open_open_spectrum_v{token(v)}_t34_soc34_only.png"
        fig.savefig(png, dpi=170)
        plt.close(fig)
        montage_paths.append(png)
        print(f"[open-open-soc34] v={v:.1f} states={len(idxs)} window={in_gap_window:.4f}")

    csv_path = out / "open_open_t34_soc34_only.csv"
    fields = ["v", "Lx", "Ly", "state_index", "energy", "abs_energy", "W_corner", "W_edge", "W_bulk", "classification"]
    write_csv(csv_path, rows, fields)
    write_csv(base / "open_open_t34_soc34_only.csv", rows, fields)

    cols = 3
    n = len(montage_paths)
    rr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rr, cols, figsize=(4.9 * cols, 3.7 * rr))
    axes = np.array(axes).reshape(rr, cols)
    for i in range(rr * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(plt.imread(montage_paths[i]))
            ax.axis("off")
            ax.set_title(montage_paths[i].stem)
        else:
            ax.axis("off")
    fig.suptitle("open-open scan (t34+soc34-only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "open_open_scan_montage_t34_soc34_only.png", dpi=170)
    fig.savefig(base / "open_open_scan_montage_t34_soc34_only.png", dpi=170)
    plt.close(fig)

    (out / "open_open_runtime_note_t34_soc34_only.txt").write_text(
        f"used_fallback={int(used_fallback)}\nLx={lx}\nLy={ly}\n",
        encoding="utf-8",
    )
    print(f"[ok] open-open summary: {csv_path}")


if __name__ == "__main__":
    main()
