#!/usr/bin/env python3
"""Open-open checks at representative fine points for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_only_common import (  # noqa: E402
    ModelParams,
    build_open_open_hamiltonian,
    cell_density,
    classify_corner_edge,
    compute_bulk_gap,
    diagonalize_open_open,
    region_masks,
    state_weights_2d,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open-open representative fine points (t34-only).")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    p.add_argument("--v-points", default="0.80,0.90,0.95,1.00,1.05,1.10")
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


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def vtok2(v: float) -> str:
    return f"{v:.2f}".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    out = base / "fine_scan_0p8_1p2" / "05_open_open"
    out.mkdir(parents=True, exist_ok=True)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_points = parse_float_list(args.v_points)

    lx, ly = args.lx, args.ly
    used_fallback = False
    try:
        _ = build_open_open_hamiltonian(lx=lx, ly=ly, v=v_points[0], params=params, termination="A")
    except Exception:
        lx = ly = args.fallback_l
        used_fallback = True

    corner, edge, bulk = region_masks(lx=lx, ly=ly, corner_size=args.corner_size, edge_width=args.edge_width)
    rows = []
    panel_paths = []
    for v in v_points:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=101, n_occ=4)
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

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot(np.arange(evals.size), evals, marker="o", linestyle="none", markersize=2.8)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(in_gap_window, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(-in_gap_window, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("state index")
        ax.set_ylabel("energy")
        ax.set_title(f"open-open spectrum v={v:.2f}, L=({lx},{ly}), win={in_gap_window:.4f}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        png = out / f"open_open_spectrum_v{vtok2(v)}_t34_only.png"
        fig.savefig(png, dpi=180)
        plt.close(fig)
        panel_paths.append(png)

        rep = int(np.argsort(np.abs(evals))[0])
        rho = cell_density(vecs[:, rep], lx=lx, ly=ly)
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(rho, origin="lower", cmap="magma")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"v={v:.2f} representative state={rep}")
        fig.colorbar(im, ax=ax, label="rho")
        fig.tight_layout()
        wf = out / f"open_open_wf_v{vtok2(v)}_t34_only.png"
        fig.savefig(wf, dpi=180)
        plt.close(fig)
        print(f"[open-open-fine-t34] v={v:.2f} states={len(idxs)} window={in_gap_window:.5f}")

    csv_path = out / "open_open_fine_points_summary_t34_only.csv"
    write_csv(
        csv_path,
        rows,
        ["v", "Lx", "Ly", "state_index", "energy", "abs_energy", "W_corner", "W_edge", "W_bulk", "classification"],
    )
    (base / "fine_scan_0p8_1p2" / "open_open_fine_points_summary_t34_only.csv").write_bytes(csv_path.read_bytes())

    cols = 3
    n = len(panel_paths)
    rr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rr, cols, figsize=(5.2 * cols, 3.9 * rr))
    axes = np.array(axes).reshape(rr, cols)
    for i in range(rr * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(plt.imread(panel_paths[i]))
            ax.axis("off")
            ax.set_title(panel_paths[i].stem)
        else:
            ax.axis("off")
    fig.suptitle("Open-open representative points (t34-only)", fontsize=12)
    fig.tight_layout()
    montage = out / "open_open_fine_points_montage_t34_only.png"
    fig.savefig(montage, dpi=180)
    fig.savefig(base / "fine_scan_0p8_1p2" / montage.name, dpi=180)
    plt.close(fig)

    (out / "open_open_fine_runtime_note_t34_only.txt").write_text(
        f"used_fallback={int(used_fallback)}\nLx={lx}\nLy={ly}\n",
        encoding="utf-8",
    )
    print(f"[ok] open-open fine summary: {csv_path}")


if __name__ == "__main__":
    main()
