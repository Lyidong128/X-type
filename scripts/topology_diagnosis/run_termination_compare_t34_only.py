#!/usr/bin/env python3
"""Termination A/B comparison for t34-only model."""

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
    build_open_open_hamiltonian,
    cell_density,
    classify_corner_edge,
    compute_bulk_gap,
    diagonalize_open_open,
    output_dirs,
    region_masks,
    state_weights_2d,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Termination comparison for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--lx", type=int, default=10)
    parser.add_argument("--ly", type=int, default=10)
    parser.add_argument("--near-k", type=int, default=120)
    parser.add_argument("--corner-size", type=int, default=2)
    parser.add_argument("--edge-width", type=int, default=2)
    return parser.parse_args()


def run_case(
    base_dir: Path,
    termination: str,
    v: float,
    params: ModelParams,
    lx: int,
    ly: int,
    near_k: int,
    corner_size: int,
    edge_width: int,
) -> tuple[list[dict], Path]:
    bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=81, n_occ=4)
    in_gap_window = max(1e-4, 0.4 * max(bulk_gap, 1e-6))
    h = build_open_open_hamiltonian(lx=lx, ly=ly, v=v, params=params, termination=termination)
    evals, vecs = diagonalize_open_open(h, near_k=near_k)
    corner, edge, bulk = region_masks(lx=lx, ly=ly, corner_size=corner_size, edge_width=edge_width)

    rows = []
    idxs = np.where(np.abs(evals) <= in_gap_window)[0]
    if idxs.size == 0:
        idxs = np.argsort(np.abs(evals))[:12]
    for idx in idxs:
        rho = cell_density(vecs[:, idx], lx=lx, ly=ly)
        wc, we, wb = state_weights_2d(rho, corner=corner, edge=edge, bulk=bulk)
        cls = classify_corner_edge(wc, we)
        rows.append(
            {
                "v": float(v),
                "termination": termination,
                "Lx": int(lx),
                "Ly": int(ly),
                "state_index": int(idx),
                "energy": float(evals[idx]),
                "abs_energy": float(abs(evals[idx])),
                "W_corner": float(wc),
                "W_edge": float(we),
                "W_bulk": float(wb),
                "classification": cls,
                "bulk_gap_used": float(bulk_gap),
                "in_gap_window": float(in_gap_window),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: float(r["abs_energy"]))
    best = rows_sorted[0]
    idx = int(best["state_index"])
    rho = cell_density(vecs[:, idx], lx=lx, ly=ly)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(np.arange(evals.size), evals, marker="o", linestyle="none", markersize=3.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(+in_gap_window, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(-in_gap_window, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(f"termination {termination}: finite spectrum v={v:.1f}")
    ax.set_xlabel("state index")
    ax.set_ylabel("energy")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    spec_png = base_dir / f"termination_{termination}_finite_spectrum_v0p5_t34_only.png"
    fig.savefig(spec_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(rho, origin="lower", cmap="magma")
    ax.set_title(
        f"termination {termination} state {idx}\n"
        f"Wc={best['W_corner']:.3f}, We={best['W_edge']:.3f}, cls={best['classification']}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="rho")
    fig.tight_layout()
    wf_png = base_dir / f"termination_{termination}_wf_state_{idx}_t34_only.png"
    fig.savefig(wf_png, dpi=180)
    plt.close(fig)
    return rows, spec_png


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["05_termination"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)

    rows_a, spec_a = run_case(
        out_dir,
        "A",
        args.v,
        params,
        args.lx,
        args.ly,
        args.near_k,
        args.corner_size,
        args.edge_width,
    )
    rows_b, spec_b = run_case(
        out_dir,
        "B",
        args.v,
        params,
        args.lx,
        args.ly,
        args.near_k,
        args.corner_size,
        args.edge_width,
    )

    write_csv(
        out_dir / "termination_A_corner_summary_t34_only.csv",
        rows_a,
        [
            "v",
            "termination",
            "Lx",
            "Ly",
            "state_index",
            "energy",
            "abs_energy",
            "W_corner",
            "W_edge",
            "W_bulk",
            "classification",
            "bulk_gap_used",
            "in_gap_window",
        ],
    )
    write_csv(
        out_dir / "termination_B_corner_summary_t34_only.csv",
        rows_b,
        [
            "v",
            "termination",
            "Lx",
            "Ly",
            "state_index",
            "energy",
            "abs_energy",
            "W_corner",
            "W_edge",
            "W_bulk",
            "classification",
            "bulk_gap_used",
            "in_gap_window",
        ],
    )
    # root copies
    (base / "termination_A_corner_summary_t34_only.csv").write_bytes(
        (out_dir / "termination_A_corner_summary_t34_only.csv").read_bytes()
    )
    (base / "termination_B_corner_summary_t34_only.csv").write_bytes(
        (out_dir / "termination_B_corner_summary_t34_only.csv").read_bytes()
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].imshow(plt.imread(spec_a))
    axes[0].set_title("termination A")
    axes[0].axis("off")
    axes[1].imshow(plt.imread(spec_b))
    axes[1].set_title("termination B")
    axes[1].axis("off")
    fig.suptitle("termination comparison t34-only (v=0.5)")
    fig.tight_layout()
    fig.savefig(out_dir / "termination_compare_montage_t34_only.png", dpi=180)
    fig.savefig(base / "termination_compare_montage_t34_only.png", dpi=180)
    plt.close(fig)
    print("[termination-t34] completed A/B comparison at v=0.5")


if __name__ == "__main__":
    main()
