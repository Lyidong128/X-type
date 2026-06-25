#!/usr/bin/env python3
"""Compare two open-open boundary terminations at v=0.5."""

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

from scripts.topology_diagnosis.common_topology import (
    build_open_open_hamiltonian,
    cell_density,
    choose_in_gap_indices,
    classify_state,
    compute_bulk_gap,
    diagonalize_open_open,
    ensure_dir,
    region_masks,
    state_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Termination comparison for open-open finite system.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs")
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--lx", type=int, default=10)
    parser.add_argument("--ly", type=int, default=10)
    parser.add_argument("--fallback-lx", type=int, default=8)
    parser.add_argument("--fallback-ly", type=int, default=8)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--bulk-nk", type=int, default=61)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrum(evals: np.ndarray, idx: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.arange(evals.size)
    ax.plot(x, evals, ".", color="tab:blue", markersize=3.0)
    if idx.size > 0:
        ax.plot(idx, evals[idx], "o", color="tab:red", markersize=4.0, label="in-gap")
        ax.legend()
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("state index")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_wf(rho: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(rho, origin="lower", cmap="magma")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="Probability density")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def analyze_termination(
    out_root: Path,
    term: str,
    v: float,
    lx: int,
    ly: int,
    t: float,
    w: float,
    lm: float,
    bulk_gap: float,
) -> tuple[list[dict], dict | None]:
    corner_mask, edge_mask, bulk_mask = region_masks(lx=lx, ly=ly, corner_size=2, edge_width=2)
    h = build_open_open_hamiltonian(lx=lx, ly=ly, v=v, t=t, w=w, lm=lm, termination=term)
    evals, evecs = diagonalize_open_open(h, num_eigs=40)
    idx, gap_window, status = choose_in_gap_indices(evals=evals, bulk_gap=bulk_gap)
    plot_spectrum(
        evals=evals,
        idx=idx,
        out_png=out_root / f"termination_{term}_finite_spectrum_v0p5.png",
        title=f"termination {term}, v={v:.1f}, window={gap_window:.3e}, status={status}",
    )

    rows: list[dict] = []
    for i in idx:
        vec = evecs[:, int(i)]
        rho = cell_density(vec, lx=lx, ly=ly)
        wc, we, wb = state_weights(rho, corner=corner_mask, edge=edge_mask, bulk=bulk_mask)
        pos = np.unravel_index(int(np.argmax(rho)), rho.shape)
        cls = classify_state(w_corner=wc, w_edge=we)
        row = {
            "termination": term,
            "v": float(v),
            "Lx": int(lx),
            "Ly": int(ly),
            "state_index": int(i),
            "energy": float(evals[int(i)]),
            "abs_energy": float(abs(evals[int(i)])),
            "W_corner": wc,
            "W_edge": we,
            "W_bulk": wb,
            "max_position_x": int(pos[1]),
            "max_position_y": int(pos[0]),
            "classification": cls,
            "in_gap_window": float(gap_window),
            "bulk_gap_used": float(bulk_gap),
            "window_status": status,
        }
        rows.append(row)
        plot_wf(
            rho,
            out_png=out_root / f"termination_{term}_wf_state_{int(i):03d}.png",
            title=f"term {term}, idx={int(i)}, E={float(evals[int(i)]):.3e}\nWc={wc:.3f}, We={we:.3f}, Wb={wb:.3f}",
        )

    rep = min(rows, key=lambda r: float(r["abs_energy"])) if rows else None
    n_corner = sum(1 for r in rows if r["classification"] == "corner_state")
    n_edge = sum(1 for r in rows if r["classification"] == "edge_state")
    print(
        f"[termination {term}] status={status} candidates={len(rows)} "
        f"corner={n_corner} edge={n_edge} bulk_gap={bulk_gap:.4e}"
    )
    return rows, rep


def make_compare_montage(out_png: Path, out_root: Path, rep_a: dict | None, rep_b: dict | None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), squeeze=False)
    specs = [
        ("A", axes[0][0]),
        ("B", axes[1][0]),
    ]
    for term, ax in specs:
        p = out_root / f"termination_{term}_finite_spectrum_v0p5.png"
        if p.exists():
            ax.imshow(plt.imread(p))
            ax.axis("off")
            ax.set_title(f"termination {term} spectrum")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    for rep, ax, term in [(rep_a, axes[0][1], "A"), (rep_b, axes[1][1], "B")]:
        if rep is None:
            ax.text(0.5, 0.5, f"termination {term}\nno in-gap state", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        p = out_root / f"termination_{term}_wf_state_{int(rep['state_index']):03d}.png"
        if p.exists():
            ax.imshow(plt.imread(p))
            ax.axis("off")
            ax.set_title(
                f"term {term} representative\n"
                f"Wc={float(rep['W_corner']):.3f}, We={float(rep['W_edge']):.3f}, Wb={float(rep['W_bulk']):.3f}"
            )
        else:
            ax.text(0.5, 0.5, "missing wf", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    fig.suptitle("Termination comparison (v=0.5)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))

    lx = int(args.fallback_lx) if (int(args.lx) * int(args.ly) * 8 > 1200) else int(args.lx)
    ly = int(args.fallback_ly) if (int(args.lx) * int(args.ly) * 8 > 1200) else int(args.ly)
    if args.quick:
        lx = min(lx, 8)
        ly = min(ly, 8)

    bulk_gap = compute_bulk_gap(v=args.v, t=args.t, w=args.w, lm=args.lm, nk=(31 if args.quick else args.bulk_nk))
    rows_a, rep_a = analyze_termination(out_root, "A", args.v, lx, ly, args.t, args.w, args.lm, bulk_gap)
    rows_b, rep_b = analyze_termination(out_root, "B", args.v, lx, ly, args.t, args.w, args.lm, bulk_gap)

    fields = [
        "termination",
        "v",
        "Lx",
        "Ly",
        "state_index",
        "energy",
        "abs_energy",
        "W_corner",
        "W_edge",
        "W_bulk",
        "max_position_x",
        "max_position_y",
        "classification",
        "in_gap_window",
        "bulk_gap_used",
        "window_status",
    ]
    write_csv(out_root / "termination_A_corner_summary.csv", rows_a, fields)
    write_csv(out_root / "termination_B_corner_summary.csv", rows_b, fields)
    make_compare_montage(out_root / "termination_compare_montage.png", out_root, rep_a, rep_b)
    print("[ok] termination comparison outputs saved")


if __name__ == "__main__":
    main()
