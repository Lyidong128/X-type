#!/usr/bin/env python3
"""Open-open finite-size corner/edge/bulk state diagnosis."""

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
    parse_float_list,
    region_masks,
    state_weights,
    token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-open finite-size corner check.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs")
    parser.add_argument("--v-list", default="0.3,0.4,0.5,0.6,0.8,1.0")
    parser.add_argument("--lx", type=int, default=10)
    parser.add_argument("--ly", type=int, default=10)
    parser.add_argument("--fallback-lx", type=int, default=8)
    parser.add_argument("--fallback-ly", type=int, default=8)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--bulk-nk", type=int, default=61)
    parser.add_argument("--num-eigs", type=int, default=40)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_spectrum(evals: np.ndarray, cand_idx: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(evals.size)
    ax.plot(x, evals, ".", color="tab:blue", markersize=3.0, alpha=0.8)
    if cand_idx.size > 0:
        ax.plot(cand_idx, evals[cand_idx], "o", color="tab:red", markersize=4.0, label="in-gap candidates")
        ax.legend()
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("state index")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_heatmap(rho: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    im = ax.imshow(rho, origin="lower", cmap="magma")
    ax.set_xlabel("x unit cell")
    ax.set_ylabel("y unit cell")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="Probability density")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_weight_bar(rows_v: list[dict], out_png: Path, v: float, lx: int, ly: int) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    if not rows_v:
        ax.text(0.5, 0.5, "No in-gap candidates", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        idx = np.arange(len(rows_v))
        wc = np.array([float(r["W_corner"]) for r in rows_v], dtype=float)
        we = np.array([float(r["W_edge"]) for r in rows_v], dtype=float)
        wb = np.array([float(r["W_bulk"]) for r in rows_v], dtype=float)
        width = 0.28
        ax.bar(idx - width, wc, width, label="W_corner")
        ax.bar(idx, we, width, label="W_edge")
        ax.bar(idx + width, wb, width, label="W_bulk")
        ax.set_xticks(idx)
        ax.set_xticklabels([str(int(r["state_index"])) for r in rows_v], rotation=45, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("weight")
        ax.legend()
    ax.set_title(f"v={v:.1f}, Lx x Ly = {lx} x {ly}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def representative_row(rows_v: list[dict]) -> dict | None:
    if not rows_v:
        return None
    return min(rows_v, key=lambda r: float(r["abs_energy"]))


def make_montage(out_png: Path, rows: list[dict], out_root: Path) -> None:
    target_vs = [0.5, 0.8]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6), squeeze=False)
    for i, v in enumerate(target_vs):
        sub = [r for r in rows if abs(float(r["v"]) - v) < 1e-12]
        rep = representative_row(sub)
        ax_spec = axes[i][0]
        ax_wf = axes[i][1]
        spec_path = out_root / f"finite_spectrum_v{token(v)}.png"
        if spec_path.exists():
            img = plt.imread(spec_path)
            ax_spec.imshow(img)
            ax_spec.set_title(f"spectrum v={v:.1f}")
            ax_spec.axis("off")
        else:
            ax_spec.text(0.5, 0.5, "missing spectrum", ha="center", va="center", transform=ax_spec.transAxes)
            ax_spec.axis("off")
        if rep is None:
            ax_wf.text(0.5, 0.5, "no in-gap state", ha="center", va="center", transform=ax_wf.transAxes)
            ax_wf.axis("off")
        else:
            wf_path = out_root / f"wf_open_open_v{token(v)}_state_{int(rep['state_index']):03d}.png"
            if wf_path.exists():
                img = plt.imread(wf_path)
                ax_wf.imshow(img)
                ax_wf.set_title(
                    f"wf v={v:.1f} idx={int(rep['state_index'])}\n"
                    f"Wc={float(rep['W_corner']):.3f}, We={float(rep['W_edge']):.3f}, Wb={float(rep['W_bulk']):.3f}"
                )
                ax_wf.axis("off")
            else:
                ax_wf.text(0.5, 0.5, "missing wf", ha="center", va="center", transform=ax_wf.transAxes)
                ax_wf.axis("off")
    fig.suptitle("Open-open comparison (v=0.5 vs v=0.8)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    v_list = parse_float_list(args.v_list)

    lx = int(args.fallback_lx) if (int(args.lx) * int(args.ly) * 8 > 1200) else int(args.lx)
    ly = int(args.fallback_ly) if (int(args.lx) * int(args.ly) * 8 > 1200) else int(args.ly)
    if args.quick:
        lx = min(lx, 8)
        ly = min(ly, 8)

    corner_mask, edge_mask, bulk_mask = region_masks(lx=lx, ly=ly, corner_size=2, edge_width=2)
    rows: list[dict] = []

    for v in v_list:
        bulk_gap = compute_bulk_gap(v=v, t=args.t, w=args.w, lm=args.lm, nk=(31 if args.quick else args.bulk_nk))
        h = build_open_open_hamiltonian(lx=lx, ly=ly, v=v, t=args.t, w=args.w, lm=args.lm, termination="A")
        evals, evecs = diagonalize_open_open(h, num_eigs=args.num_eigs)
        cand_idx, in_gap_window, window_status = choose_in_gap_indices(evals=evals, bulk_gap=bulk_gap)

        plot_spectrum(
            evals=evals,
            cand_idx=cand_idx,
            out_png=out_root / f"finite_spectrum_v{token(v)}.png",
            title=(
                f"open-open spectrum v={v:.1f}, L={lx}x{ly}\n"
                f"bulk_gap={bulk_gap:.4e}, in_gap_window={in_gap_window:.4e}, status={window_status}"
            ),
        )

        rows_v: list[dict] = []
        for idx in cand_idx:
            vec = evecs[:, int(idx)]
            rho = cell_density(vec, lx=lx, ly=ly)
            wc, we, wb = state_weights(rho, corner=corner_mask, edge=edge_mask, bulk=bulk_mask)
            norm_err = abs((wc + we + wb) - 1.0)
            if norm_err > 1e-6:
                print(f"[warning] normalization error at v={v:.1f}, state={idx}: {norm_err:.3e}")
            max_pos = np.unravel_index(int(np.argmax(rho)), rho.shape)
            cls = classify_state(w_corner=wc, w_edge=we)
            row = {
                "v": float(v),
                "Lx": int(lx),
                "Ly": int(ly),
                "state_index": int(idx),
                "energy": float(evals[int(idx)]),
                "abs_energy": float(abs(evals[int(idx)])),
                "W_corner": wc,
                "W_edge": we,
                "W_bulk": wb,
                "max_position_x": int(max_pos[1]),
                "max_position_y": int(max_pos[0]),
                "classification": cls,
                "in_gap_window": float(in_gap_window),
                "bulk_gap_used": float(bulk_gap),
                "window_status": window_status,
            }
            rows.append(row)
            rows_v.append(row)
            plot_heatmap(
                rho=rho,
                out_png=out_root / f"wf_open_open_v{token(v)}_state_{int(idx):03d}.png",
                title=(
                    f"v={v:.1f}, idx={int(idx)}, E={float(evals[int(idx)]):.4e}\n"
                    f"Wc={wc:.3f}, We={we:.3f}, Wb={wb:.3f}"
                ),
            )

        plot_weight_bar(
            rows_v=rows_v,
            out_png=out_root / f"corner_weight_bar_v{token(v)}.png",
            v=v,
            lx=lx,
            ly=ly,
        )
        n_corner = sum(1 for r in rows_v if r["classification"] == "corner_state")
        n_edge = sum(1 for r in rows_v if r["classification"] == "edge_state")
        print(
            f"[open-open] v={v:.1f} bulk_gap={bulk_gap:.4e} "
            f"window={in_gap_window:.4e} status={window_status} "
            f"candidates={len(rows_v)} corner={n_corner} edge={n_edge}"
        )

    write_csv(
        out_root / "open_open_corner_summary.csv",
        rows,
        [
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
        ],
    )
    make_montage(out_png=out_root / "open_open_corner_montage.png", rows=rows, out_root=out_root)
    print(f"[ok] open-open summary rows={len(rows)} saved at {out_root / 'open_open_corner_summary.csv'}")


if __name__ == "__main__":
    main()
