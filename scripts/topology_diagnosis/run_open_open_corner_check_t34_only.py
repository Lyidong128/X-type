#!/usr/bin/env python3
"""Open-open corner/edge diagnostics for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.topology_diagnosis.t34_only_common import (
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
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-open corner check for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.3,0.4,0.5,0.6,0.8,1.0")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--lx", type=int, default=10)
    parser.add_argument("--ly", type=int, default=10)
    parser.add_argument("--fallback-l", type=int, default=8)
    parser.add_argument("--near-k", type=int, default=120)
    parser.add_argument("--corner-size", type=int, default=2)
    parser.add_argument("--edge-width", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["04_open_open"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)

    v_list = parse_float_list(args.v_list)
    rows = []
    montage_paths = []

    lx, ly = args.lx, args.ly
    used_fallback = False
    try:
        _ = build_open_open_hamiltonian(lx=lx, ly=ly, v=v_list[0], params=params, termination="A")
    except Exception:
        lx = ly = args.fallback_l
        used_fallback = True

    for v in v_list:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=81, n_occ=4)
        in_gap_window = max(1e-4, 0.4 * max(bulk_gap, 1e-6))
        h = build_open_open_hamiltonian(lx=lx, ly=ly, v=v, params=params, termination="A")
        evals, vecs = diagonalize_open_open(h, near_k=args.near_k)
        corner, edge, bulk = region_masks(lx=lx, ly=ly, corner_size=args.corner_size, edge_width=args.edge_width)

        # spectrum plot
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot(np.arange(evals.size), evals, marker="o", linestyle="none", markersize=3.0)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(+in_gap_window, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(-in_gap_window, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("state index")
        ax.set_ylabel("energy")
        ax.set_title(f"open-open finite spectrum v={v:.1f}, L=({lx},{ly}), window={in_gap_window:.3f}")
        ax.grid(alpha=0.24)
        fig.tight_layout()
        spec_png = out_dir / f"finite_spectrum_v{token(v)}_t34_only.png"
        fig.savefig(spec_png, dpi=180)
        plt.close(fig)
        montage_paths.append(spec_png)
        (base / spec_png.name).write_bytes(spec_png.read_bytes())

        # choose in-gap states
        idxs = np.where(np.abs(evals) <= in_gap_window)[0]
        if idxs.size == 0:
            idxs = np.argsort(np.abs(evals))[:12]
        chosen_rows = []
        for idx in idxs:
            vec = vecs[:, idx]
            rho = cell_density(vec, lx=lx, ly=ly)
            wc, we, wb = state_weights_2d(rho, corner=corner, edge=edge, bulk=bulk)
            cls = classify_corner_edge(wc, we)
            maxpos = np.unravel_index(int(np.argmax(rho)), rho.shape)
            row = {
                "v": float(v),
                "Lx": int(lx),
                "Ly": int(ly),
                "state_index": int(idx),
                "energy": float(evals[idx]),
                "abs_energy": float(abs(evals[idx])),
                "W_corner": float(wc),
                "W_edge": float(we),
                "W_bulk": float(wb),
                "max_position_x": int(maxpos[1]),
                "max_position_y": int(maxpos[0]),
                "classification": cls,
                "bulk_gap_used": float(bulk_gap),
                "in_gap_window": float(in_gap_window),
            }
            rows.append(row)
            chosen_rows.append(row)

        # representative wavefunction: closest |E| state
        chosen_rows.sort(key=lambda r: float(r["abs_energy"]))
        best = chosen_rows[0]
        idx = int(best["state_index"])
        rho = cell_density(vecs[:, idx], lx=lx, ly=ly)
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        im = ax.imshow(rho, origin="lower", cmap="magma")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"v={v:.1f}, state={idx}, E={best['energy']:.3e}\n"
            f"Wc={best['W_corner']:.3f}, We={best['W_edge']:.3f}, Wb={best['W_bulk']:.3f}"
        )
        fig.colorbar(im, ax=ax, label="rho")
        fig.tight_layout()
        wf_png = out_dir / f"wf_open_open_v{token(v)}_state_{idx}_t34_only.png"
        fig.savefig(wf_png, dpi=180)
        plt.close(fig)

        # corner-edge-bulk bar for representative
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        ax.bar(["W_corner", "W_edge", "W_bulk"], [best["W_corner"], best["W_edge"], best["W_bulk"]])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"v={v:.1f} representative state {idx} ({best['classification']})")
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        bar_png = out_dir / f"corner_weight_bar_v{token(v)}_t34_only.png"
        fig.savefig(bar_png, dpi=180)
        plt.close(fig)
        (base / bar_png.name).write_bytes(bar_png.read_bytes())
        print(
            f"[open-open-t34] v={v:.1f} states={len(idxs)} "
            f"best_class={best['classification']} Wc={best['W_corner']:.3f} We={best['W_edge']:.3f}"
        )

    write_csv(
        out_dir / "open_open_corner_summary_t34_only.csv",
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
            "bulk_gap_used",
            "in_gap_window",
        ],
    )
    write_csv(
        base / "open_open_corner_summary_t34_only.csv",
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
            "bulk_gap_used",
            "in_gap_window",
        ],
    )

    # montage
    n = len(montage_paths)
    cols = 3
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.0 * cols, 3.7 * rows_n))
    axes = np.array(axes).reshape(rows_n, cols)
    for i in range(rows_n * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(plt.imread(montage_paths[i]))
            ax.axis("off")
            ax.set_title(montage_paths[i].stem)
        else:
            ax.axis("off")
    fig.suptitle("open-open finite spectra montage (t34-only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "open_open_corner_montage_t34_only.png", dpi=180)
    fig.savefig(base / "open_open_corner_montage_t34_only.png", dpi=180)
    plt.close(fig)

    # fallback note
    note = out_dir / "open_open_runtime_note_t34_only.txt"
    note.write_text(
        f"used_fallback={used_fallback}\nLx={lx}\nLy={ly}\n",
        encoding="utf-8",
    )
    print(f"[ok] open-open summary: {out_dir / 'open_open_corner_summary_t34_only.csv'}")


if __name__ == "__main__":
    main()
