#!/usr/bin/env python3
"""Mark summed near-zero states in OBC E-vs-index plots (strict t34+soc34-only)."""

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
    compute_bulk_gap,
    diagonalize_open_open,
    ensure_dir,
    parse_float_list,
    token,
    write_csv,
    write_model_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mark summed near-zero states on OBC index plots.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only")
    p.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    p.add_argument("--lx", type=int, default=10)
    p.add_argument("--ly", type=int, default=10)
    p.add_argument("--fallback-l", type=int, default=8)
    p.add_argument("--near-k", type=int, default=120)
    p.add_argument("--energy-window-floor", type=float, default=0.01)
    p.add_argument("--window-gap-factor", type=float, default=0.30)
    p.add_argument("--max-states-per-v", type=int, default=24)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    out = ensure_dir(base / "07_zero_energy_wf")
    write_model_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_list = parse_float_list(args.v_list)

    lx, ly = args.lx, args.ly
    used_fallback = False
    try:
        _ = build_open_open_hamiltonian(lx=lx, ly=ly, v=v_list[0], params=params, termination="A")
    except Exception:
        lx = ly = args.fallback_l
        used_fallback = True

    rows = []
    panel_paths: list[Path] = []
    for v in v_list:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=101, n_occ=4)
        win = max(args.energy_window_floor, args.window_gap_factor * max(bulk_gap, 1e-6))
        h = build_open_open_hamiltonian(lx=lx, ly=ly, v=v, params=params, termination="A")
        evals, _ = diagonalize_open_open(h, near_k=args.near_k)
        x = np.arange(evals.size)

        idxs = np.where(np.abs(evals) <= win)[0]
        if idxs.size == 0:
            idxs = np.argsort(np.abs(evals))[:2]
        if idxs.size > args.max_states_per_v:
            order = np.argsort(np.abs(evals[idxs]))
            idxs = idxs[order[: args.max_states_per_v]]
        idxs = np.array(sorted(set(int(i) for i in idxs)), dtype=int)

        for idx in idxs:
            rows.append(
                {
                    "v": float(v),
                    "Lx": int(lx),
                    "Ly": int(ly),
                    "bulk_gap": float(bulk_gap),
                    "energy_window": float(win),
                    "state_index": int(idx),
                    "energy": float(evals[idx]),
                    "abs_energy": float(abs(evals[idx])),
                }
            )

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.plot(x, evals, marker="o", linestyle="none", markersize=2.4, color="#2a66b8", alpha=0.75, label="all states")
        ax.scatter(idxs, evals[idxs], s=26, color="red", zorder=5, label="summed near-zero states")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(win, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(-win, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("state index")
        ax.set_ylabel("energy")
        ax.set_title(
            f"OBC E vs index (v={v:.1f}) with summed states in red\n"
            f"L=({lx},{ly}), window={win:.4f}, count={len(idxs)}"
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        png = out / f"open_open_spectrum_mark_sum_v{token(v)}_t34_soc34_only.png"
        fig.savefig(png, dpi=180)
        plt.close(fig)
        panel_paths.append(png)
        print(f"[obc-mark-sum] v={v:.1f} window={win:.4f} marked_states={len(idxs)}")

    csv_path = out / "obc_marked_sum_states_t34_soc34_only.csv"
    fields = ["v", "Lx", "Ly", "bulk_gap", "energy_window", "state_index", "energy", "abs_energy"]
    write_csv(csv_path, rows, fields)
    write_csv(base / "obc_marked_sum_states_t34_soc34_only.csv", rows, fields)

    if panel_paths:
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
        fig.suptitle("OBC E-index with summed states highlighted (red)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "open_open_spectrum_mark_sum_montage_t34_soc34_only.png", dpi=180)
        fig.savefig(base / "open_open_spectrum_mark_sum_montage_t34_soc34_only.png", dpi=180)
        plt.close(fig)

    (out / "obc_mark_sum_runtime_note_t34_soc34_only.txt").write_text(
        f"used_fallback={int(used_fallback)}\nLx={lx}\nLy={ly}\n",
        encoding="utf-8",
    )
    print(f"[ok] marked-sum OBC plots: {csv_path}")


if __name__ == "__main__":
    main()
