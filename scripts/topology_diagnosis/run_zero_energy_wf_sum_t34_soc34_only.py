#!/usr/bin/env python3
"""Sum near-zero-energy wavefunction densities for strict t34+soc34 model."""

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
    ensure_dir,
    parse_float_list,
    region_masks,
    state_weights_2d,
    write_csv,
    write_model_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summed near-zero-energy density maps (t34+soc34-only).")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only")
    p.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    p.add_argument("--lx", type=int, default=10)
    p.add_argument("--ly", type=int, default=10)
    p.add_argument("--fallback-l", type=int, default=8)
    p.add_argument("--near-k", type=int, default=120)
    p.add_argument("--corner-size", type=int, default=2)
    p.add_argument("--edge-width", type=int, default=2)
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

    corner, edge, bulk = region_masks(lx=lx, ly=ly, corner_size=args.corner_size, edge_width=args.edge_width)

    rows = []
    panel_paths: list[Path] = []
    for v in v_list:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=101, n_occ=4)
        win = max(args.energy_window_floor, args.window_gap_factor * max(bulk_gap, 1e-6))
        h = build_open_open_hamiltonian(lx=lx, ly=ly, v=v, params=params, termination="A")
        evals, vecs = diagonalize_open_open(h, near_k=args.near_k)

        idxs = np.where(np.abs(evals) <= win)[0]
        if idxs.size == 0:
            idxs = np.argsort(np.abs(evals))[:2]
        if idxs.size > args.max_states_per_v:
            order = np.argsort(np.abs(evals[idxs]))
            idxs = idxs[order[: args.max_states_per_v]]

        rho_sum = np.zeros((ly, lx), dtype=float)
        for idx in idxs:
            rho_sum += cell_density(vecs[:, idx], lx=lx, ly=ly)
        rho_sum = rho_sum / max(np.sum(rho_sum), 1e-15)

        wc, we, wb = state_weights_2d(rho_sum, corner=corner, edge=edge, bulk=bulk)
        cls = classify_corner_edge(wc, we)
        rows.append(
            {
                "v": float(v),
                "Lx": int(lx),
                "Ly": int(ly),
                "bulk_gap": float(bulk_gap),
                "energy_window": float(win),
                "num_states_summed": int(len(idxs)),
                "min_abs_energy_in_window": float(np.min(np.abs(evals[idxs]))),
                "max_abs_energy_in_window": float(np.max(np.abs(evals[idxs]))),
                "W_corner_sum": float(wc),
                "W_edge_sum": float(we),
                "W_bulk_sum": float(wb),
                "classification_sum": cls,
            }
        )

        fig, ax = plt.subplots(figsize=(4.8, 4.1))
        im = ax.imshow(rho_sum, origin="lower", cmap="magma")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"v={v:.1f}, summed near-zero states={len(idxs)}\n"
            f"window={win:.4f}, Wc={wc:.3f}, We={we:.3f}, Wb={wb:.3f}"
        )
        fig.colorbar(im, ax=ax, label=r"$\sum |\psi_n|^2$")
        fig.tight_layout()
        png = out / f"wf_near_zero_sum_v{v:.1f}_t34_soc34_only.png"
        fig.savefig(png, dpi=180)
        plt.close(fig)
        panel_paths.append(png)
        print(
            f"[zero-wf-sum-soc34] v={v:.1f} states={len(idxs)} "
            f"window={win:.4f} Wc={wc:.3f} We={we:.3f} Wb={wb:.3f}"
        )

    fields = [
        "v",
        "Lx",
        "Ly",
        "bulk_gap",
        "energy_window",
        "num_states_summed",
        "min_abs_energy_in_window",
        "max_abs_energy_in_window",
        "W_corner_sum",
        "W_edge_sum",
        "W_bulk_sum",
        "classification_sum",
    ]
    csv_path = out / "zero_energy_wavefunction_sum_summary_t34_soc34_only.csv"
    write_csv(csv_path, rows, fields)
    write_csv(base / "zero_energy_wavefunction_sum_summary_t34_soc34_only.csv", rows, fields)

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
        fig.suptitle("Summed near-zero-energy density maps", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "zero_energy_wavefunction_sum_montage_t34_soc34_only.png", dpi=180)
        fig.savefig(base / "zero_energy_wavefunction_sum_montage_t34_soc34_only.png", dpi=180)
        plt.close(fig)

    (out / "zero_energy_wf_sum_runtime_note_t34_soc34_only.txt").write_text(
        f"used_fallback={int(used_fallback)}\nLx={lx}\nLy={ly}\n",
        encoding="utf-8",
    )
    print(f"[ok] summed near-zero wavefunction summary: {csv_path}")


if __name__ == "__main__":
    main()
