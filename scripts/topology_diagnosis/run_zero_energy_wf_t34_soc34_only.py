#!/usr/bin/env python3
"""Plot near-zero-energy wavefunction distributions for strict t34+soc34 model."""

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
    p = argparse.ArgumentParser(description="Near-zero wavefunction maps for t34+soc34-only model.")
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
    p.add_argument("--max-states-per-v", type=int, default=6)
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
    representative_imgs: list[Path] = []
    for v in v_list:
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=101, n_occ=4)
        win = max(args.energy_window_floor, args.window_gap_factor * max(bulk_gap, 1e-6))
        h = build_open_open_hamiltonian(lx=lx, ly=ly, v=v, params=params, termination="A")
        evals, vecs = diagonalize_open_open(h, near_k=args.near_k)

        idxs = np.where(np.abs(evals) <= win)[0]
        if idxs.size == 0:
            idxs = np.argsort(np.abs(evals))[: min(2, evals.size)]
        if idxs.size > args.max_states_per_v:
            order = np.argsort(np.abs(evals[idxs]))
            idxs = idxs[order[: args.max_states_per_v]]

        rep_img = None
        for idx in idxs:
            vec = vecs[:, idx]
            rho = cell_density(vec, lx=lx, ly=ly)
            rho = rho / max(np.sum(rho), 1e-15)
            wc, we, wb = state_weights_2d(rho, corner=corner, edge=edge, bulk=bulk)
            cls = classify_corner_edge(wc, we)
            pos = np.unravel_index(int(np.argmax(rho)), rho.shape)
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
                    "W_corner": float(wc),
                    "W_edge": float(we),
                    "W_bulk": float(wb),
                    "max_position_x": int(pos[1]),
                    "max_position_y": int(pos[0]),
                    "classification": cls,
                }
            )

            fig, ax = plt.subplots(figsize=(4.8, 4.1))
            im = ax.imshow(rho, origin="lower", cmap="magma")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(
                f"v={v:.1f}, state={idx}, E={evals[idx]:.4e}\n"
                f"Wc={wc:.3f}, We={we:.3f}, Wb={wb:.3f}, {cls}"
            )
            fig.colorbar(im, ax=ax, label="|psi|^2")
            fig.tight_layout()
            png = out / f"wf_near_zero_v{v:.1f}_state_{idx}_t34_soc34_only.png"
            fig.savefig(png, dpi=180)
            plt.close(fig)
            if rep_img is None:
                rep_img = png

        if rep_img is not None:
            representative_imgs.append(rep_img)
        print(f"[zero-wf-soc34] v={v:.1f} bulk_gap={bulk_gap:.4f} window={win:.4f} states={len(idxs)}")

    fields = [
        "v",
        "Lx",
        "Ly",
        "bulk_gap",
        "energy_window",
        "state_index",
        "energy",
        "abs_energy",
        "W_corner",
        "W_edge",
        "W_bulk",
        "max_position_x",
        "max_position_y",
        "classification",
    ]
    csv_path = out / "zero_energy_wavefunction_summary_t34_soc34_only.csv"
    write_csv(csv_path, rows, fields)
    write_csv(base / "zero_energy_wavefunction_summary_t34_soc34_only.csv", rows, fields)

    if representative_imgs:
        cols = 3
        n = len(representative_imgs)
        rr = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rr, cols, figsize=(5.2 * cols, 3.9 * rr))
        axes = np.array(axes).reshape(rr, cols)
        for i in range(rr * cols):
            ax = axes[i // cols, i % cols]
            if i < n:
                ax.imshow(plt.imread(representative_imgs[i]))
                ax.axis("off")
                ax.set_title(representative_imgs[i].stem)
            else:
                ax.axis("off")
        fig.suptitle("Near-zero wavefunction maps (representative per v)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "zero_energy_wavefunction_montage_t34_soc34_only.png", dpi=180)
        fig.savefig(base / "zero_energy_wavefunction_montage_t34_soc34_only.png", dpi=180)
        plt.close(fig)

    (out / "zero_energy_wf_runtime_note_t34_soc34_only.txt").write_text(
        f"used_fallback={int(used_fallback)}\nLx={lx}\nLy={ly}\n",
        encoding="utf-8",
    )
    print(f"[ok] near-zero wavefunction summary: {csv_path}")


if __name__ == "__main__":
    main()
