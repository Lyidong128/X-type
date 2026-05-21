#!/usr/bin/env python3
"""OBC near-zero spectrum and real-space wavefunction maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.next_stage.common import (
    auto_select_points,
    cell_probability,
    classify_distribution,
    compute_ipr,
    solve_obc_modes,
    summarize_points,
    write_csv,
    write_json,
    write_text,
)


def parse_sizes(raw: str) -> list[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        v = int(token)
        if v >= 4:
            vals.append(v)
    vals = sorted(set(vals))
    return vals if vals else [20]


def save_density(path: Path, grid: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.1))
    im = ax.imshow(grid, origin="lower", cmap="magma")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="Probability density")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="OBC near-zero wavefunction analysis.")
    parser.add_argument("--model-file", type=Path, default=Path("/workspace/models/xtype_model.py"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/outputs/next_stage_analysis/obc_wavefunction"))
    parser.add_argument("--sizes", default="20")
    parser.add_argument("--near-states", type=int, default=8)
    parser.add_argument("--dense-max-dim", type=int, default=2200)
    parser.add_argument("--near-k", type=int, default=96)
    parser.add_argument("--edge-width", type=int, default=2)
    parser.add_argument("--corner-size", type=int, default=3)
    parser.add_argument("--edge-like-thr", type=float, default=0.45)
    parser.add_argument("--corner-like-thr", type=float, default=0.28)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    sizes = parse_sizes("12,20" if args.quick else args.sizes)
    near_states = 4 if args.quick else args.near_states
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    points, notes = auto_select_points(model_path=args.model_file)
    point_map = {p.label: p for p in points}
    labels = ["A", "B", "C", "D", "E_A", "E_B"]
    summary_rows = []
    run_logs = []

    for label in labels:
        p = point_map.get(label)
        if p is None:
            run_logs.append(f"[skip] {label}: point not available")
            continue
        for L in sizes:
            try:
                vals, vecs = solve_obc_modes(
                    v=p.v,
                    lm=p.lm,
                    t=p.t,
                    Lx=L,
                    Ly=L,
                    near_k=args.near_k,
                    dense_max_dim=args.dense_max_dim,
                )
            except Exception as exc:
                run_logs.append(f"[fail] {label} L={L}: solve_obc_modes failed ({exc!r})")
                continue

            order = np.argsort(np.abs(vals))
            pick = order[: min(near_states, len(order))]
            csv_rows = []
            subspace = np.zeros((L, L), dtype=float)
            point_json = {
                "point_label": label,
                "params": p.as_dict(),
                "L": L,
                "near_state_indices": [int(i) for i in pick],
                "thresholds": {
                    "edge_width": args.edge_width,
                    "corner_size": args.corner_size,
                    "edge_like_thr": args.edge_like_thr,
                    "corner_like_thr": args.corner_like_thr,
                },
                "states": [],
            }

            for local_idx, idx in enumerate(pick):
                vec = vecs[:, idx]
                grid = cell_probability(vec, Lx=L, Ly=L)
                subspace += grid
                dist = classify_distribution(
                    grid,
                    edge_width=args.edge_width,
                    corner_size=args.corner_size,
                    edge_like_thr=args.edge_like_thr,
                    corner_like_thr=args.corner_like_thr,
                )
                ipr = compute_ipr(vec)

                save_density(
                    out_root / f"{label}_L{L}_state_{local_idx:02d}_density.png",
                    grid,
                    title=(
                        f"{label} L={L} state#{local_idx} "
                        f"E={float(vals[idx]):.3e} ({dist['classification']})"
                    ),
                )

                row = {
                    "point_label": label,
                    "v": p.v,
                    "lm": p.lm,
                    "t": p.t,
                    "w": p.w,
                    "L": L,
                    "state_index": int(idx),
                    "energy": float(vals[idx]),
                    "abs_energy": float(abs(vals[idx])),
                    "edge_weight": float(dist["edge_weight"]),
                    "corner_weight": float(dist["corner_weight"]),
                    "ipr": float(ipr),
                    "classification": dist["classification"],
                }
                csv_rows.append(row)
                summary_rows.append(row)
                point_json["states"].append(row)

            # subspace map
            total = float(np.sum(subspace))
            if total > 0:
                subspace /= total
            subspace_dist = classify_distribution(
                subspace,
                edge_width=args.edge_width,
                corner_size=args.corner_size,
                edge_like_thr=args.edge_like_thr,
                corner_like_thr=args.corner_like_thr,
            )
            save_density(
                out_root / f"{label}_L{L}_nearzero_subspace_density.png",
                subspace,
                title=f"{label} L={L} near-zero subspace ({subspace_dist['classification']})",
            )
            point_json["subspace"] = {
                "classification": subspace_dist["classification"],
                "edge_weight": float(subspace_dist["edge_weight"]),
                "corner_weight": float(subspace_dist["corner_weight"]),
                "bulk_weight": float(subspace_dist["bulk_weight"]),
            }

            # spectrum csv for available eigenpairs
            spec_rows = [{"mode_index": int(i), "energy": float(vals[i]), "abs_energy": float(abs(vals[i]))} for i in range(len(vals))]
            write_csv(
                out_root / f"{label}_L{L}_spectrum.csv",
                spec_rows,
                ["mode_index", "energy", "abs_energy"],
            )
            write_json(out_root / f"{label}_L{L}_wavefunction_summary.json", point_json)
            run_logs.append(f"[ok] {label} L={L}: solved {len(vals)} modes, near={len(pick)}")

    write_csv(
        out_root / "obc_wavefunction_summary.csv",
        summary_rows,
        [
            "point_label",
            "v",
            "lm",
            "t",
            "w",
            "L",
            "state_index",
            "energy",
            "abs_energy",
            "edge_weight",
            "corner_weight",
            "ipr",
            "classification",
        ],
    )
    write_text(
        out_root / "selection_notes.txt",
        summarize_points(points, notes)
        + f"\n\nsizes={sizes}, near_states={near_states}, dense_max_dim={args.dense_max_dim}, near_k={args.near_k}\n",
    )
    write_text(out_root / "run_log.txt", "\n".join(run_logs) + "\n")
    print(f"[ok] obc wavefunction outputs at {out_root}")


if __name__ == "__main__":
    main()
