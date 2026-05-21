#!/usr/bin/env python3
"""Finite-size scaling for min|E|, near-zero counts, IPR and localization weights."""

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
    fit_exponential,
    fit_powerlaw,
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
        val = int(token)
        if val >= 4:
            vals.append(val)
    vals = sorted(set(vals))
    return vals if vals else [12, 16, 20, 24, 28, 32]


def save_line_plot(path: Path, x: np.ndarray, ys: list[np.ndarray], labels: list[str], title: str, y_label: str, logy: bool = False, logx: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for y, lb in zip(ys, labels):
        ax.plot(x, y, marker="o", linewidth=1.2, label=lb)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("L")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-size scaling + IPR diagnostics.")
    parser.add_argument("--model-file", type=Path, default=Path("/workspace/models/xtype_model.py"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/outputs/next_stage_analysis/finite_size_ipr"))
    parser.add_argument("--sizes", default="12,16,20,24,28,32")
    parser.add_argument("--near-k", type=int, default=96)
    parser.add_argument("--dense-max-dim", type=int, default=2200)
    parser.add_argument("--edge-width", type=int, default=2)
    parser.add_argument("--corner-size", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    sizes = parse_sizes("12,16,20" if args.quick else args.sizes)
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    points, notes = auto_select_points(model_path=args.model_file)
    point_map = {p.label: p for p in points}
    labels = ["A", "B", "C", "D", "E_A", "E_B"]

    all_rows = []
    fit_summary: dict[str, dict] = {}
    run_logs = []

    for label in labels:
        p = point_map.get(label)
        if p is None:
            run_logs.append(f"[skip] {label}: point not available")
            continue

        point_rows = []
        for L in sizes:
            if args.quick is False and L >= 32 and (L * L * 8) > 9000:
                run_logs.append(f"[skip] {label} L={L}: dimension too large for current runtime budget")
                continue

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
                run_logs.append(f"[fail] {label} L={L}: solver failed ({exc!r})")
                continue

            abs_vals = np.abs(vals)
            min_idx = int(np.argmin(abs_vals))
            min_abs = float(abs_vals[min_idx])
            n001 = int(np.sum(abs_vals <= 0.01))
            n002 = int(np.sum(abs_vals <= 0.02))
            closest_vec = vecs[:, min_idx]
            ipr_closest = compute_ipr(closest_vec)

            near_mask = abs_vals <= 0.02
            if np.sum(near_mask) == 0:
                near_mask[min_idx] = True
            near_indices = np.where(near_mask)[0]
            subspace = np.zeros((L, L), dtype=float)
            for idx in near_indices:
                subspace += cell_probability(vecs[:, idx], Lx=L, Ly=L)
            total = float(np.sum(subspace))
            if total > 0:
                subspace /= total
            subspace_ipr = float(np.sum(subspace**2))
            dist = classify_distribution(
                subspace,
                edge_width=args.edge_width,
                corner_size=args.corner_size,
            )

            row = {
                "point_label": label,
                "v": p.v,
                "lm": p.lm,
                "t": p.t,
                "w": p.w,
                "L": L,
                "min_abs_E": min_abs,
                "nearzero_count_0p01": n001,
                "nearzero_count_0p02": n002,
                "ipr_closest_state": ipr_closest,
                "ipr_nearzero_subspace": subspace_ipr,
                "edge_weight_subspace": float(dist["edge_weight"]),
                "corner_weight_subspace": float(dist["corner_weight"]),
                "classification_subspace": dist["classification"],
            }
            point_rows.append(row)
            all_rows.append(row)
            run_logs.append(f"[ok] {label} L={L}: min|E|={min_abs:.3e}, N0.02={n002}")

        if not point_rows:
            continue

        point_rows = sorted(point_rows, key=lambda r: int(r["L"]))
        x = np.array([int(r["L"]) for r in point_rows], dtype=float)
        y_min = np.array([float(r["min_abs_E"]) for r in point_rows], dtype=float)
        y_n01 = np.array([float(r["nearzero_count_0p01"]) for r in point_rows], dtype=float)
        y_n02 = np.array([float(r["nearzero_count_0p02"]) for r in point_rows], dtype=float)
        y_ipr = np.array([float(r["ipr_closest_state"]) for r in point_rows], dtype=float)
        y_ipr_sub = np.array([float(r["ipr_nearzero_subspace"]) for r in point_rows], dtype=float)
        y_edge = np.array([float(r["edge_weight_subspace"]) for r in point_rows], dtype=float)
        y_corner = np.array([float(r["corner_weight_subspace"]) for r in point_rows], dtype=float)

        save_line_plot(
            out_root / f"{label}_min_abs_E_vs_L.png",
            x,
            [y_min],
            ["min|E|"],
            f"{label}: min |E| vs L",
            "min |E|",
        )
        save_line_plot(
            out_root / f"{label}_log_min_abs_E_vs_L.png",
            x,
            [y_min],
            ["min|E|"],
            f"{label}: log(min|E|) vs L",
            "min |E|",
            logy=True,
        )
        save_line_plot(
            out_root / f"{label}_loglog_min_abs_E_vs_L.png",
            x,
            [y_min],
            ["min|E|"],
            f"{label}: log-log min|E| vs L",
            "min |E|",
            logx=True,
            logy=True,
        )
        save_line_plot(
            out_root / f"{label}_nearzero_count_vs_L.png",
            x,
            [y_n01, y_n02],
            ["N(|E|<0.01)", "N(|E|<0.02)"],
            f"{label}: near-zero count vs L",
            "count",
        )
        save_line_plot(
            out_root / f"{label}_ipr_vs_L.png",
            x,
            [y_ipr, y_ipr_sub],
            ["IPR closest", "IPR near-zero subspace"],
            f"{label}: IPR vs L",
            "IPR",
        )
        save_line_plot(
            out_root / f"{label}_edge_corner_weight_vs_L.png",
            x,
            [y_edge, y_corner],
            ["edge_weight", "corner_weight"],
            f"{label}: edge/corner weight vs L",
            "weight",
        )

        fit_exp = fit_exponential(x, y_min)
        fit_pow = fit_powerlaw(x, y_min)
        best_model = "undetermined"
        if fit_exp and fit_pow:
            best_model = "exponential" if fit_exp["r2"] >= fit_pow["r2"] else "powerlaw"
        elif fit_exp:
            best_model = "exponential"
        elif fit_pow:
            best_model = "powerlaw"
        fit_summary[label] = {
            "point": p.as_dict(),
            "fit_exponential": fit_exp,
            "fit_powerlaw": fit_pow,
            "best_model": best_model,
            "n_sizes": int(len(x)),
        }

    write_csv(
        out_root / "finite_size_summary.csv",
        all_rows,
        [
            "point_label",
            "v",
            "lm",
            "t",
            "w",
            "L",
            "min_abs_E",
            "nearzero_count_0p01",
            "nearzero_count_0p02",
            "ipr_closest_state",
            "ipr_nearzero_subspace",
            "edge_weight_subspace",
            "corner_weight_subspace",
            "classification_subspace",
        ],
    )
    write_json(out_root / "finite_size_fit_summary.json", fit_summary)
    write_text(
        out_root / "selection_notes.txt",
        summarize_points(points, notes)
        + f"\n\nsizes={sizes}, near_k={args.near_k}, dense_max_dim={args.dense_max_dim}\n",
    )
    write_text(out_root / "run_log.txt", "\n".join(run_logs) + "\n")
    print(f"[ok] finite-size ipr outputs at {out_root}")


if __name__ == "__main__":
    main()
