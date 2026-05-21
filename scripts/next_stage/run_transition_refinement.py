#!/usr/bin/env python3
"""Refine transition corridors around A/B with dense local scans."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.next_stage.common import (
    auto_select_points,
    compute_bulk_gap_kmin,
    compute_dynamic_w,
    load_model,
    set_params,
    summarize_points,
    write_csv,
    write_text,
)
from scripts.run_scan import compute_bulk_gap, compute_chern_number, compute_wilson_loop_and_z2


def build_values(center: float, lo: float, hi: float, step: float) -> np.ndarray:
    start = max(lo, center - (center - lo))
    vals = np.arange(lo, hi + 1e-12, step)
    return np.array(vals, dtype=float)


def grid_from_rows(rows: list[dict], x_key: str, y_key: str, z_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = sorted({float(r[x_key]) for r in rows})
    ys = sorted({float(r[y_key]) for r in rows})
    x_map = {x: i for i, x in enumerate(xs)}
    y_map = {y: j for j, y in enumerate(ys)}
    grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
    for r in rows:
        i = x_map[float(r[x_key])]
        j = y_map[float(r[y_key])]
        grid[j, i] = float(r[z_key])
    return np.array(xs), np.array(ys), grid


def save_map(rows: list[dict], z_key: str, out_png: Path, title: str, cmap: str = "viridis") -> None:
    x, y, z = grid_from_rows(rows, "v", "lm", z_key)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap=cmap,
    )
    ax.set_xlabel("v")
    ax.set_ylabel("lm")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=z_key)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def refine_center(
    model,
    label: str,
    center_v: float,
    center_lm: float,
    v_range: tuple[float, float],
    lm_range: tuple[float, float],
    step: float,
    nk_gap: int,
    nk_topo: int,
) -> list[dict]:
    rows = []
    v_vals = np.arange(v_range[0], v_range[1] + 1e-12, step)
    lm_vals = np.arange(lm_range[0], lm_range[1] + 1e-12, step)
    total = len(v_vals) * len(lm_vals)
    done = 0
    for v in v_vals:
        for lm in lm_vals:
            done += 1
            p = set_params(model, v=float(v), lm=float(lm), t=0.5)
            indirect_gap = float(compute_bulk_gap(model, nk=max(7, nk_gap // 2), n_occ=4))
            direct_gap, kx_min, ky_min = compute_bulk_gap_kmin(model, nk=nk_gap, n_occ=4)
            chern = float(compute_chern_number(model, nk=nk_topo, n_occ=4))
            wilson, z2 = compute_wilson_loop_and_z2(model, nk=max(9, nk_topo), n_occ=4)
            rows.append(
                {
                    "center_label": label,
                    "v": float(v),
                    "lm": float(lm),
                    "t": 0.5,
                    "w": float(p["w"]),
                    "bulk_gap": indirect_gap,
                    "direct_gap": float(direct_gap),
                    "indirect_gap": indirect_gap,
                    "min_direct_gap": float(direct_gap),
                    "kx_min": float(kx_min),
                    "ky_min": float(ky_min),
                    "chern": chern,
                    "z2": int(z2),
                    "wilson": float(wilson),
                    "occupied_band_index": 3,
                    "unoccupied_band_index": 4,
                    "distance_to_center": float(np.hypot(v - center_v, lm - center_lm)),
                }
            )
            if done % 200 == 0 or done == total:
                print(f"[{label}] processed {done}/{total}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Transition corridor refinement around A/B points.")
    parser.add_argument("--model-file", type=Path, default=Path("/workspace/models/xtype_model.py"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/outputs/next_stage_analysis/transition_refinement"),
    )
    parser.add_argument("--a-v-range", default="0.80,1.00")
    parser.add_argument("--a-lm-range", default="0.10,0.30")
    parser.add_argument("--b-v-range", default="1.00,1.20")
    parser.add_argument("--b-lm-range", default="0.10,0.30")
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--nk-gap", type=int, default=21)
    parser.add_argument("--nk-topo", type=int, default=15)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    def parse_range(raw: str) -> tuple[float, float]:
        a, b = [float(x.strip()) for x in raw.split(",")]
        return min(a, b), max(a, b)

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    step = 0.01 if args.quick else args.step
    nk_gap = 13 if args.quick else args.nk_gap
    nk_topo = 9 if args.quick else args.nk_topo

    model = load_model(args.model_file)
    points, notes = auto_select_points(model_path=args.model_file)
    pmap = {p.label: p for p in points}
    if "A" not in pmap or "B" not in pmap:
        raise RuntimeError("A/B points unavailable for transition refinement.")

    all_rows = []
    candidate_rows = []
    for lbl, vr_raw, lr_raw in (
        ("A", args.a_v_range, args.a_lm_range),
        ("B", args.b_v_range, args.b_lm_range),
    ):
        p = pmap[lbl]
        v_range = parse_range(vr_raw)
        lm_range = parse_range(lr_raw)
        rows = refine_center(
            model=model,
            label=lbl,
            center_v=p.v,
            center_lm=p.lm,
            v_range=v_range,
            lm_range=lm_range,
            step=step,
            nk_gap=nk_gap,
            nk_topo=nk_topo,
        )
        all_rows.extend(rows)
        write_csv(
            out_root / f"{lbl}_transition_scan.csv",
            rows,
            [
                "center_label",
                "v",
                "lm",
                "t",
                "w",
                "bulk_gap",
                "direct_gap",
                "indirect_gap",
                "min_direct_gap",
                "kx_min",
                "ky_min",
                "chern",
                "z2",
                "wilson",
                "occupied_band_index",
                "unoccupied_band_index",
                "distance_to_center",
            ],
        )
        save_map(rows, "bulk_gap", out_root / f"{lbl}_gap_map.png", f"{lbl}: bulk gap map", cmap="coolwarm")
        save_map(rows, "z2", out_root / f"{lbl}_z2_map.png", f"{lbl}: Z2 map", cmap="viridis")
        save_map(rows, "kx_min", out_root / f"{lbl}_gap_closing_kx_map.png", f"{lbl}: gap-closing kx map", cmap="plasma")
        save_map(rows, "ky_min", out_root / f"{lbl}_gap_closing_ky_map.png", f"{lbl}: gap-closing ky map", cmap="plasma")

        # top-10 candidates by minimum direct/indirect gap and closeness to center.
        sorted_rows = sorted(rows, key=lambda r: (abs(float(r["min_direct_gap"])), abs(float(r["bulk_gap"])), float(r["distance_to_center"])))
        for rank, row in enumerate(sorted_rows[:10], start=1):
            candidate_rows.append(
                {
                    "rank": rank,
                    "center_label": lbl,
                    "v": row["v"],
                    "lm": row["lm"],
                    "t": row["t"],
                    "w": row["w"],
                    "gap": row["bulk_gap"],
                    "kx_min": row["kx_min"],
                    "ky_min": row["ky_min"],
                    "chern": row["chern"],
                    "z2": row["z2"],
                    "comment": f"direct_gap={float(row['direct_gap']):.3e}",
                }
            )

    write_csv(
        out_root / "transition_candidates.csv",
        candidate_rows,
        ["rank", "center_label", "v", "lm", "t", "w", "gap", "kx_min", "ky_min", "chern", "z2", "comment"],
    )
    write_text(
        out_root / "selection_notes.txt",
        summarize_points(points, notes)
        + f"\n\nstep={step}, nk_gap={nk_gap}, nk_topo={nk_topo}\n",
    )
    print(f"[ok] transition refinement outputs at {out_root}")


if __name__ == "__main__":
    main()
