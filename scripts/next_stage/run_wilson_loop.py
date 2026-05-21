#!/usr/bin/env python3
"""Wilson loop / Wannier center flow for selected analysis points."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.next_stage.common import (
    auto_select_points,
    estimate_z2_from_wannier_centers,
    load_model,
    set_params,
    summarize_points,
    write_csv,
    write_text,
    wilson_loop_flow,
)


def save_flow_plot(
    scan_vals: np.ndarray,
    centers: np.ndarray,
    out_png: Path,
    title: str,
    scan_label: str,
    estimated_z2: int,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for band in range(centers.shape[1]):
        ax.plot(scan_vals, centers[:, band], linewidth=1.0)
    ax.set_xlabel(scan_label)
    ax.set_ylabel("Wannier center")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{title}\nestimated_z2={estimated_z2}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Wilson-loop / Wannier center flow.")
    parser.add_argument("--model-file", type=Path, default=Path("/workspace/models/xtype_model.py"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/outputs/next_stage_analysis/wilson_loop"))
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--nk-loop", type=int, default=81)
    parser.add_argument("--nk-scan", type=int, default=81)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    nk_loop = 41 if args.quick else args.nk_loop
    nk_scan = 41 if args.quick else args.nk_scan

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_file)

    points, notes = auto_select_points(model_path=args.model_file)
    point_map = {p.label: p for p in points}
    requested = ["A", "B", "C", "D", "E_A", "E_B"]

    summary_rows = []
    for label in requested:
        if label not in point_map:
            summary_rows.append(
                {
                    "point_label": label,
                    "v": "",
                    "lm": "",
                    "t": "",
                    "w": "",
                    "direction": "",
                    "estimated_z2": "",
                    "comment": "point_not_available",
                }
            )
            continue

        p = point_map[label]
        set_params(model, v=p.v, lm=p.lm, t=p.t)

        csv_rows = []
        for direction, scan_name, png_name in (
            ("kx", "ky", f"{label}_wilson_kx_vs_ky.png"),
            ("ky", "kx", f"{label}_wilson_ky_vs_kx.png"),
        ):
            scan_vals, centers = wilson_loop_flow(
                model=model,
                direction=direction,
                nk_loop=nk_loop,
                nk_scan=nk_scan,
                n_occ=args.n_occ,
            )
            est_z2 = estimate_z2_from_wannier_centers(centers)
            save_flow_plot(
                scan_vals=scan_vals,
                centers=centers,
                out_png=out_root / png_name,
                title=f"{label}: v={p.v:.3f}, lm={p.lm:.3f}, t={p.t:.3f}, w={p.w:.3f}",
                scan_label=scan_name,
                estimated_z2=est_z2,
            )
            for i in range(scan_vals.size):
                row = {"point_label": label, "direction": direction, "scan_value": float(scan_vals[i])}
                for b in range(centers.shape[1]):
                    row[f"center_{b}"] = float(centers[i, b])
                csv_rows.append(row)

            summary_rows.append(
                {
                    "point_label": label,
                    "v": p.v,
                    "lm": p.lm,
                    "t": p.t,
                    "w": p.w,
                    "direction": direction,
                    "estimated_z2": int(est_z2),
                    "comment": p.note if p.note else p.source,
                }
            )

        center_fields = ["point_label", "direction", "scan_value"] + [f"center_{i}" for i in range(args.n_occ)]
        write_csv(out_root / f"{label}_wannier_centers.csv", csv_rows, center_fields)

    write_csv(
        out_root / "wilson_loop_summary.csv",
        summary_rows,
        ["point_label", "v", "lm", "t", "w", "direction", "estimated_z2", "comment"],
    )
    write_text(
        out_root / "selection_notes.txt",
        summarize_points(points, notes)
        + f"\n\nwilson_grid: nk_loop={nk_loop}, nk_scan={nk_scan}, n_occ={args.n_occ}\n",
    )
    print(f"[ok] wilson outputs at {out_root}")


if __name__ == "__main__":
    main()
