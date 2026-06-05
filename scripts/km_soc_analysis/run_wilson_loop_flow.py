#!/usr/bin/env python3
"""Generate full Wilson-loop/Wannier-center flows for selected points."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import (
    compute_wilson_centers,
    crossing_parity,
    ensure_dir,
    load_model,
    param_token,
    plot_wilson_centers,
    set_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wilson loop flow analysis.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    return parser.parse_args()


def _pick_points(scan_csv: Path) -> list[tuple[str, float]]:
    rows = list(csv.DictReader(scan_csv.open("r", encoding="utf-8")))
    lm_rows = [r for r in rows if abs(float(r["lm"]) - 0.1) < 1e-12]
    by_v = {round(float(r["v"]), 10): r for r in lm_rows}
    points: list[tuple[str, float]] = [
        ("A", 0.10),
        ("B", 0.30),
        ("C", 0.50),
        ("D", 0.80),
        ("E", 1.00),
    ]
    mins = sorted(lm_rows, key=lambda r: float(r["direct_gap"]))
    used = {p[1] for p in points}
    extra = []
    for r in mins:
        v = round(float(r["v"]), 10)
        if v in used:
            continue
        extra.append(v)
        used.add(v)
        if len(extra) >= 2:
            break
    if len(extra) >= 1:
        points.append(("F", extra[0]))
    if len(extra) >= 2:
        points.append(("G", extra[1]))
    return points


def _point_gap(scan_rows: list[dict[str, str]], v: float, lm: float) -> float:
    best = min(
        (r for r in scan_rows if abs(float(r["lm"]) - lm) < 1e-12),
        key=lambda r: abs(float(r["v"]) - v),
    )
    return float(best["direct_gap"])


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root) / "wilson_loop")
    scan_csv = Path(args.output_root) / "gap_z2_scan" / "gap_z2_scan.csv"
    if not scan_csv.exists():
        raise FileNotFoundError(f"Missing prerequisite scan: {scan_csv}")

    scan_rows = list(csv.DictReader(scan_csv.open("r", encoding="utf-8")))
    model = load_model(args.model_file)
    points = _pick_points(scan_csv)

    nk_scan = 41 if args.quick else 81
    nk_loop = 61 if args.quick else 121
    summary_rows = []

    for label, v in points:
        set_params(model, v=v, lm=args.lm, t=args.t, w=args.w, j=0.0)
        gap = _point_gap(scan_rows, v=v, lm=args.lm)
        vtok = param_token(v)
        ltok = param_token(args.lm)

        for direction, suffix, xlabel in (
            ("kx_scan_ky", "wilson_kx_scan_ky", r"$k_y$ (fraction of b2)"),
            ("ky_scan_kx", "wilson_ky_scan_kx", r"$k_x$ (fraction of b1)"),
        ):
            scan_values, centers = compute_wilson_centers(
                model,
                nk_scan=nk_scan,
                nk_loop=nk_loop,
                n_occ=4,
                direction=direction,
            )
            parity = crossing_parity(centers, reference=0.5)
            est_z2 = int(parity % 2)
            reliability = "reliable" if gap >= 1e-4 else "uncertain"
            comment = "Gap below threshold; Wilson parity may be unstable." if gap < 1e-4 else ""

            fig_path = out_root / f"v_{vtok}_lm_{ltok}_{suffix}.png"
            plot_wilson_centers(
                scan_values=scan_values,
                centers=centers,
                save_path=fig_path,
                title=f"{label}: v={v:.3f}, lm={args.lm:.3f}, dir={direction}",
                xlabel=xlabel,
            )

            csv_path = out_root / f"wilson_centers_v_{vtok}_lm_{ltok}.csv"
            mode = "a" if csv_path.exists() else "w"
            with csv_path.open(mode, newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["point_label", "v", "lm", "direction", "scan_param", "center_index", "wannier_center"],
                )
                if mode == "w":
                    writer.writeheader()
                for i in range(centers.shape[0]):
                    for j in range(centers.shape[1]):
                        writer.writerow(
                            {
                                "point_label": label,
                                "v": v,
                                "lm": args.lm,
                                "direction": direction,
                                "scan_param": float(scan_values[i]),
                                "center_index": int(j),
                                "wannier_center": float(centers[i, j]),
                            }
                        )

            summary_rows.append(
                {
                    "point_label": label,
                    "v": float(v),
                    "lm": float(args.lm),
                    "t": float(args.t),
                    "w": float(args.w),
                    "gap": float(gap),
                    "direction": direction,
                    "crossing_parity": int(parity),
                    "estimated_z2": int(est_z2),
                    "reliability": reliability,
                    "comment": comment,
                }
            )

    summary_csv = out_root / "wilson_loop_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "point_label",
                "v",
                "lm",
                "t",
                "w",
                "gap",
                "direction",
                "crossing_parity",
                "estimated_z2",
                "reliability",
                "comment",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[ok] summary={summary_csv}")
    print(f"[ok] out_dir={out_root}")


if __name__ == "__main__":
    main()
