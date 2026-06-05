#!/usr/bin/env python3
"""Analyze gap closing/reopening and classify transition candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gap-closing candidates.")
    parser.add_argument("--output-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _classify(z2_left: int, z2_right: int, gap_min: float) -> tuple[str, str, str]:
    z2_changed = z2_left != z2_right and z2_left >= 0 and z2_right >= 0
    if z2_changed and gap_min < 1e-4:
        return "z2_topological_transition", "high", "Z2 changes with near-closing gap."
    if z2_changed and gap_min >= 1e-4:
        return "possible_numerical_issue", "low", "Z2 changes without a clear gap closure."
    if gap_min < 1e-3:
        return "gapless_region", "medium", "Gap is effectively closed over this interval."
    if gap_min < 1e-4:
        return "ordinary_gap_reopening", "medium", "Gap closes/reopens without Z2 change."
    if gap_min < 5e-4:
        return "gapless_region", "low", "Extended small-gap region."
    return "ordinary_gap_reopening", "low", "No robust closure detected."


def _plot_zoom(rows: list[dict[str, str]], lm: float, vc: float, idx: int, out_dir: Path) -> None:
    subset = [
        r
        for r in rows
        if abs(float(r["lm"]) - lm) < 1e-12 and abs(float(r["v"]) - vc) <= 0.05
    ]
    subset = sorted(subset, key=lambda r: float(r["v"]))
    if not subset:
        return
    v = np.array([float(r["v"]) for r in subset], dtype=float)
    dg = np.array([float(r["direct_gap"]) for r in subset], dtype=float)
    z2 = np.array([float(r["z2"]) if int(r["z2_reliable"]) == 1 else np.nan for r in subset], dtype=float)
    kx = np.array([float(r["kx_min"]) for r in subset], dtype=float)
    ky = np.array([float(r["ky_min"]) for r in subset], dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(v, dg, marker="o", linewidth=1.1)
    ax.axhline(1e-4, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("v")
    ax.set_ylabel("direct gap")
    ax.set_title(f"Transition {idx}: lm={lm:.1f} gap zoom")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"gap_zoom_transition_{idx}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.step(v, z2, where="mid", linewidth=1.1)
    ax.set_xlabel("v")
    ax.set_ylabel("Z2 (reliable only)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"Transition {idx}: lm={lm:.1f} Z2 zoom")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"z2_zoom_transition_{idx}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(v, kx, marker="o", linewidth=1.1, label="kx_min")
    ax.plot(v, ky, marker="s", linewidth=1.1, label="ky_min")
    ax.set_xlabel("v")
    ax.set_ylabel("k at min direct gap")
    ax.set_title(f"Transition {idx}: lm={lm:.1f} k-min zoom")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"kmin_zoom_transition_{idx}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "gap_closing")
    scan_csv = Path(args.output_root) / "gap_z2_scan" / "gap_z2_scan.csv"
    if not scan_csv.exists():
        raise FileNotFoundError(f"Missing prerequisite scan: {scan_csv}")

    rows = list(csv.DictReader(scan_csv.open("r", encoding="utf-8")))
    transitions = []

    for lm in (0.0, 0.1):
        lm_rows = sorted([r for r in rows if abs(float(r["lm"]) - lm) < 1e-12], key=lambda r: float(r["v"]))
        if len(lm_rows) < 3:
            continue

        intervals: list[tuple[float, float, str]] = []

        gapless_idx = [i for i, r in enumerate(lm_rows) if float(r["direct_gap"]) < 1e-3]
        if gapless_idx:
            start = gapless_idx[0]
            prev = start
            for idx in gapless_idx[1:]:
                if idx == prev + 1:
                    prev = idx
                    continue
                lo = float(lm_rows[max(0, start - 1)]["v"])
                hi = float(lm_rows[min(len(lm_rows) - 1, prev + 1)]["v"])
                intervals.append((lo, hi, "gapless_region"))
                start = idx
                prev = idx
            lo = float(lm_rows[max(0, start - 1)]["v"])
            hi = float(lm_rows[min(len(lm_rows) - 1, prev + 1)]["v"])
            intervals.append((lo, hi, "gapless_region"))

        for i in range(len(lm_rows) - 1):
            a, b = lm_rows[i], lm_rows[i + 1]
            if int(a["z2_reliable"]) == 1 and int(b["z2_reliable"]) == 1 and int(a["z2"]) != int(b["z2"]):
                intervals.append((float(a["v"]), float(b["v"]), "z2_change"))
        for i in range(1, len(lm_rows) - 1):
            gm = float(lm_rows[i]["direct_gap"])
            gl = float(lm_rows[i - 1]["direct_gap"])
            gr = float(lm_rows[i + 1]["direct_gap"])
            if gm <= gl and gm <= gr and (gm < 0.01) and (gm >= 1e-3):
                intervals.append((float(lm_rows[i - 1]["v"]), float(lm_rows[i + 1]["v"]), "local_min"))

        # Merge overlaps.
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        for lo, hi, src in intervals:
            if not merged or lo > merged[-1][1] + 1e-12:
                merged.append([lo, hi, {src}])
            else:
                merged[-1][1] = max(merged[-1][1], hi)
                merged[-1][2].add(src)

        for lo, hi, sources in merged:
            local = [r for r in lm_rows if lo - 1e-12 <= float(r["v"]) <= hi + 1e-12]
            if not local:
                continue
            min_row = min(local, key=lambda r: float(r["direct_gap"]))
            gap_min = float(min_row["direct_gap"])
            vc = float(min_row["v"])
            z2_left = int(local[0]["z2"]) if int(local[0]["z2_reliable"]) == 1 else -1
            z2_right = int(local[-1]["z2"]) if int(local[-1]["z2_reliable"]) == 1 else -1
            if "gapless_region" in sources:
                ttype, reliability, comment = "gapless_region", "medium", "Contiguous gapless interval."
            else:
                ttype, reliability, comment = _classify(z2_left, z2_right, gap_min)
            transitions.append(
                {
                    "transition_id": f"T{len(transitions) + 1}",
                    "lm": float(lm),
                    "v_left": float(lo),
                    "v_right": float(hi),
                    "z2_left": int(z2_left),
                    "z2_right": int(z2_right),
                    "gap_min": float(gap_min),
                    "estimated_vc": float(vc),
                    "kx_min": float(min_row["kx_min"]),
                    "ky_min": float(min_row["ky_min"]),
                    "transition_type": ttype,
                    "reliability": reliability,
                    "comment": comment,
                }
            )

    csv_path = out_dir / "transition_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "transition_id",
                "lm",
                "v_left",
                "v_right",
                "z2_left",
                "z2_right",
                "gap_min",
                "estimated_vc",
                "kx_min",
                "ky_min",
                "transition_type",
                "reliability",
                "comment",
            ],
        )
        writer.writeheader()
        for row in transitions:
            writer.writerow(row)

    for idx, tr in enumerate(transitions, start=1):
        _plot_zoom(rows, lm=float(tr["lm"]), vc=float(tr["estimated_vc"]), idx=idx, out_dir=out_dir)

    print(f"[ok] csv={csv_path}")
    print(f"[ok] out_dir={out_dir}")


if __name__ == "__main__":
    main()
