#!/usr/bin/env python3
"""Post-process 3D summary results into phase labels and SOC deltas."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def classify_phase(row: dict[str, str], small_gap_threshold: float) -> tuple[str, str]:
    gap_min = to_float(row.get("gap_kz_min"))
    z2_count = to_int(row.get("z2_kz_ones_count"))
    soc_state = row.get("soc_state", "")

    if gap_min <= 0.0:
        return "overlap_or_semimetal_like", "gap_kz_min <= 0"
    if gap_min < small_gap_threshold:
        if z2_count > 0:
            return "near_transition_z2_active", f"0 < gap_kz_min < {small_gap_threshold} and z2_kz_ones_count > 0"
        return "near_transition_trivial_like", f"0 < gap_kz_min < {small_gap_threshold} and z2_kz_ones_count = 0"
    if z2_count > 0:
        return "gapped_z2_like_candidate", f"gap_kz_min >= {small_gap_threshold} and z2_kz_ones_count > 0"
    return "gapped_trivial_like", f"gap_kz_min >= {small_gap_threshold} and z2_kz_ones_count = 0 ({soc_state})"


def build_soc_delta_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float, float], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (
            round(to_float(row.get("v")), 10),
            round(to_float(row.get("t")), 10),
            round(to_float(row.get("w")), 10),
            round(to_float(row.get("J")), 10),
        )
        grouped.setdefault(key, {})[row.get("soc_state", "unknown")] = row

    delta_rows: list[dict[str, Any]] = []
    for (v, t, w, j), pair in sorted(grouped.items()):
        on = pair.get("soc_on")
        off = pair.get("soc_off")
        if on is None or off is None:
            delta_rows.append(
                {
                    "v": v,
                    "t": t,
                    "w": w,
                    "J": j,
                    "has_soc_on": int(on is not None),
                    "has_soc_off": int(off is not None),
                    "delta_gap_kz_min": "",
                    "delta_z2_kz_ones_count": "",
                    "delta_wilson_kz0": "",
                    "comment": "missing_soc_pair",
                }
            )
            continue

        on_gap = to_float(on.get("gap_kz_min"))
        off_gap = to_float(off.get("gap_kz_min"))
        on_z2_count = to_int(on.get("z2_kz_ones_count"))
        off_z2_count = to_int(off.get("z2_kz_ones_count"))
        on_wil = to_float(on.get("wilson_kz0"))
        off_wil = to_float(off.get("wilson_kz0"))

        delta_rows.append(
            {
                "v": v,
                "t": t,
                "w": w,
                "J": j,
                "has_soc_on": 1,
                "has_soc_off": 1,
                "gap_kz_min_soc_on": on_gap,
                "gap_kz_min_soc_off": off_gap,
                "delta_gap_kz_min": on_gap - off_gap,
                "z2_kz_ones_soc_on": on_z2_count,
                "z2_kz_ones_soc_off": off_z2_count,
                "delta_z2_kz_ones_count": on_z2_count - off_z2_count,
                "wilson_kz0_soc_on": on_wil,
                "wilson_kz0_soc_off": off_wil,
                "delta_wilson_kz0": on_wil - off_wil,
                "comment": "ok",
            }
        )
    return delta_rows


def save_transition_plot(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_soc: dict[str, list[dict[str, str]]] = {"soc_on": [], "soc_off": []}
    for row in rows:
        soc = row.get("soc_state", "")
        if soc in by_soc:
            by_soc[soc].append(row)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)

    for soc, color in (("soc_on", "tab:blue"), ("soc_off", "tab:orange")):
        pts = sorted(by_soc[soc], key=lambda r: to_float(r.get("v")))
        if not pts:
            continue
        v = np.array([to_float(p.get("v")) for p in pts], dtype=float)
        gap_min = np.array([to_float(p.get("gap_kz_min")) for p in pts], dtype=float)
        z2_count = np.array([to_int(p.get("z2_kz_ones_count")) for p in pts], dtype=float)
        axes[0].plot(v, gap_min, marker="o", linewidth=1.3, color=color, label=soc)
        axes[1].plot(v, z2_count, marker="o", linewidth=1.3, color=color, label=soc)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("gap_kz_min")
    axes[0].set_title("3D transition corridor summary")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel("v")
    axes[1].set_ylabel("z2_kz_ones_count")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def pick_key_points(
    phase_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    small_gap_threshold: float,
) -> list[str]:
    lines: list[str] = []
    lines.append("# key_points_3d")
    lines.append("")
    lines.append(f"- classification_small_gap_threshold: `{small_gap_threshold}`")
    lines.append("")

    # smallest positive gap in each soc state
    for soc in ("soc_on", "soc_off"):
        cand = [
            r for r in phase_rows if r.get("soc_state") == soc and to_float(r.get("gap_kz_min")) > 0
        ]
        if cand:
            best = min(cand, key=lambda r: to_float(r.get("gap_kz_min")))
            lines.append(
                f"- smallest_positive_gap_{soc}: {best['point_id']} (v={best['v']}, gap_kz_min={to_float(best['gap_kz_min']):.6e}, phase={best['phase_label']})"
            )

    robust = [r for r in phase_rows if r.get("phase_label") == "gapped_z2_like_candidate"]
    robust = sorted(robust, key=lambda r: to_float(r.get("gap_kz_min")), reverse=True)
    if robust:
        lines.append("- strongest_gapped_z2_like_candidates:")
        for r in robust[:3]:
            lines.append(
                f"  - {r['point_id']} (v={r['v']}, soc={r['soc_state']}, gap_kz_min={to_float(r['gap_kz_min']):.6e}, z2_count={to_int(r['z2_kz_ones_count'])})"
            )

    if delta_rows:
        valid = [r for r in delta_rows if r.get("comment") == "ok"]
        if valid:
            max_gap_open = max(valid, key=lambda r: to_float(r.get("delta_gap_kz_min")))
            max_z2_shift = max(valid, key=lambda r: abs(to_float(r.get("delta_z2_kz_ones_count"))))
            lines.append("")
            lines.append(
                f"- strongest_soc_gap_opening: v={max_gap_open['v']} (delta_gap_kz_min={to_float(max_gap_open['delta_gap_kz_min']):.6e})"
            )
            lines.append(
                f"- strongest_soc_z2_count_shift: v={max_z2_shift['v']} (delta_z2_kz_ones_count={to_float(max_z2_shift['delta_z2_kz_ones_count']):.1f})"
            )
    return lines


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    summary_path = output_root / "summary_3d.csv"
    rows = read_csv(summary_path)
    if not rows:
        raise RuntimeError(f"summary_3d.csv not found or empty: {summary_path}")

    phase_rows: list[dict[str, Any]] = []
    for row in rows:
        label, reason = classify_phase(row, small_gap_threshold=args.small_gap_threshold)
        phase_row = dict(row)
        phase_row["phase_label"] = label
        phase_row["phase_reason"] = reason
        phase_rows.append(phase_row)

    phase_csv = output_root / "phase_label_3d.csv"
    phase_fields = list(rows[0].keys()) + ["phase_label", "phase_reason"]
    write_csv(phase_csv, phase_rows, phase_fields)

    delta_rows = build_soc_delta_table(rows)
    delta_csv = output_root / "soc_delta_table.csv"
    delta_fields = [
        "v",
        "t",
        "w",
        "J",
        "has_soc_on",
        "has_soc_off",
        "gap_kz_min_soc_on",
        "gap_kz_min_soc_off",
        "delta_gap_kz_min",
        "z2_kz_ones_soc_on",
        "z2_kz_ones_soc_off",
        "delta_z2_kz_ones_count",
        "wilson_kz0_soc_on",
        "wilson_kz0_soc_off",
        "delta_wilson_kz0",
        "comment",
    ]
    write_csv(delta_csv, delta_rows, delta_fields)

    plot_path = output_root / "transition_corridor_3d.png"
    save_transition_plot(rows, output_path=plot_path)

    key_md = output_root / "key_points_3d.md"
    key_lines = pick_key_points(phase_rows, delta_rows, small_gap_threshold=args.small_gap_threshold)
    key_md.write_text("\n".join(key_lines) + "\n", encoding="utf-8")

    run_log = output_root / "post_analysis_log.txt"
    run_log.write_text(
        "\n".join(
            [
                "3D post analysis generated",
                f"input_summary={summary_path}",
                f"phase_label={phase_csv}",
                f"soc_delta={delta_csv}",
                f"transition_plot={plot_path}",
                f"key_points={key_md}",
                f"small_gap_threshold={args.small_gap_threshold}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] phase_label_3d.csv: {phase_csv}")
    print(f"[ok] soc_delta_table.csv: {delta_csv}")
    print(f"[ok] transition_corridor_3d.png: {plot_path}")
    print(f"[ok] key_points_3d.md: {key_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive 3D post-analysis files from summary_3d.csv.")
    parser.add_argument("--output-root", default="/workspace/outputs/soc_comparison_3d")
    parser.add_argument("--small-gap-threshold", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
