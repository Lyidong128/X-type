#!/usr/bin/env python3
"""Run spin-Chern convergence analysis near M-driven transition windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.M_point_spin_chern_analysis.common import (
    compute_spin_chern_pair,
    direct_gap_global,
    ensure_dir,
    load_model,
    nearest_value,
    set_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spin Chern convergence scan.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/M_point_spin_chern_analysis")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _pick_points(output_root: Path) -> list[float]:
    base = [0.55, 0.58, 0.595, 0.600, 0.605, 0.62, 0.65, 0.80, 1.00]
    g_csv = output_root / "global_gap_scan.csv"
    if not g_csv.exists():
        return sorted(set(base + [1.08, 1.09, 1.093, 1.096, 1.10]))

    rows = list(csv.DictReader(g_csv.open("r", encoding="utf-8")))
    rows_2nd = [r for r in rows if 1.08 <= float(r["v"]) <= 1.10]
    if not rows_2nd:
        return sorted(set(base + [1.08, 1.09, 1.093, 1.096, 1.10]))

    v_candidates = sorted({float(r["v"]) for r in rows_2nd})
    v_min = float(min(rows_2nd, key=lambda r: float(r["Delta_global"]))["v"])
    extra = [
        nearest_value(v_min - 0.003, v_candidates),
        v_min,
        nearest_value(v_min + 0.003, v_candidates),
        nearest_value(1.09, v_candidates),
    ]
    return sorted(set(base + extra))


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    model = load_model(args.model_file)

    points = _pick_points(out_root)
    nk_list = [21, 31, 41] if args.quick else [31, 51, 81, 101]
    nk_gap = 31 if args.quick else 61

    raw_by_v = {}
    for v in points:
        set_params(model, v=v, lm=args.lm, t=args.t, w=args.w, j=0.0)
        gap, _, _ = direct_gap_global(model, nk=nk_gap, n_occ=4)
        rows_v = []
        for nk in nk_list:
            cup, cdn = compute_spin_chern_pair(model, nk=nk)
            ctot = float(cup + cdn)
            cspin = float(0.5 * (cup - cdn))
            rows_v.append(
                {
                    "v": float(v),
                    "lm": float(args.lm),
                    "t": float(args.t),
                    "w": float(args.w),
                    "Nk": int(nk),
                    "gap": float(gap),
                    "C_up": float(cup),
                    "C_down": float(cdn),
                    "C_total": float(ctot),
                    "C_spin": float(cspin),
                }
            )
        raw_by_v[float(v)] = rows_v

    # Determine convergence reliability using largest-Nk rounded integers as targets.
    output_rows = []
    for v, rows_v in raw_by_v.items():
        rows_sorted = sorted(rows_v, key=lambda r: int(r["Nk"]))
        ref = rows_sorted[-1]
        rcu = int(round(float(ref["C_up"])))
        rcd = int(round(float(ref["C_down"])))
        rcs = int(round(float(ref["C_spin"])))
        max_dev = max(
            max(
                abs(float(r["C_up"]) - rcu),
                abs(float(r["C_down"]) - rcd),
                abs(float(r["C_spin"]) - rcs),
            )
            for r in rows_sorted
        )
        for r in rows_sorted:
            if float(r["gap"]) < 1e-4:
                reliability = "critical_or_gapless"
                comment = "Gap below threshold; invariant at this point is not strictly stable."
            elif max_dev < 0.05:
                reliability = "reliable"
                comment = "Spin Chern converges to integers across Nk."
            elif max_dev < 0.15:
                reliability = "semi_reliable"
                comment = "Moderate Nk dependence; near-integer but not fully converged."
            else:
                reliability = "uncertain"
                comment = "Strong Nk dependence; convergence not established."
            output_rows.append(
                {
                    **r,
                    "rounded_C_up": rcu,
                    "rounded_C_down": rcd,
                    "rounded_C_spin": rcs,
                    "reliability": reliability,
                    "comment": comment,
                }
            )

    csv_path = out_root / "spin_chern_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "lm",
                "t",
                "w",
                "Nk",
                "gap",
                "C_up",
                "C_down",
                "C_total",
                "C_spin",
                "rounded_C_up",
                "rounded_C_down",
                "rounded_C_spin",
                "reliability",
                "comment",
            ],
        )
        writer.writeheader()
        for row in sorted(output_rows, key=lambda r: (float(r["v"]), int(r["Nk"]))):
            writer.writerow(row)

    # Plot 1: C_up / C_down versus v using largest Nk.
    top_rows = []
    for v in sorted(raw_by_v.keys()):
        rows_v = sorted(raw_by_v[v], key=lambda r: int(r["Nk"]))
        top_rows.append(rows_v[-1])
    v_arr = np.array([float(r["v"]) for r in top_rows], dtype=float)
    cup_arr = np.array([float(r["C_up"]) for r in top_rows], dtype=float)
    cdn_arr = np.array([float(r["C_down"]) for r in top_rows], dtype=float)
    cs_arr = np.array([float(r["C_spin"]) for r in top_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    ax.plot(v_arr, cup_arr, marker="o", linewidth=1.2, label="C_up")
    ax.plot(v_arr, cdn_arr, marker="s", linewidth=1.2, label="C_down")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("v")
    ax.set_ylabel("Chern")
    ax.set_title(f"Spin-resolved Chern vs v (Nk={max(nk_list)})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "Cup_Cdown_vs_v.png", dpi=180)
    plt.close(fig)

    # Plot 2: C_spin versus v using largest Nk.
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    ax.plot(v_arr, cs_arr, marker="o", linewidth=1.2, label="C_spin")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("v")
    ax.set_ylabel("C_spin")
    ax.set_title(f"Spin Chern vs v (Nk={max(nk_list)})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "Cspin_vs_v.png", dpi=180)
    plt.close(fig)

    # Plot 3: convergence metric versus Nk.
    nk_sorted = sorted(nk_list)
    dev_up = []
    dev_dn = []
    dev_spin = []
    for nk in nk_sorted:
        subset = [r for r in output_rows if int(r["Nk"]) == nk]
        dev_up.append(np.mean([abs(float(r["C_up"]) - float(r["rounded_C_up"])) for r in subset]))
        dev_dn.append(np.mean([abs(float(r["C_down"]) - float(r["rounded_C_down"])) for r in subset]))
        dev_spin.append(np.mean([abs(float(r["C_spin"]) - float(r["rounded_C_spin"])) for r in subset]))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(nk_sorted, dev_up, marker="o", linewidth=1.2, label="mean |C_up - round|")
    ax.plot(nk_sorted, dev_dn, marker="s", linewidth=1.2, label="mean |C_down - round|")
    ax.plot(nk_sorted, dev_spin, marker="^", linewidth=1.2, label="mean |C_spin - round|")
    ax.set_xlabel("Nk")
    ax.set_ylabel("Convergence deviation")
    ax.set_title("Spin Chern convergence versus k-mesh size")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "spin_chern_convergence_vs_Nk.png", dpi=180)
    plt.close(fig)

    print(f"[ok] csv={csv_path}")
    print(f"[ok] points={len(points)} nk_list={nk_list}")


if __name__ == "__main__":
    main()
