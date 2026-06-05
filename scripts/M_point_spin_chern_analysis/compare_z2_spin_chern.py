#!/usr/bin/env python3
"""Compare spin-Chern-derived Z2 with Fu-Kane and Wilson diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Z2 sources against spin Chern.")
    parser.add_argument("--output-root", default="/workspace/outputs/M_point_spin_chern_analysis")
    parser.add_argument("--legacy-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def _nearest_row(rows: list[dict[str, str]], v: float, v_key: str = "v"):
    if not rows:
        return None
    return min(rows, key=lambda r: abs(float(r[v_key]) - float(v)))


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    spin_csv = out_root / "spin_chern_convergence.csv"
    spin_rows = _read_csv(spin_csv)
    if not spin_rows:
        raise FileNotFoundError(f"Missing prerequisite spin Chern file: {spin_csv}")

    # Keep highest-Nk row for each v.
    by_v = {}
    for row in spin_rows:
        v = round(float(row["v"]), 10)
        nk = int(row["Nk"])
        if (v not in by_v) or (nk > int(by_v[v]["Nk"])):
            by_v[v] = row
    spin_top = [by_v[v] for v in sorted(by_v.keys())]

    fu_rows = _read_csv(Path(args.legacy_root) / "fu_kane" / "fu_kane_summary.csv")
    gap_rows = _read_csv(Path(args.legacy_root) / "gap_z2_scan" / "gap_z2_scan.csv")
    gap_rows = [r for r in gap_rows if abs(float(r.get("lm", "nan")) - 0.1) < 1e-12]

    out_rows = []
    for row in spin_top:
        v = float(row["v"])
        cup = float(row["rounded_C_up"])
        cdn = float(row["rounded_C_down"])
        cspin = float(row["rounded_C_spin"])
        z2_spin = int(cup) % 2

        fu = _nearest_row(fu_rows, v) if fu_rows else None
        z2_fu = int(fu["z2_fu_kane"]) if fu is not None else -1
        if fu is not None and abs(float(fu["v"]) - v) > 0.03:
            z2_fu = -1

        gap = _nearest_row(gap_rows, v) if gap_rows else None
        z2_w = -1
        if gap is not None:
            # Prefer explicit Wilson channel if present.
            if "z2_wilson" in gap and str(gap["z2_wilson"]).strip() != "":
                z2_w = int(float(gap["z2_wilson"]))
            elif int(gap.get("z2_reliable", "0")) == 1:
                z2_w = int(float(gap["z2"]))
            if abs(float(gap["v"]) - v) > 0.02:
                z2_w = -1

        mismatch_fu = (z2_fu >= 0) and (z2_fu != z2_spin)
        mismatch_w = (z2_w >= 0) and (z2_w != z2_spin)
        if (z2_fu < 0) and (z2_w < 0):
            label = "z2_unavailable"
            comment = "Fu-Kane/Wilson unavailable near this v."
        elif not mismatch_fu and not mismatch_w:
            label = "all_consistent"
            comment = "Spin-Chern-derived Z2 agrees with available references."
        elif mismatch_fu and mismatch_w:
            label = "spin_chern_vs_fu_kane_inconsistent"
            comment = (
                "Both Fu-Kane and Wilson disagree with spin Chern; check inversion parity implementation "
                "and Wilson gauge tracking."
            )
        elif mismatch_fu:
            label = "spin_chern_vs_fu_kane_inconsistent"
            comment = "Fu-Kane differs from spin Chern; check inversion operator / parity implementation."
        else:
            label = "spin_chern_vs_wilson_inconsistent"
            comment = "Wilson differs from spin Chern; check Wilson gauge continuity / crossing counting."

        out_rows.append(
            {
                "v": v,
                "lm": float(row["lm"]),
                "C_up": cup,
                "C_down": cdn,
                "C_spin": cspin,
                "z2_from_spin_chern": z2_spin,
                "z2_fu_kane": z2_fu,
                "z2_wilson": z2_w,
                "consistency_label": label,
                "comment": comment,
            }
        )

    out_csv = out_root / "z2_spin_chern_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "lm",
                "C_up",
                "C_down",
                "C_spin",
                "z2_from_spin_chern",
                "z2_fu_kane",
                "z2_wilson",
                "consistency_label",
                "comment",
            ],
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"[ok] csv={out_csv}")


if __name__ == "__main__":
    main()
