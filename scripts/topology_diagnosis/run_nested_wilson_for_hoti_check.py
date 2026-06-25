#!/usr/bin/env python3
"""Nested Wilson reliability check for HOTI diagnosis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.common_topology import ensure_dir
from scripts.topology_diagnosis.run_nested_wilson_vscan_strict import (
    nested_polarization,
    track_sector_frames,
    wilson_data_x,
    wilson_data_y,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested Wilson loop only where allowed.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs/wilson_polarization_check")
    parser.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--nkx", type=int, default=101)
    parser.add_argument("--nky", type=int, default=101)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    v_list = parse_float_list(args.v_list)

    pol_path = out_root / "wilson_polarization_summary.csv"
    if not pol_path.exists():
        raise FileNotFoundError(
            f"{pol_path} not found. Please run run_wilson_polarization_check.py first."
        )
    pol_rows = read_csv(pol_path)
    pol_by_v = {float(r["v"]): r for r in pol_rows}

    rows: list[dict] = []

    for v in v_list:
        if v not in pol_by_v:
            raise RuntimeError(f"v={v:.3f} not found in {pol_path}")
        p = pol_by_v[v]
        gap_x = float(p["wannier_gap_x"])
        gap_y = float(p["wannier_gap_y"])
        nested_allowed = int(p["nested_allowed"]) == 1

        if not nested_allowed:
            rows.append(
                {
                    "v": float(v),
                    "p_y_nu_x": "nan",
                    "p_x_nu_y": "nan",
                    "qxy": "nan",
                    "wannier_gap_x": float(gap_x),
                    "wannier_gap_y": float(gap_y),
                    "min_sector_overlap_x": 0.0,
                    "min_sector_overlap_y": 0.0,
                    "nested_reliable": 0,
                    "nested_supports_hoti": 0,
                    "reason": "Wannier sector gap closed or too small",
                }
            )
            print(f"[nested-hoti] v={v:.1f} skipped (nested_allowed=False)")
            continue

        ky, centers_x, eigvec_x, occ_base_x = wilson_data_x(
            v=v,
            t=args.t,
            w=args.w,
            lm=args.lm,
            n_occ=args.n_occ,
            nkx=args.nkx,
            nky=args.nky,
        )
        kx, centers_y, eigvec_y, occ_base_y = wilson_data_y(
            v=v,
            t=args.t,
            w=args.w,
            lm=args.lm,
            n_occ=args.n_occ,
            nkx=args.nkx,
            nky=args.nky,
        )
        _ = ky, kx  # keep explicit use for clarity

        track_x, min_ov_x = track_sector_frames(centers_x, eigvec_x, n_occ=args.n_occ)
        track_y, min_ov_y = track_sector_frames(centers_y, eigvec_y, n_occ=args.n_occ)

        p_y_nu_x = float(nested_polarization(track_x, occ_base_x))
        p_x_nu_y = float(nested_polarization(track_y, occ_base_y))
        qxy = float(np.mod(0.5 * (p_y_nu_x + p_x_nu_y), 1.0))

        nested_reliable = bool(gap_x > 0.05 and gap_y > 0.05 and min_ov_x > 0.5 and min_ov_y > 0.5)
        nested_supports_hoti = bool(nested_reliable and abs(qxy - 0.5) < 0.05)

        reason = "ok"
        if not nested_reliable:
            reason_parts = []
            if gap_x <= 0.05 or gap_y <= 0.05:
                reason_parts.append("Wannier sector gap closed or too small")
            if min_ov_x <= 0.5 or min_ov_y <= 0.5:
                reason_parts.append("Wannier sector tracking overlap too small")
            reason = "; ".join(reason_parts) if reason_parts else "nested reliability thresholds not satisfied"

        rows.append(
            {
                "v": float(v),
                "p_y_nu_x": float(p_y_nu_x),
                "p_x_nu_y": float(p_x_nu_y),
                "qxy": float(qxy),
                "wannier_gap_x": float(gap_x),
                "wannier_gap_y": float(gap_y),
                "min_sector_overlap_x": float(min_ov_x),
                "min_sector_overlap_y": float(min_ov_y),
                "nested_reliable": int(nested_reliable),
                "nested_supports_hoti": int(nested_supports_hoti),
                "reason": reason,
            }
        )
        print(
            f"[nested-hoti] v={v:.1f} qxy={qxy:.4f} "
            f"reliable={nested_reliable} overlap=({min_ov_x:.3f},{min_ov_y:.3f})"
        )

    write_csv(
        out_root / "nested_hoti_check_summary.csv",
        rows,
        [
            "v",
            "p_y_nu_x",
            "p_x_nu_y",
            "qxy",
            "wannier_gap_x",
            "wannier_gap_y",
            "min_sector_overlap_x",
            "min_sector_overlap_y",
            "nested_reliable",
            "nested_supports_hoti",
            "reason",
        ],
    )
    print(f"[ok] saved {out_root / 'nested_hoti_check_summary.csv'}")


if __name__ == "__main__":
    main()
