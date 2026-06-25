#!/usr/bin/env python3
"""Spin-Chern convergence check for QSH diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.common_topology import chern_fukui, ensure_dir, h_spin_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug spin Chern convergence.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs/z2_debug")
    parser.add_argument("--v-list", default="0.8,1.0")
    parser.add_argument("--nk-list", default="41,61,81")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    v_list = parse_float_list(args.v_list)
    nk_list = parse_int_list(args.nk_list)

    rows: list[dict] = []
    for v in v_list:
        for nk in nk_list:
            cup = chern_fukui(
                lambda kx, ky: h_spin_block(kx, ky, v=v, t=args.t, w=args.w, lm=args.lm, spin_sign=1),
                n_occ=2,
                nk=nk,
            )
            cdn = chern_fukui(
                lambda kx, ky: h_spin_block(kx, ky, v=v, t=args.t, w=args.w, lm=args.lm, spin_sign=-1),
                n_occ=2,
                nk=nk,
            )
            cspin = 0.5 * (cup - cdn)
            cup_r = int(round(cup))
            cdn_r = int(round(cdn))
            rows.append(
                {
                    "v": float(v),
                    "nk": int(nk),
                    "C_up": float(cup),
                    "C_down": float(cdn),
                    "C_spin": float(cspin),
                    "C_up_rounded": int(cup_r),
                    "C_down_rounded": int(cdn_r),
                    "integer_error_up": float(abs(cup - cup_r)),
                    "integer_error_down": float(abs(cdn - cdn_r)),
                }
            )
            print(
                f"[spin-chern] v={v:.1f} nk={nk} "
                f"C_up={cup:.6f} C_down={cdn:.6f} C_spin={cspin:.6f} "
                f"err_up={abs(cup-cup_r):.2e} err_dn={abs(cdn-cdn_r):.2e}"
            )

    write_csv(
        out_root / "spin_chern_convergence.csv",
        rows,
        [
            "v",
            "nk",
            "C_up",
            "C_down",
            "C_spin",
            "C_up_rounded",
            "C_down_rounded",
            "integer_error_up",
            "integer_error_down",
        ],
    )
    print(f"[ok] spin chern convergence csv saved at {out_root / 'spin_chern_convergence.csv'}")


if __name__ == "__main__":
    main()
