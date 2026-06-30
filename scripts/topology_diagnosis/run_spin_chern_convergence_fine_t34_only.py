#!/usr/bin/env python3
"""Spin Chern convergence checks near transition for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_only_common import (  # noqa: E402
    ModelParams,
    chern_fukui,
    h_spin_block_t34,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spin Chern convergence around fine-scan transition.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    p.add_argument("--v-check", default="0.88,0.90,0.92,0.94,0.96,0.98,1.00,1.02")
    p.add_argument("--nk-list", default="41,61,81,101")
    p.add_argument("--integer-threshold", type=float, default=1e-3)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    out = base / "fine_scan_0p8_1p2" / "02_spin_chern_z2"
    out.mkdir(parents=True, exist_ok=True)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_check = parse_float_list(args.v_check)
    nk_list = parse_int_list(args.nk_list)

    rows = []
    for v in v_check:
        for nk in nk_list:
            c_up = chern_fukui(
                lambda kx, ky: h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=+1),
                n_occ=2,
                nk=nk,
            )
            c_dn = chern_fukui(
                lambda kx, ky: h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=-1),
                n_occ=2,
                nk=nk,
            )
            c_spin = 0.5 * (c_up - c_dn)
            cup_r = int(np.rint(c_up))
            cdn_r = int(np.rint(c_dn))
            csp_r = int(np.rint(c_spin))
            err_up = abs(c_up - cup_r)
            err_dn = abs(c_dn - cdn_r)
            conv = bool(err_up < args.integer_threshold and err_dn < args.integer_threshold)
            rows.append(
                {
                    "v": float(v),
                    "nk": int(nk),
                    "C_up": float(c_up),
                    "C_down": float(c_dn),
                    "C_spin": float(c_spin),
                    "C_up_rounded": int(cup_r),
                    "C_down_rounded": int(cdn_r),
                    "C_spin_rounded": int(csp_r),
                    "integer_error_up": float(err_up),
                    "integer_error_down": float(err_dn),
                    "converged": int(conv),
                }
            )
            print(f"[chern-conv-t34] v={v:.2f} nk={nk} C_spin={c_spin:.4f} conv={conv}")

    fields = [
        "v",
        "nk",
        "C_up",
        "C_down",
        "C_spin",
        "C_up_rounded",
        "C_down_rounded",
        "C_spin_rounded",
        "integer_error_up",
        "integer_error_down",
        "converged",
    ]
    csv_path = out / "spin_chern_convergence_fine_t34_only.csv"
    write_csv(csv_path, rows, fields)
    (base / "fine_scan_0p8_1p2" / "spin_chern_convergence_fine_t34_only.csv").write_bytes(csv_path.read_bytes())

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for v in v_check:
        sub = sorted([r for r in rows if abs(float(r["v"]) - v) < 1e-9], key=lambda x: int(x["nk"]))
        x = [int(r["nk"]) for r in sub]
        y = [float(r["C_spin"]) for r in sub]
        ax.plot(x, y, marker="o", linewidth=1.1, label=f"v={v:.2f}")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("nk")
    ax.set_ylabel("C_spin")
    ax.set_title("Spin Chern convergence near transition (t34-only)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    png = out / "spin_chern_convergence_fine_t34_only.png"
    fig.savefig(png, dpi=180)
    fig.savefig(base / "fine_scan_0p8_1p2" / "spin_chern_convergence_fine_t34_only.png", dpi=180)
    plt.close(fig)
    print(f"[ok] spin-chern convergence summary: {csv_path}")


if __name__ == "__main__":
    main()
