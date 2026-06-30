#!/usr/bin/env python3
"""Spin Chern and Z2 diagnostics for strict t34 + soc34-only model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_soc34_only_common import (  # noqa: E402
    ModelParams,
    chern_fukui,
    commutator_norm_h_sz,
    compute_bulk_gap,
    output_dirs,
    parse_float_list,
    h_spin_block_t34_soc34,
    write_csv,
    write_model_check,
    z2_crossing_raw,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QSH invariants for strict t34+soc34-only model.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only")
    p.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    p.add_argument("--nk-chern", type=int, default=61)
    p.add_argument("--nk-z2", type=int, default=101)
    p.add_argument("--spin-threshold", type=float, default=1e-8)
    p.add_argument("--integer-threshold", type=float, default=1e-3)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def load_bulk_gap(path: Path) -> dict[float, float]:
    out: dict[float, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            out[round(float(r["v"]), 10)] = float(r["bulk_gap"])
    return out


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_model_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out = dirs["02_spin_chern_z2"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_list = parse_float_list(args.v_list)
    bg = load_bulk_gap(base / "bulk_gap_t34_soc34_only.csv")

    rows = []
    for v in v_list:
        gap = bg.get(round(v, 10))
        if gap is None:
            gap, _, _ = compute_bulk_gap(v=v, params=params, nk=101, n_occ=4)
        comm = commutator_norm_h_sz(v=v, params=params, nk=17)
        spin_conserved = bool(comm < args.spin_threshold)

        c_up = chern_fukui(
            lambda kx, ky: h_spin_block_t34_soc34(kx, ky, v=v, params=params, spin_sign=+1),
            n_occ=2,
            nk=args.nk_chern,
        )
        c_dn = chern_fukui(
            lambda kx, ky: h_spin_block_t34_soc34(kx, ky, v=v, params=params, spin_sign=-1),
            n_occ=2,
            nk=args.nk_chern,
        )
        c_spin = 0.5 * (c_up - c_dn)
        cup_r = int(np.rint(c_up))
        cdn_r = int(np.rint(c_dn))
        csp_r = int(np.rint(c_spin))
        err_up = abs(c_up - cup_r)
        err_dn = abs(c_dn - cdn_r)
        integer_conv = bool(err_up < args.integer_threshold and err_dn < args.integer_threshold)

        z2_raw, (c45, c50, c55) = z2_crossing_raw(v=v, params=params, nkx=args.nk_z2, nky=args.nk_z2)
        z2_sc = int(csp_r % 2)
        if spin_conserved and integer_conv:
            z2_final = z2_sc
            warning = ""
            if z2_raw != z2_sc:
                warning = (
                    "WARNING: Wilson crossing Z2 disagrees with spin Chern parity. "
                    "Since spin is conserved, final Z2 follows C_spin mod 2."
                )
        else:
            z2_final = z2_raw
            warning = "use_crossing_z2_due_to_nonideal_spin_or_noninteger_chern"
        qsh_supported = int(csp_r == 1 and z2_final == 1)

        rows.append(
            {
                "v": float(v),
                "bulk_gap": float(gap),
                "spin_conserved": int(spin_conserved),
                "commutator_norm_H_sz": float(comm),
                "C_up": float(c_up),
                "C_down": float(c_dn),
                "C_spin": float(c_spin),
                "C_up_rounded": int(cup_r),
                "C_down_rounded": int(cdn_r),
                "C_spin_rounded": int(csp_r),
                "integer_error_up": float(err_up),
                "integer_error_down": float(err_dn),
                "crossings_ref_0p45": int(c45),
                "crossings_ref_0p50": int(c50),
                "crossings_ref_0p55": int(c55),
                "Z2_crossing_raw": int(z2_raw),
                "Z2_from_spin_chern": int(z2_sc),
                "Z2_final": int(z2_final),
                "QSH_supported": int(qsh_supported),
                "warning": warning,
            }
        )
        print(
            f"[qsh-soc34] v={v:.1f} C_spin={c_spin:.3f}->{csp_r} "
            f"Z2(raw/final)={z2_raw}/{z2_final} QSH={bool(qsh_supported)}"
        )

    fields = [
        "v",
        "bulk_gap",
        "spin_conserved",
        "commutator_norm_H_sz",
        "C_up",
        "C_down",
        "C_spin",
        "C_up_rounded",
        "C_down_rounded",
        "C_spin_rounded",
        "integer_error_up",
        "integer_error_down",
        "crossings_ref_0p45",
        "crossings_ref_0p50",
        "crossings_ref_0p55",
        "Z2_crossing_raw",
        "Z2_from_spin_chern",
        "Z2_final",
        "QSH_supported",
        "warning",
    ]
    csv_path = out / "qsh_invariant_t34_soc34_only.csv"
    write_csv(csv_path, rows, fields)
    write_csv(base / "qsh_invariant_t34_soc34_only.csv", rows, fields)

    v = np.array([float(r["v"]) for r in rows], dtype=float)
    gap = np.array([float(r["bulk_gap"]) for r in rows], dtype=float)
    cspin = np.array([float(r["C_spin"]) for r in rows], dtype=float)
    z2 = np.array([float(r["Z2_final"]) for r in rows], dtype=float)
    qsh = np.array([float(r["QSH_supported"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.6), sharex=True)
    axes[0].plot(v, gap, marker="o")
    axes[0].set_ylabel("bulk gap")
    axes[0].set_title("qsh invariants (t34+soc34-only)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(v, cspin, marker="o", label="C_spin")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("C_spin")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].step(v, z2, where="mid", label="Z2_final")
    axes[2].step(v, qsh, where="mid", linestyle="--", label="QSH_supported")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_ylabel("Z2 / QSH")
    axes[2].set_xlabel("v")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "qsh_invariant_t34_soc34_only.png", dpi=170)
    fig.savefig(base / "qsh_invariant_t34_soc34_only.png", dpi=170)
    plt.close(fig)
    print(f"[ok] qsh summary: {csv_path}")


if __name__ == "__main__":
    main()
