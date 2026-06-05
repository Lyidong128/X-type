#!/usr/bin/env python3
"""Extract M-point effective mass proxy across the scanned v windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.M_point_spin_chern_analysis.common import ensure_dir, load_model, m_point_from_model, set_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract M-point effective mass proxy.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/M_point_spin_chern_analysis")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _normalize_phase(vec_ref: np.ndarray, vec_new: np.ndarray) -> np.ndarray:
    phase = np.vdot(vec_ref, vec_new)
    if abs(phase) < 1e-16:
        return vec_new
    return vec_new * np.exp(-1j * np.angle(phase))


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    m_csv = out_root / "M_gap_scan.csv"
    if not m_csv.exists():
        skip = out_root / "skip_reason.txt"
        skip.write_text("Missing M_gap_scan.csv; run run_M_gap_scan.py first.\n", encoding="utf-8")
        print(f"[skip] {skip}")
        return

    rows = list(csv.DictReader(m_csv.open("r", encoding="utf-8")))
    if not rows:
        skip = out_root / "skip_reason.txt"
        skip.write_text("M_gap_scan.csv is empty.\n", encoding="utf-8")
        print(f"[skip] {skip}")
        return

    model = load_model(args.model_file)
    mpt = m_point_from_model(model)

    # reference point: smallest v in primary window [0.55,0.65], fallback global minimum-v.
    primary = sorted([float(r["v"]) for r in rows if 0.55 <= float(r["v"]) <= 0.65])
    v_ref = primary[0] if primary else min(float(r["v"]) for r in rows)
    set_params(model, v=v_ref, lm=args.lm, t=args.t, w=args.w, j=0.0)
    _, evecs_ref = np.linalg.eigh(model.Hxtype(mpt))
    ref_val = evecs_ref[:, 3]
    ref_con = evecs_ref[:, 4]

    q = 5e-3 if args.quick else 2e-3
    output = []
    for row in sorted(rows, key=lambda r: float(r["v"])):
        v = float(row["v"])
        set_params(model, v=v, lm=args.lm, t=args.t, w=args.w, j=0.0)
        evals, evecs = np.linalg.eigh(model.Hxtype(mpt))
        val = evecs[:, 3]
        con = evecs[:, 4]

        ov_vv = float(abs(np.vdot(ref_val, val)))
        ov_vc = float(abs(np.vdot(ref_con, val)))
        sign = 1 if ov_vv >= ov_vc else -1
        delta_m = float(np.real(evals[4] - evals[3]))
        signed_mass_proxy = float(sign * 0.5 * delta_m)

        # rough velocity scale around M from low-energy splitting gradients.
        kpx = mpt + np.array([q, 0.0, 0.0], dtype=float)
        kmx = mpt + np.array([-q, 0.0, 0.0], dtype=float)
        kpy = mpt + np.array([0.0, q, 0.0], dtype=float)
        kmy = mpt + np.array([0.0, -q, 0.0], dtype=float)
        ep_x = np.linalg.eigvalsh(model.Hxtype(kpx))
        em_x = np.linalg.eigvalsh(model.Hxtype(kmx))
        ep_y = np.linalg.eigvalsh(model.Hxtype(kpy))
        em_y = np.linalg.eigvalsh(model.Hxtype(kmy))
        vx = float(0.5 * abs((ep_x[4] - ep_x[3]) - (em_x[4] - em_x[3])) / (2 * q))
        vy = float(0.5 * abs((ep_y[4] - ep_y[3]) - (em_y[4] - em_y[3])) / (2 * q))

        output.append(
            {
                "v": v,
                "lm": float(args.lm),
                "t": float(args.t),
                "w": float(args.w),
                "Delta_M": delta_m,
                "band_character_sign": int(sign),
                "signed_mass_proxy": signed_mass_proxy,
                "overlap_val_ref_val": ov_vv,
                "overlap_val_ref_con": ov_vc,
                "vx_est": vx,
                "vy_est": vy,
                "comment": "signed_mass_proxy from M-point component continuity",
            }
        )

    out_csv = out_root / "M_effective_mass.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "lm",
                "t",
                "w",
                "Delta_M",
                "band_character_sign",
                "signed_mass_proxy",
                "overlap_val_ref_val",
                "overlap_val_ref_con",
                "vx_est",
                "vy_est",
                "comment",
            ],
        )
        writer.writeheader()
        for row in output:
            writer.writerow(row)

    v_arr = np.array([float(r["v"]) for r in output], dtype=float)
    m_arr = np.array([float(r["signed_mass_proxy"]) for r in output], dtype=float)
    idx_min_abs = int(np.argmin(np.abs(m_arr)))
    vc_mass = float(v_arr[idx_min_abs])

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(v_arr, m_arr, linewidth=1.2, label="signed_mass_proxy")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(vc_mass, color="red", linestyle=":", linewidth=0.9, label=f"near sign-change v={vc_mass:.6f}")
    ax.set_xlabel("v")
    ax.set_ylabel("effective mass proxy")
    ax.set_title("M-point effective mass proxy vs v")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "M_effective_mass_vs_v.png", dpi=180)
    plt.close(fig)

    print(f"[ok] csv={out_csv}")
    print(f"[ok] vc_mass_proxy={vc_mass:.6f}")


if __name__ == "__main__":
    main()
