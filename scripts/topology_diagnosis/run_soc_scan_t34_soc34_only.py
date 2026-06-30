#!/usr/bin/env python3
"""SOC-strength scan for strict t34+soc34-only model."""

from __future__ import annotations

import argparse
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
    h_spin_block_t34_soc34,
    parse_float_list,
    write_csv,
    write_model_check,
    z2_crossing_raw,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan SOC strength lm for strict t34+soc34-only model.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only_lm_scan")
    p.add_argument("--lm-list", default="0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--nk-bulk", type=int, default=81)
    p.add_argument("--nk-chern", type=int, default=61)
    p.add_argument("--nk-z2", type=int, default=81)
    p.add_argument("--spin-threshold", type=float, default=1e-8)
    p.add_argument("--integer-threshold", type=float, default=1e-3)
    return p.parse_args()


def qsh_interval(rows: list[dict]) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    v = np.array([float(r["v"]) for r in rows], dtype=float)
    q = np.array([int(r["QSH_supported"]) for r in rows], dtype=int)
    idx = np.where(q == 1)[0]
    if idx.size == 0:
        return None, None
    return float(v[idx[0]]), float(v[idx[-1]])


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    base.mkdir(parents=True, exist_ok=True)
    lm_list = parse_float_list(args.lm_list)
    v_list = parse_float_list(args.v_list)

    all_rows: list[dict] = []
    lm_rows: list[dict] = []

    for lm in lm_list:
        params = ModelParams(t34=args.t34, w=args.w, lm=lm)
        # write/overwrite check file for each lm subdir
        lm_dir = base / f"lm_{lm:.1f}".replace(".", "p")
        lm_dir.mkdir(parents=True, exist_ok=True)
        write_model_check(lm_dir, params)

        rows_this_lm: list[dict] = []
        for v in v_list:
            bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=args.nk_bulk, n_occ=4)
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
            c_up_r = int(np.rint(c_up))
            c_dn_r = int(np.rint(c_dn))
            c_spin_r = int(np.rint(c_spin))
            err_up = abs(c_up - c_up_r)
            err_dn = abs(c_dn - c_dn_r)
            integer_conv = bool(err_up < args.integer_threshold and err_dn < args.integer_threshold)

            z2_raw, (c45, c50, c55) = z2_crossing_raw(v=v, params=params, nkx=args.nk_z2, nky=args.nk_z2)
            z2_sc = int(c_spin_r % 2)
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
            qsh = int(c_spin_r == 1 and z2_final == 1)

            row = {
                "lm": float(lm),
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "spin_conserved": int(spin_conserved),
                "commutator_norm_H_sz": float(comm),
                "C_up": float(c_up),
                "C_down": float(c_dn),
                "C_spin": float(c_spin),
                "C_up_rounded": int(c_up_r),
                "C_down_rounded": int(c_dn_r),
                "C_spin_rounded": int(c_spin_r),
                "integer_error_up": float(err_up),
                "integer_error_down": float(err_dn),
                "crossings_ref_0p45": int(c45),
                "crossings_ref_0p50": int(c50),
                "crossings_ref_0p55": int(c55),
                "Z2_crossing_raw": int(z2_raw),
                "Z2_from_spin_chern": int(z2_sc),
                "Z2_final": int(z2_final),
                "QSH_supported": int(qsh),
                "warning": warning,
            }
            all_rows.append(row)
            rows_this_lm.append(row)
            print(
                f"[soc-scan] lm={lm:.1f} v={v:.1f} gap={bulk_gap:.4f} "
                f"C_spin={c_spin_r} Z2={z2_final} QSH={bool(qsh)}"
            )

        # per-lm summary
        g = np.array([float(r["bulk_gap"]) for r in rows_this_lm], dtype=float)
        vv = np.array([float(r["v"]) for r in rows_this_lm], dtype=float)
        i_min = int(np.argmin(g))
        v_on, v_off = qsh_interval(rows_this_lm)
        v08 = next((r for r in rows_this_lm if abs(float(r["v"]) - 0.8) < 1e-9), None)
        v10 = next((r for r in rows_this_lm if abs(float(r["v"]) - 1.0) < 1e-9), None)
        v12 = next((r for r in rows_this_lm if abs(float(r["v"]) - 1.2) < 1e-9), None)
        lm_rows.append(
            {
                "lm": float(lm),
                "v_gap_min": float(vv[i_min]),
                "bulk_gap_min": float(g[i_min]),
                "v_qsh_onset": "" if v_on is None else float(v_on),
                "v_qsh_offset": "" if v_off is None else float(v_off),
                "qsh_state_count": int(np.sum([int(r["QSH_supported"]) for r in rows_this_lm])),
                "qsh_at_v0p8": int(v08["QSH_supported"]) if v08 else -1,
                "qsh_at_v1p0": int(v10["QSH_supported"]) if v10 else -1,
                "qsh_at_v1p2": int(v12["QSH_supported"]) if v12 else -1,
            }
        )
        # Save per-lm details
        write_csv(
            lm_dir / "qsh_invariant_t34_soc34_only.csv",
            rows_this_lm,
            [
                "lm",
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
            ],
        )
        print(
            f"[soc-scan-summary] lm={lm:.1f} min_gap=({vv[i_min]:.2f},{g[i_min]:.6f}) "
            f"QSH_window={[v_on, v_off]}"
        )

    detail_csv = base / "soc_scan_qsh_detail_t34_soc34_only.csv"
    write_csv(
        detail_csv,
        all_rows,
        [
            "lm",
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
        ],
    )
    summary_csv = base / "soc_scan_summary_t34_soc34_only.csv"
    write_csv(
        summary_csv,
        lm_rows,
        [
            "lm",
            "v_gap_min",
            "bulk_gap_min",
            "v_qsh_onset",
            "v_qsh_offset",
            "qsh_state_count",
            "qsh_at_v0p8",
            "qsh_at_v1p0",
            "qsh_at_v1p2",
        ],
    )

    # plot summary
    lms = np.array([float(r["lm"]) for r in lm_rows], dtype=float)
    gmin = np.array([float(r["bulk_gap_min"]) for r in lm_rows], dtype=float)
    qcnt = np.array([float(r["qsh_state_count"]) for r in lm_rows], dtype=float)
    q08 = np.array([float(r["qsh_at_v0p8"]) for r in lm_rows], dtype=float)
    q10 = np.array([float(r["qsh_at_v1p0"]) for r in lm_rows], dtype=float)
    q12 = np.array([float(r["qsh_at_v1p2"]) for r in lm_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes[0, 0].plot(lms, gmin, marker="o")
    axes[0, 0].set_title("min bulk gap vs lm")
    axes[0, 0].set_xlabel("lm")
    axes[0, 0].set_ylabel("bulk_gap_min")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(lms, qcnt, marker="o")
    axes[0, 1].set_title("QSH-supported v-point count vs lm")
    axes[0, 1].set_xlabel("lm")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(lms, q08, marker="o", label="QSH@v=0.8")
    axes[1, 0].plot(lms, q10, marker="s", label="QSH@v=1.0")
    axes[1, 0].plot(lms, q12, marker="^", label="QSH@v=1.2")
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].set_title("QSH indicator at selected v")
    axes[1, 0].set_xlabel("lm")
    axes[1, 0].set_ylabel("QSH_supported")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    # Heatmap lm-v of C_spin_rounded
    v_vals = sorted({float(r["v"]) for r in all_rows})
    lm_vals = sorted({float(r["lm"]) for r in all_rows})
    mat = np.zeros((len(lm_vals), len(v_vals)), dtype=float)
    for i, lm in enumerate(lm_vals):
        for j, v in enumerate(v_vals):
            rr = next(r for r in all_rows if abs(float(r["lm"]) - lm) < 1e-9 and abs(float(r["v"]) - v) < 1e-9)
            mat[i, j] = float(rr["C_spin_rounded"])
    im = axes[1, 1].imshow(mat, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_title("C_spin_rounded heatmap (lm vs v)")
    axes[1, 1].set_xlabel("v index")
    axes[1, 1].set_ylabel("lm index")
    axes[1, 1].set_xticks(range(len(v_vals)))
    axes[1, 1].set_xticklabels([f"{x:.1f}" for x in v_vals], rotation=45, ha="right", fontsize=8)
    axes[1, 1].set_yticks(range(len(lm_vals)))
    axes[1, 1].set_yticklabels([f"{x:.1f}" for x in lm_vals], fontsize=8)
    fig.colorbar(im, ax=axes[1, 1], label="C_spin_rounded")

    fig.tight_layout()
    fig.savefig(base / "soc_scan_summary_t34_soc34_only.png", dpi=180)
    plt.close(fig)

    # Text report
    lines = []
    lines.append("SOC strength scan report (strict t34+soc34-only)")
    lines.append("============================================")
    lines.append(f"lm_list={','.join([f'{x:.1f}' for x in lm_list])}")
    lines.append(f"v_list={','.join([f'{x:.1f}' for x in v_list])}")
    lines.append(f"nk_bulk={args.nk_bulk}, nk_chern={args.nk_chern}, nk_z2={args.nk_z2}")
    lines.append("")
    for r in lm_rows:
        lines.append(
            f"lm={float(r['lm']):.1f}: min_gap(v={float(r['v_gap_min']):.2f})={float(r['bulk_gap_min']):.6f}, "
            f"QSH_count={int(r['qsh_state_count'])}, "
            f"QSH(v=0.8/1.0/1.2)={int(r['qsh_at_v0p8'])}/{int(r['qsh_at_v1p0'])}/{int(r['qsh_at_v1p2'])}"
        )
    (base / "soc_scan_report_t34_soc34_only.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] SOC scan outputs: {summary_csv}")


if __name__ == "__main__":
    main()
