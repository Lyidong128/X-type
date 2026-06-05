#!/usr/bin/env python3
"""Scan bulk gap / Chern / Z2 versus v for lm=0 and lm=0.1."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import (
    GapResult,
    build_scan_values,
    compute_gap_result,
    compute_total_chern,
    compute_wilson_det_result,
    default_inversion_operator,
    ensure_dir,
    fu_kane_z2,
    generate_model_basis_report,
    inversion_error,
    load_model,
    set_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kane-Mele SOC gap/Z2 scan.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=1.1)
    parser.add_argument("--coarse-dv", type=float, default=0.01)
    parser.add_argument("--fine-dv", type=float, default=0.002)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _phase_label(direct_gap: float, indirect_gap: float, chern: float, z2: int, reliable: bool) -> str:
    if min(direct_gap, indirect_gap) < 1e-4:
        return "critical_or_gapless"
    if not reliable:
        return "critical_or_gapless"
    if abs(chern) >= 0.5:
        return "chern_phase"
    if int(z2) == 1:
        return "z2_like_topological"
    return "trivial_insulator"


def _compute_row(
    model,
    v: float,
    lm: float,
    t: float,
    w: float,
    p_op,
    inversion_ok: bool,
    nk_gap: int,
    nk_chern: int,
    nk_wilson_scan: int,
    nk_wilson_loop: int,
) -> dict[str, float | int | str]:
    set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
    gap: GapResult = compute_gap_result(model, nk=nk_gap, n_occ=4)
    chern_total = compute_total_chern(model, nk=nk_chern, n_occ=4)
    wilson_kx = compute_wilson_det_result(
        model,
        nk_scan=nk_wilson_scan,
        nk_loop=nk_wilson_loop,
        n_occ=4,
        direction="kx_scan_ky",
    )
    wilson_ky = compute_wilson_det_result(
        model,
        nk_scan=nk_wilson_scan,
        nk_loop=nk_wilson_loop,
        n_occ=4,
        direction="ky_scan_kx",
    )
    winding_kx_near_int = abs(wilson_kx.winding - round(wilson_kx.winding)) < 0.20
    winding_ky_near_int = abs(wilson_ky.winding - round(wilson_ky.winding)) < 0.20
    z2_wilson = int(wilson_kx.z2) if int(wilson_kx.z2) == int(wilson_ky.z2) else -1
    wilson_reliable = (z2_wilson >= 0) and winding_kx_near_int and winding_ky_near_int

    z2_fu_kane = -1
    fu_quality = np.nan
    fu_reliable = False
    if inversion_ok:
        z2_fu_kane, _, fu_quality = fu_kane_z2(model, p_op=p_op, n_occ=4)
        fu_reliable = float(fu_quality) < 1e-3

    gap_open = min(gap.direct_gap, gap.indirect_gap) >= 1e-4
    z2 = int(z2_fu_kane) if fu_reliable else int(z2_wilson)
    z2_reliable = int(gap_open and (fu_reliable or wilson_reliable) and (z2 >= 0))
    phase_label = _phase_label(
        direct_gap=gap.direct_gap,
        indirect_gap=gap.indirect_gap,
        chern=chern_total,
        z2=z2,
        reliable=bool(z2_reliable),
    )
    comment_parts = []
    if not gap_open:
        comment_parts.append("gap below reliability threshold")
    if fu_reliable:
        comment_parts.append("z2 source=fu_kane")
    elif wilson_reliable:
        comment_parts.append("z2 source=wilson")
    else:
        comment_parts.append("z2 unreliable")
    comment = "; ".join(comment_parts)
    return {
        "v": float(v),
        "t": float(t),
        "w": float(w),
        "lm": float(lm),
        "direct_gap": float(gap.direct_gap),
        "indirect_gap": float(gap.indirect_gap),
        "kx_min": float(gap.kx_min),
        "ky_min": float(gap.ky_min),
        "chern_total": float(chern_total),
        "z2": int(z2),
        "z2_reliable": int(z2_reliable),
        "phase_label": phase_label,
        "comment": comment,
        "z2_wilson": int(z2_wilson),
        "z2_fu_kane": int(z2_fu_kane),
        "wilson_winding_kx": float(wilson_kx.winding),
        "wilson_winding_ky": float(wilson_ky.winding),
        "fu_kane_quality": float(fu_quality) if inversion_ok else np.nan,
        "k_u_min": float(gap.k_u_min),
        "k_v_min": float(gap.k_v_min),
    }


def _collect_refined_points(rows: list[dict[str, float | int | str]], fine_dv: float) -> list[float]:
    rows = sorted(rows, key=lambda r: float(r["v"]))
    refine_points: set[float] = set()
    for i in range(len(rows) - 1):
        a = rows[i]
        b = rows[i + 1]
        va = float(a["v"])
        vb = float(b["v"])
        gap_small = min(float(a["direct_gap"]), float(b["direct_gap"])) < 0.03
        z2_change = (
            int(a["z2_reliable"]) == 1
            and int(b["z2_reliable"]) == 1
            and int(a["z2"]) != int(b["z2"])
        )
        if gap_small or z2_change:
            lo = max(va - 0.02, 0.0)
            hi = min(vb + 0.02, 1.10)
            x = lo
            while x <= hi + 1e-12:
                refine_points.add(round(x, 10))
                x += fine_dv
    return sorted(refine_points)


def _plot_gap(rows: list[dict[str, float | int | str]], lm: float, out_path: Path) -> None:
    arr = sorted([r for r in rows if abs(float(r["lm"]) - lm) < 1e-12], key=lambda r: float(r["v"]))
    v = np.array([float(r["v"]) for r in arr], dtype=float)
    dg = np.array([float(r["direct_gap"]) for r in arr], dtype=float)
    ig = np.array([float(r["indirect_gap"]) for r in arr], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(v, dg, linewidth=1.2, label="direct gap")
    ax.plot(v, ig, linewidth=1.2, label="indirect gap")
    ax.axhline(1e-4, color="black", linestyle="--", linewidth=0.8, label="reliability threshold")
    ax.set_xlabel("v")
    ax.set_ylabel("Gap")
    ax.set_title(f"Gap vs v (lm={lm:.1f}, t=0.3, w=1.0)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_z2(rows: list[dict[str, float | int | str]], lm: float, out_path: Path) -> None:
    arr = sorted([r for r in rows if abs(float(r["lm"]) - lm) < 1e-12], key=lambda r: float(r["v"]))
    v = np.array([float(r["v"]) for r in arr], dtype=float)
    z = np.array([float(r["z2"]) if int(r["z2_reliable"]) == 1 else np.nan for r in arr], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.step(v, z, where="mid", linewidth=1.2, label="Z2 (reliable points)")
    ax.set_xlabel("v")
    ax.set_ylabel("Z2")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"Z2 vs v (lm={lm:.1f}, reliable only)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_chern(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for lm, color in ((0.0, "tab:blue"), (0.1, "tab:orange")):
        arr = sorted([r for r in rows if abs(float(r["lm"]) - lm) < 1e-12], key=lambda r: float(r["v"]))
        v = np.array([float(r["v"]) for r in arr], dtype=float)
        c = np.array([float(r["chern_total"]) for r in arr], dtype=float)
        ax.plot(v, c, linewidth=1.2, color=color, label=f"lm={lm:.1f}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("v")
    ax.set_ylabel("Chern_total")
    ax.set_title("Total Chern vs v")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_phase(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    phase_map = {
        "critical_or_gapless": 0,
        "trivial_insulator": 1,
        "z2_like_topological": 2,
        "chern_phase": 3,
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for lm, marker in ((0.0, "o"), (0.1, "s")):
        arr = sorted([r for r in rows if abs(float(r["lm"]) - lm) < 1e-12], key=lambda r: float(r["v"]))
        v = np.array([float(r["v"]) for r in arr], dtype=float)
        p = np.array([phase_map.get(str(r["phase_label"]), 0) for r in arr], dtype=float)
        ax.plot(v, p, marker=marker, linewidth=1.0, markersize=2.8, label=f"lm={lm:.1f}")
    ax.set_xlabel("v")
    ax.set_ylabel("phase index")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["critical", "trivial", "z2-like", "chern"])
    ax.set_title("Phase summary vs v")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    gap_root = ensure_dir(out_root / "gap_z2_scan")

    generate_model_basis_report(
        output_path=out_root / "model_basis_report.txt",
        model_file=args.model_file,
        t=args.t,
        w=args.w,
        v=0.5,
        lm=0.1,
    )

    model = load_model(args.model_file)
    p_op = default_inversion_operator()
    set_params(model, v=0.5, lm=0.1, t=args.t, w=args.w, j=0.0)
    inv_err = inversion_error(model, p_op=p_op, nk=5 if args.quick else 7)
    inversion_ok = inv_err < 1e-8
    coarse_dv = 0.02 if args.quick else args.coarse_dv
    fine_dv = 0.005 if args.quick else args.fine_dv
    nk_gap = 25 if args.quick else 41
    nk_chern = 17 if args.quick else 31
    nk_wscan = 21 if args.quick else 41
    nk_wloop = 41 if args.quick else 81

    all_rows: list[dict[str, float | int | str]] = []
    for lm in (0.0, 0.1):
        coarse_vs = build_scan_values(args.v_min, args.v_max, coarse_dv)
        coarse_rows = [
            _compute_row(
                model=model,
                v=v,
                lm=lm,
                t=args.t,
                w=args.w,
                p_op=p_op,
                inversion_ok=inversion_ok,
                nk_gap=nk_gap,
                nk_chern=nk_chern,
                nk_wilson_scan=nk_wscan,
                nk_wilson_loop=nk_wloop,
            )
            for v in coarse_vs
        ]
        refine_vs = _collect_refined_points(coarse_rows, fine_dv=fine_dv)
        v_all = sorted(set(coarse_vs + refine_vs))
        lm_rows = []
        for v in v_all:
            lm_rows.append(
                _compute_row(
                    model=model,
                    v=v,
                    lm=lm,
                    t=args.t,
                    w=args.w,
                    p_op=p_op,
                    inversion_ok=inversion_ok,
                    nk_gap=nk_gap,
                    nk_chern=nk_chern,
                    nk_wilson_scan=nk_wscan,
                    nk_wilson_loop=nk_wloop,
                )
            )
        all_rows.extend(lm_rows)

    all_rows.sort(key=lambda r: (float(r["lm"]), float(r["v"])))
    csv_path = gap_root / "gap_z2_scan.csv"
    fields = [
        "v",
        "t",
        "w",
        "lm",
        "direct_gap",
        "indirect_gap",
        "kx_min",
        "ky_min",
        "chern_total",
        "z2",
        "z2_reliable",
        "phase_label",
        "comment",
        "z2_wilson",
        "z2_fu_kane",
        "wilson_winding_kx",
        "wilson_winding_ky",
        "fu_kane_quality",
        "k_u_min",
        "k_v_min",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    _plot_gap(all_rows, lm=0.0, out_path=gap_root / "gap_vs_v_lm0.png")
    _plot_gap(all_rows, lm=0.1, out_path=gap_root / "gap_vs_v_lm0p1.png")
    _plot_z2(all_rows, lm=0.0, out_path=gap_root / "z2_vs_v_lm0.png")
    _plot_z2(all_rows, lm=0.1, out_path=gap_root / "z2_vs_v_lm0p1.png")
    _plot_chern(all_rows, out_path=gap_root / "chern_vs_v.png")
    _plot_phase(all_rows, out_path=gap_root / "phase_summary_vs_v.png")

    print(f"[ok] csv={csv_path}")
    print(f"[ok] out_dir={gap_root}")


if __name__ == "__main__":
    main()
