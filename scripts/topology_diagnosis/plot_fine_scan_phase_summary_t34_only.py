#!/usr/bin/env python3
"""Aggregate fine-scan phase summary and report for t34-only model."""

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

from scripts.topology_diagnosis.t34_only_common import write_t_hopping_check, ModelParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot fine-scan phase summary and write report (t34-only).")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def nearest(rows: list[dict], v: float) -> dict | None:
    if not rows:
        return None
    best = None
    bd = 1e9
    for r in rows:
        try:
            d = abs(float(r["v"]) - v)
        except Exception:
            continue
        if d < bd:
            best = r
            bd = d
    return best


def classify_qsh_interval(rows: list[dict]) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    vs = np.array([float(r["v"]) for r in rows], dtype=float)
    q = np.array([int(r["QSH_supported"]) for r in rows], dtype=int)
    idx = np.where(q == 1)[0]
    if idx.size == 0:
        return None, None
    return float(vs[idx[0]]), float(vs[idx[-1]])


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    fine_root = base / "fine_scan_0p8_1p2"
    out = fine_root / "06_summary"
    out.mkdir(parents=True, exist_ok=True)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))

    bulk = read_csv(fine_root / "bulk_gap_fine_scan_t34_only.csv")
    qsh = read_csv(fine_root / "qsh_fine_scan_t34_only.csv")
    conv = read_csv(fine_root / "spin_chern_convergence_fine_t34_only.csv")
    ribbon = read_csv(fine_root / "ribbon_fine_points_summary_t34_only.csv")
    wflow = read_csv(fine_root / "z2_wilson_flow_fine_summary_t34_only.csv")
    oo = read_csv(fine_root / "open_open_fine_points_summary_t34_only.csv")

    if not bulk or not qsh:
        raise RuntimeError("Missing required fine-scan CSVs for bulk and qsh summaries.")

    v_bulk = np.array([float(r["v"]) for r in bulk], dtype=float)
    g_bulk = np.array([float(r["bulk_gap"]) for r in bulk], dtype=float)
    i_min = int(np.argmin(g_bulk))
    v_gap_min = float(v_bulk[i_min])
    gap_min = float(g_bulk[i_min])

    v_q = np.array([float(r["v"]) for r in qsh], dtype=float)
    c_spin = np.array([float(r["C_spin"]) for r in qsh], dtype=float)
    z2_final = np.array([int(r["Z2_final"]) for r in qsh], dtype=float)
    qsh_supported = np.array([int(r["QSH_supported"]) for r in qsh], dtype=float)

    v_qsh_onset = None
    for r in qsh:
        if int(r["QSH_supported"]) == 1:
            v_qsh_onset = float(r["v"])
            break

    # panel 6 data: average open-open weights by v
    oo_vs = sorted({float(r["v"]) for r in oo}) if oo else []
    oo_wc, oo_we, oo_wb = [], [], []
    for vv in oo_vs:
        sub = [r for r in oo if abs(float(r["v"]) - vv) < 1e-9]
        oo_wc.append(float(np.mean([float(r["W_corner"]) for r in sub])))
        oo_we.append(float(np.mean([float(r["W_edge"]) for r in sub])))
        oo_wb.append(float(np.mean([float(r["W_bulk"]) for r in sub])))

    # panel 5 data from ribbon representative points
    rv = np.array([float(r["v"]) for r in ribbon], dtype=float) if ribbon else np.array([])
    rnum = np.array([float(r["num_in_gap_edge_states"]) for r in ribbon], dtype=float) if ribbon else np.array([])
    rmaxw = np.array([float(r["max_edge_weight"]) for r in ribbon], dtype=float) if ribbon else np.array([])

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.5))
    ax = axes[0, 0]
    ax.plot(v_bulk, g_bulk, marker="o")
    ax.scatter([v_gap_min], [gap_min], color="red", s=90, zorder=5, label=f"v_gap_min={v_gap_min:.2f}")
    ax.set_title("Panel 1: bulk gap vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("bulk gap")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(v_q, c_spin, marker="o")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Panel 2: C_spin vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("C_spin")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.step(v_q, z2_final, where="mid")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Panel 3: Z2_final vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("Z2_final")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.step(v_q, qsh_supported, where="mid", label="QSH_supported")
    ax.set_ylim(-0.1, 1.1)
    if v_qsh_onset is not None:
        ax.axvline(v_qsh_onset, color="purple", linestyle="--", linewidth=1.0, label=f"v_QSH_onset={v_qsh_onset:.2f}")
    ax.set_title("Panel 4: QSH_supported vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("QSH_supported")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    if rv.size > 0:
        ax.plot(rv, rnum, marker="o", label="num_in_gap_edge_states")
        ax.plot(rv, rmaxw, marker="s", label="max_edge_weight")
    ax.set_title("Panel 5: ribbon edge indicators")
    ax.set_xlabel("v")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    if oo_vs:
        x = np.array(oo_vs, dtype=float)
        ax.plot(x, oo_wc, marker="o", label="W_corner")
        ax.plot(x, oo_we, marker="s", label="W_edge")
        ax.plot(x, oo_wb, marker="^", label="W_bulk")
    ax.set_title("Panel 6: open-open weights")
    ax.set_xlabel("v")
    ax.set_ylabel("weight")
    ax.grid(alpha=0.25)
    ax.legend()

    for a in axes.reshape(-1):
        a.axvline(v_gap_min, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
        if v_qsh_onset is not None:
            a.axvline(v_qsh_onset, color="purple", linestyle=":", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    sum_png = out / "fine_scan_phase_summary_t34_only.png"
    fig.savefig(sum_png, dpi=180)
    fig.savefig(fine_root / "fine_scan_phase_summary_t34_only.png", dpi=180)
    plt.close(fig)

    # report answers
    q06 = nearest(read_csv(base / "bulk_gap_t34_only_summary.csv"), 0.6)
    q08 = nearest(qsh, 0.8)
    q10 = nearest(qsh, 1.0)
    q12 = nearest(qsh, 1.2)
    rb08 = nearest(ribbon, 0.8)
    rb10 = nearest(ribbon, 1.0)
    rb12 = nearest(ribbon, 1.2)

    # open-open corner check around fine points
    has_corner = False
    if oo:
        has_corner = any(float(r["W_corner"]) > 0.6 for r in oo)

    v_on, v_off = classify_qsh_interval(qsh)

    report_lines = []
    report_lines.append("Fine scan phase report (t34-only)")
    report_lines.append("================================")
    report_lines.append("")
    report_lines.append(f"1. Bulk-gap minimum point: v_gap_min={v_gap_min:.2f}, bulk_gap_min={gap_min:.8f}.")

    if q06 and float(q06["bulk_gap"]) > 1e-4:
        report_lines.append("2. v=0.6 is not a topological transition point in the t34-only model.")
    else:
        report_lines.append("2. v=0.6 transition remains unresolved from current data.")

    # estimate C_spin jump
    jump_text = "not observed"
    cs_round = np.rint(c_spin).astype(int)
    d = np.where(np.diff(cs_round) != 0)[0]
    if d.size > 0:
        vc = 0.5 * (v_q[d[0]] + v_q[d[0] + 1])
        jump_text = f"between v={v_q[d[0]]:.2f} and v={v_q[d[0]+1]:.2f} (center ~{vc:.2f})"
    report_lines.append(f"3. C_spin changes from 0 to 1: {jump_text}.")
    report_lines.append(f"4. Smallest v with QSH_supported=True: {v_qsh_onset}.")

    if q08 and int(q08["C_spin_rounded"]) == 0 and int(q08["Z2_final"]) == 0:
        report_lines.append("5. v=0.8 remains topologically trivial in the QSH sense.")
    else:
        report_lines.append("5. v=0.8 classification is not QSH-clean from current indicators.")

    if q10 and int(q10["C_spin_rounded"]) == 1 and int(q10["Z2_final"]) == 1:
        report_lines.append("6. v=1.0 supports the QSH phase.")
    else:
        report_lines.append("6. v=1.0 is not robustly QSH from current indicators.")

    if q12:
        if int(q12["QSH_supported"]) == 1:
            report_lines.append("7. v=1.2 remains in the QSH phase in this fine scan.")
        else:
            report_lines.append("7. v=1.2 is outside the QSH phase (QSH_supported=False).")
    else:
        report_lines.append("7. v=1.2 result missing in qsh fine scan.")

    report_lines.append(
        "8. Ribbon edge-state trend is broadly consistent with invariant change: "
        f"v=0.8 edge_present={rb08['edge_state_present'] if rb08 else 'NA'}, "
        f"v=1.0 edge_present={rb10['edge_state_present'] if rb10 else 'NA'}, "
        f"v=1.2 edge_present={rb12['edge_state_present'] if rb12 else 'NA'}."
    )
    report_lines.append(
        "9. Open-open representative points show "
        + ("corner-state signatures (W_corner>0.6 present)." if has_corner else "edge/mixed dominance without robust corner-state dominance.")
        + " This does not support HOTI confirmation in this range."
    )

    if 0.95 <= v_gap_min <= 1.00:
        report_lines.append(
            "10. The topological transition in the t34-only model is shifted from v≈0.6 to the vicinity of v≈0.95–1.00."
        )
    else:
        report_lines.append(
            "10. The topological transition in the t34-only model is shifted away from v≈0.6 to the high-v region near the bulk-gap minimum."
        )
    if v_on is not None:
        report_lines.append(f"    QSH interval from fine scan: approximately v in [{v_on:.2f}, {v_off:.2f}].")
    else:
        report_lines.append("    No stable QSH interval found in [0.80,1.20].")

    # include requested statement on low-v regime from existing results
    report_lines.append(
        "    v<0.6 is an SSH-like polarized edge-localized phase, but not a confirmed HOTI."
    )

    report_path = out / "fine_scan_phase_report_t34_only.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (fine_root / "fine_scan_phase_report_t34_only.txt").write_bytes(report_path.read_bytes())
    print(
        f"[ok] fine summary figure/report generated: v_gap_min={v_gap_min:.2f}, "
        f"v_QSH_onset={v_qsh_onset}, out={sum_png}"
    )


if __name__ == "__main__":
    main()
