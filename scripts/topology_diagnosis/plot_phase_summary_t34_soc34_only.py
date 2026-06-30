#!/usr/bin/env python3
"""Total phase summary/report for strict t34 + soc34-only model and A/B/C comparison."""

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

from scripts.topology_diagnosis.t34_soc34_only_common import ModelParams, write_model_check  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase summary/report for t34+soc34-only model.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only")
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


def qsh_interval(rows: list[dict], key: str = "QSH_supported") -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    vs = np.array([float(r["v"]) for r in rows], dtype=float)
    q = np.array([int(r[key]) for r in rows], dtype=int)
    idx = np.where(q == 1)[0]
    if idx.size == 0:
        return None, None
    return float(vs[idx[0]]), float(vs[idx[-1]])


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    write_model_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out = base / "06_summary"
    out.mkdir(parents=True, exist_ok=True)

    bulk = read_csv(base / "bulk_gap_t34_soc34_only.csv")
    ribbon = read_csv(base / "ribbon_t34_soc34_only.csv")
    oo = read_csv(base / "open_open_t34_soc34_only.csv")
    wilson = read_csv(base / "wilson_polarization_t34_soc34_only.csv")
    nested = read_csv(base / "nested_wilson_t34_soc34_only.csv")
    qsh = read_csv(base / "qsh_invariant_t34_soc34_only.csv")

    if not bulk or not qsh:
        raise RuntimeError("Missing essential CSV files for summary.")

    vb = np.array([float(r["v"]) for r in bulk], dtype=float)
    gb = np.array([float(r["bulk_gap"]) for r in bulk], dtype=float)
    i_min = int(np.argmin(gb))
    v_gap_min = float(vb[i_min])
    gap_min = float(gb[i_min])

    vq = np.array([float(r["v"]) for r in qsh], dtype=float)
    cspin = np.array([float(r["C_spin_rounded"]) for r in qsh], dtype=float)
    z2 = np.array([float(r["Z2_final"]) for r in qsh], dtype=float)
    qsh_sup = np.array([float(r["QSH_supported"]) for r in qsh], dtype=float)
    v_qsh_on, v_qsh_off = qsh_interval(qsh, key="QSH_supported")

    # open-open aggregate weights
    oo_vs = sorted({float(r["v"]) for r in oo}) if oo else []
    oo_wc, oo_we, oo_wb = [], [], []
    for v in oo_vs:
        sub = [r for r in oo if abs(float(r["v"]) - v) < 1e-9]
        oo_wc.append(float(np.mean([float(r["W_corner"]) for r in sub])))
        oo_we.append(float(np.mean([float(r["W_edge"]) for r in sub])))
        oo_wb.append(float(np.mean([float(r["W_bulk"]) for r in sub])))

    # ribbon summary vectors
    rv = np.array([float(r["v"]) for r in ribbon], dtype=float) if ribbon else np.array([])
    rnum = np.array([float(r["num_in_gap_edge_states"]) for r in ribbon], dtype=float) if ribbon else np.array([])
    rmaxw = np.array([float(r["max_edge_weight"]) for r in ribbon], dtype=float) if ribbon else np.array([])

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 12.3))
    axes[0, 0].plot(vb, gb, marker="o")
    axes[0, 0].scatter([v_gap_min], [gap_min], color="red", s=80, label=f"v_gap_min={v_gap_min:.1f}")
    axes[0, 0].set_title("Panel 1: bulk gap vs v")
    axes[0, 0].set_xlabel("v")
    axes[0, 0].set_ylabel("bulk gap")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(vq, cspin, marker="o", label="C_spin_rounded")
    axes[0, 1].set_title("Panel 2: C_spin vs v")
    axes[0, 1].set_xlabel("v")
    axes[0, 1].set_ylabel("C_spin")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].step(vq, z2, where="mid")
    axes[1, 0].set_title("Panel 3: Z2_final vs v")
    axes[1, 0].set_xlabel("v")
    axes[1, 0].set_ylabel("Z2_final")
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].step(vq, qsh_sup, where="mid", label="QSH_supported")
    if v_qsh_on is not None:
        axes[1, 1].axvline(v_qsh_on, color="purple", linestyle="--", linewidth=1.0, label=f"QSH_on={v_qsh_on:.1f}")
    axes[1, 1].set_title("Panel 4: QSH_supported")
    axes[1, 1].set_xlabel("v")
    axes[1, 1].set_ylabel("QSH")
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    if rv.size > 0:
        axes[2, 0].plot(rv, rnum, marker="o", label="num_in_gap_edge_states")
        axes[2, 0].plot(rv, rmaxw, marker="s", label="max_edge_weight")
    axes[2, 0].set_title("Panel 5: ribbon edge indicators")
    axes[2, 0].set_xlabel("v")
    axes[2, 0].set_ylabel("value")
    axes[2, 0].grid(alpha=0.25)
    axes[2, 0].legend()

    if oo_vs:
        x = np.array(oo_vs, dtype=float)
        axes[2, 1].plot(x, oo_wc, marker="o", label="W_corner")
        axes[2, 1].plot(x, oo_we, marker="s", label="W_edge")
        axes[2, 1].plot(x, oo_wb, marker="^", label="W_bulk")
    axes[2, 1].set_title("Panel 6: open-open weights")
    axes[2, 1].set_xlabel("v")
    axes[2, 1].set_ylabel("weight")
    axes[2, 1].grid(alpha=0.25)
    axes[2, 1].legend()

    for ax in axes.reshape(-1):
        ax.axvline(v_gap_min, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
        if v_qsh_on is not None:
            ax.axvline(v_qsh_on, color="purple", linestyle=":", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out / "total_phase_summary_t34_soc34_only.png", dpi=170)
    fig.savefig(base / "total_phase_summary_t34_soc34_only.png", dpi=170)
    plt.close(fig)

    # Compare models A/B/C
    a_rows = read_csv(Path("/workspace/topology_diagnosis_outputs/z2_debug/largest_gap_fix/qsh_invariant_summary_fixed.csv"))
    b_rows = read_csv(Path("/workspace/topology_diagnosis_outputs_t34_only/fine_scan_0p8_1p2/qsh_fine_scan_t34_only.csv"))
    c_rows = qsh

    # A interval: from available points only
    a_on = None
    a_off = None
    if a_rows:
        av = np.array([float(r["v"]) for r in a_rows], dtype=float)
        aq = np.array([int(r["QSH_supported"]) for r in a_rows], dtype=int)
        ai = np.where(aq == 1)[0]
        if ai.size > 0:
            a_on, a_off = float(av[ai[0]]), float(av[ai[-1]])

    b_on, b_off = qsh_interval(b_rows, key="QSH_supported")
    c_on, c_off = qsh_interval(c_rows, key="QSH_supported")

    # mandatory question points
    q06 = nearest(read_csv(Path("/workspace/topology_diagnosis_outputs_t34_only/bulk_gap_t34_only_summary.csv")), 0.6)
    c08 = nearest(c_rows, 0.8)
    c10 = nearest(c_rows, 1.0)
    c12 = nearest(c_rows, 1.2)
    small_p = nearest(wilson, 0.4)
    nested_small = nearest(nested, 0.4)
    max_wc = max([float(r["W_corner"]) for r in oo], default=0.0)

    lines = []
    lines.append("Topological diagnosis report for strict t34 + soc34-only model")
    lines.append("============================================================")
    lines.append("")
    lines.append("Model definition:")
    lines.append("H = H_SSH(v,w) + H_t34 + H_SOC34, with only 3-4 channel for both t and SOC.")
    lines.append("")
    lines.append(f"1) bulk-gap minimum at v={v_gap_min:.2f}, bulk_gap={gap_min:.8f}.")
    if q06 and float(q06["bulk_gap"]) > 1e-4:
        lines.append("2) v=0.6 is not a topological transition point in the t34-only model.")
    else:
        lines.append("2) v=0.6 transition remains unresolved from available data.")
    if c08 and int(c08["C_spin_rounded"]) == 0 and int(c08["Z2_final"]) == 0:
        lines.append("3) v=0.8 remains topologically trivial in the QSH sense.")
    else:
        lines.append("3) v=0.8 is not robustly QSH-trivial under current metrics.")
    if c10 and int(c10["C_spin_rounded"]) == 1 and int(c10["Z2_final"]) == 1:
        lines.append("4) v=1.0 supports the QSH phase.")
    else:
        lines.append("4) v=1.0 does not satisfy QSH criterion in this strict model.")
    if c12:
        if int(c12["QSH_supported"]) == 1:
            lines.append("5) v=1.2 remains in the QSH phase.")
        else:
            lines.append("5) v=1.2 is outside QSH support (QSH_supported=False).")

    if c_on is not None:
        lines.append(f"6) QSH interval in model C (t34-only + soc34-only): approx v in [{c_on:.2f}, {c_off:.2f}].")
    else:
        lines.append("6) Model C shows no QSH interval in scanned v range.")

    if b_on is not None:
        lines.append(f"7) Model B (t34-only + full SOC) QSH interval from fine scan: approx [{b_on:.2f}, {b_off:.2f}].")
    else:
        lines.append("7) Model B QSH interval unavailable from current files.")
    if a_on is not None:
        lines.append(f"8) Model A (full t + full SOC) QSH-supported points indicate interval roughly [{a_on:.2f}, {a_off:.2f}] (available samples).")
    else:
        lines.append("8) Model A QSH interval unavailable from current files.")

    if b_on is not None and c_on is not None:
        if c_off - c_on < b_off - b_on - 1e-9:
            lines.append("9) Compared with model B, model C shows a further narrowed QSH window.")
        elif c_off - c_on > b_off - b_on + 1e-9:
            lines.append("9) Compared with model B, model C shows a broader QSH window in the scanned set.")
        else:
            lines.append("9) Compared with model B, model C shows similar QSH window width.")
    elif b_on is not None and c_on is None:
        lines.append("9) Compared with model B, model C shows a collapsed/disappeared QSH window in the scanned range.")
    else:
        lines.append("9) QSH-window comparison between model B and C is inconclusive from available files.")

    if small_p and nested_small:
        px, py = float(small_p["p_x"]), float(small_p["p_y"])
        rel = int(float(nested_small["nested_reliable"])) if str(nested_small["nested_reliable"]) != "" else 0
        if abs(px - 0.5) < 0.05 and abs(py - 0.5) < 0.05 and rel == 0:
            lines.append(
                "10) small-v region remains SSH-like polarized; nested Wilson is unreliable, "
                "and any corner-like finite-size localization is not robust HOTI evidence."
            )
        else:
            lines.append("10) small-v region classification is mixed; no robust HOTI corner-state evidence in open-open weights.")

    report = out / "topology_diagnosis_report_t34_soc34_only.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (base / "topology_diagnosis_report_t34_soc34_only.txt").write_bytes(report.read_bytes())
    print(
        f"[ok] phase summary/report done: v_gap_min={v_gap_min:.2f}, "
        f"QSH_C=[{c_on},{c_off}], report={report}"
    )


if __name__ == "__main__":
    main()
