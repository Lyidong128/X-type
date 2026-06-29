#!/usr/bin/env python3
"""Aggregate total phase diagnosis for t34-only outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.topology_diagnosis.t34_only_common import output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Total topology summary for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def nearest_row(rows: list[dict], v: float) -> dict | None:
    if not rows:
        return None
    best = None
    best_d = 1e9
    for r in rows:
        try:
            d = abs(float(r["v"]) - v)
        except Exception:
            continue
        if d < best_d:
            best = r
            best_d = d
    return best


def mean_weights_by_v(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])
    vs = sorted({float(r["v"]) for r in rows})
    wc = []
    we = []
    wb = []
    for v in vs:
        sub = [r for r in rows if abs(float(r["v"]) - v) < 1e-9]
        wc.append(float(np.mean([float(r["W_corner"]) for r in sub])))
        we.append(float(np.mean([float(r["W_edge"]) for r in sub])))
        wb.append(float(np.mean([float(r["W_bulk"]) for r in sub])))
    return np.array(vs), np.array(wc), np.array(we), np.array(wb)


def parse_z2_final(z: str) -> float:
    try:
        return float(int(z))
    except Exception:
        return np.nan


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    out_dir = dirs["10_summary"]

    bulk = read_csv(base / "bulk_gap_t34_only_summary.csv")
    ribbon = read_csv(base / "ribbon_t34_only_summary.csv")
    wilson = read_csv(base / "wilson_polarization_summary_t34_only.csv")
    open_open = read_csv(base / "open_open_corner_summary_t34_only.csv")
    qsh = read_csv(base / "qsh_invariant_summary_t34_only.csv")
    nested = read_csv(base / "nested_hoti_check_summary_t34_only.csv")
    z2fix = read_csv(base / "qsh_invariant_summary_fixed_t34_only.csv")

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 13.0))

    # Panel 1: bulk gap vs v
    if bulk:
        v_bulk = np.array([float(r["v"]) for r in bulk], dtype=float)
        g_bulk = np.array([float(r["bulk_gap"]) for r in bulk], dtype=float)
        axes[0, 0].plot(v_bulk, g_bulk, marker="o")
        axes[0, 0].axhline(1e-4, color="gray", linestyle="--", linewidth=0.9)
    axes[0, 0].set_title("Panel 1: bulk gap vs v")
    axes[0, 0].set_xlabel("v")
    axes[0, 0].set_ylabel("bulk gap")
    axes[0, 0].grid(alpha=0.25)

    # Panel 2: ribbon edge indicator vs v
    if ribbon:
        v = np.array([float(r["v"]) for r in ribbon], dtype=float)
        edge_ind = np.array([float(r["edge_state_present"]) for r in ribbon], dtype=float)
        n_ingap = np.array([float(r["num_in_gap_edge_states"]) for r in ribbon], dtype=float)
        axes[0, 1].plot(v, edge_ind, marker="o", label="edge_state_present")
        axes[0, 1].plot(v, n_ingap / max(np.max(n_ingap), 1.0), marker="s", label="normalized in-gap edge count")
        axes[0, 1].legend()
    axes[0, 1].set_title("Panel 2: ribbon edge-state indicator")
    axes[0, 1].set_xlabel("v")
    axes[0, 1].set_ylabel("indicator")
    axes[0, 1].grid(alpha=0.25)

    # Panel 3: px,py vs v
    if wilson:
        v = np.array([float(r["v"]) for r in wilson], dtype=float)
        px = np.array([float(r["p_x"]) for r in wilson], dtype=float)
        py = np.array([float(r["p_y"]) for r in wilson], dtype=float)
        axes[1, 0].plot(v, px, marker="o", label="p_x")
        axes[1, 0].plot(v, py, marker="s", label="p_y")
        axes[1, 0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axes[1, 0].legend()
    axes[1, 0].set_title("Panel 3: polarization p_x,p_y")
    axes[1, 0].set_xlabel("v")
    axes[1, 0].set_ylabel("polarization")
    axes[1, 0].grid(alpha=0.25)

    # Panel 4: W_corner,W_edge,W_bulk vs v
    vo, wc, we, wb = mean_weights_by_v(open_open)
    if vo.size > 0:
        axes[1, 1].plot(vo, wc, marker="o", label="W_corner")
        axes[1, 1].plot(vo, we, marker="s", label="W_edge")
        axes[1, 1].plot(vo, wb, marker="^", label="W_bulk")
        axes[1, 1].legend()
    axes[1, 1].set_title("Panel 4: open-open weights")
    axes[1, 1].set_xlabel("v")
    axes[1, 1].set_ylabel("weight")
    axes[1, 1].grid(alpha=0.25)

    # Panel 5: C_spin and Z2 vs v
    if qsh:
        v = np.array([float(r["v"]) for r in qsh], dtype=float)
        cs = np.array([float(r["C_spin"]) for r in qsh], dtype=float)
        z2c = np.array([float(r["Z2_crossing"]) for r in qsh], dtype=float)
        axes[2, 0].plot(v, cs, marker="o", label="C_spin")
        axes[2, 0].plot(v, z2c, marker="s", label="Z2_crossing")
    if z2fix:
        vf = np.array([float(r["v"]) for r in z2fix], dtype=float)
        z2f = np.array([parse_z2_final(r["Z2_final"]) for r in z2fix], dtype=float)
        axes[2, 0].plot(vf, z2f, marker="^", label="Z2_final(debug)")
    axes[2, 0].set_title("Panel 5: C_spin and Z2")
    axes[2, 0].set_xlabel("v")
    axes[2, 0].set_ylabel("invariant")
    axes[2, 0].grid(alpha=0.25)
    axes[2, 0].legend()

    # Panel 6: qxy / nested reliability
    if nested:
        v = np.array([float(r["v"]) for r in nested], dtype=float)
        qxy = np.array([float(r["qxy"]) if r["qxy"] not in ("", "nan", "NaN") else np.nan for r in nested], dtype=float)
        rel = np.array([float(r["nested_reliable"]) for r in nested], dtype=float)
        axes[2, 1].plot(v, qxy, marker="o", label="qxy")
        axes[2, 1].plot(v, rel, marker="s", label="nested_reliable")
        axes[2, 1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axes[2, 1].legend()
    axes[2, 1].set_title("Panel 6: qxy and nested reliability")
    axes[2, 1].set_xlabel("v")
    axes[2, 1].set_ylabel("value")
    axes[2, 1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "total_phase_diagnosis_t34_only.png", dpi=180)
    fig.savefig(base / "total_phase_diagnosis_t34_only.png", dpi=180)
    plt.close(fig)

    # report synthesis with requested logic
    rep = []
    rep.append("Topological diagnosis report (t34-only)")
    rep.append("===================================")
    rep.append("")
    check_path = base / "t_hopping_check_after_t34_only.txt"
    check_ok = check_path.exists() and ("CHECK_STATUS = PASS" in check_path.read_text(encoding="utf-8"))
    rep.append(f"1) Hamiltonian t34-only check: {'PASS' if check_ok else 'FAIL'}")

    vc = None
    if bulk:
        g = np.array([float(r["bulk_gap"]) for r in bulk], dtype=float)
        v = np.array([float(r["v"]) for r in bulk], dtype=float)
        imin = int(np.argmin(g))
        vc = float(v[imin])
        rep.append(f"2) bulk gap minimum located near v={vc:.3f}, gap={g[imin]:.6e}.")
    else:
        rep.append("2) bulk gap summary missing.")

    # v<0.6 logic
    v_probe = 0.5
    wrow = nearest_row(wilson, v_probe)
    qrow = nearest_row(qsh, v_probe)
    nrow = nearest_row(nested, v_probe)
    o_sub = [r for r in open_open if abs(float(r["v"]) - v_probe) < 1e-9]
    wc_mean = float(np.mean([float(r["W_corner"]) for r in o_sub])) if o_sub else np.nan
    we_mean = float(np.mean([float(r["W_edge"]) for r in o_sub])) if o_sub else np.nan
    rep.append(
        f"3) v<0.6 polarization at v=0.5: "
        f"p_x={float(wrow['p_x']):.4f}, p_y={float(wrow['p_y']):.4f}." if wrow else "3) missing wilson data."
    )
    rep.append(
        f"4) open-open at v=0.5: mean W_corner={wc_mean:.4f}, mean W_edge={we_mean:.4f} "
        f"-> {'edge_state-dominant' if we_mean > wc_mean else 'corner_state-dominant'}."
    )
    if nrow:
        rep.append(
            f"5) nested Wilson at v=0.5: reliable={int(float(nrow['nested_reliable']))}, "
            f"qxy={nrow['qxy']}, supports_hoti={int(float(nrow['nested_supports_hoti']))}."
        )
    else:
        rep.append("5) nested Wilson result missing.")

    # decision text by user logic
    if wrow and qrow and nrow:
        px = float(wrow["p_x"])
        py = float(wrow["p_y"])
        cspin = float(qrow["C_spin"])
        z2 = int(float(qrow["Z2_crossing"]))
        nrel = int(float(nrow["nested_reliable"]))
        qxy = float(nrow["qxy"]) if nrow["qxy"] not in ("nan", "NaN", "") else np.nan
        if abs(px - 0.5) < 0.05 and abs(py - 0.5) < 0.05 and abs(cspin) < 0.25 and z2 == 0 and we_mean > wc_mean and nrel == 0:
            rep.append("6) v<0.6 is a Z2-trivial SSH-like edge-localized phase.")
        elif (
            abs(px - 0.5) < 0.05
            and abs(py - 0.5) < 0.05
            and nrel == 1
            and (not np.isnan(qxy))
            and abs(qxy - 0.5) < 0.05
            and wc_mean > 0.6
        ):
            rep.append("6) v<0.6 supports HOTI.")
        elif (
            (abs(px - 0.5) >= 0.05 or abs(py - 0.5) >= 0.05)
            and abs(wc_mean - we_mean) < 0.2
        ):
            rep.append("6) v<0.6 is likely a termination-induced boundary state regime.")
        else:
            rep.append("6) v<0.6 classification: mixed indicators, see panel data.")
    else:
        rep.append("6) v<0.6 classification unavailable due to missing summaries.")

    if vc is not None:
        rep.append(
            f"7) transition near v≈0.6: {'YES (gap minimum around 0.6)' if abs(vc - 0.6) < 0.12 else 'NO clear gap closing at 0.6'}."
        )
    else:
        rep.append("7) transition near v≈0.6 unresolved.")

    vq = 0.8
    qrow2 = nearest_row(qsh, vq)
    zfix2 = nearest_row(z2fix, vq)
    if qrow2:
        cspin2 = float(qrow2["C_spin"])
        z22 = int(float(qrow2["Z2_crossing"]))
        rep.append(f"8) v>0.6 representative (v=0.8): C_spin={cspin2:.3f}, Z2_crossing={z22}.")
    if zfix2:
        rep.append(
            f"9) z2-debug (v=0.8): Z2_final={zfix2['Z2_final']}, QSH_supported={zfix2['QSH_supported']}."
        )

    # old-vs-new qualitative statement
    rep.append(
        "10) Compared with previous model (multiple t links), t34-only model removes non-3-4 extra hopping "
        "and can shift gap minima, edge-state counts, and Z2/HOTI diagnostics quantitatively."
    )

    report_path = out_dir / "topology_diagnosis_report_t34_only.txt"
    report_path.write_text("\n".join(rep) + "\n", encoding="utf-8")
    (base / "topology_diagnosis_report_t34_only.txt").write_bytes(report_path.read_bytes())
    print(f"[ok] total summary figure: {out_dir / 'total_phase_diagnosis_t34_only.png'}")
    print(f"[ok] report: {report_path}")


if __name__ == "__main__":
    main()
