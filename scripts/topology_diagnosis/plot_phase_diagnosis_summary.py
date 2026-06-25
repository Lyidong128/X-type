#!/usr/bin/env python3
"""Generate integrated phase-diagnosis figure and final text report."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and report topology diagnosis summary.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs")
    return parser.parse_args()


def read_csv_optional(path: Path) -> tuple[list[dict], str | None]:
    if not path.exists():
        return [], f"missing file: {path.name}"
    try:
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        return rows, None
    except Exception as exc:  # noqa: BLE001
        return [], f"failed to read {path.name}: {exc}"


def pick_closest_zero_state(rows: list[dict], v: float) -> dict | None:
    sub = [r for r in rows if abs(float(r.get("v", "nan")) - v) < 1e-12]
    if not sub:
        return None
    return min(sub, key=lambda r: float(r["abs_energy"]))


def safe_float(row: dict | None, key: str, default: float = np.nan) -> float:
    if row is None:
        return default
    try:
        return float(row[key])
    except Exception:  # noqa: BLE001
        return default


def make_summary_plot(
    out_png: Path,
    qsh_rows: list[dict],
    open_rows: list[dict],
    nested_rows: list[dict],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.4), squeeze=False)

    # Panel 1: bulk gap
    ax = axes[0][0]
    if qsh_rows:
        v = np.array([float(r["v"]) for r in qsh_rows], dtype=float)
        gap = np.array([float(r["bulk_gap"]) for r in qsh_rows], dtype=float)
        ax.plot(v, gap, marker="o", linewidth=1.2, label="bulk_gap")
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    ax.set_title("Panel 1: bulk_gap vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("bulk gap")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    # Panel 2: C_spin and Z2
    ax = axes[0][1]
    if qsh_rows:
        v = np.array([float(r["v"]) for r in qsh_rows], dtype=float)
        cspin = np.array([np.nan if str(r["C_spin"]) == "nan" else float(r["C_spin"]) for r in qsh_rows], dtype=float)
        z2 = np.array([float(r["Z2"]) for r in qsh_rows], dtype=float)
        ax.plot(v, cspin, marker="o", linewidth=1.2, label="C_spin")
        ax.step(v, z2, where="mid", linewidth=1.1, label="Z2")
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    ax.set_title("Panel 2: C_spin and Z2 vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("invariants")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    # Panel 3: W_corner/W_edge/W_bulk from closest-to-zero in-gap state
    ax = axes[1][0]
    if open_rows:
        v_unique = sorted({float(r["v"]) for r in open_rows})
        wc, we, wb = [], [], []
        for vv in v_unique:
            rep = pick_closest_zero_state(open_rows, vv)
            wc.append(safe_float(rep, "W_corner"))
            we.append(safe_float(rep, "W_edge"))
            wb.append(safe_float(rep, "W_bulk"))
        v = np.array(v_unique, dtype=float)
        ax.plot(v, np.array(wc, dtype=float), marker="o", linewidth=1.1, label="W_corner")
        ax.plot(v, np.array(we, dtype=float), marker="s", linewidth=1.1, label="W_edge")
        ax.plot(v, np.array(wb, dtype=float), marker="^", linewidth=1.1, label="W_bulk")
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    ax.set_title("Panel 3: weights vs v (closest-to-zero in-gap state)")
    ax.set_xlabel("v")
    ax.set_ylabel("weight")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    # Panel 4: qxy and nested polarizations with reliability markers
    ax = axes[1][1]
    if nested_rows:
        v = np.array([float(r["v"]) for r in nested_rows], dtype=float)
        qxy = np.array([float(r["qxy"]) for r in nested_rows], dtype=float)
        py = np.array([float(r["p_y_nu_x"]) for r in nested_rows], dtype=float)
        px = np.array([float(r["p_x_nu_y"]) for r in nested_rows], dtype=float)
        reliable = np.array([int(r["reliable"]) == 1 for r in nested_rows], dtype=bool)
        for y, label in [(qxy, "qxy"), (py, "p_y_nu_x"), (px, "p_x_nu_y")]:
            ax.plot(v, y, linewidth=1.0, alpha=0.8, label=label)
            ax.scatter(v[reliable], y[reliable], s=44, facecolors="tab:blue", edgecolors="black")
            ax.scatter(v[~reliable], y[~reliable], s=44, facecolors="none", edgecolors="gray")
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    ax.set_title("Panel 4: nested indicators vs v (filled=reliable)")
    ax.set_xlabel("v")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    open_rows, open_warn = read_csv_optional(out_root / "open_open_corner_summary.csv")
    nested_rows, nested_warn = read_csv_optional(out_root / "nested_wilson_strict_summary.csv")
    qsh_rows, qsh_warn = read_csv_optional(out_root / "qsh_invariant_summary.csv")
    term_a_rows, term_a_warn = read_csv_optional(out_root / "termination_A_corner_summary.csv")
    term_b_rows, term_b_warn = read_csv_optional(out_root / "termination_B_corner_summary.csv")

    warns = [w for w in [open_warn, nested_warn, qsh_warn, term_a_warn, term_b_warn] if w]

    make_summary_plot(
        out_png=out_root / "phase_diagnosis_summary.png",
        qsh_rows=qsh_rows,
        open_rows=open_rows,
        nested_rows=nested_rows,
    )

    # Extract key rows
    rep_v05 = pick_closest_zero_state(open_rows, 0.5)
    rep_v08 = pick_closest_zero_state(open_rows, 0.8)
    nested_v05 = next((r for r in nested_rows if abs(float(r["v"]) - 0.5) < 1e-12), None)
    qsh_v08 = next((r for r in qsh_rows if abs(float(r["v"]) - 0.8) < 1e-12), None)

    # Final judgments by requested criteria.
    w_corner_05 = safe_float(rep_v05, "W_corner")
    w_edge_05 = safe_float(rep_v05, "W_edge")
    nested_reliable_05 = int(nested_v05["reliable"]) == 1 if nested_v05 else False
    qxy_05 = safe_float(nested_v05, "qxy")
    py_05 = safe_float(nested_v05, "p_y_nu_x")
    px_05 = safe_float(nested_v05, "p_x_nu_y")

    if w_corner_05 > 0.6 and nested_reliable_05 and abs(qxy_05 - 0.5) < 0.1:
        verdict_v05 = "v=0.5 supports SSH-like higher-order topology."
    elif w_edge_05 > 0.6 and w_corner_05 < 0.6:
        verdict_v05 = "v=0.5 has edge-localized SSH-like boundary states, but no direct evidence of higher-order corner states."
    else:
        verdict_v05 = "v=0.5 does not show clear in-gap corner states in the current finite-size calculation."

    if qsh_v08 is not None:
        cspin_08 = np.nan if str(qsh_v08["C_spin"]) == "nan" else float(qsh_v08["C_spin"])
        z2_08 = int(qsh_v08["Z2"])
        cons_08 = int(qsh_v08["consistent"]) == 1
        spin_cons_08 = int(qsh_v08["spin_conserved"]) == 1
    else:
        cspin_08 = np.nan
        z2_08 = -1
        cons_08 = False
        spin_cons_08 = False

    if spin_cons_08 and not np.isnan(cspin_08) and int(round(cspin_08)) == 1 and z2_08 == 1 and cons_08:
        verdict_v08 = "v=0.8 supports QSH phase."
    elif spin_cons_08 and not np.isnan(cspin_08) and (int(round(cspin_08)) != z2_08 or not cons_08):
        verdict_v08 = "QSH diagnosis at v=0.8 is inconclusive because spin Chern and Z2 are inconsistent."
    elif (not spin_cons_08) and z2_08 == 1:
        verdict_v08 = "v=0.8 supports QSH-like phase based on Z2, while spin Chern is not well-defined due to spin mixing."
    else:
        verdict_v08 = "QSH diagnosis at v=0.8 is inconclusive."

    transition_statement = "The closing and reopening of the bulk gap near v≈0.6 indicates a topological phase transition."
    if qsh_rows:
        v = np.array([float(r["v"]) for r in qsh_rows], dtype=float)
        g = np.array([float(r["bulk_gap"]) for r in qsh_rows], dtype=float)
        idx06 = int(np.argmin(np.abs(v - 0.6)))
        cond = (g[idx06] < 1e-3) and np.any(g[v < 0.6] > 1e-3) and np.any(g[v > 0.6] > 1e-3)
        if not cond:
            transition_statement = "Bulk gap behavior near v≈0.6 does not show a clean close-reopen pattern with the current grid."

    used_sizes = sorted({(int(r["Lx"]), int(r["Ly"])) for r in open_rows}) if open_rows else []

    report_lines = [
        "Topology diagnosis report",
        "=======================",
        "",
        "1) Hamiltonian source / function names",
        "- Reused existing model definitions from scripts.hoti_v_lt_0p6.run_hoti_v_lt_0p6:",
        "  h8_k(), h0_k(), h_soc_orbital(), unitary_part().",
        "- No physical Hamiltonian redefinition was introduced.",
        "",
        "2) Parameters",
        "- t=0.3, w=1.0, lm=0.1, occupied bands n_occ=4.",
        "",
        "3) finite open-open system size",
        f"- Used sizes: {used_sizes if used_sizes else 'N/A'}.",
        "",
        "4) v=0.5 in-gap-state type",
        f"- closest-to-zero classification: {rep_v05['classification'] if rep_v05 else 'N/A'}",
        "",
        "5) v=0.5 weights",
        f"- W_corner={w_corner_05:.6f}, W_edge={w_edge_05:.6f}, W_bulk={safe_float(rep_v05,'W_bulk'):.6f}",
        "",
        "6) nested Wilson reliability",
        f"- reliable(v=0.5)={nested_reliable_05}",
        "",
        "7) v=0.5 nested values",
        f"- qxy={qxy_05:.6f}, p_y_nu_x={py_05:.6f}, p_x_nu_y={px_05:.6f}",
        "",
        "8) v=0.8 QSH invariants",
        f"- C_spin={cspin_08}, Z2={z2_08}, consistent={cons_08}",
        "",
        "9) v<0.6 SSH-like HOTI support",
        f"- {verdict_v05}",
        "",
        "10) v>0.6 QSH support",
        f"- {verdict_v08}",
        "",
        "11) missing evidence / caveats",
        "- nested Wilson points remain unreliable when Wannier-sector gaps or sector overlaps fail thresholds.",
        "- open-open v=0.5 representative state is edge-dominant, not corner-dominant.",
        "",
        "Transition near v≈0.6:",
        f"- {transition_statement}",
        "",
        "Warnings:",
    ]
    if warns:
        report_lines.extend([f"- {w}" for w in warns])
    else:
        report_lines.append("- none")

    (out_root / "topology_diagnosis_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[ok] summary plot: {out_root / 'phase_diagnosis_summary.png'}")
    print(f"[ok] report: {out_root / 'topology_diagnosis_report.txt'}")


if __name__ == "__main__":
    main()
