#!/usr/bin/env python3
"""Check candidate chiral symmetry breaking with/without Kane-Mele SOC."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import build_scan_values, ensure_dir, load_model, set_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check chiral-symmetry breaking by SOC.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    return parser.parse_args()


def _candidate_gammas() -> list[np.ndarray]:
    out = []
    idx_all = list(range(4))
    for pos in combinations(idx_all, 2):
        signs = np.full(4, -1.0)
        signs[list(pos)] = 1.0
        gamma_orb = np.diag(signs)
        gamma = np.kron(gamma_orb, np.eye(2))
        out.append(gamma.astype(complex))
    return out


def _delta_for_gamma(model, gamma: np.ndarray, v: float, lm: float, t: float, w: float, nk: int) -> tuple[float, float, float, float]:
    set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
    deltas = []
    rb = []
    for i in range(nk):
        u = i / max(1, nk - 1)
        for j in range(nk):
            vv = j / max(1, nk - 1)
            k = u * model.b1 + vv * model.b2
            h = np.array(model.Hxtype(k), dtype=complex)
            ghg = gamma @ h @ gamma
            denom = max(float(np.linalg.norm(h)), 1e-16)
            delta = float(np.linalg.norm(ghg + h) / denom)
            h_break = 0.5 * (h + ghg)
            r_break = float(np.linalg.norm(h_break) / denom)
            deltas.append(delta)
            rb.append(r_break)
    return float(np.mean(deltas)), float(np.max(deltas)), float(np.mean(rb)), float(np.max(rb))


def _pick_best_gamma(model, t: float, w: float, nk: int) -> tuple[np.ndarray, str, float]:
    best_gamma = None
    best_label = ""
    best_score = float("inf")
    for idx, gamma in enumerate(_candidate_gammas(), start=1):
        dm, _, _, _ = _delta_for_gamma(model, gamma, v=0.5, lm=0.0, t=t, w=w, nk=nk)
        if dm < best_score:
            best_score = dm
            best_gamma = gamma
            best_label = f"candidate_{idx}"
    assert best_gamma is not None
    return best_gamma, best_label, best_score


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "chiral_symmetry")
    model = load_model(args.model_file)

    nk_select = 11 if args.quick else 21
    nk_eval = 21 if args.quick else 41
    gamma, gamma_label, gamma_score = _pick_best_gamma(model, t=args.t, w=args.w, nk=nk_select)

    v_values = build_scan_values(0.0, 1.1, 0.05 if args.quick else 0.01)
    rows = []
    for lm in (0.0, 0.1):
        for v in v_values:
            dm, dx, rm, rx = _delta_for_gamma(model, gamma, v=v, lm=lm, t=args.t, w=args.w, nk=nk_eval)
            comment = ""
            if gamma_score > 0.2:
                comment = "No near-exact chiral candidate found; treating Gamma as approximate."
            rows.append(
                {
                    "v": float(v),
                    "lm": float(lm),
                    "delta_chi_mean": float(dm),
                    "delta_chi_max": float(dx),
                    "r_break_mean": float(rm),
                    "r_break_max": float(rx),
                    "comment": comment,
                }
            )

    csv_path = out_dir / "chiral_breaking_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "lm",
                "delta_chi_mean",
                "delta_chi_max",
                "r_break_mean",
                "r_break_max",
                "comment",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Comparison plot for r_break_mean.
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for lm, color in ((0.0, "tab:blue"), (0.1, "tab:orange")):
        arr = sorted([r for r in rows if abs(float(r["lm"]) - lm) < 1e-12], key=lambda r: float(r["v"]))
        v = np.array([float(r["v"]) for r in arr], dtype=float)
        rb = np.array([float(r["r_break_mean"]) for r in arr], dtype=float)
        ax.plot(v, rb, linewidth=1.2, color=color, label=f"lm={lm:.1f}")
    ax.set_xlabel("v")
    ax.set_ylabel("r_break_mean")
    ax.set_title("Chiral-breaking ratio vs v")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "r_break_vs_v.png", dpi=180)
    plt.close(fig)

    # report
    report = out_dir / "gamma_candidate_report.txt"
    report.write_text(
        "\n".join(
            [
                "Candidate chiral operator selection",
                f"selected_gamma={gamma_label}",
                f"selection_score_delta_chi_mean(lm=0,v=0.5)={gamma_score:.6e}",
                "Gamma built as kron(diag([±1,±1,±1,±1]), I_spin) with 2+ and 2- signs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] csv={csv_path}")
    print(f"[ok] out_dir={out_dir}")


if __name__ == "__main__":
    main()
