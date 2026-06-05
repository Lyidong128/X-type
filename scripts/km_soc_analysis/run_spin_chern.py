#!/usr/bin/env python3
"""Compute spin-resolved Chern diagnostics for Kane-Mele SOC analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import (
    chern_number_general,
    compute_gap_result,
    ensure_dir,
    load_model,
    set_params,
    split_spin_indices,
    spin_mixing_ratio,
    sz_operator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spin Chern analysis.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    return parser.parse_args()


def _select_v_points(scan_csv: Path) -> list[float]:
    rows = list(csv.DictReader(scan_csv.open("r", encoding="utf-8")))
    lm_rows = sorted([r for r in rows if abs(float(r["lm"]) - 0.1) < 1e-12], key=lambda r: float(r["v"]))
    base = {0.10, 0.30, 0.50, 0.80, 1.00}
    for i in range(len(lm_rows) - 1):
        a = lm_rows[i]
        b = lm_rows[i + 1]
        if int(a["z2_reliable"]) == 1 and int(b["z2_reliable"]) == 1 and int(a["z2"]) != int(b["z2"]):
            base.add(float(a["v"]))
            base.add(float(b["v"]))

    # Also add points just before/after contiguous near-gapless windows.
    small_gap_idx = [i for i, r in enumerate(lm_rows) if float(r["direct_gap"]) < 1e-2]
    if small_gap_idx:
        start = small_gap_idx[0]
        prev = start
        for idx in small_gap_idx[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            base.add(float(lm_rows[max(0, start - 1)]["v"]))
            base.add(float(lm_rows[min(len(lm_rows) - 1, prev + 1)]["v"]))
            start = idx
            prev = idx
        base.add(float(lm_rows[max(0, start - 1)]["v"]))
        base.add(float(lm_rows[min(len(lm_rows) - 1, prev + 1)]["v"]))
    return sorted(base)


def _projected_spin_gap(model, v: float, lm: float, t: float, w: float, nk: int = 9, n_occ: int = 4) -> float:
    set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
    dim = model.Hxtype(np.zeros(3)).shape[0]
    sz = sz_operator(dim).astype(complex)
    mins = []
    for i in range(nk):
        u = i / max(1, nk - 1)
        for j in range(nk):
            vv = j / max(1, nk - 1)
            k = u * model.b1 + vv * model.b2
            _, vecs = np.linalg.eigh(model.Hxtype(k))
            occ = vecs[:, :n_occ]
            m = occ.conj().T @ sz @ occ
            ev = np.linalg.eigvalsh(0.5 * (m + m.conj().T))
            mins.append(float(np.min(np.abs(ev))))
    return float(np.min(mins))


def _chern_spin_blocks(model, v: float, lm: float, t: float, w: float, nk: int) -> tuple[float, float, float]:
    set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
    h0 = np.array(model.Hxtype(np.zeros(3)), dtype=complex)
    up, dn = split_spin_indices(h0.shape[0])

    def h_up(k):
        hk = np.array(model.Hxtype(k), dtype=complex)
        return hk[np.ix_(up, up)]

    def h_dn(k):
        hk = np.array(model.Hxtype(k), dtype=complex)
        return hk[np.ix_(dn, dn)]

    cup = chern_number_general(h_up, model.b1, model.b2, nk=nk, n_occ=2)
    cdn = chern_number_general(h_dn, model.b1, model.b2, nk=nk, n_occ=2)
    mixing = spin_mixing_ratio(h0)
    return cup, cdn, mixing


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root) / "spin_chern")
    scan_csv = Path(args.output_root) / "gap_z2_scan" / "gap_z2_scan.csv"
    if not scan_csv.exists():
        raise FileNotFoundError(f"Missing prerequisite scan: {scan_csv}")

    model = load_model(args.model_file)
    nk_gap = 17 if args.quick else 31
    nk_spin_gap = 7 if args.quick else 11
    nk_chern = 17 if args.quick else 31

    rows = []
    for v in _select_v_points(scan_csv):
        set_params(model, v=v, lm=args.lm, t=args.t, w=args.w, j=0.0)
        gap = compute_gap_result(model, nk=nk_gap, n_occ=4).direct_gap
        cup, cdn, mixing = _chern_spin_blocks(model, v=v, lm=args.lm, t=args.t, w=args.w, nk=nk_chern)
        ctot = float(cup + cdn)
        cspin = float(0.5 * (cup - cdn))
        p_gap = _projected_spin_gap(model, v=v, lm=args.lm, t=args.t, w=args.w, nk=nk_spin_gap, n_occ=4)

        if mixing < 1e-7 and p_gap > 1e-3 and gap > 1e-4:
            reliability = "reliable"
            comment = "Sz approximately conserved; spin blocks decoupled."
        elif mixing < 1e-5 and p_gap > 1e-3:
            reliability = "semi_reliable"
            comment = "Weak spin mixing; projected-spin gap open."
        else:
            reliability = "uncertain"
            comment = "Projected spin gap small or spin mixing non-negligible."

        rows.append(
            {
                "v": float(v),
                "lm": float(args.lm),
                "t": float(args.t),
                "w": float(args.w),
                "gap": float(gap),
                "C_up": float(cup),
                "C_down": float(cdn),
                "C_total": float(ctot),
                "C_spin": float(cspin),
                "projected_spin_gap": float(p_gap),
                "reliability": reliability,
                "comment": comment,
                "spin_mixing_ratio": float(mixing),
            }
        )

    csv_path = out_root / "spin_chern_summary.csv"
    fields = [
        "v",
        "lm",
        "t",
        "w",
        "gap",
        "C_up",
        "C_down",
        "C_total",
        "C_spin",
        "projected_spin_gap",
        "reliability",
        "comment",
        "spin_mixing_ratio",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if rows:
        rows_sorted = sorted(rows, key=lambda r: float(r["v"]))
        v = np.array([float(r["v"]) for r in rows_sorted], dtype=float)
        cup = np.array([float(r["C_up"]) for r in rows_sorted], dtype=float)
        cdn = np.array([float(r["C_down"]) for r in rows_sorted], dtype=float)
        cspin = np.array([float(r["C_spin"]) for r in rows_sorted], dtype=float)

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(v, cup, marker="o", linewidth=1.2, label="C_up")
        ax.plot(v, cdn, marker="s", linewidth=1.2, label="C_down")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("v")
        ax.set_ylabel("Chern")
        ax.set_title("Spin-resolved Chern vs v (lm=0.1)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_root / "Cup_Cdown_vs_v.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(v, cspin, marker="o", linewidth=1.2, label="C_spin")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("v")
        ax.set_ylabel("C_spin")
        ax.set_title("Spin Chern vs v (lm=0.1)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_root / "spin_chern_vs_v.png", dpi=180)
        plt.close(fig)

    print(f"[ok] csv={csv_path}")
    print(f"[ok] out_dir={out_root}")


if __name__ == "__main__":
    main()
