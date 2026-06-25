#!/usr/bin/env python3
"""Unified QSH invariant diagnostics (bulk gap, spin Chern, Wilson-loop Z2)."""

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

from scripts.topology_diagnosis.common_topology import (
    chern_fukui,
    commutator_norm_h_sz,
    compute_bulk_gap,
    ensure_dir,
    h8_k,
    h_spin_block,
    parse_float_list,
    unitary_part,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QSH invariant consistency check.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs")
    parser.add_argument("--v-list", default="0.3,0.4,0.5,0.6,0.8,1.0")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--nk", type=int, default=61)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def crossing_parity(centers: np.ndarray, reference: float = 0.5) -> int:
    crossings = 0
    for i in range(centers.shape[0] - 1):
        a = centers[i, :] - reference
        b = centers[i + 1, :] - reference
        crossings += int(np.sum(a * b < 0.0))
    return int(crossings % 2)


def wilson_centers_x(v: float, t: float, w: float, lm: float, n_occ: int, nk: int) -> np.ndarray:
    kx_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    centers = np.zeros((nk, n_occ), dtype=float)

    def occ(kx: float, ky: float) -> np.ndarray:
        _, vecs = np.linalg.eigh(h8_k(kx, ky, v=v, t=t, w=w, lm=lm))
        return vecs[:, :n_occ]

    for j, ky in enumerate(ky_grid):
        wmat = np.eye(n_occ, dtype=complex)
        for i, kx in enumerate(kx_grid):
            kx_next = float(kx_grid[(i + 1) % nk])
            oa = occ(float(kx), float(ky))
            ob = occ(kx_next, float(ky))
            q = unitary_part(oa.conj().T @ ob)
            wmat = q @ wmat
        eigvals = np.linalg.eigvals(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        centers[j, :] = np.sort(nu)
    return centers


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_qsh(rows: list[dict], out_png: Path) -> None:
    v = np.array([float(r["v"]) for r in rows], dtype=float)
    gap = np.array([float(r["bulk_gap"]) for r in rows], dtype=float)
    cspin = np.array(
        [np.nan if str(r["C_spin"]) == "nan" else float(r["C_spin"]) for r in rows],
        dtype=float,
    )
    z2 = np.array([float(r["Z2"]) for r in rows], dtype=float)
    consistent = np.array([1 if int(r["consistent"]) == 1 else 0 for r in rows], dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 9.2), sharex=True)
    axes[0].plot(v, gap, marker="o", linewidth=1.2, label="bulk_gap")
    axes[0].axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("bulk gap")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(v, cspin, marker="o", linewidth=1.2, label="C_spin")
    axes[1].step(v, z2, where="mid", linewidth=1.2, label="Z2")
    axes[1].axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("invariants")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    colors = ["tab:green" if c == 1 else "tab:red" for c in consistent]
    axes[2].scatter(v, consistent, c=colors, s=50)
    axes[2].axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["inconsistent", "consistent"])
    axes[2].set_xlabel("v")
    axes[2].set_ylabel("consistency")
    axes[2].grid(alpha=0.25)
    axes[2].set_title("QSH invariant diagnosis summary")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    v_list = parse_float_list(args.v_list)
    nk = 31 if args.quick else int(args.nk)

    rows: list[dict] = []
    for v in v_list:
        bulk_gap = compute_bulk_gap(v=v, t=args.t, w=args.w, lm=args.lm, nk=nk)
        comm_norm = commutator_norm_h_sz(v=v, t=args.t, w=args.w, lm=args.lm, nk=11 if args.quick else 15)
        spin_conserved = int(comm_norm < 1e-8)

        cup = np.nan
        cdn = np.nan
        cspin = np.nan
        warning = ""
        if spin_conserved:
            cup = chern_fukui(
                lambda kx, ky: h_spin_block(kx, ky, v=v, t=args.t, w=args.w, lm=args.lm, spin_sign=1),
                n_occ=2,
                nk=nk,
            )
            cdn = chern_fukui(
                lambda kx, ky: h_spin_block(kx, ky, v=v, t=args.t, w=args.w, lm=args.lm, spin_sign=-1),
                n_occ=2,
                nk=nk,
            )
            cspin = 0.5 * (round(cup) - round(cdn))
        else:
            warning = "spin_not_conserved; block Chern undefined"

        centers = wilson_centers_x(v=v, t=args.t, w=args.w, lm=args.lm, n_occ=4, nk=nk)
        z2 = int(crossing_parity(centers, reference=0.5) % 2)

        if spin_conserved:
            consistent = int((int(round(cspin)) % 2) == z2)
            if consistent == 0:
                warning = (
                    warning + "; " if warning else ""
                ) + "WARNING: spin Chern and Z2 are inconsistent. Check occupied bands, gauge, Wilson loop implementation, or Hamiltonian spin block."
        else:
            consistent = 0

        rows.append(
            {
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "spin_conserved": spin_conserved,
                "commutator_norm_H_sz": float(comm_norm),
                "C_up": float(cup) if not np.isnan(cup) else "nan",
                "C_down": float(cdn) if not np.isnan(cdn) else "nan",
                "C_spin": float(cspin) if not np.isnan(cspin) else "nan",
                "Z2": int(z2),
                "consistent": int(consistent),
                "warning": warning,
            }
        )
        print(
            f"[qsh] v={v:.1f} bulk_gap={bulk_gap:.4e} spin_conserved={spin_conserved} "
            f"C_spin={cspin if not np.isnan(cspin) else 'nan'} Z2={z2} consistent={consistent}"
        )

    write_csv(
        out_root / "qsh_invariant_summary.csv",
        rows,
        [
            "v",
            "bulk_gap",
            "spin_conserved",
            "commutator_norm_H_sz",
            "C_up",
            "C_down",
            "C_spin",
            "Z2",
            "consistent",
            "warning",
        ],
    )
    plot_qsh(rows, out_root / "qsh_invariant_vs_v.png")
    print(f"[ok] qsh summary saved at {out_root / 'qsh_invariant_summary.csv'}")


if __name__ == "__main__":
    main()
