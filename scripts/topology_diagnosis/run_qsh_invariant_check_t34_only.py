#!/usr/bin/env python3
"""Spin-Chern and Wilson-Z2 checks for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.topology_diagnosis.t34_only_common import (
    ModelParams,
    chern_fukui,
    commutator_norm_h_sz,
    compute_bulk_gap,
    crossing_count,
    h_spin_block_t34,
    output_dirs,
    parse_float_list,
    track_wilson_branches,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QSH invariant check for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.3,0.4,0.5,0.6,0.8,1.0")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--nk", type=int, default=61)
    parser.add_argument("--spin-threshold", type=float, default=1e-8)
    return parser.parse_args()


def wilson_half_bz_tracked(v: float, params: ModelParams, nkx: int, nky: int) -> np.ndarray:
    ky_list = np.linspace(0.0, np.pi, nky)
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    centers = np.zeros((nky, 4), dtype=float)
    eigvecs = []
    for j, ky in enumerate(ky_list):
        wmat = np.eye(4, dtype=complex)
        for i, kx in enumerate(kx_grid):
            kx2 = float(kx_grid[(i + 1) % nkx])
            _, va = np.linalg.eigh(
                np.kron(
                    h_spin_block_t34(float(kx), float(ky), v=v, params=params, spin_sign=+1),
                    np.array([[1, 0], [0, 0]], dtype=complex),
                )
                + np.kron(
                    h_spin_block_t34(float(kx), float(ky), v=v, params=params, spin_sign=-1),
                    np.array([[0, 0], [0, 1]], dtype=complex),
                )
            )
            _, vb = np.linalg.eigh(
                np.kron(
                    h_spin_block_t34(float(kx2), float(ky), v=v, params=params, spin_sign=+1),
                    np.array([[1, 0], [0, 0]], dtype=complex),
                )
                + np.kron(
                    h_spin_block_t34(float(kx2), float(ky), v=v, params=params, spin_sign=-1),
                    np.array([[0, 0], [0, 1]], dtype=complex),
                )
            )
            oa = va[:, :4]
            ob = vb[:, :4]
            u, _, vh = np.linalg.svd(oa.conj().T @ ob)
            q = u @ vh
            wmat = q @ wmat
        ew, ev = np.linalg.eig(wmat)
        nu = np.mod(np.angle(ew) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        eigvecs.append(ev[:, order])
    return track_wilson_branches(centers, eigvecs)


def z2_crossing(v: float, params: ModelParams, nk: int) -> int:
    tracked = wilson_half_bz_tracked(v=v, params=params, nkx=nk, nky=nk)
    counts = [crossing_count(tracked, ref) for ref in (0.45, 0.50, 0.55)]
    parities = [c % 2 for c in counts]
    if parities[0] == parities[1] == parities[2]:
        return int(parities[1])
    return int(parities[1])


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["08_qsh_invariant"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)

    rows = []
    for v in parse_float_list(args.v_list):
        bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=args.nk, n_occ=4)
        comm_norm = commutator_norm_h_sz(v=v, params=params, nk=17)
        spin_conserved = bool(comm_norm < args.spin_threshold)
        c_up = float("nan")
        c_dn = float("nan")
        c_spin = float("nan")
        if spin_conserved:
            c_up = chern_fukui(
                lambda kx, ky: h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=+1),
                n_occ=2,
                nk=args.nk,
            )
            c_dn = chern_fukui(
                lambda kx, ky: h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=-1),
                n_occ=2,
                nk=args.nk,
            )
            c_spin = 0.5 * (c_up - c_dn)
        z2 = z2_crossing(v=v, params=params, nk=max(41, args.nk))
        consistent = True
        warning = ""
        if spin_conserved:
            expected = int(np.rint(c_spin)) % 2
            consistent = bool(expected == z2)
            if not consistent:
                warning = f"inconsistent_with_spin_chern(expected={expected}, got={z2})"
        row = {
            "v": float(v),
            "bulk_gap": float(bulk_gap),
            "spin_conserved": int(spin_conserved),
            "commutator_norm_H_sz": float(comm_norm),
            "C_up": float(c_up),
            "C_down": float(c_dn),
            "C_spin": float(c_spin),
            "Z2_crossing": int(z2),
            "consistent": int(consistent),
            "warning": warning,
        }
        rows.append(row)
        print(
            f"[qsh-t34] v={v:.1f} gap={bulk_gap:.4f} spin_conserved={spin_conserved} "
            f"C_spin={c_spin:.3f} Z2={z2} consistent={consistent}"
        )

    fields = [
        "v",
        "bulk_gap",
        "spin_conserved",
        "commutator_norm_H_sz",
        "C_up",
        "C_down",
        "C_spin",
        "Z2_crossing",
        "consistent",
        "warning",
    ]
    write_csv(out_dir / "qsh_invariant_summary_t34_only.csv", rows, fields)
    write_csv(base / "qsh_invariant_summary_t34_only.csv", rows, fields)

    v = np.array([float(r["v"]) for r in rows], dtype=float)
    cspin = np.array([float(r["C_spin"]) for r in rows], dtype=float)
    z2 = np.array([int(r["Z2_crossing"]) for r in rows], dtype=float)
    gap = np.array([float(r["bulk_gap"]) for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    axes[0].plot(v, cspin, marker="o", label="C_spin")
    axes[0].plot(v, z2, marker="s", label="Z2_crossing")
    axes[0].set_ylabel("Invariant")
    axes[0].set_title("QSH invariant summary (t34-only)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(v, gap, marker="o")
    axes[1].set_ylabel("bulk gap")
    axes[1].set_xlabel("v")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "qsh_invariant_vs_v_t34_only.png", dpi=180)
    fig.savefig(base / "qsh_invariant_vs_v_t34_only.png", dpi=180)
    plt.close(fig)
    print(f"[ok] qsh summary: {out_dir / 'qsh_invariant_summary_t34_only.csv'}")


if __name__ == "__main__":
    main()
