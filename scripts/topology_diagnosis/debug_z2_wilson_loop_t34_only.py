#!/usr/bin/env python3
"""Robust Z2 debug and largest-gap fix for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_only_common import (
    ModelParams,
    chern_fukui,
    crossing_count,
    h_spin_block_t34,
    largest_gap_center_from_row,
    output_dirs,
    parse_float_list,
    token,
    track_wilson_branches,
    unwrap_mod_curve,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Wilson-loop Z2 for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.5,0.8,1.0")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--nkx", type=int, default=101)
    parser.add_argument("--nky", type=int, default=101)
    return parser.parse_args()


def build_h8_from_spin_blocks(kx: float, ky: float, v: float, params: ModelParams) -> np.ndarray:
    pup = np.array([[1, 0], [0, 0]], dtype=complex)
    pdn = np.array([[0, 0], [0, 1]], dtype=complex)
    return np.kron(h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=+1), pup) + np.kron(
        h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=-1),
        pdn,
    )


def tracked_wilson_branches(v: float, params: ModelParams, nkx: int, nky: int) -> tuple[np.ndarray, np.ndarray]:
    ky_list = np.linspace(0.0, np.pi, nky)
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    centers = np.zeros((nky, 4), dtype=float)
    eigvecs = []
    for j, ky in enumerate(ky_list):
        wmat = np.eye(4, dtype=complex)
        for i, kx in enumerate(kx_grid):
            kx2 = float(kx_grid[(i + 1) % nkx])
            _, va = np.linalg.eigh(build_h8_from_spin_blocks(float(kx), float(ky), v=v, params=params))
            _, vb = np.linalg.eigh(build_h8_from_spin_blocks(float(kx2), float(ky), v=v, params=params))
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
    tracked = track_wilson_branches(centers, eigvecs)
    return ky_list, tracked


def largest_gap_fixed(tracked: np.ndarray) -> tuple[int, np.ndarray, float]:
    g_mod = np.array([largest_gap_center_from_row(row)[0] for row in tracked], dtype=float)
    g_unwrap = unwrap_mod_curve(g_mod)
    delta = float(g_unwrap[-1] - g_unwrap[0])
    winding = int(np.rint(delta))
    z2 = int(abs(winding) % 2)
    return z2, g_unwrap, delta


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["09_z2_debug"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)

    z2_debug_rows = []
    lg_rows = []
    fixed_rows = []

    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    axes1 = np.array(axes1).reshape(-1)
    axes2 = np.array(axes2).reshape(-1)

    for i, v in enumerate(parse_float_list(args.v_list)):
        ky, tracked = tracked_wilson_branches(v=v, params=params, nkx=args.nkx, nky=args.nky)
        c45 = crossing_count(tracked, 0.45)
        c50 = crossing_count(tracked, 0.50)
        c55 = crossing_count(tracked, 0.55)
        z2_cross = int(c50 % 2)
        z2_lg, g_unwrap, delta = largest_gap_fixed(tracked)
        reliable_lg = bool(abs(delta - np.rint(delta)) < 0.1)

        c_up = chern_fukui(
            lambda kx, ky_: h_spin_block_t34(kx, ky_, v=v, params=params, spin_sign=+1),
            n_occ=2,
            nk=61,
        )
        c_dn = chern_fukui(
            lambda kx, ky_: h_spin_block_t34(kx, ky_, v=v, params=params, spin_sign=-1),
            n_occ=2,
            nk=61,
        )
        c_spin = 0.5 * (c_up - c_dn)
        z2_expected = int(np.rint(c_spin)) % 2

        if z2_cross == z2_lg:
            z2_final = z2_cross
            comment = "crossing_and_largest_gap_consistent"
        elif z2_cross == z2_expected:
            z2_final = z2_cross
            comment = "largest_gap_differs_use_crossing_with_spin_chern"
        elif z2_lg == z2_expected:
            z2_final = z2_lg
            comment = "crossing_differs_use_largest_gap_with_spin_chern"
        else:
            z2_final = -1
            comment = "unresolved_crossing_vs_largest_gap_vs_spin_chern"
        qsh_supported = int((z2_final == 1) and (int(np.rint(c_spin)) == 1))

        z2_debug_rows.append(
            {
                "v": float(v),
                "C_spin": float(c_spin),
                "crossings_ref_0p45": int(c45),
                "crossings_ref_0p50": int(c50),
                "crossings_ref_0p55": int(c55),
                "Z2_crossing": int(z2_cross),
                "comment": comment,
            }
        )
        lg_rows.append(
            {
                "v": float(v),
                "largest_gap_delta": float(delta),
                "Z2_largest_gap_fixed": int(z2_lg),
                "largest_gap_reliable": int(reliable_lg),
            }
        )
        fixed_rows.append(
            {
                "v": float(v),
                "C_spin": float(c_spin),
                "Z2_expected_from_spin_chern": int(z2_expected),
                "Z2_crossing": int(z2_cross),
                "Z2_largest_gap_fixed": int(z2_lg),
                "Z2_final": int(z2_final),
                "QSH_supported": int(qsh_supported),
                "comment": comment,
            }
        )

        ax = axes1[i]
        for b in range(tracked.shape[1]):
            ax.plot(ky / np.pi, tracked[:, b], linewidth=1.0)
        for ref, color in [(0.45, "gray"), (0.5, "black"), (0.55, "gray")]:
            ax.axhline(ref, color=color, linestyle="--", linewidth=0.8)
        ax.set_title(f"v={v:.1f}, Z2_cross={z2_cross}")
        ax.set_xlabel(r"$k_y/\pi$")
        ax.grid(alpha=0.2)

        ax2 = axes2[i]
        ax2.plot(ky / np.pi, np.mod(g_unwrap, 1.0), marker="o", markersize=2.5, linewidth=1.0)
        ax2.set_title(f"v={v:.1f}, Z2_lg={z2_lg}, Δ={delta:.3f}")
        ax2.set_xlabel(r"$k_y/\pi$")
        ax2.grid(alpha=0.2)

        print(
            f"[z2-debug-t34] v={v:.1f} C_spin={c_spin:.3f} "
            f"Z2_cross={z2_cross} Z2_lg={z2_lg} Z2_final={z2_final}"
        )

    axes1[0].set_ylabel("tracked Wannier center")
    axes2[0].set_ylabel("largest-gap center mod 1")
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(out_dir / "z2_wilson_debug_montage_t34_only.png", dpi=180)
    fig1.savefig(base / "z2_wilson_debug_montage_t34_only.png", dpi=180)
    fig2.savefig(out_dir / "largest_gap_fix_montage_t34_only.png", dpi=180)
    fig2.savefig(base / "largest_gap_fix_montage_t34_only.png", dpi=180)
    plt.close(fig1)
    plt.close(fig2)

    write_csv(
        out_dir / "z2_debug_summary_t34_only.csv",
        z2_debug_rows,
        ["v", "C_spin", "crossings_ref_0p45", "crossings_ref_0p50", "crossings_ref_0p55", "Z2_crossing", "comment"],
    )
    write_csv(
        out_dir / "largest_gap_fix_summary_t34_only.csv",
        lg_rows,
        ["v", "largest_gap_delta", "Z2_largest_gap_fixed", "largest_gap_reliable"],
    )
    write_csv(
        out_dir / "qsh_invariant_summary_fixed_t34_only.csv",
        fixed_rows,
        [
            "v",
            "C_spin",
            "Z2_expected_from_spin_chern",
            "Z2_crossing",
            "Z2_largest_gap_fixed",
            "Z2_final",
            "QSH_supported",
            "comment",
        ],
    )

    # root-level copies
    for name in [
        "z2_debug_summary_t34_only.csv",
        "largest_gap_fix_summary_t34_only.csv",
        "qsh_invariant_summary_fixed_t34_only.csv",
    ]:
        (base / name).write_bytes((out_dir / name).read_bytes())
    print(f"[ok] z2-debug summary: {out_dir / 'qsh_invariant_summary_fixed_t34_only.csv'}")


if __name__ == "__main__":
    main()
