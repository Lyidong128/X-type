#!/usr/bin/env python3
"""Fine scan of spin Chern and Z2 for t34-only model."""

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

from scripts.topology_diagnosis.t34_only_common import (  # noqa: E402
    ModelParams,
    chern_fukui,
    commutator_norm_h_sz,
    compute_bulk_gap,
    crossing_count,
    h_spin_block_t34,
    track_wilson_branches,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine scan QSH invariants in 0.8-1.2 (t34-only).")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    p.add_argument("--v-min", type=float, default=0.80)
    p.add_argument("--v-max", type=float, default=1.20)
    p.add_argument("--v-step", type=float, default=0.02)
    p.add_argument("--dense-step", type=float, default=0.01)
    p.add_argument("--use-dense", action="store_true")
    p.add_argument("--nk-chern", type=int, default=61)
    p.add_argument("--nk-z2", type=int, default=101)
    p.add_argument("--spin-threshold", type=float, default=1e-8)
    p.add_argument("--integer-threshold", type=float, default=1e-3)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def make_v_list(vmin: float, vmax: float, step: float) -> list[float]:
    n = int(round((vmax - vmin) / step))
    return [round(vmin + i * step, 10) for i in range(n + 1)]


def load_bulk_gap_map(path: Path) -> dict[float, float]:
    if not path.exists():
        return {}
    out: dict[float, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out[round(float(row["v"]), 10)] = float(row["bulk_gap"])
    return out


def build_h8_from_spin_blocks(kx: float, ky: float, v: float, params: ModelParams) -> np.ndarray:
    pup = np.array([[1, 0], [0, 0]], dtype=complex)
    pdn = np.array([[0, 0], [0, 1]], dtype=complex)
    return np.kron(h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=+1), pup) + np.kron(
        h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=-1),
        pdn,
    )


def z2_crossing_raw(v: float, params: ModelParams, nkx: int, nky: int) -> int:
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
    c45 = crossing_count(tracked, 0.45)
    c50 = crossing_count(tracked, 0.50)
    c55 = crossing_count(tracked, 0.55)
    par = [c45 % 2, c50 % 2, c55 % 2]
    if par[0] == par[1] == par[2]:
        return int(par[1])
    return int(par[1])


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    out = base / "fine_scan_0p8_1p2" / "02_spin_chern_z2"
    out.mkdir(parents=True, exist_ok=True)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))

    step = args.dense_step if args.use_dense else args.v_step
    v_list = make_v_list(args.v_min, args.v_max, step)
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    bulk_map = load_bulk_gap_map(base / "fine_scan_0p8_1p2" / "bulk_gap_fine_scan_t34_only.csv")

    rows = []
    for v in v_list:
        bulk_gap = bulk_map.get(round(v, 10))
        if bulk_gap is None:
            bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=101, n_occ=4)
        comm = commutator_norm_h_sz(v=v, params=params, nk=17)
        spin_conserved = bool(comm < args.spin_threshold)

        c_up = chern_fukui(
            lambda kx, ky: h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=+1),
            n_occ=2,
            nk=args.nk_chern,
        )
        c_dn = chern_fukui(
            lambda kx, ky: h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=-1),
            n_occ=2,
            nk=args.nk_chern,
        )
        c_spin = 0.5 * (c_up - c_dn)
        c_up_r = int(np.rint(c_up))
        c_dn_r = int(np.rint(c_dn))
        c_spin_r = int(np.rint(c_spin))
        err_up = abs(c_up - c_up_r)
        err_dn = abs(c_dn - c_dn_r)
        integer_converged = bool(err_up < args.integer_threshold and err_dn < args.integer_threshold)

        z2_raw = z2_crossing_raw(v=v, params=params, nkx=args.nk_z2, nky=args.nk_z2)
        z2_from_spin = int(c_spin_r % 2)

        warning = ""
        if spin_conserved and integer_converged:
            z2_final = z2_from_spin
            if z2_raw != z2_from_spin:
                warning = (
                    "WARNING: Wilson crossing Z2 disagrees with spin Chern parity. "
                    "Since spin is conserved, final Z2 follows C_spin mod 2."
                )
        else:
            z2_final = z2_raw
            if not spin_conserved:
                warning = "spin_not_conserved_use_crossing_z2"
            elif not integer_converged:
                warning = "spin_chern_not_integer_converged_use_crossing_z2"

        qsh_supported = bool(c_spin_r == 1 and z2_final == 1)
        rows.append(
            {
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "spin_conserved": int(spin_conserved),
                "commutator_norm_H_sz": float(comm),
                "C_up": float(c_up),
                "C_down": float(c_dn),
                "C_spin": float(c_spin),
                "C_up_rounded": int(c_up_r),
                "C_down_rounded": int(c_dn_r),
                "C_spin_rounded": int(c_spin_r),
                "integer_error_up": float(err_up),
                "integer_error_down": float(err_dn),
                "Z2_crossing_raw": int(z2_raw),
                "Z2_from_spin_chern": int(z2_from_spin),
                "Z2_final": int(z2_final),
                "QSH_supported": int(qsh_supported),
                "warning": warning,
            }
        )
        print(
            f"[qsh-fine-t34] v={v:.2f} C_spin={c_spin:.4f}->{c_spin_r} "
            f"Z2(raw/final)={z2_raw}/{z2_final} QSH={qsh_supported}"
        )

    fields = [
        "v",
        "bulk_gap",
        "spin_conserved",
        "commutator_norm_H_sz",
        "C_up",
        "C_down",
        "C_spin",
        "C_up_rounded",
        "C_down_rounded",
        "C_spin_rounded",
        "integer_error_up",
        "integer_error_down",
        "Z2_crossing_raw",
        "Z2_from_spin_chern",
        "Z2_final",
        "QSH_supported",
        "warning",
    ]
    csv_path = out / "qsh_fine_scan_t34_only.csv"
    write_csv(csv_path, rows, fields)
    (base / "fine_scan_0p8_1p2" / "qsh_fine_scan_t34_only.csv").write_bytes(csv_path.read_bytes())

    v = np.array([float(r["v"]) for r in rows], dtype=float)
    gap = np.array([float(r["bulk_gap"]) for r in rows], dtype=float)
    cspin = np.array([float(r["C_spin"]) for r in rows], dtype=float)
    z2 = np.array([int(r["Z2_final"]) for r in rows], dtype=float)
    qsh = np.array([int(r["QSH_supported"]) for r in rows], dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)
    axes[0].plot(v, gap, marker="o")
    axes[0].set_ylabel("bulk gap")
    axes[0].set_title("Fine QSH scan (t34-only)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(v, cspin, marker="o", label="C_spin")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    diff_idx = np.where(np.diff(np.rint(cspin).astype(int)) != 0)[0]
    if diff_idx.size > 0:
        vc = 0.5 * (v[diff_idx[0]] + v[diff_idx[0] + 1])
        axes[1].axvline(vc, color="purple", linestyle="--", linewidth=1.0, label=f"C_spin jump ~{vc:.2f}")
    axes[1].set_ylabel("C_spin")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].step(v, z2, where="mid", label="Z2_final")
    axes[2].set_ylabel("Z2_final")
    axes[2].set_xlabel("v")
    axes[2].set_ylim(-0.1, 1.1)
    for i in range(len(v) - 1):
        if qsh[i] == 1:
            axes[2].axvspan(v[i], v[i + 1], color="green", alpha=0.15)
    axes[2].step(v, qsh, where="mid", label="QSH_supported", linestyle="--")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    png = out / "qsh_fine_scan_t34_only.png"
    fig.savefig(png, dpi=180)
    fig.savefig(base / "fine_scan_0p8_1p2" / "qsh_fine_scan_t34_only.png", dpi=180)
    plt.close(fig)

    onset = None
    for r in rows:
        if int(r["QSH_supported"]) == 1:
            onset = float(r["v"])
            break
    (out / "qsh_fine_scan_summary_t34_only.txt").write_text(
        "\n".join(
            [
                f"nk_chern={args.nk_chern}",
                f"nk_z2={args.nk_z2}",
                f"spin_threshold={args.spin_threshold}",
                f"integer_threshold={args.integer_threshold}",
                f"v_QSH_onset={onset if onset is not None else 'None'}",
                f"used_dense_scan={int(args.use_dense)}",
                f"scan_step={step:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] qsh fine summary: v_QSH_onset={onset}")


if __name__ == "__main__":
    main()
