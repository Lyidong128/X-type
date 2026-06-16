#!/usr/bin/env python3
"""Compute HOTI-related invariants for H_chiral and H_original."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GAMMA = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)


@dataclass
class NestedResult:
    q_xy: float
    wannier_gap: float
    reliability: str
    p_y_from_nux: float
    p_x_from_nuy: float


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOTI invariant analysis for chiral/original models.")
    parser.add_argument("--output-root", default="/workspace/outputs/hoti_invariants")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--v-list", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--model-types", default="H_chiral,H_original")
    parser.add_argument("--nk-wilson", type=int, default=81)
    parser.add_argument("--ribbon-L", type=int, default=40)
    parser.add_argument("--ribbon-nk", type=int, default=101)
    parser.add_argument("--edge-cells", type=int, default=4)
    parser.add_argument("--corner-L-list", default="10,20,30")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def token(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "m").replace(".", "p")


def idx(ix: int, iy: int, orb: int, lx: int) -> int:
    return ((iy * lx + ix) * 4) + orb


def idx_1d(i: int, orb: int) -> int:
    return i * 4 + orb


def add_hop(h: np.ndarray, i: int, j: int, amp: complex) -> None:
    h[i, j] += amp
    h[j, i] += np.conjugate(amp)


def h_original_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    h = np.zeros((4, 4), dtype=complex)
    h[0, 1] = t
    h[0, 2] = t
    h[0, 3] = v + w * np.exp(-1j * kx)
    h[1, 2] = v + w * np.exp(-1j * ky)
    h[1, 3] = t
    h[2, 3] = t
    h = h + h.conj().T
    np.fill_diagonal(h, 0.0)
    return h


def h_chiral_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    h0 = h_original_k(kx, ky, v, t, w)
    return 0.5 * (h0 - GAMMA @ h0 @ GAMMA)


def get_hk(model_type: str, v: float, t: float, w: float):
    if model_type == "H_chiral":
        return lambda kx, ky: h_chiral_k(kx, ky, v=v, t=t, w=w)
    if model_type == "H_original":
        return lambda kx, ky: h_original_k(kx, ky, v=v, t=t, w=w)
    raise ValueError(f"Unsupported model_type: {model_type}")


def occ_subspace(hk, kx: float, ky: float, n_occ: int = 2) -> np.ndarray:
    _, vecs = np.linalg.eigh(hk(kx, ky))
    return vecs[:, :n_occ]


def unitary_part(mat: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(mat)
    return u @ vh


def wilson_matrix_x(hk, ky: float, kx_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_occ = 2
    wmat = np.eye(n_occ, dtype=complex)
    occ_base = occ_subspace(hk, float(kx_grid[0]), float(ky), n_occ=n_occ)
    for i, kx in enumerate(kx_grid):
        kx_next = float(kx_grid[(i + 1) % len(kx_grid)])
        occ_a = occ_subspace(hk, float(kx), float(ky), n_occ=n_occ)
        occ_b = occ_subspace(hk, kx_next, float(ky), n_occ=n_occ)
        link = unitary_part(occ_a.conj().T @ occ_b)
        wmat = link @ wmat
    return wmat, occ_base


def wilson_matrix_y(hk, kx: float, ky_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_occ = 2
    wmat = np.eye(n_occ, dtype=complex)
    occ_base = occ_subspace(hk, float(kx), float(ky_grid[0]), n_occ=n_occ)
    for i, ky in enumerate(ky_grid):
        ky_next = float(ky_grid[(i + 1) % len(ky_grid)])
        occ_a = occ_subspace(hk, float(kx), float(ky), n_occ=n_occ)
        occ_b = occ_subspace(hk, float(kx), ky_next, n_occ=n_occ)
        link = unitary_part(occ_a.conj().T @ occ_b)
        wmat = link @ wmat
    return wmat, occ_base


def wilson_spectrum_x(hk, nk: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    kx_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    centers = np.zeros((nk, 2), dtype=float)
    eigvec_occ: list[np.ndarray] = []
    occ_bases: list[np.ndarray] = []
    for j, ky in enumerate(ky_grid):
        wmat, occ_base = wilson_matrix_x(hk, float(ky), kx_grid)
        eigvals, eigvecs = np.linalg.eig(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        eigvec_occ.append(eigvecs[:, order])
        occ_bases.append(occ_base)
    return ky_grid, centers, eigvec_occ, occ_bases


def wilson_spectrum_y(hk, nk: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    ky_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    centers = np.zeros((nk, 2), dtype=float)
    eigvec_occ: list[np.ndarray] = []
    occ_bases: list[np.ndarray] = []
    for j, kx in enumerate(kx_grid):
        wmat, occ_base = wilson_matrix_y(hk, float(kx), ky_grid)
        eigvals, eigvecs = np.linalg.eig(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        eigvec_occ.append(eigvecs[:, order])
        occ_bases.append(occ_base)
    return kx_grid, centers, eigvec_occ, occ_bases


def circular_two_band_gap(centers: np.ndarray) -> float:
    d = np.abs(centers[:, 1] - centers[:, 0])
    d = np.minimum(d, 1.0 - d)
    return float(np.min(d))


def track_wannier_sector_vectors(
    centers: np.ndarray,
    eigvec_occ: list[np.ndarray],
    occ_bases: list[np.ndarray],
) -> tuple[list[np.ndarray], float]:
    # Build physical 4D vectors for two sectors at each scan point.
    states_by_k: list[list[np.ndarray]] = []
    for j in range(len(centers)):
        occ = occ_bases[j]
        vecs = eigvec_occ[j]
        states = []
        for s in range(2):
            vec = occ @ vecs[:, s]
            vec = vec / max(np.linalg.norm(vec), 1e-15)
            states.append(vec)
        states_by_k.append(states)

    # Start from lower Wannier center branch.
    prev_idx = int(np.argmin(centers[0, :]))
    tracked: list[np.ndarray] = [states_by_k[0][prev_idx]]
    min_overlap = 1.0
    for j in range(1, len(centers)):
        cand0 = states_by_k[j][0]
        cand1 = states_by_k[j][1]
        ov0 = abs(np.vdot(tracked[-1], cand0))
        ov1 = abs(np.vdot(tracked[-1], cand1))
        if ov1 > ov0:
            chosen = cand1
            min_overlap = min(min_overlap, float(ov1))
            prev_idx = 1
        else:
            chosen = cand0
            min_overlap = min(min_overlap, float(ov0))
            prev_idx = 0
        tracked.append(chosen)

    # Enforce loop closure phase consistency.
    if len(tracked) > 1:
        overlap_close = abs(np.vdot(tracked[-1], tracked[0]))
        min_overlap = min(min_overlap, float(overlap_close))
    return tracked, float(min_overlap)


def berry_phase_from_tracked_states(states: list[np.ndarray]) -> float:
    phase_prod = 1.0 + 0.0j
    n = len(states)
    for j in range(n):
        a = states[j]
        b = states[(j + 1) % n]
        ov = np.vdot(a, b)
        if abs(ov) > 1e-14:
            phase_prod *= ov / abs(ov)
    return float(np.mod(np.angle(phase_prod) / (2.0 * np.pi), 1.0))


def nested_result(hk, nk: int) -> NestedResult:
    _, centers_x, eigvec_x, occ_base_x = wilson_spectrum_x(hk, nk=nk)
    _, centers_y, eigvec_y, occ_base_y = wilson_spectrum_y(hk, nk=nk)
    gap_x = circular_two_band_gap(centers_x)
    gap_y = circular_two_band_gap(centers_y)
    wannier_gap = float(min(gap_x, gap_y))

    tracked_x, min_ov_x = track_wannier_sector_vectors(centers_x, eigvec_x, occ_base_x)
    tracked_y, min_ov_y = track_wannier_sector_vectors(centers_y, eigvec_y, occ_base_y)
    p_y_from_nux = berry_phase_from_tracked_states(tracked_x)
    p_x_from_nuy = berry_phase_from_tracked_states(tracked_y)
    q_xy = float(np.mod(0.5 * (p_y_from_nux + p_x_from_nuy), 1.0))

    if wannier_gap < 0.03:
        reliability = "low_wannier_gap"
    elif min(min_ov_x, min_ov_y) < 0.60:
        reliability = "medium_sector_tracking"
    else:
        reliability = "high"
    return NestedResult(
        q_xy=q_xy,
        wannier_gap=wannier_gap,
        reliability=reliability,
        p_y_from_nux=p_y_from_nux,
        p_x_from_nuy=p_x_from_nuy,
    )


def build_ribbon_x_open(ky: float, lx: int, model_type: str, v: float, t: float, w: float) -> np.ndarray:
    h = np.zeros((4 * lx, 4 * lx), dtype=complex)
    t_same = 0.0 if model_type == "H_chiral" else t
    vky = v + w * np.exp(-1j * ky)
    for ix in range(lx):
        i0 = idx_1d(ix, 0)
        i1 = idx_1d(ix, 1)
        i2 = idx_1d(ix, 2)
        i3 = idx_1d(ix, 3)
        add_hop(h, i0, i1, t_same)
        add_hop(h, i0, i2, t)
        add_hop(h, i0, i3, v)
        add_hop(h, i1, i2, vky)
        add_hop(h, i1, i3, t)
        add_hop(h, i2, i3, t_same)
        if ix > 0:
            add_hop(h, i0, idx_1d(ix - 1, 3), w)
    return h


def build_ribbon_y_open(kx: float, ly: int, model_type: str, v: float, t: float, w: float) -> np.ndarray:
    h = np.zeros((4 * ly, 4 * ly), dtype=complex)
    t_same = 0.0 if model_type == "H_chiral" else t
    vkx = v + w * np.exp(-1j * kx)
    for iy in range(ly):
        i0 = idx_1d(iy, 0)
        i1 = idx_1d(iy, 1)
        i2 = idx_1d(iy, 2)
        i3 = idx_1d(iy, 3)
        add_hop(h, i0, i1, t_same)
        add_hop(h, i0, i2, t)
        add_hop(h, i0, i3, vkx)
        add_hop(h, i1, i2, v)
        add_hop(h, i1, i3, t)
        add_hop(h, i2, i3, t_same)
        if iy > 0:
            add_hop(h, i1, idx_1d(iy - 1, 2), w)
    return h


def ribbon_edge_polarization(
    model_type: str,
    v: float,
    t: float,
    w: float,
    l_ribbon: int,
    nk: int,
    edge_cells: int,
) -> tuple[float, float, float, float, float, float]:
    k_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    rho_x = np.zeros(l_ribbon, dtype=float)
    rho_y = np.zeros(l_ribbon, dtype=float)
    n_occ = 2 * l_ribbon

    for k in k_grid:
        hx = build_ribbon_x_open(float(k), l_ribbon, model_type, v, t, w)
        ex, vx = np.linalg.eigh(hx)
        occx = vx[:, :n_occ]
        for ix in range(l_ribbon):
            s = idx_1d(ix, 0)
            rho_x[ix] += float(np.sum(np.abs(occx[s : s + 4, :]) ** 2))

        hy = build_ribbon_y_open(float(k), l_ribbon, model_type, v, t, w)
        ey, vy = np.linalg.eigh(hy)
        occy = vy[:, :n_occ]
        for iy in range(l_ribbon):
            s = idx_1d(iy, 0)
            rho_y[iy] += float(np.sum(np.abs(occy[s : s + 4, :]) ** 2))

    rho_x /= nk
    rho_y /= nk
    trim = max(edge_cells, 2)
    bulk_slice = slice(trim, l_ribbon - trim)
    bg_x = float(np.mean(rho_x[bulk_slice]))
    bg_y = float(np.mean(rho_y[bulk_slice]))
    dx = rho_x - bg_x
    dy = rho_y - bg_y
    ql_x = float(np.sum(dx[:edge_cells]))
    qr_x = float(np.sum(dx[-edge_cells:]))
    ql_y = float(np.sum(dy[:edge_cells]))
    qr_y = float(np.sum(dy[-edge_cells:]))

    # A practical edge-polarization proxy from edge excess charge asymmetry.
    p_x_edge = 0.5 * (qr_x - ql_x)
    p_y_edge = 0.5 * (qr_y - ql_y)
    return p_x_edge, p_y_edge, ql_x, qr_x, ql_y, qr_y


def build_obc_hamiltonian(l: int, model_type: str, v: float, t: float, w: float) -> np.ndarray:
    n = 4 * l * l
    h = np.zeros((n, n), dtype=float)
    t_same = 0.0 if model_type == "H_chiral" else t
    for iy in range(l):
        for ix in range(l):
            i0 = idx(ix, iy, 0, l)
            i1 = idx(ix, iy, 1, l)
            i2 = idx(ix, iy, 2, l)
            i3 = idx(ix, iy, 3, l)
            add_hop(h, i0, i1, t_same)
            add_hop(h, i0, i2, t)
            add_hop(h, i0, i3, v)
            add_hop(h, i1, i2, v)
            add_hop(h, i1, i3, t)
            add_hop(h, i2, i3, t_same)
            if ix > 0:
                add_hop(h, i0, idx(ix - 1, iy, 3, l), w)
            if iy > 0:
                add_hop(h, i1, idx(ix, iy - 1, 2, l), w)
    return h


def corner_charge_for_case(l: int, model_type: str, v: float, t: float, w: float) -> dict[str, float | int | str]:
    h = build_obc_hamiltonian(l, model_type, v, t, w)
    evals, evecs = np.linalg.eigh(h)
    n_occ = h.shape[0] // 2
    occ = evecs[:, :n_occ]

    rho = np.zeros((l, l), dtype=float)
    for iy in range(l):
        for ix in range(l):
            s = idx(ix, iy, 0, l)
            rho[iy, ix] = float(np.sum(np.abs(occ[s : s + 4, :]) ** 2))

    rho_bg = float(n_occ / (l * l))
    delta = rho - rho_bg
    lc = max(2, l // 6)
    c1 = float(np.sum(delta[:lc, :lc]))
    c2 = float(np.sum(delta[:lc, -lc:]))
    c3 = float(np.sum(delta[-lc:, :lc]))
    c4 = float(np.sum(delta[-lc:, -lc:]))
    corner_mean = float(np.mean([c1, c2, c3, c4]))
    corner_abs_mean = float(np.mean(np.abs([c1, c2, c3, c4])))
    return {
        "model_type": model_type,
        "v": v,
        "L": l,
        "corner_patch_size": lc,
        "Q_corner_1": c1,
        "Q_corner_2": c2,
        "Q_corner_3": c3,
        "Q_corner_4": c4,
        "Q_corner_mean": corner_mean,
        "Q_corner_abs_mean": corner_abs_mean,
        "Q_corner_total_4corners": float(c1 + c2 + c3 + c4),
    }


def load_corner_weight_reference(path: Path) -> dict[tuple[str, float], dict[str, float]]:
    ref: dict[tuple[str, float], dict[str, float]] = {}
    if not path.exists():
        return ref
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows.extend(reader)
    for model_type, alpha in [("H_chiral", 0.0), ("H_original", 1.0)]:
        for v in sorted({float(r["v"]) for r in rows}):
            sub = [
                r
                for r in rows
                if abs(float(r["v"]) - v) < 1e-12
                and abs(float(r["alpha"]) - alpha) < 1e-12
                and int(r["rank_by_absE"]) < 4
            ]
            if not sub:
                continue
            ref[(model_type, v)] = {
                "W_corner_mean4": float(np.mean([float(r["W_corner"]) for r in sub])),
                "W_edge_mean4": float(np.mean([float(r["W_edge"]) for r in sub])),
                "W_bulk_mean4": float(np.mean([float(r["W_bulk"]) for r in sub])),
            }
    return ref


def plot_wannier_spectrum_panels(
    save_path: Path,
    spectra: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]],
    model_types: list[str],
    v_list: list[float],
    axis_label: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(len(model_types), len(v_list), figsize=(4.2 * len(v_list), 3.6 * len(model_types)), sharey=True)
    if len(model_types) == 1:
        axes = np.array([axes])
    if len(v_list) == 1:
        axes = axes.reshape(len(model_types), 1)
    for i, model_type in enumerate(model_types):
        for j, v in enumerate(v_list):
            ax = axes[i, j]
            k, centers = spectra[(model_type, v)]
            ax.plot(k, centers[:, 0], linewidth=0.9, color="tab:blue")
            ax.plot(k, centers[:, 1], linewidth=0.9, color="tab:orange")
            ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.22)
            ax.set_title(f"{model_type}, v={v:.2f}", fontsize=10)
            ax.set_xlabel(axis_label)
            if j == 0:
                ax.set_ylabel("Wannier center")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantized_to_half(x: float, tol: float = 0.10) -> bool:
    d0 = min(abs(x - 0.0), abs(x - 1.0))
    d1 = abs(x - 0.5)
    return d1 + 1e-12 < d0 and d1 < tol


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    t = float(args.t)
    w = float(args.w)
    v_list = parse_float_list(args.v_list)
    model_types = [x.strip() for x in args.model_types.split(",") if x.strip()]
    nk_wilson = 41 if args.quick else int(args.nk_wilson)
    ribbon_l = 24 if args.quick else int(args.ribbon_L)
    ribbon_nk = 41 if args.quick else int(args.ribbon_nk)
    edge_cells = min(int(args.edge_cells), max(2, ribbon_l // 3))
    corner_l_list = [10] if args.quick else parse_int_list(args.corner_L_list)

    # ---------- Task 1: ordinary Wilson loop ----------
    wx_spectra: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    wy_spectra: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    nested_rows: list[dict] = []

    for model_type in model_types:
        for v in v_list:
            hk = get_hk(model_type, v=v, t=t, w=w)
            ky, centers_x, _, _ = wilson_spectrum_x(hk, nk=nk_wilson)
            kx, centers_y, _, _ = wilson_spectrum_y(hk, nk=nk_wilson)
            wx_spectra[(model_type, v)] = (ky, centers_x)
            wy_spectra[(model_type, v)] = (kx, centers_y)

            nr = nested_result(hk, nk=nk_wilson)
            nested_rows.append(
                {
                    "model_type": model_type,
                    "v": v,
                    "q_xy": nr.q_xy,
                    "wannier_gap": nr.wannier_gap,
                    "reliability": nr.reliability,
                    "p_y_from_nu_x": nr.p_y_from_nux,
                    "p_x_from_nu_y": nr.p_x_from_nuy,
                }
            )

    plot_wannier_spectrum_panels(
        save_path=out_root / "wannier_spectrum_x.png",
        spectra=wx_spectra,
        model_types=model_types,
        v_list=v_list,
        axis_label=r"$k_y$",
        title=f"Wannier spectrum nu_x(k_y), t={t:.1f}, w={w:.1f}",
    )
    plot_wannier_spectrum_panels(
        save_path=out_root / "wannier_spectrum_y.png",
        spectra=wy_spectra,
        model_types=model_types,
        v_list=v_list,
        axis_label=r"$k_x$",
        title=f"Wannier spectrum nu_y(k_x), t={t:.1f}, w={w:.1f}",
    )

    write_csv(
        out_root / "nested_wilson_summary.csv",
        nested_rows,
        fieldnames=[
            "model_type",
            "v",
            "q_xy",
            "wannier_gap",
            "reliability",
            "p_y_from_nu_x",
            "p_x_from_nu_y",
        ],
    )

    # ---------- Task 3: edge polarization ----------
    edge_rows: list[dict] = []
    for model_type in model_types:
        for v in v_list:
            px, py, qlx, qrx, qly, qry = ribbon_edge_polarization(
                model_type=model_type,
                v=v,
                t=t,
                w=w,
                l_ribbon=ribbon_l,
                nk=ribbon_nk,
                edge_cells=edge_cells,
            )
            edge_rows.append(
                {
                    "model_type": model_type,
                    "v": v,
                    "L_ribbon": ribbon_l,
                    "nk": ribbon_nk,
                    "edge_cells": edge_cells,
                    "P_x_edge": px,
                    "P_y_edge": py,
                    "Q_left_x_edge": qlx,
                    "Q_right_x_edge": qrx,
                    "Q_bottom_y_edge": qly,
                    "Q_top_y_edge": qry,
                }
            )

    write_csv(
        out_root / "edge_polarization_summary.csv",
        edge_rows,
        fieldnames=[
            "model_type",
            "v",
            "L_ribbon",
            "nk",
            "edge_cells",
            "P_x_edge",
            "P_y_edge",
            "Q_left_x_edge",
            "Q_right_x_edge",
            "Q_bottom_y_edge",
            "Q_top_y_edge",
        ],
    )

    # ---------- Task 4: corner charge ----------
    corner_rows: list[dict] = []
    for model_type in model_types:
        for v in v_list:
            for l in corner_l_list:
                corner_rows.append(corner_charge_for_case(l=l, model_type=model_type, v=v, t=t, w=w))

    write_csv(
        out_root / "corner_charge_summary.csv",
        corner_rows,
        fieldnames=[
            "model_type",
            "v",
            "L",
            "corner_patch_size",
            "Q_corner_1",
            "Q_corner_2",
            "Q_corner_3",
            "Q_corner_4",
            "Q_corner_mean",
            "Q_corner_abs_mean",
            "Q_corner_total_4corners",
        ],
    )

    # ---------- Task 5: compare with existing corner-localization ----------
    ref_metrics = load_corner_weight_reference(Path("/workspace/outputs/chiral_symmetric_model/near_zero_state_metrics.csv"))
    nested_map = {(r["model_type"], float(r["v"])): r for r in nested_rows}
    edge_map = {(r["model_type"], float(r["v"])): r for r in edge_rows}

    report_lines = [
        "# HOTI_INVARIANT_REPORT",
        "",
        "## Setup",
        f"- model_types={model_types}",
        f"- v_list={v_list}",
        f"- parameters: t={t:.3f}, w={w:.3f}, lm=0.0",
        f"- Wilson grid nk={nk_wilson}; ribbon L={ribbon_l}, nk={ribbon_nk}; corner L_list={corner_l_list}",
        "",
        "## 1) Ordinary Wilson loop / Wannier spectrum",
        "- `wannier_spectrum_x.png` and `wannier_spectrum_y.png` show nu_x(ky), nu_y(kx) for both models and all v.",
        "- `nested_wilson_summary.csv` gives q_xy and Wannier-gap based reliability.",
        "",
        "## 2) Nested Wilson / quadrupole proxy",
    ]

    for model_type in model_types:
        report_lines.append(f"### {model_type}")
        for v in v_list:
            r = nested_map[(model_type, v)]
            qxy = float(r["q_xy"])
            gap = float(r["wannier_gap"])
            rel = str(r["reliability"])
            half_tag = "near-1/2" if quantized_to_half(qxy) else "not-near-1/2"
            report_lines.append(
                f"- v={v:.2f}: q_xy={qxy:.6f}, wannier_gap={gap:.4f}, reliability={rel}, verdict={half_tag}"
            )

    report_lines += [
        "",
        "## 3) Edge polarization (ribbon proxy)",
        "- `edge_polarization_summary.csv` reports P_x_edge / P_y_edge from edge excess-charge asymmetry proxies.",
    ]
    for model_type in model_types:
        report_lines.append(f"### {model_type}")
        for v in v_list:
            e = edge_map[(model_type, v)]
            report_lines.append(
                f"- v={v:.2f}: P_x_edge={float(e['P_x_edge']):.6f}, P_y_edge={float(e['P_y_edge']):.6f}"
            )

    report_lines += [
        "",
        "## 4) Corner charge scaling (OBC)",
        "- `corner_charge_summary.csv` gives corner charge in corner patches for L=10,20,30.",
    ]
    for model_type in model_types:
        report_lines.append(f"### {model_type}")
        for v in v_list:
            sub = [r for r in corner_rows if r["model_type"] == model_type and abs(float(r["v"]) - v) < 1e-12]
            sub = sorted(sub, key=lambda r: int(r["L"]))
            for r in sub:
                report_lines.append(
                    f"- v={v:.2f}, L={int(r['L'])}: Q_corner_mean={float(r['Q_corner_mean']):.6f}, "
                    f"Q_corner_abs_mean={float(r['Q_corner_abs_mean']):.6f}"
                )

    report_lines += [
        "",
        "## 5) Compare with existing near-zero corner weights",
        "- Existing reference from `outputs/chiral_symmetric_model/near_zero_state_metrics.csv` (rank_by_absE<4).",
    ]
    for model_type in model_types:
        report_lines.append(f"### {model_type}")
        for v in v_list:
            key = (model_type, v)
            if key in ref_metrics:
                m = ref_metrics[key]
                report_lines.append(
                    f"- v={v:.2f}: <W_corner,W_edge,W_bulk>=({m['W_corner_mean4']:.3f},"
                    f"{m['W_edge_mean4']:.3f},{m['W_bulk_mean4']:.3f})"
                )
            else:
                report_lines.append(f"- v={v:.2f}: no reference W_corner data found")

    # Final judgment requested by user.
    report_lines += [
        "",
        "## Final judgments",
    ]
    # 1) H_chiral q_xy ~ 1/2?
    chiral_rows = [nested_map[("H_chiral", v)] for v in v_list if ("H_chiral", v) in nested_map]
    n_half_chiral = sum(1 for r in chiral_rows if quantized_to_half(float(r["q_xy"])))
    report_lines.append(
        f"1. H_chiral 是否量子化 q_xy=1/2：{n_half_chiral}/{len(chiral_rows)} 个 v 点接近 1/2（按 |q_xy-0.5|<0.1 判据）。"
    )
    orig_rows = [nested_map[("H_original", v)] for v in v_list if ("H_original", v) in nested_map]
    n_half_orig = sum(1 for r in orig_rows if quantized_to_half(float(r["q_xy"])))
    report_lines.append(
        f"2. H_original 是否保留二阶拓扑特征：{n_half_orig}/{len(orig_rows)} 个 v 点接近 1/2；需结合 reliability 与 corner charge 尺度行为判断稳健性。"
    )

    mixed_tags = []
    for v in v_list:
        key = ("H_original", v)
        if key not in ref_metrics:
            continue
        mm = ref_metrics[key]
        if mm["W_edge_mean4"] >= mm["W_corner_mean4"]:
            mixed_tags.append(v)
    report_lines.append(
        f"3. 原始模型近零态是否 edge-corner mixed：在 v={mixed_tags} 上 W_edge>=W_corner，显示明显混合。"
    )

    # Conservative naming judgment.
    strong_hoti_points = 0
    for v in v_list:
        nk = nested_map.get(("H_chiral", v))
        if nk is None:
            continue
        if quantized_to_half(float(nk["q_xy"])) and str(nk["reliability"]).startswith("high"):
            strong_hoti_points += 1
    if strong_hoti_points >= 2:
        label = "可称为“高阶拓扑角态候选”，但仍建议做更大尺寸与边界条件鲁棒性验证。"
    else:
        label = "现有证据更偏向“近零角局域/边角混合态”，不足以强称高阶拓扑角态。"
    report_lines.append(f"4. 是否可称为高阶拓扑角态：{label}")

    (out_root / "HOTI_INVARIANT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[ok] output_root={out_root}")
    print(f"[ok] nested_rows={len(nested_rows)} edge_rows={len(edge_rows)} corner_rows={len(corner_rows)}")


if __name__ == "__main__":
    main()
