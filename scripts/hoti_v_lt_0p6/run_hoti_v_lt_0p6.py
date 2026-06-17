#!/usr/bin/env python3
"""Validate HOTI signatures for original SOC model in v<0.6 region."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOTI check for original SOC model at v<0.6.")
    parser.add_argument("--output-root", default="/workspace/outputs/hoti_v_lt_0p6")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--v-list", default="0.3,0.4,0.5")
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--bulk-nk", type=int, default=101)
    parser.add_argument("--bulk-path-nseg", type=int, default=220)
    parser.add_argument("--ribbon-width", type=int, default=40)
    parser.add_argument("--ribbon-nk", type=int, default=161)
    parser.add_argument("--edge-cells", type=int, default=3)
    parser.add_argument("--L-list", default="20,30,40")
    parser.add_argument("--near-k", type=int, default=120)
    parser.add_argument("--corner-size-ratio", type=float, default=0.12)
    parser.add_argument("--edge-width-ratio", type=float, default=0.12)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def token(v: float) -> str:
    return f"{v:.1f}".replace("-", "m").replace(".", "p")


def h_soc_orbital(lm: float) -> np.ndarray:
    h1 = np.zeros((4, 4), dtype=complex)
    h1[0, 1] = 1j * lm
    h1[0, 2] = -1j * lm
    h1[1, 0] = -1j * lm
    h1[1, 3] = 1j * lm
    h1[2, 0] = 1j * lm
    h1[2, 3] = -1j * lm
    h1[3, 1] = -1j * lm
    h1[3, 2] = 1j * lm
    return h1


def h0_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v + w * np.exp(-1j * kx)
    h0[1, 0] = t
    h0[1, 2] = v + w * np.exp(-1j * ky)
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v + w * np.exp(1j * ky)
    h0[2, 3] = t
    h0[3, 0] = v + w * np.exp(1j * kx)
    h0[3, 1] = t
    h0[3, 2] = t
    return h0


def h8_k(kx: float, ky: float, v: float, t: float, w: float, lm: float) -> np.ndarray:
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(h0_k(kx, ky, v, t, w), s0) + np.kron(h_soc_orbital(lm), sz)


def build_high_symmetry_path(nseg: int) -> tuple[np.ndarray, np.ndarray, list[float], list[str], int]:
    b1 = np.array([2.0 * np.pi, 0.0], dtype=float)
    b2 = np.array([0.0, 2.0 * np.pi], dtype=float)
    g = np.array([0.0, 0.0], dtype=float)
    x = 0.5 * b1
    y = 0.5 * b2
    m = 0.5 * (b1 + b2)
    nodes = [g, x, y, g, m, g]
    labels = [r"$\Gamma$", "X", "Y", r"$\Gamma$", "M", r"$\Gamma$"]

    k_list = []
    tick_indices = [0]
    for i in range(len(nodes) - 1):
        a = nodes[i]
        b = nodes[i + 1]
        for j in range(nseg):
            u = j / nseg
            k_list.append((1.0 - u) * a + u * b)
        tick_indices.append(len(k_list))
    k_list.append(nodes[-1])

    kpts = np.array(k_list, dtype=float)
    xcoords = np.zeros(len(kpts), dtype=float)
    for i in range(1, len(kpts)):
        xcoords[i] = xcoords[i - 1] + np.linalg.norm(kpts[i] - kpts[i - 1])
    tick_positions = [float(xcoords[i]) for i in tick_indices]
    m_idx = tick_indices[labels.index("M")]
    return kpts, xcoords, tick_positions, labels, m_idx


def compute_bulk_summary(v: float, t: float, w: float, lm: float, nk: int, n_occ: int) -> dict[str, float | str | int]:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    valence_max = -np.inf
    conduction_min = np.inf
    direct_min = np.inf
    k_direct = (0.0, 0.0)
    for kx in kgrid:
        for ky in kgrid:
            e = np.linalg.eigvalsh(h8_k(float(kx), float(ky), v=v, t=t, w=w, lm=lm))
            e = np.real(e)
            val = float(e[n_occ - 1])
            con = float(e[n_occ])
            valence_max = max(valence_max, val)
            conduction_min = min(conduction_min, con)
            dg = con - val
            if dg < direct_min:
                direct_min = float(dg)
                k_direct = (float(kx), float(ky))

    m_e = np.linalg.eigvalsh(h8_k(np.pi, np.pi, v=v, t=t, w=w, lm=lm))
    m_gap = float(np.real(m_e[n_occ] - m_e[n_occ - 1]))
    global_gap = float(conduction_min - valence_max)
    return {
        "v": v,
        "global_bulk_gap": global_gap,
        "direct_bulk_gap_min": float(direct_min),
        "M_point_gap": m_gap,
        "band_min_k": f"({k_direct[0]:.6f},{k_direct[1]:.6f})",
        "band_min_kx": k_direct[0],
        "band_min_ky": k_direct[1],
        "is_bulk_gapped_insulator": int(global_gap > 1e-4),
    }


def plot_bulk_band(v: float, t: float, w: float, lm: float, nseg: int, out_png: Path) -> None:
    kpts, xcoords, ticks, labels, _m_idx = build_high_symmetry_path(nseg=nseg)
    bands = []
    for k in kpts:
        bands.append(np.real(np.linalg.eigvalsh(h8_k(k[0], k[1], v=v, t=t, w=w, lm=lm))))
    bands = np.array(bands, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for ib in range(bands.shape[1]):
        ax.plot(xcoords, bands[:, ib], linewidth=0.9, color="tab:blue")
    for x in ticks:
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.65, alpha=0.7)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy")
    ax.set_title(f"Bulk band (SOC) v={v:.1f}, t={t:.1f}, w={w:.1f}, lm={lm:.1f}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_onsite_xopen(ky: float, v: float, t: float, w: float, lm: float) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v
    h0[1, 0] = t
    h0[1, 2] = v + w * np.exp(-1j * ky)
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v + w * np.exp(1j * ky)
    h0[2, 3] = t
    h0[3, 0] = v
    h0[3, 1] = t
    h0[3, 2] = t
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(h0, s0) + np.kron(h_soc_orbital(lm), sz)


def build_ribbon_xopen(ky: float, nx: int, v: float, t: float, w: float, lm: float) -> np.ndarray:
    h = np.zeros((8 * nx, 8 * nx), dtype=complex)
    onsite = build_onsite_xopen(ky=ky, v=v, t=t, w=w, lm=lm)
    tx = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[0 * 2 + spin, 3 * 2 + spin] = w
    for x in range(nx):
        s = x * 8
        h[s : s + 8, s : s + 8] += onsite
        if x > 0:
            p = (x - 1) * 8
            h[s : s + 8, p : p + 8] += tx
            h[p : p + 8, s : s + 8] += tx.conj().T
    return h


def build_ribbon_yopen(kx: float, ny: int, v: float, t: float, w: float, lm: float) -> np.ndarray:
    h = np.zeros((8 * ny, 8 * ny), dtype=complex)
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v + w * np.exp(-1j * kx)
    h0[1, 0] = t
    h0[1, 2] = v
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v
    h0[2, 3] = t
    h0[3, 0] = v + w * np.exp(1j * kx)
    h0[3, 1] = t
    h0[3, 2] = t
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    onsite = np.kron(h0, s0) + np.kron(h_soc_orbital(lm), sz)
    ty = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        ty[1 * 2 + spin, 2 * 2 + spin] = w
    for y in range(ny):
        s = y * 8
        h[s : s + 8, s : s + 8] += onsite
        if y > 0:
            p = (y - 1) * 8
            h[s : s + 8, p : p + 8] += ty
            h[p : p + 8, s : s + 8] += ty.conj().T
    return h


def ribbon_spectrum_xopen(v: float, t: float, w: float, lm: float, nx: int, nk: int, edge_cells: int):
    ky_vals = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    evals_all = []
    edge_w_all = []
    for ky in ky_vals:
        h = build_ribbon_xopen(float(ky), nx=nx, v=v, t=t, w=w, lm=lm)
        evals, vecs = np.linalg.eigh(h)
        evals = np.real(evals)
        prob = np.abs(vecs) ** 2
        prob_cell = prob.reshape(nx, 8, prob.shape[1]).sum(axis=1)
        edge_w = prob_cell[:edge_cells, :].sum(axis=0) + prob_cell[-edge_cells:, :].sum(axis=0)
        evals_all.append(evals)
        edge_w_all.append(edge_w)
    return ky_vals, np.array(evals_all), np.array(edge_w_all)


def ribbon_spectrum_yopen(v: float, t: float, w: float, lm: float, ny: int, nk: int, edge_cells: int):
    kx_vals = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    evals_all = []
    edge_w_all = []
    for kx in kx_vals:
        h = build_ribbon_yopen(float(kx), ny=ny, v=v, t=t, w=w, lm=lm)
        evals, vecs = np.linalg.eigh(h)
        evals = np.real(evals)
        prob = np.abs(vecs) ** 2
        prob_cell = prob.reshape(ny, 8, prob.shape[1]).sum(axis=1)
        edge_w = prob_cell[:edge_cells, :].sum(axis=0) + prob_cell[-edge_cells:, :].sum(axis=0)
        evals_all.append(evals)
        edge_w_all.append(edge_w)
    return kx_vals, np.array(evals_all), np.array(edge_w_all)


def edge_gap_metrics(
    evals: np.ndarray,
    edge_weights: np.ndarray,
    bulk_gap: float,
    edge_thr: float = 0.60,
) -> tuple[float, int, float, float]:
    flat_e = evals.reshape(-1)
    neg = flat_e[flat_e < 0.0]
    pos = flat_e[flat_e > 0.0]
    if neg.size == 0 or pos.size == 0:
        return 0.0, 1, np.nan, np.nan
    edge_gap = float(np.min(pos) - np.max(neg))

    flat_w = edge_weights.reshape(-1)
    edge_e = flat_e[flat_w >= edge_thr]
    if edge_e.size == 0:
        return edge_gap, 0, np.nan, np.nan

    min_abs_edge = float(np.min(np.abs(edge_e)))
    crossing = int(np.any(edge_e < 0.0) and np.any(edge_e > 0.0))
    tol = max(1e-3, 0.05 * max(bulk_gap, 1e-6))
    gapless = int(crossing and min_abs_edge < tol)
    return edge_gap, gapless, float(np.min(edge_e)), float(np.max(edge_e))


def plot_ribbon(
    kvals: np.ndarray,
    evals: np.ndarray,
    edge_w: np.ndarray,
    out_png: Path,
    title: str,
    k_label: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    x = np.repeat(kvals / np.pi, evals.shape[1])
    y = evals.reshape(-1)
    c = edge_w.reshape(-1)
    sc = ax.scatter(x, y, c=c, s=5, cmap="viridis", edgecolors="none", vmin=0.0, vmax=1.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(k_label)
    ax.set_ylabel("Energy")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="edge weight")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def obc_index(ix: int, iy: int, orb: int, spin: int, l: int) -> int:
    return ((iy * l + ix) * 8) + orb * 2 + spin


def add_hop_spin_independent(h: lil_matrix, i: int, j: int, amp: float) -> None:
    h[i, j] += amp
    h[j, i] += amp


def build_obc_sparse(l: int, v: float, t: float, w: float, lm: float) -> csr_matrix:
    n = 8 * l * l
    h = lil_matrix((n, n), dtype=np.float64)

    # orbital onsite block with SOC (spin-dependent)
    h0 = np.zeros((4, 4), dtype=float)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v
    h0[1, 0] = t
    h0[1, 2] = v
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v
    h0[2, 3] = t
    h0[3, 0] = v
    h0[3, 1] = t
    h0[3, 2] = t
    hs = h_soc_orbital(lm)

    for iy in range(l):
        for ix in range(l):
            # onsite 8x8: H0⊗s0 + HSOC⊗sz
            for a in range(4):
                for b in range(4):
                    val0 = float(np.real(h0[a, b]))
                    vals = hs[a, b]
                    if abs(val0) > 0 or abs(vals) > 0:
                        # up spin
                        iu = obc_index(ix, iy, a, 0, l)
                        ju = obc_index(ix, iy, b, 0, l)
                        h[iu, ju] += val0 + np.real(vals)
                        # down spin
                        idn = obc_index(ix, iy, a, 1, l)
                        jdn = obc_index(ix, iy, b, 1, l)
                        h[idn, jdn] += val0 - np.real(vals)
                        # imag SOC off-diagonal terms
                        if abs(np.imag(vals)) > 0:
                            h[iu, ju] += 1j * np.imag(vals)
                            h[idn, jdn] -= 1j * np.imag(vals)

            # inter-cell x: orb0 <-> left-cell orb3 with w (spin-independent)
            if ix > 0:
                for spin in range(2):
                    i = obc_index(ix, iy, 0, spin, l)
                    j = obc_index(ix - 1, iy, 3, spin, l)
                    add_hop_spin_independent(h, i, j, w)

            # inter-cell y: orb1 <-> down-cell orb2 with w (spin-independent)
            if iy > 0:
                for spin in range(2):
                    i = obc_index(ix, iy, 1, spin, l)
                    j = obc_index(ix, iy - 1, 2, spin, l)
                    add_hop_spin_independent(h, i, j, w)

    h = h.tocsr()
    # Ensure Hermitian numerically.
    h = 0.5 * (h + h.getH())
    return h


def eigs_near_zero(h: csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = h.shape[0]
    k_use = min(max(8, k), n - 2)
    try:
        vals, vecs = eigsh(h, k=k_use, sigma=0.0, which="LM")
    except (ArpackNoConvergence, RuntimeError, ValueError):
        vals, vecs = eigsh(h, k=k_use, which="SM")
    vals = np.real(vals)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def cell_density(vec: np.ndarray, l: int) -> np.ndarray:
    rho = np.zeros((l, l), dtype=float)
    for iy in range(l):
        for ix in range(l):
            s = ((iy * l + ix) * 8)
            rho[iy, ix] = float(np.sum(np.abs(vec[s : s + 8]) ** 2))
    return rho


def corner_edge_bulk_masks(l: int, corner_size: int, edge_width: int):
    corner = np.zeros((l, l), dtype=bool)
    corner[:corner_size, :corner_size] = True
    corner[:corner_size, l - corner_size :] = True
    corner[l - corner_size :, :corner_size] = True
    corner[l - corner_size :, l - corner_size :] = True
    edge = np.zeros((l, l), dtype=bool)
    edge[:edge_width, :] = True
    edge[l - edge_width :, :] = True
    edge[:, :edge_width] = True
    edge[:, l - edge_width :] = True
    edge &= ~corner
    bulk = ~(corner | edge)
    return corner, edge, bulk


def state_weights(rho: np.ndarray, corner: np.ndarray, edge: np.ndarray, bulk: np.ndarray) -> tuple[float, float, float, float]:
    wc = float(np.sum(rho[corner]))
    we = float(np.sum(rho[edge]))
    wb = float(np.sum(rho[bulk]))
    ipr = float(np.sum(rho**2))
    return wc, we, wb, ipr


def classify_state(wc: float, we: float) -> str:
    if wc > 0.6 and wc > we:
        return "corner-like in-gap state candidate"
    if we > wc:
        return "edge-like or edge-corner mixed state"
    return "mixed/other"


def unitary_part(mat: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(mat)
    return u @ vh


def occ_subspace(kx: float, ky: float, v: float, t: float, w: float, lm: float, n_occ: int) -> np.ndarray:
    _, vecs = np.linalg.eigh(h8_k(kx, ky, v=v, t=t, w=w, lm=lm))
    return vecs[:, :n_occ]


def wilson_x_data(v: float, t: float, w: float, lm: float, n_occ: int, nk: int):
    kx_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    centers = np.zeros((nk, n_occ), dtype=float)
    sector_bases = []
    min_sector_ov = 1.0
    for j, ky in enumerate(ky_grid):
        wmat = np.eye(n_occ, dtype=complex)
        occ0 = occ_subspace(float(kx_grid[0]), float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
        for i, kx in enumerate(kx_grid):
            kx_next = float(kx_grid[(i + 1) % nk])
            oa = occ_subspace(float(kx), float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            ob = occ_subspace(kx_next, float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            q = unitary_part(oa.conj().T @ ob)
            wmat = q @ wmat
        eigvals, eigvecs = np.linalg.eig(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        ev = eigvecs[:, order]
        sec = occ0 @ ev[:, :2]  # lower Wannier sector
        sec, _ = np.linalg.qr(sec)
        sector_bases.append(sec)
        if j > 0:
            sv = np.linalg.svd(sector_bases[j - 1].conj().T @ sec, compute_uv=False)
            min_sector_ov = min(min_sector_ov, float(np.min(sv)))
    return ky_grid, centers, sector_bases, float(min_sector_ov)


def wilson_y_data(v: float, t: float, w: float, lm: float, n_occ: int, nk: int):
    kx_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    centers = np.zeros((nk, n_occ), dtype=float)
    sector_bases = []
    min_sector_ov = 1.0
    for j, kx in enumerate(kx_grid):
        wmat = np.eye(n_occ, dtype=complex)
        occ0 = occ_subspace(float(kx), float(ky_grid[0]), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
        for i, ky in enumerate(ky_grid):
            ky_next = float(ky_grid[(i + 1) % nk])
            oa = occ_subspace(float(kx), float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            ob = occ_subspace(float(kx), ky_next, v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            q = unitary_part(oa.conj().T @ ob)
            wmat = q @ wmat
        eigvals, eigvecs = np.linalg.eig(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        ev = eigvecs[:, order]
        sec = occ0 @ ev[:, :2]
        sec, _ = np.linalg.qr(sec)
        sector_bases.append(sec)
        if j > 0:
            sv = np.linalg.svd(sector_bases[j - 1].conj().T @ sec, compute_uv=False)
            min_sector_ov = min(min_sector_ov, float(np.min(sv)))
    return kx_grid, centers, sector_bases, float(min_sector_ov)


def nested_phase_from_sector(sector_bases: list[np.ndarray]) -> float:
    phase = 1.0 + 0.0j
    n = len(sector_bases)
    for i in range(n):
        a = sector_bases[i]
        b = sector_bases[(i + 1) % n]
        q = unitary_part(a.conj().T @ b)
        detq = np.linalg.det(q)
        if abs(detq) > 1e-14:
            phase *= detq / abs(detq)
    return float(np.mod(np.angle(phase) / (2.0 * np.pi), 1.0))


def wannier_split_gap(centers: np.ndarray) -> float:
    # Sector gap between 2nd and 3rd sorted wannier bands (n_occ=4).
    gap = centers[:, 2] - centers[:, 1]
    return float(np.min(gap))


def plot_wannier(kvals: np.ndarray, centers: np.ndarray, out_png: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    for i in range(centers.shape[1]):
        ax.plot(kvals / np.pi, centers[:, i], linewidth=0.9)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Wannier center")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    t = float(args.t)
    w = float(args.w)
    lm = float(args.lm)
    v_list = parse_float_list(args.v_list)
    n_occ = int(args.n_occ)
    bulk_nk = 41 if args.quick else int(args.bulk_nk)
    bulk_path_nseg = 80 if args.quick else int(args.bulk_path_nseg)
    ribbon_width = 24 if args.quick else int(args.ribbon_width)
    ribbon_nk = 81 if args.quick else int(args.ribbon_nk)
    edge_cells = min(int(args.edge_cells), max(2, ribbon_width // 3))
    l_list = [20] if args.quick else parse_int_list(args.L_list)
    near_k = 80 if args.quick else int(args.near_k)

    # -------- Task 1: bulk gap + band plots --------
    bulk_rows: list[dict] = []
    for v in v_list:
        brow = compute_bulk_summary(v=v, t=t, w=w, lm=lm, nk=bulk_nk, n_occ=n_occ)
        bulk_rows.append(brow)
        plot_bulk_band(
            v=v,
            t=t,
            w=w,
            lm=lm,
            nseg=bulk_path_nseg,
            out_png=out_root / f"bulk_band_v{token(v)}.png",
        )
    write_csv(
        out_root / "bulk_gap_summary.csv",
        bulk_rows,
        fieldnames=[
            "v",
            "global_bulk_gap",
            "direct_bulk_gap_min",
            "M_point_gap",
            "band_min_k",
            "band_min_kx",
            "band_min_ky",
            "is_bulk_gapped_insulator",
        ],
    )
    bulk_map = {float(r["v"]): r for r in bulk_rows}

    # -------- Task 2: ribbon gaps --------
    edge_rows: list[dict] = []
    for v in v_list:
        kx, ex, wx = ribbon_spectrum_xopen(v=v, t=t, w=w, lm=lm, nx=ribbon_width, nk=ribbon_nk, edge_cells=edge_cells)
        ky, ey, wy = ribbon_spectrum_yopen(v=v, t=t, w=w, lm=lm, ny=ribbon_width, nk=ribbon_nk, edge_cells=edge_cells)

        bulk_gap = float(bulk_map[v]["global_bulk_gap"])
        edge_gap_x, gapless_x, eemin_x, eemax_x = edge_gap_metrics(ex, wx, bulk_gap=bulk_gap)
        edge_gap_y, gapless_y, eemin_y, eemax_y = edge_gap_metrics(ey, wy, bulk_gap=bulk_gap)
        has_gapless = int(gapless_x or gapless_y)
        edge_rows.append(
            {
                "v": v,
                "edge_gap_x": edge_gap_x,
                "edge_gap_y": edge_gap_y,
                "has_gapless_edge_states": has_gapless,
                "edge_localized_energy_min_x": eemin_x,
                "edge_localized_energy_max_x": eemax_x,
                "edge_localized_energy_min_y": eemin_y,
                "edge_localized_energy_max_y": eemax_y,
            }
        )

        plot_ribbon(
            kvals=kx,
            evals=ex,
            edge_w=wx,
            out_png=out_root / f"ribbon_xopen_v{token(v)}.png",
            title=f"x-open ribbon (v={v:.1f}, lm={lm:.1f})",
            k_label=r"$k_y/\pi$",
        )
        plot_ribbon(
            kvals=ky,
            evals=ey,
            edge_w=wy,
            out_png=out_root / f"ribbon_yopen_v{token(v)}.png",
            title=f"y-open ribbon (v={v:.1f}, lm={lm:.1f})",
            k_label=r"$k_x/\pi$",
        )

    write_csv(
        out_root / "edge_gap_summary.csv",
        edge_rows,
        fieldnames=[
            "v",
            "edge_gap_x",
            "edge_gap_y",
            "has_gapless_edge_states",
            "edge_localized_energy_min_x",
            "edge_localized_energy_max_x",
            "edge_localized_energy_min_y",
            "edge_localized_energy_max_y",
        ],
    )
    edge_map = {float(r["v"]): r for r in edge_rows}

    # -------- Task 3: OBC states --------
    obc_rows: list[dict] = []
    finite_rows: list[dict] = []
    candidate_plots: list[tuple[Path, np.ndarray, str]] = []

    for v in v_list:
        for l in l_list:
            corner_size = max(3, int(args.corner_size_ratio * l))
            edge_width = max(3, int(args.edge_width_ratio * l))
            corner_mask, edge_mask, bulk_mask = corner_edge_bulk_masks(l=l, corner_size=corner_size, edge_width=edge_width)

            h = build_obc_sparse(l=l, v=v, t=t, w=w, lm=lm)
            vals, vecs = eigs_near_zero(h, k=near_k)

            bulk_gap = float(bulk_map[v]["global_bulk_gap"])
            edge_gap_x = float(edge_map[v]["edge_gap_x"])
            edge_gap_y = float(edge_map[v]["edge_gap_y"])
            positives = [g for g in [bulk_gap, edge_gap_x, edge_gap_y] if g > 0]
            gap_ref = min(positives) if positives else 0.1
            in_gap_thr = max(1e-4, 0.5 * gap_ref)

            state_rows_case = []
            for i in range(vals.size):
                e = float(vals[i])
                vec = vecs[:, i]
                rho = cell_density(vec, l=l)
                wc, we, wb, ipr = state_weights(rho, corner_mask, edge_mask, bulk_mask)
                classification = classify_state(wc, we)
                is_in_gap = int(abs(e) <= in_gap_thr)
                row = {
                    "v": v,
                    "L": l,
                    "state_index": int(i),
                    "energy": e,
                    "abs_energy": abs(e),
                    "W_corner": wc,
                    "W_edge": we,
                    "W_bulk": wb,
                    "IPR": ipr,
                    "classification": classification,
                    "is_in_gap": is_in_gap,
                }
                state_rows_case.append(row)

            in_gap_rows = [r for r in state_rows_case if int(r["is_in_gap"]) == 1]
            if not in_gap_rows:
                in_gap_rows = sorted(state_rows_case, key=lambda r: float(r["abs_energy"]))[:8]
            in_gap_rows = sorted(in_gap_rows, key=lambda r: float(r["abs_energy"]))

            for row in in_gap_rows:
                obc_rows.append(
                    {
                        "v": row["v"],
                        "L": row["L"],
                        "state_index": row["state_index"],
                        "energy": row["energy"],
                        "abs_energy": row["abs_energy"],
                        "W_corner": row["W_corner"],
                        "W_edge": row["W_edge"],
                        "W_bulk": row["W_bulk"],
                        "IPR": row["IPR"],
                        "classification": row["classification"],
                    }
                )

            corner_like_rows = [
                r for r in in_gap_rows if str(r["classification"]) == "corner-like in-gap state candidate"
            ]
            best = max(in_gap_rows, key=lambda r: float(r["W_corner"]))
            finite_rows.append(
                {
                    "v": v,
                    "L": l,
                    "corner_like_count": len(corner_like_rows),
                    "max_W_corner": float(best["W_corner"]),
                    "max_W_edge": float(best["W_edge"]),
                    "energy_at_max_corner": float(best["energy"]),
                }
            )

            # candidate wavefunction plot
            best_idx = int(best["state_index"])
            rho_best = cell_density(vecs[:, best_idx], l=l)
            candidate_path = out_root / f"wavefunction_v{token(v)}_L{l}_candidate.png"
            title = (
                f"candidate v={v:.1f}, L={l}, E={float(best['energy']):.3e}\n"
                f"Wc={float(best['W_corner']):.3f}, We={float(best['W_edge']):.3f}, Wb={float(best['W_bulk']):.3f}"
            )
            candidate_plots.append((candidate_path, rho_best, title))

    write_csv(
        out_root / "obc_state_metrics.csv",
        obc_rows,
        fieldnames=["v", "L", "state_index", "energy", "abs_energy", "W_corner", "W_edge", "W_bulk", "IPR", "classification"],
    )
    for path, rho, title in candidate_plots:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(rho, origin="lower", cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="rho(x,y)")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    # -------- Task 4: finite-size scaling --------
    write_csv(
        out_root / "finite_size_corner_summary.csv",
        finite_rows,
        fieldnames=["v", "L", "corner_like_count", "max_W_corner", "max_W_edge", "energy_at_max_corner"],
    )

    # plots vs L
    unique_v = sorted({float(r["v"]) for r in finite_rows})
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for v in unique_v:
        sub = sorted([r for r in finite_rows if abs(float(r["v"]) - v) < 1e-12], key=lambda r: int(r["L"]))
        lvals = np.array([int(r["L"]) for r in sub], dtype=int)
        wc = np.array([float(r["max_W_corner"]) for r in sub], dtype=float)
        we = np.array([float(r["max_W_edge"]) for r in sub], dtype=float)
        ax.plot(lvals, wc, marker="o", label=f"max W_corner v={v:.1f}")
        ax.plot(lvals, we, marker="s", linestyle="--", label=f"W_edge@max_corner v={v:.1f}")
    ax.set_xlabel("L")
    ax.set_ylabel("weight")
    ax.set_title("Corner / edge weight vs L")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_root / "corner_weight_vs_L.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for v in unique_v:
        sub = sorted([r for r in finite_rows if abs(float(r["v"]) - v) < 1e-12], key=lambda r: int(r["L"]))
        lvals = np.array([int(r["L"]) for r in sub], dtype=int)
        e = np.array([float(r["energy_at_max_corner"]) for r in sub], dtype=float)
        ax.plot(lvals, e, marker="o", label=f"v={v:.1f}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("L")
    ax.set_ylabel("energy of max-corner state")
    ax.set_title("Corner-state energy vs L")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "corner_state_energy_vs_L.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for v in unique_v:
        sub = sorted([r for r in finite_rows if abs(float(r["v"]) - v) < 1e-12], key=lambda r: int(r["L"]))
        lvals = np.array([int(r["L"]) for r in sub], dtype=int)
        c = np.array([int(r["corner_like_count"]) for r in sub], dtype=int)
        ax.plot(lvals, c, marker="o", label=f"v={v:.1f}")
    ax.set_xlabel("L")
    ax.set_ylabel("corner-like in-gap state count")
    ax.set_title("Corner-like state count vs L")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "corner_state_count_vs_L.png", dpi=180)
    plt.close(fig)

    # -------- Task 5: Wilson + nested --------
    nested_rows = []
    wilson_nk = 41 if args.quick else 81
    for v in v_list:
        ky_x, centers_x, sec_x, ov_x = wilson_x_data(v=v, t=t, w=w, lm=lm, n_occ=n_occ, nk=wilson_nk)
        kx_y, centers_y, sec_y, ov_y = wilson_y_data(v=v, t=t, w=w, lm=lm, n_occ=n_occ, nk=wilson_nk)
        gap_x = wannier_split_gap(centers_x)
        gap_y = wannier_split_gap(centers_y)

        p_y_nux = nested_phase_from_sector(sec_x)
        p_x_nuy = nested_phase_from_sector(sec_y)
        q_xy = float(np.mod(0.5 * (p_y_nux + p_x_nuy), 1.0))

        if gap_x < 0.03 or gap_y < 0.03 or min(ov_x, ov_y) < 0.20:
            reliability = "unreliable"
        else:
            reliability = "reliable"
        nested_rows.append(
            {
                "v": v,
                "wannier_gap_x": gap_x,
                "wannier_gap_y": gap_y,
                "q_xy": q_xy,
                "reliability": reliability,
            }
        )

        plot_wannier(
            ky_x,
            centers_x,
            out_root / f"wannier_spectrum_x_v{token(v)}.png",
            title=f"nu_x(k_y), v={v:.1f}",
            xlabel=r"$k_y/\pi$",
        )
        plot_wannier(
            kx_y,
            centers_y,
            out_root / f"wannier_spectrum_y_v{token(v)}.png",
            title=f"nu_y(k_x), v={v:.1f}",
            xlabel=r"$k_x/\pi$",
        )

    write_csv(
        out_root / "nested_wilson_summary.csv",
        nested_rows,
        fieldnames=["v", "wannier_gap_x", "wannier_gap_y", "q_xy", "reliability"],
    )
    nested_map = {float(r["v"]): r for r in nested_rows}

    # -------- Task 6: corner charge (conditional) --------
    corner_charge_rows: list[dict] = []
    for v in v_list:
        edge_row = edge_map[v]
        edge_gapped = (
            float(edge_row["edge_gap_x"]) > 1e-4
            and float(edge_row["edge_gap_y"]) > 1e-4
            and int(edge_row["has_gapless_edge_states"]) == 0
        )
        has_corner_like = any(
            int(r["corner_like_count"]) > 0 for r in finite_rows if abs(float(r["v"]) - v) < 1e-12
        )

        for l in l_list:
            corner_size = max(3, int(args.corner_size_ratio * l))
            edge_width = max(3, int(args.edge_width_ratio * l))
            corner_mask, edge_mask, bulk_mask = corner_edge_bulk_masks(l=l, corner_size=corner_size, edge_width=edge_width)

            if not (edge_gapped and has_corner_like):
                corner_charge_rows.append(
                    {
                        "v": v,
                        "L": l,
                        "Q_corner_mean": np.nan,
                        "Q_corner_abs_mean": np.nan,
                        "Q_corner_1": np.nan,
                        "Q_corner_2": np.nan,
                        "Q_corner_3": np.nan,
                        "Q_corner_4": np.nan,
                        "status": "skipped_no_edge_gapped_cornerlike_condition",
                    }
                )
                continue

            n = 8 * l * l
            # Exact occupied density only for moderate sizes.
            if n > 8000:
                corner_charge_rows.append(
                    {
                        "v": v,
                        "L": l,
                        "Q_corner_mean": np.nan,
                        "Q_corner_abs_mean": np.nan,
                        "Q_corner_1": np.nan,
                        "Q_corner_2": np.nan,
                        "Q_corner_3": np.nan,
                        "Q_corner_4": np.nan,
                        "status": "skipped_exact_occ_too_large",
                    }
                )
                continue

            h_dense = build_obc_sparse(l=l, v=v, t=t, w=w, lm=lm).toarray()
            evals, evecs = np.linalg.eigh(h_dense)
            n_occ_total = n // 2
            occ = evecs[:, :n_occ_total]
            rho = np.zeros((l, l), dtype=float)
            for iy in range(l):
                for ix in range(l):
                    s = ((iy * l + ix) * 8)
                    rho[iy, ix] = float(np.sum(np.abs(occ[s : s + 8, :]) ** 2))
            rho_bg = float(np.mean(rho[bulk_mask]))
            drho = rho - rho_bg
            c1 = float(np.sum(drho[:corner_size, :corner_size]))
            c2 = float(np.sum(drho[:corner_size, l - corner_size :]))
            c3 = float(np.sum(drho[l - corner_size :, :corner_size]))
            c4 = float(np.sum(drho[l - corner_size :, l - corner_size :]))
            corner_charge_rows.append(
                {
                    "v": v,
                    "L": l,
                    "Q_corner_mean": float(np.mean([c1, c2, c3, c4])),
                    "Q_corner_abs_mean": float(np.mean(np.abs([c1, c2, c3, c4]))),
                    "Q_corner_1": c1,
                    "Q_corner_2": c2,
                    "Q_corner_3": c3,
                    "Q_corner_4": c4,
                    "status": "ok_exact",
                }
            )

    write_csv(
        out_root / "corner_charge_summary.csv",
        corner_charge_rows,
        fieldnames=[
            "v",
            "L",
            "Q_corner_mean",
            "Q_corner_abs_mean",
            "Q_corner_1",
            "Q_corner_2",
            "Q_corner_3",
            "Q_corner_4",
            "status",
        ],
    )

    # -------- Task 7: report --------
    report_lines = [
        "# HOTI_V_LT_0P6_REPORT",
        "",
        "## Bulk gap summary",
    ]
    for r in bulk_rows:
        report_lines.append(
            f"- v={float(r['v']):.1f}: global_bulk_gap={float(r['global_bulk_gap']):.6e}, "
            f"M_point_gap={float(r['M_point_gap']):.6e}, band_min_k={r['band_min_k']}, "
            f"bulk_gapped={bool(int(r['is_bulk_gapped_insulator']))}"
        )

    report_lines += ["", "## Edge/ribbon summary"]
    for r in edge_rows:
        report_lines.append(
            f"- v={float(r['v']):.1f}: edge_gap_x={float(r['edge_gap_x']):.6e}, "
            f"edge_gap_y={float(r['edge_gap_y']):.6e}, gapless_edges={int(r['has_gapless_edge_states'])}"
        )

    report_lines += ["", "## OBC corner-like summary (from in-gap candidate states)"]
    for r in finite_rows:
        report_lines.append(
            f"- v={float(r['v']):.1f}, L={int(r['L'])}: corner_like_count={int(r['corner_like_count'])}, "
            f"max_W_corner={float(r['max_W_corner']):.3f}, max_W_edge={float(r['max_W_edge']):.3f}, "
            f"E@max_corner={float(r['energy_at_max_corner']):.3e}"
        )

    report_lines += ["", "## Nested Wilson summary"]
    for r in nested_rows:
        report_lines.append(
            f"- v={float(r['v']):.1f}: wannier_gap_x={float(r['wannier_gap_x']):.4f}, "
            f"wannier_gap_y={float(r['wannier_gap_y']):.4f}, q_xy={float(r['q_xy']):.6f}, "
            f"reliability={r['reliability']}"
        )

    report_lines += ["", "## Corner charge summary"]
    for r in corner_charge_rows:
        report_lines.append(
            f"- v={float(r['v']):.1f}, L={int(r['L'])}: Q_corner_mean={r['Q_corner_mean']}, "
            f"Q_corner_abs_mean={r['Q_corner_abs_mean']}, status={r['status']}"
        )

    # Question-based answers
    report_lines += ["", "## Answers to requested questions"]
    all_bulk_gapped = all(int(r["is_bulk_gapped_insulator"]) == 1 for r in bulk_rows)
    report_lines.append(f"1) v<0.6 区域是否 bulk gapped: {'是' if all_bulk_gapped else '否'}.")

    all_edge_gapped = all(float(r["edge_gap_x"]) > 1e-4 and float(r["edge_gap_y"]) > 1e-4 for r in edge_rows)
    report_lines.append(f"2) v<0.6 区域是否 edge gapped: {'是' if all_edge_gapped else '否'}.")

    has_gapless_helical = any(int(r["has_gapless_edge_states"]) == 1 for r in edge_rows)
    report_lines.append(f"3) 是否存在穿过 gap 的一阶 helical edge states: {'是' if has_gapless_helical else '否'} (基于 edge-localized crossing 判据).")

    stable_corner = {}
    for v in v_list:
        sub = sorted([r for r in finite_rows if abs(float(r["v"]) - v) < 1e-12], key=lambda r: int(r["L"]))
        stable_corner[v] = all(int(r["corner_like_count"]) > 0 and float(r["max_W_corner"]) > 0.6 for r in sub)
    report_lines.append(
        "4) OBC 中是否存在稳定 corner-localized in-gap states: "
        + ", ".join([f"v={v:.1f}:{'是' if stable_corner[v] else '否'}" for v in v_list])
        + "."
    )

    report_lines.append(
        "5) corner-like 态是否随 L=20,30,40 稳定: "
        + ", ".join([f"v={v:.1f}:{'稳定' if stable_corner[v] else '不稳定'}" for v in v_list])
        + "."
    )

    qxy_quantized = {}
    for v in v_list:
        q = float(nested_map[v]["q_xy"])
        rel = str(nested_map[v]["reliability"])
        qxy_quantized[v] = rel == "reliable" and abs(q - 0.5) < 0.1
    report_lines.append(
        "6) nested Wilson 的 q_xy 是否量子化: "
        + ", ".join([f"v={v:.1f}:{'~0.5' if qxy_quantized[v] else '非0.5/不可靠'}" for v in v_list])
        + "."
    )

    qcorner_quantized = {}
    for v in v_list:
        sub = [r for r in corner_charge_rows if abs(float(r["v"]) - v) < 1e-12 and str(r["status"]) == "ok_exact"]
        if not sub:
            qcorner_quantized[v] = False
            continue
        qabs = np.array([abs(float(r["Q_corner_mean"])) for r in sub], dtype=float)
        qcorner_quantized[v] = bool(np.any(np.abs(qabs - 0.5) < 0.1))
    report_lines.append(
        "7) corner charge 是否接近量子化值: "
        + ", ".join([f"v={v:.1f}:{'是' if qcorner_quantized[v] else '否/未满足条件'}" for v in v_list])
        + "."
    )

    # final class
    class_rows = []
    for v in v_list:
        bulk_gapped = bool(int(bulk_map[v]["is_bulk_gapped_insulator"]))
        edge_gapped = bool(float(edge_map[v]["edge_gap_x"]) > 1e-4 and float(edge_map[v]["edge_gap_y"]) > 1e-4)
        gapless = bool(int(edge_map[v]["has_gapless_edge_states"]) == 1)
        corner_ok = stable_corner[v]
        qhalf = qxy_quantized[v]
        q0_like = abs(float(nested_map[v]["q_xy"])) < 0.15 or abs(float(nested_map[v]["q_xy"]) - 1.0) < 0.15
        if bulk_gapped and edge_gapped and (not gapless) and (not corner_ok) and q0_like:
            cls = "A. 普通平庸绝缘体"
            why = "bulk/edge gapped, no stable corner in-gap state, q_xy~0"
        elif bulk_gapped and edge_gapped and (not gapless) and corner_ok and (qhalf or qcorner_quantized[v]):
            cls = "B. 二阶拓扑候选"
            why = "bulk/edge gapped, stable corner state, q_xy~0.5 or quantized corner charge"
        else:
            cls = "C. 不确定"
            why = "gap or Wannier/corner-size-scaling evidence not jointly conclusive"
        class_rows.append((v, cls, why))

    report_lines.append("8) 是否可认为 v<0.6 存在二阶拓扑角态: 见下方逐 v 分类。")
    report_lines.append("9) 若不能，类型更可能为何: 见下方理由。")
    report_lines += ["", "## Final classification (A/B/C)"]
    for v, cls, why in class_rows:
        report_lines.append(f"- v={v:.1f}: {cls}. reason: {why}.")

    (out_root / "HOTI_V_LT_0P6_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[ok] output_root={out_root}")
    for v, cls, _why in class_rows:
        print(f"[ok] v={v:.1f} -> {cls}")


if __name__ == "__main__":
    main()
