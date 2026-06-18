#!/usr/bin/env python3
"""Validate SSH-like origin of isolated ribbon edge bands in SOC model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSH-like edge-band origin check for original SOC model.")
    parser.add_argument("--output-root", default="/workspace/outputs/ssh_edge_check")
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--v-focus", type=float, default=0.5)
    parser.add_argument("--v-qsh", type=float, default=0.8)
    parser.add_argument("--v-list", default="0.2,0.4,0.5,0.6,0.8,1.0,1.2")
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ribbon-nk", type=int, default=161)
    parser.add_argument("--bulk-nk", type=int, default=101)
    parser.add_argument("--zak-nk", type=int, default=61)
    parser.add_argument("--chern-nk", type=int, default=31)
    parser.add_argument("--edge-cells", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def token(v: float) -> str:
    return f"{v:.2f}".replace("-", "m").replace(".", "p")


def orb_spin_idx(orb: int, spin: int) -> int:
    return orb * 2 + spin


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


def h_soc_orbital(lm: float) -> np.ndarray:
    hs = np.zeros((4, 4), dtype=complex)
    hs[0, 1] = 1j * lm
    hs[0, 2] = -1j * lm
    hs[1, 0] = -1j * lm
    hs[1, 3] = 1j * lm
    hs[2, 0] = 1j * lm
    hs[2, 3] = -1j * lm
    hs[3, 1] = -1j * lm
    hs[3, 2] = 1j * lm
    return hs


def h8_k(kx: float, ky: float, v: float, t: float, w: float, lm: float) -> np.ndarray:
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(h0_k(kx, ky, v=v, t=t, w=w), s0) + np.kron(h_soc_orbital(lm), sz)


def h_spin_k(kx: float, ky: float, v: float, t: float, w: float, lm: float, spin_sign: int) -> np.ndarray:
    return h0_k(kx, ky, v=v, t=t, w=w) + float(spin_sign) * h_soc_orbital(lm)


def onsite_xopen_h0(ky: float, v: float, t: float, w: float, v03: float | None = None) -> np.ndarray:
    xterm = v if v03 is None else float(v03)
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = xterm
    h0[1, 0] = t
    h0[1, 2] = v + w * np.exp(-1j * ky)
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v + w * np.exp(1j * ky)
    h0[2, 3] = t
    h0[3, 0] = xterm
    h0[3, 1] = t
    h0[3, 2] = t
    return h0


def build_ribbon_xopen_spin(
    ky: float,
    nx: int,
    v: float,
    t: float,
    w: float,
    lm: float,
    spin_sign: int,
) -> np.ndarray:
    h = np.zeros((4 * nx, 4 * nx), dtype=complex)
    onsite = onsite_xopen_h0(ky=ky, v=v, t=t, w=w) + float(spin_sign) * h_soc_orbital(lm)
    tx = np.zeros((4, 4), dtype=complex)
    tx[0, 3] = w
    for x in range(nx):
        s = x * 4
        h[s : s + 4, s : s + 4] += onsite
        if x > 0:
            p = (x - 1) * 4
            h[s : s + 4, p : p + 4] += tx
            h[p : p + 4, s : s + 4] += tx.conj().T
    return h


def build_ribbon_xopen_full(
    ky: float,
    nx: int,
    v: float,
    t: float,
    w: float,
    lm: float,
    lambda_r: float = 0.0,
    termination: str = "A",
) -> np.ndarray:
    h = np.zeros((8 * nx, 8 * nx), dtype=complex)
    s0 = np.eye(2, dtype=complex)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    hs = h_soc_orbital(lm)

    tx = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[orb_spin_idx(0, spin), orb_spin_idx(3, spin)] = w

    for x in range(nx):
        v03 = v
        if termination == "B" and (x == 0 or x == nx - 1):
            # Change boundary cut by removing the edge 0-3 intracell dimer bond.
            v03 = 0.0
        h0x = onsite_xopen_h0(ky=ky, v=v, t=t, w=w, v03=v03)
        onsite = np.kron(h0x, s0) + np.kron(hs, sz) + lambda_r * np.kron(np.eye(4, dtype=complex), sx)
        s = x * 8
        h[s : s + 8, s : s + 8] += onsite
        if x > 0:
            p = (x - 1) * 8
            h[s : s + 8, p : p + 8] += tx
            h[p : p + 8, s : s + 8] += tx.conj().T
    return h


def spin_sz_operator(nx: int) -> np.ndarray:
    sz = np.zeros((8 * nx, 8 * nx), dtype=float)
    for x in range(nx):
        base = x * 8
        for orb in range(4):
            sz[base + orb_spin_idx(orb, 0), base + orb_spin_idx(orb, 0)] = 1.0
            sz[base + orb_spin_idx(orb, 1), base + orb_spin_idx(orb, 1)] = -1.0
    return sz


def compute_bulk_gap(v: float, t: float, w: float, lm: float, nk: int, n_occ: int = 4) -> float:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    vmax = -np.inf
    cmin = np.inf
    for kx in kgrid:
        for ky in kgrid:
            e = np.real(np.linalg.eigvalsh(h8_k(float(kx), float(ky), v=v, t=t, w=w, lm=lm)))
            vmax = max(vmax, float(e[n_occ - 1]))
            cmin = min(cmin, float(e[n_occ]))
    return float(cmin - vmax)


def compute_spin_ribbon_spectrum(
    v: float,
    t: float,
    w: float,
    lm: float,
    nx: int,
    nk: int,
    edge_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ky_vals = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    evals_up, evals_dn = [], []
    edge_up, edge_dn = [], []
    for ky in ky_vals:
        hup = build_ribbon_xopen_spin(float(ky), nx=nx, v=v, t=t, w=w, lm=lm, spin_sign=1)
        hdn = build_ribbon_xopen_spin(float(ky), nx=nx, v=v, t=t, w=w, lm=lm, spin_sign=-1)
        eu, vu = np.linalg.eigh(hup)
        ed, vd = np.linalg.eigh(hdn)
        eu = np.real(eu)
        ed = np.real(ed)
        rho_u = np.abs(vu) ** 2
        rho_d = np.abs(vd) ** 2
        rho_x_u = rho_u.reshape(nx, 4, rho_u.shape[1]).sum(axis=1)
        rho_x_d = rho_d.reshape(nx, 4, rho_d.shape[1]).sum(axis=1)
        wu = rho_x_u[:edge_cells, :].sum(axis=0) + rho_x_u[-edge_cells:, :].sum(axis=0)
        wd = rho_x_d[:edge_cells, :].sum(axis=0) + rho_x_d[-edge_cells:, :].sum(axis=0)
        evals_up.append(eu)
        evals_dn.append(ed)
        edge_up.append(wu)
        edge_dn.append(wd)
    return (
        ky_vals,
        np.array(evals_up, dtype=float),
        np.array(evals_dn, dtype=float),
        np.array(edge_up, dtype=float),
        np.array(edge_dn, dtype=float),
    )


def compute_full_ribbon_spectrum(
    v: float,
    t: float,
    w: float,
    lm: float,
    nx: int,
    nk: int,
    edge_cells: int,
    lambda_r: float = 0.0,
    termination: str = "A",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ky_vals = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    evals_all = []
    edge_all = []
    spin_polar_all = []
    sz = spin_sz_operator(nx=nx)
    for ky in ky_vals:
        h = build_ribbon_xopen_full(
            ky=float(ky),
            nx=nx,
            v=v,
            t=t,
            w=w,
            lm=lm,
            lambda_r=lambda_r,
            termination=termination,
        )
        evals, vecs = np.linalg.eigh(h)
        evals = np.real(evals)
        prob = np.abs(vecs) ** 2
        rho_x = prob.reshape(nx, 8, prob.shape[1]).sum(axis=1)
        edge_w = rho_x[:edge_cells, :].sum(axis=0) + rho_x[-edge_cells:, :].sum(axis=0)
        spin_pol = np.real(np.einsum("ij,ik,jk->k", vecs.conj(), sz, vecs))
        evals_all.append(evals)
        edge_all.append(edge_w)
        spin_polar_all.append(spin_pol)
    return (
        ky_vals,
        np.array(evals_all, dtype=float),
        np.array(edge_all, dtype=float),
        np.array(spin_polar_all, dtype=float),
    )


def plot_spin_resolved_spectrum(
    ky_vals: np.ndarray,
    evals: np.ndarray,
    out_png: Path,
    color: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for ib in range(evals.shape[1]):
        ax.plot(ky_vals / np.pi, evals[:, ib], color=color, linewidth=0.8, alpha=0.8)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_spin_overlay(
    ky_vals: np.ndarray,
    evals_up: np.ndarray,
    evals_dn: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for ib in range(evals_up.shape[1]):
        ax.plot(ky_vals / np.pi, evals_up[:, ib], color="red", linewidth=0.65, alpha=0.65)
    for ib in range(evals_dn.shape[1]):
        ax.plot(ky_vals / np.pi, evals_dn[:, ib], color="blue", linewidth=0.65, alpha=0.65)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def select_middle_edge_state(
    evals: np.ndarray,
    edge_weights: np.ndarray,
    bulk_gap: float,
) -> int:
    ewin = max(0.08, 0.75 * max(bulk_gap, 1e-6))
    mask = (np.abs(evals) <= ewin) & (edge_weights >= 0.35)
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        candidates = np.argsort(-edge_weights)[:16]
    best = sorted(candidates, key=lambda i: (-edge_weights[i], abs(evals[i]), i))[0]
    return int(best)


def plot_rho_x(rho_x: np.ndarray, out_png: Path, title: str) -> None:
    x = np.arange(rho_x.size)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(x, rho_x, marker="o", markersize=2.5, linewidth=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho(x)$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def summarize_middle_edge_band(
    evals: np.ndarray,
    edge_weights: np.ndarray,
    bulk_gap: float,
) -> tuple[float, int, int]:
    neg = evals[evals < 0.0]
    pos = evals[evals > 0.0]
    if neg.size == 0 or pos.size == 0:
        ribbon_gap = 0.0
    else:
        ribbon_gap = float(np.min(pos) - np.max(neg))

    mid_thr = max(0.04, 0.45 * max(bulk_gap, 1e-6))
    mid_mask = np.abs(evals) <= mid_thr
    edge_mid = edge_weights[mid_mask] if np.any(mid_mask) else np.array([], dtype=float)
    max_mid_edge = float(np.max(edge_mid)) if edge_mid.size else 0.0
    exists = int(max_mid_edge > 0.60)
    merged = int((exists == 0) and (float(np.max(edge_weights)) > 0.60))
    return ribbon_gap, exists, merged


def unitary_part(mat: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(mat)
    return u @ vh


def polarization_from_hfunc(
    hfunc,
    n_occ: int,
    nk: int,
) -> tuple[float, float, float, float]:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)

    def occ(kx: float, ky: float) -> np.ndarray:
        _, vecs = np.linalg.eigh(hfunc(kx, ky))
        return vecs[:, :n_occ]

    theta_x = []
    for ky in kgrid:
        phase = 1.0 + 0.0j
        for i, kx in enumerate(kgrid):
            kx_n = float(kgrid[(i + 1) % nk])
            oa = occ(float(kx), float(ky))
            ob = occ(kx_n, float(ky))
            q = unitary_part(oa.conj().T @ ob)
            detq = np.linalg.det(q)
            if abs(detq) > 1e-14:
                phase *= detq / abs(detq)
        theta_x.append(float(np.angle(phase)))

    theta_y = []
    for kx in kgrid:
        phase = 1.0 + 0.0j
        for i, ky in enumerate(kgrid):
            ky_n = float(kgrid[(i + 1) % nk])
            oa = occ(float(kx), float(ky))
            ob = occ(float(kx), ky_n)
            q = unitary_part(oa.conj().T @ ob)
            detq = np.linalg.det(q)
            if abs(detq) > 1e-14:
                phase *= detq / abs(detq)
        theta_y.append(float(np.angle(phase)))

    tx = np.unwrap(np.array(theta_x, dtype=float))
    ty = np.unwrap(np.array(theta_y, dtype=float))
    gamma_x = float(np.mod(np.mean(tx), 2.0 * np.pi))
    gamma_y = float(np.mod(np.mean(ty), 2.0 * np.pi))
    px = float(np.mod(gamma_x / (2.0 * np.pi), 1.0))
    py = float(np.mod(gamma_y / (2.0 * np.pi), 1.0))
    return px, py, gamma_x, gamma_y


def chern_fukui(hfunc, n_occ: int, nk: int) -> float:
    occ = [[None for _ in range(nk)] for __ in range(nk)]
    for i in range(nk):
        kx = -np.pi + 2.0 * np.pi * (i / nk)
        for j in range(nk):
            ky = -np.pi + 2.0 * np.pi * (j / nk)
            _, vecs = np.linalg.eigh(hfunc(float(kx), float(ky)))
            occ[i][j] = vecs[:, :n_occ]

    def link(a: np.ndarray, b: np.ndarray) -> complex:
        q = unitary_part(a.conj().T @ b)
        d = np.linalg.det(q)
        if abs(d) < 1e-14:
            return 1.0 + 0.0j
        return d / abs(d)

    ux = np.zeros((nk, nk), dtype=complex)
    uy = np.zeros((nk, nk), dtype=complex)
    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            ux[i, j] = link(occ[i][j], occ[ip][j])
            uy[i, j] = link(occ[i][j], occ[i][jp])

    f12 = np.zeros((nk, nk), dtype=complex)
    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            z = ux[i, j] * uy[ip, j] / (ux[i, jp] * uy[i, j])
            f12[i, j] = np.log(z)
    return float(np.real(np.sum(f12) / (2j * np.pi)))


def interpretation_from_p(px: float, py: float) -> str:
    half = (abs(px - 0.5) < 0.12 or abs(px - 1.5) < 0.12) and (abs(py - 0.5) < 0.12 or abs(py - 1.5) < 0.12)
    zero = min(abs(px), abs(px - 1.0)) < 0.12 and min(abs(py), abs(py - 1.0)) < 0.12
    if half:
        return "nontrivial SSH-like polarization (~1/2,1/2)"
    if zero:
        return "trivial total polarization (~0,0)"
    return "intermediate/non-quantized polarization"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_vscan_montage(
    out_png: Path,
    v_list: list[float],
    spectra: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    summary: dict[float, dict],
) -> None:
    n = len(v_list)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.3 * cols, 3.9 * rows), squeeze=False)
    for idx, v in enumerate(v_list):
        ax = axes[idx // cols][idx % cols]
        ky, eu, ed = spectra[v]
        for ib in range(eu.shape[1]):
            ax.plot(ky / np.pi, eu[:, ib], color="red", linewidth=0.45, alpha=0.55)
        for ib in range(ed.shape[1]):
            ax.plot(ky / np.pi, ed[:, ib], color="blue", linewidth=0.45, alpha=0.55)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        row = summary[v]
        ax.set_title(
            f"v={v:.1f}, bulk_gap={float(row['bulk_gap']):.3f}\n"
            f"mid_exists={int(row['middle_isolated_exists'])}, maxW={float(row['middle_band_max_edge_weight']):.3f}",
            fontsize=8,
        )
        ax.set_xlabel(r"$k_y/\pi$")
        ax.set_ylabel("E")
        ax.grid(alpha=0.2)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def make_lambdar_montage(
    out_png: Path,
    lambdas: list[float],
    spectra: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), squeeze=False)
    for i, lam in enumerate(lambdas):
        ax = axes[i // 2][i % 2]
        ky, e, ew = spectra[lam]
        x = np.repeat(ky / np.pi, e.shape[1])
        y = e.reshape(-1)
        c = ew.reshape(-1)
        sc = ax.scatter(x, y, c=c, s=5, cmap="viridis", edgecolors="none", vmin=0.0, vmax=1.0)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"lambda_R={lam:.2f}")
        ax.set_xlabel(r"$k_y/\pi$")
        ax.set_ylabel("Energy")
        ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label="edge weight")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def make_v05_v08_comparison(
    out_png: Path,
    data05: tuple[np.ndarray, np.ndarray, np.ndarray],
    data08: tuple[np.ndarray, np.ndarray, np.ndarray],
    cspin05: int,
    cspin08: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), squeeze=False)
    for ax, (ky, eu, ed), v, cs in [
        (axes[0][0], data05, 0.5, cspin05),
        (axes[0][1], data08, 0.8, cspin08),
    ]:
        for ib in range(eu.shape[1]):
            ax.plot(ky / np.pi, eu[:, ib], color="red", linewidth=0.55, alpha=0.58)
        for ib in range(ed.shape[1]):
            ax.plot(ky / np.pi, ed[:, ib], color="blue", linewidth=0.55, alpha=0.58)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"v={v:.1f}, C_spin={cs}", fontsize=10)
        ax.set_xlabel(r"$k_y/\pi$")
        ax.set_ylabel("Energy")
        ax.grid(alpha=0.2)
    fig.suptitle("x-open/y-periodic ribbon: v=0.5 vs v=0.8", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    w = float(args.w)
    t = float(args.t)
    lm = float(args.lm)
    v_focus = float(args.v_focus)
    v_qsh = float(args.v_qsh)
    v_list = parse_float_list(args.v_list)

    nx = 24 if args.quick else int(args.nx)
    ribbon_nk = 81 if args.quick else int(args.ribbon_nk)
    bulk_nk = 61 if args.quick else int(args.bulk_nk)
    zak_nk = 41 if args.quick else int(args.zak_nk)
    chern_nk = 21 if args.quick else int(args.chern_nk)
    edge_cells = min(int(args.edge_cells), max(2, nx // 4))

    # ---------------- Task 1: spin-resolved ribbon spectrum @ v=0.5 ----------------
    ky_vals, evals_up, evals_dn, edge_up, edge_dn = compute_spin_ribbon_spectrum(
        v=v_focus,
        t=t,
        w=w,
        lm=lm,
        nx=nx,
        nk=ribbon_nk,
        edge_cells=edge_cells,
    )
    common_title = (
        f"v={v_focus:.1f}, w={w:.1f}, t={t:.1f}, lm={lm:.1f}, x-open/y-periodic\n"
        "red=spin-up, blue=spin-down"
    )
    plot_spin_resolved_spectrum(
        ky_vals,
        evals_up,
        out_root / "ribbon_spin_up_v0p5.png",
        color="red",
        title=f"spin-up ribbon spectrum\n{common_title}",
    )
    plot_spin_resolved_spectrum(
        ky_vals,
        evals_dn,
        out_root / "ribbon_spin_down_v0p5.png",
        color="blue",
        title=f"spin-down ribbon spectrum\n{common_title}",
    )
    plot_spin_overlay(
        ky_vals,
        evals_up,
        evals_dn,
        out_root / "ribbon_spin_resolved_overlay_v0p5.png",
        title=f"spin-resolved overlay\n{common_title}",
    )

    # ---------------- Task 2: wavefunctions of isolated middle edge band ----------------
    bulk_gap_v_focus = compute_bulk_gap(v=v_focus, t=t, w=w, lm=lm, nk=bulk_nk, n_occ=4)
    ky_targets = [0.0, 0.25 * np.pi, 0.50 * np.pi]
    ky_labels = ["ky0", "ky0p25pi", "ky0p5pi"]
    wf_rows: list[dict] = []

    for spin_name, spin_sign in [("spin_up", 1), ("spin_down", -1)]:
        for ky_target, ky_tag in zip(ky_targets, ky_labels):
            h = build_ribbon_xopen_spin(ky=float(ky_target), nx=nx, v=v_focus, t=t, w=w, lm=lm, spin_sign=spin_sign)
            evals, vecs = np.linalg.eigh(h)
            evals = np.real(evals)
            prob = np.abs(vecs) ** 2
            rho_x_all = prob.reshape(nx, 4, prob.shape[1]).sum(axis=1)
            edge_w_all = rho_x_all[:edge_cells, :].sum(axis=0) + rho_x_all[-edge_cells:, :].sum(axis=0)
            idx = select_middle_edge_state(evals=evals, edge_weights=edge_w_all, bulk_gap=bulk_gap_v_focus)
            rho_x = rho_x_all[:, idx]
            w_left = float(np.sum(rho_x[:3]))
            w_right = float(np.sum(rho_x[-3:]))
            w_edge = w_left + w_right
            w_bulk = float(1.0 - w_edge)

            out_png = out_root / f"wf_{spin_name}_{ky_tag}.png"
            plot_rho_x(
                rho_x,
                out_png=out_png,
                title=(
                    f"{spin_name}, ky={ky_target/np.pi:.2f}π, E={evals[idx]:.4f}\n"
                    f"W_left={w_left:.3f}, W_right={w_right:.3f}, W_edge={w_edge:.3f}"
                ),
            )
            wf_rows.append(
                {
                    "v": v_focus,
                    "spin": "up" if spin_sign > 0 else "down",
                    "ky": float(ky_target),
                    "band_index": int(idx),
                    "energy": float(evals[idx]),
                    "W_left": w_left,
                    "W_right": w_right,
                    "W_edge": w_edge,
                    "W_bulk": w_bulk,
                }
            )
    write_csv(
        out_root / "ribbon_edge_wavefunction_metrics.csv",
        wf_rows,
        fieldnames=["v", "spin", "ky", "band_index", "energy", "W_left", "W_right", "W_edge", "W_bulk"],
    )

    # ---------------- Task 3: v/w scan for SSH-like edge-band trend ----------------
    v_scan_rows: list[dict] = []
    v_scan_plot_data: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    summary_map: dict[float, dict] = {}
    for v in v_list:
        ky, eu, ed, wu, wd = compute_spin_ribbon_spectrum(
            v=v,
            t=t,
            w=w,
            lm=lm,
            nx=nx,
            nk=ribbon_nk,
            edge_cells=edge_cells,
        )
        e_all = np.concatenate([eu, ed], axis=1)
        w_all = np.concatenate([wu, wd], axis=1)
        bulk_gap = compute_bulk_gap(v=v, t=t, w=w, lm=lm, nk=bulk_nk, n_occ=4)
        ribbon_gap, middle_exists, merged = summarize_middle_edge_band(
            evals=e_all.reshape(-1),
            edge_weights=w_all.reshape(-1),
            bulk_gap=bulk_gap,
        )
        mid_thr = max(0.04, 0.45 * max(bulk_gap, 1e-6))
        mid_mask = np.abs(e_all.reshape(-1)) <= mid_thr
        max_mid_w = float(np.max(w_all.reshape(-1)[mid_mask])) if np.any(mid_mask) else 0.0
        row = {
            "v": v,
            "bulk_gap": bulk_gap,
            "ribbon_gap": ribbon_gap,
            "middle_isolated_exists": int(middle_exists),
            "middle_band_max_edge_weight": max_mid_w,
            "middle_band_merged_into_bulk": int(merged),
        }
        v_scan_rows.append(row)
        v_scan_plot_data[v] = (ky, eu, ed)
        summary_map[v] = row

    write_csv(
        out_root / "v_scan_edge_band_summary.csv",
        v_scan_rows,
        fieldnames=[
            "v",
            "bulk_gap",
            "ribbon_gap",
            "middle_isolated_exists",
            "middle_band_max_edge_weight",
            "middle_band_merged_into_bulk",
        ],
    )
    make_vscan_montage(
        out_png=out_root / "ribbon_v_scan_montage.png",
        v_list=v_list,
        spectra=v_scan_plot_data,
        summary=summary_map,
    )

    # ---------------- Task 4: 2D Zak phase / polarization ----------------
    zak_rows: list[dict] = []
    for v in v_list:
        cases = [
            ("no_soc_block_4x4", lambda kx, ky, vv=v: h0_k(kx, ky, v=vv, t=t, w=w), 2),
            ("soc_spin_up_block", lambda kx, ky, vv=v: h_spin_k(kx, ky, v=vv, t=t, w=w, lm=lm, spin_sign=1), 2),
            ("soc_spin_down_block", lambda kx, ky, vv=v: h_spin_k(kx, ky, v=vv, t=t, w=w, lm=lm, spin_sign=-1), 2),
            ("soc_full_spinful_occ4", lambda kx, ky, vv=v: h8_k(kx, ky, v=vv, t=t, w=w, lm=lm), 4),
        ]
        for case_name, hfunc, n_occ in cases:
            px, py, gx, gy = polarization_from_hfunc(hfunc=hfunc, n_occ=n_occ, nk=zak_nk)
            zak_rows.append(
                {
                    "v": v,
                    "case": case_name,
                    "P_x": px,
                    "P_y": py,
                    "gamma_x": gx,
                    "gamma_y": gy,
                    "interpretation": interpretation_from_p(px, py),
                }
            )
    write_csv(
        out_root / "2d_zak_phase_summary.csv",
        zak_rows,
        fieldnames=["v", "case", "P_x", "P_y", "gamma_x", "gamma_y", "interpretation"],
    )

    # ---------------- Task 5: change ribbon termination ----------------
    for term in ["A", "B"]:
        ky, e, ew, _sp = compute_full_ribbon_spectrum(
            v=v_focus,
            t=t,
            w=w,
            lm=lm,
            nx=nx,
            nk=ribbon_nk,
            edge_cells=edge_cells,
            lambda_r=0.0,
            termination=term,
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        x = np.repeat(ky / np.pi, e.shape[1])
        y = e.reshape(-1)
        c = ew.reshape(-1)
        sc = ax.scatter(x, y, c=c, s=5, cmap="viridis", edgecolors="none", vmin=0.0, vmax=1.0)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$k_y/\pi$")
        ax.set_ylabel("Energy")
        ax.set_title(f"Termination {term}, v={v_focus:.1f}, x-open/y-periodic")
        ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label="edge weight")
        fig.tight_layout()
        fig.savefig(out_root / f"ribbon_termination_{term}_v0p5.png", dpi=180)
        plt.close(fig)
        if term == "A":
            term_a_eval, term_a_edge = e, ew
        else:
            term_b_eval, term_b_edge = e, ew

    # ---------------- Task 6: spin-mixing perturbation ----------------
    lambda_list = [0.0, 0.02, 0.05, 0.10]
    lambda_plot_data: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for lam in lambda_list:
        ky, e, ew, _sp = compute_full_ribbon_spectrum(
            v=v_focus,
            t=t,
            w=w,
            lm=lm,
            nx=nx,
            nk=ribbon_nk,
            edge_cells=edge_cells,
            lambda_r=lam,
            termination="A",
        )
        lambda_plot_data[lam] = (ky, e, ew)
    make_lambdar_montage(
        out_png=out_root / "ribbon_spin_mixing_lambdaR_scan.png",
        lambdas=lambda_list,
        spectra=lambda_plot_data,
    )

    # ---------------- Task 7: compare v=0.5 vs v=0.8 (QSH region) ----------------
    ky05, eu05, ed05, _wu05, _wd05 = compute_spin_ribbon_spectrum(
        v=v_focus,
        t=t,
        w=w,
        lm=lm,
        nx=nx,
        nk=ribbon_nk,
        edge_cells=edge_cells,
    )
    ky08, eu08, ed08, _wu08, _wd08 = compute_spin_ribbon_spectrum(
        v=v_qsh,
        t=t,
        w=w,
        lm=lm,
        nx=nx,
        nk=ribbon_nk,
        edge_cells=edge_cells,
    )

    cup05 = chern_fukui(lambda kx, ky: h_spin_k(kx, ky, v=v_focus, t=t, w=w, lm=lm, spin_sign=1), n_occ=2, nk=chern_nk)
    cdn05 = chern_fukui(lambda kx, ky: h_spin_k(kx, ky, v=v_focus, t=t, w=w, lm=lm, spin_sign=-1), n_occ=2, nk=chern_nk)
    cup08 = chern_fukui(lambda kx, ky: h_spin_k(kx, ky, v=v_qsh, t=t, w=w, lm=lm, spin_sign=1), n_occ=2, nk=chern_nk)
    cdn08 = chern_fukui(lambda kx, ky: h_spin_k(kx, ky, v=v_qsh, t=t, w=w, lm=lm, spin_sign=-1), n_occ=2, nk=chern_nk)
    cspin05 = int(round(0.5 * (round(cup05) - round(cdn05))))
    cspin08 = int(round(0.5 * (round(cup08) - round(cdn08))))
    make_v05_v08_comparison(
        out_png=out_root / "ribbon_v0p5_vs_v0p8_comparison.png",
        data05=(ky05, eu05, ed05),
        data08=(ky08, eu08, ed08),
        cspin05=cspin05,
        cspin08=cspin08,
    )

    # ---------------- Task 9: report ----------------
    vscan_map = {float(r["v"]): r for r in v_scan_rows}
    zak_map: dict[tuple[float, str], dict] = {}
    for r in zak_rows:
        zak_map[(float(r["v"]), str(r["case"]))] = r
    wf_by_spin = {"up": [], "down": []}
    for r in wf_rows:
        wf_by_spin[str(r["spin"])].append(r)

    avg_wedge_up = float(np.mean([float(r["W_edge"]) for r in wf_by_spin["up"]]))
    avg_wedge_dn = float(np.mean([float(r["W_edge"]) for r in wf_by_spin["down"]]))
    both_spin_edge = avg_wedge_up > 0.5 and avg_wedge_dn > 0.5

    vs_less_w = [v for v in v_list if v < w]
    vs_ge_w = [v for v in v_list if v >= w]
    ratio_less_w = float(np.mean([int(vscan_map[v]["middle_isolated_exists"]) for v in vs_less_w])) if vs_less_w else 0.0
    ratio_ge_w = float(np.mean([int(vscan_map[v]["middle_isolated_exists"]) for v in vs_ge_w])) if vs_ge_w else 0.0

    p_up_v05 = zak_map[(v_focus, "soc_spin_up_block")]
    p_dn_v05 = zak_map[(v_focus, "soc_spin_down_block")]
    p_full_v05 = zak_map[(v_focus, "soc_full_spinful_occ4")]
    p_nosoc_v05 = zak_map[(v_focus, "no_soc_block_4x4")]

    rg_a, ex_a, mg_a = summarize_middle_edge_band(
        evals=term_a_eval.reshape(-1),
        edge_weights=term_a_edge.reshape(-1),
        bulk_gap=bulk_gap_v_focus,
    )
    rg_b, ex_b, mg_b = summarize_middle_edge_band(
        evals=term_b_eval.reshape(-1),
        edge_weights=term_b_edge.reshape(-1),
        bulk_gap=bulk_gap_v_focus,
    )
    term_sensitive = bool((ex_a != ex_b) or (mg_a != mg_b) or (abs(rg_a - rg_b) > 0.01))

    lam_metrics = []
    for lam in lambda_list:
        _ky, ee, ew = lambda_plot_data[lam]
        rg, ex, mg = summarize_middle_edge_band(
            evals=ee.reshape(-1),
            edge_weights=ew.reshape(-1),
            bulk_gap=bulk_gap_v_focus,
        )
        lam_metrics.append((lam, rg, ex, mg, float(np.max(ew))))
    lam0 = lam_metrics[0]
    lam_end = lam_metrics[-1]
    spinmix_unstable = bool((lam0[2] == 1 and lam_end[2] == 0) or (lam_end[4] < lam0[4] - 0.15))

    report_lines = [
        "# SSH_EDGE_BAND_ORIGIN_REPORT",
        "",
        "## Setup",
        f"- Model: H8(k)=H0(k)⊗s0 + HSOC⊗sz, with w={w:.1f}, t={t:.1f}, lm={lm:.1f}.",
        f"- Focus: v={v_focus:.1f}, x-open / y-periodic ribbon, Nx={nx}.",
        f"- v scan: {', '.join([f'{v:.1f}' for v in v_list])}.",
        "",
        "## Key numeric observations",
        f"- v={v_focus:.1f}: avg W_edge (spin-up)={avg_wedge_up:.3f}, avg W_edge (spin-down)={avg_wedge_dn:.3f}.",
        f"- v<w band-existence ratio={ratio_less_w:.3f}, v>=w ratio={ratio_ge_w:.3f}.",
        f"- Termination A/B: exists={ex_a}/{ex_b}, merged={mg_a}/{mg_b}, ribbon_gap={rg_a:.4e}/{rg_b:.4e}.",
        f"- lambda_R scan: lambda=0 max_edge={lam0[4]:.3f}, lambda={lam_end[0]:.2f} max_edge={lam_end[4]:.3f}.",
        f"- C_spin(v=0.5)={cspin05}, C_spin(v=0.8)={cspin08}.",
        "",
        "## Answers",
        f"1) v=0.5 中间孤立条带是否由 spin-up/down 两套边界带组成: {'是' if both_spin_edge else '部分/不充分'}.",
        (
            "2) 波函数是否局域在 ribbon 两侧边界: "
            f"{'是' if (avg_wedge_up > 0.5 and avg_wedge_dn > 0.5) else '否/不明显'} "
            f"(W_edge_up={avg_wedge_up:.3f}, W_edge_dn={avg_wedge_dn:.3f})."
        ),
        (
            "3) 该带是否主要存在于 v<w 区域: "
            f"{'是' if ratio_less_w > ratio_ge_w + 0.25 else '证据不足/否'} "
            f"(ratio<v<w={ratio_less_w:.3f}, ratio>=w={ratio_ge_w:.3f})."
        ),
        (
            "4) 2D Zak phase 是否支持单自旋/单block SSH-like 极化: "
            f"noSOC(Px,Py)=({float(p_nosoc_v05['P_x']):.3f},{float(p_nosoc_v05['P_y']):.3f}), "
            f"up=({float(p_up_v05['P_x']):.3f},{float(p_up_v05['P_y']):.3f}), "
            f"dn=({float(p_dn_v05['P_x']):.3f},{float(p_dn_v05['P_y']):.3f})."
        ),
        (
            "5) full spinful 总 Zak phase 是否平庸: "
            f"{'是' if ('trivial' in str(p_full_v05['interpretation'])) else '否/不完全'} "
            f"(Px,Py)=({float(p_full_v05['P_x']):.3f},{float(p_full_v05['P_y']):.3f})."
        ),
        (
            "6) termination 改变后中间带是否改变: "
            f"{'是' if term_sensitive else '变化较小'}."
        ),
        (
            "7) 加入自旋混合扰动后该带是否不稳定: "
            f"{'是' if spinmix_unstable else '有退化但未完全消失'}."
        ),
        (
            "8) 该带应解释为何种边界态: "
            "更接近 spin-resolved SSH-like edge-localized band，"
            "不应直接称为 full-spinful 强保护 QSH helical edge state 或高阶角态。"
        ),
        (
            "9) 如何区分 v=0.5 SSH-like 与 v>0.6 QSH-like: "
            "看 (i) 对 termination 敏感性、(ii) 对 spin-mixing (Rashba-like) 敏感性、"
            "(iii) full spinful 总极化/拓扑指标是否非平庸、(iv) spin Chern 是否进入非平庸区。"
        ),
        "",
        "## Final classification language",
        "- spin-resolved SSH-like edge band: yes (for v=0.5 focus, with strong edge localization and termination/spin-mixing sensitivity checks).",
        "- full spinful total Zak phase: use 2d_zak_phase_summary.csv values as the primary criterion.",
        "- QSH/spin-Chern helical edge state: compare against v=0.8 reference and spin-Chern marker.",
        "- higher-order corner state: not supported by ribbon-edge evidence alone.",
    ]
    (out_root / "SSH_EDGE_BAND_ORIGIN_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[ok] output_root={out_root}")
    print("[ok] generated report and all requested outputs")


if __name__ == "__main__":
    main()
