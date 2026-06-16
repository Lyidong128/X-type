#!/usr/bin/env python3
"""Minimal plotting script for original 4x4 model: bulk + x-open ribbon."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Edit only these parameters
# ============================================================
v = 0.6
t = 0.3
w = 1.0

nseg = 220          # k points per high-symmetry segment (bulk)
lx = 60             # ribbon width (x-open)
nk = 201            # ky sampling for ribbon

output_dir = Path("/workspace/outputs/manual_original_bands")
# ============================================================


def h_original_k(kx: float, ky: float, v_: float, t_: float, w_: float) -> np.ndarray:
    h = np.zeros((4, 4), dtype=complex)
    h[0, 1] = t_
    h[0, 2] = t_
    h[0, 3] = v_ + w_ * np.exp(-1j * kx)

    h[1, 0] = t_
    h[1, 2] = v_ + w_ * np.exp(-1j * ky)
    h[1, 3] = t_

    h[2, 0] = t_
    h[2, 1] = v_ + w_ * np.exp(1j * ky)
    h[2, 3] = t_

    h[3, 0] = v_ + w_ * np.exp(1j * kx)
    h[3, 1] = t_
    h[3, 2] = t_
    return h


def build_path(nseg_: int):
    # square lattice reciprocal vectors
    b1 = np.array([2.0 * np.pi, 0.0], dtype=float)
    b2 = np.array([0.0, 2.0 * np.pi], dtype=float)

    g = np.array([0.0, 0.0], dtype=float)
    x = 0.5 * b1
    y = 0.5 * b2
    m = 0.5 * (b1 + b2)

    # Gamma -> X -> Y -> Gamma -> M -> Gamma
    nodes = [g, x, y, g, m, g]
    labels = [r"$\Gamma$", "X", "Y", r"$\Gamma$", "M", r"$\Gamma$"]

    k_list = []
    tick_indices = [0]
    for i in range(len(nodes) - 1):
        a = nodes[i]
        b = nodes[i + 1]
        for j in range(nseg_):
            u = j / nseg_
            k_list.append((1.0 - u) * a + u * b)
        tick_indices.append(len(k_list))
    k_list.append(nodes[-1])
    kpts = np.array(k_list, dtype=float)

    xcoords = np.zeros(len(kpts), dtype=float)
    for i in range(1, len(kpts)):
        xcoords[i] = xcoords[i - 1] + np.linalg.norm(kpts[i] - kpts[i - 1])

    tick_positions = [xcoords[i] for i in tick_indices]
    return kpts, xcoords, tick_positions, labels


def plot_bulk() -> None:
    kpts, xcoords, tick_positions, labels = build_path(nseg)

    bands = []
    for k in kpts:
        evals = np.linalg.eigvalsh(h_original_k(k[0], k[1], v, t, w))
        bands.append(np.real(evals))
    bands = np.array(bands, dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for ib in range(bands.shape[1]):
        ax.plot(xcoords, bands[:, ib], lw=1.0, color="tab:blue")
    for xp in tick_positions:
        ax.axvline(xp, color="gray", ls="--", lw=0.7, alpha=0.7)
    ax.axhline(0.0, color="black", ls=":", lw=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy")
    ax.set_title(f"Original bulk band: v={v:.2f}, t={t:.2f}, w={w:.2f}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "bulk_original.png", dpi=180)
    plt.close(fig)


def idx_1d(i: int, orb: int) -> int:
    return i * 4 + orb


def add_hop(H: np.ndarray, i: int, j: int, amp: complex) -> None:
    H[i, j] += amp
    H[j, i] += np.conjugate(amp)


def build_ribbon_xopen(ky: float, lx_: int, v_: float, t_: float, w_: float) -> np.ndarray:
    # x-open / y-periodic
    H = np.zeros((4 * lx_, 4 * lx_), dtype=complex)
    vky = v_ + w_ * np.exp(-1j * ky)
    for ix in range(lx_):
        i0 = idx_1d(ix, 0)
        i1 = idx_1d(ix, 1)
        i2 = idx_1d(ix, 2)
        i3 = idx_1d(ix, 3)

        # intra-cell (original model)
        add_hop(H, i0, i1, t_)
        add_hop(H, i0, i2, t_)
        add_hop(H, i0, i3, v_)
        add_hop(H, i1, i2, vky)
        add_hop(H, i1, i3, t_)
        add_hop(H, i2, i3, t_)

        # inter-cell along x
        if ix > 0:
            add_hop(H, i0, idx_1d(ix - 1, 3), w_)
    return H


def plot_ribbon() -> None:
    ky_list = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    e_all = []
    for ky in ky_list:
        H = build_ribbon_xopen(ky, lx, v, t, w)
        evals = np.linalg.eigvalsh(H)
        e_all.append(np.real(evals))
    e_all = np.array(e_all, dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.repeat(ky_list / np.pi, e_all.shape[1])
    y = e_all.reshape(-1)
    ax.scatter(x, y, s=4, color="tab:blue", alpha=0.75)
    ax.axhline(0.0, color="black", ls="--", lw=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(f"Original ribbon x-open: v={v:.2f}, t={t:.2f}, w={w:.2f}, Lx={lx}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "ribbon_xopen_original.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_bulk()
    plot_ribbon()
    print(f"[ok] saved: {output_dir / 'bulk_original.png'}")
    print(f"[ok] saved: {output_dir / 'ribbon_xopen_original.png'}")


if __name__ == "__main__":
    main()
