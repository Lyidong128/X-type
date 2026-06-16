#!/usr/bin/env python3
"""Plot bulk and ribbon bands for 4x4 spinless model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GAMMA = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def token(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "m").replace(".", "p")


def h_original_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    h = np.zeros((4, 4), dtype=complex)
    h[0, 1] = t
    h[0, 2] = t
    h[0, 3] = v + w * np.exp(-1j * kx)

    h[1, 0] = t
    h[1, 2] = v + w * np.exp(-1j * ky)
    h[1, 3] = t

    h[2, 0] = t
    h[2, 1] = v + w * np.exp(1j * ky)
    h[2, 3] = t

    h[3, 0] = v + w * np.exp(1j * kx)
    h[3, 1] = t
    h[3, 2] = t
    return h


def h_chiral_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    h0 = h_original_k(kx, ky, v, t, w)
    return 0.5 * (h0 - GAMMA @ h0 @ GAMMA)


def hk(model_type: str, kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    if model_type == "H_chiral":
        return h_chiral_k(kx, ky, v, t, w)
    if model_type == "H_original":
        return h_original_k(kx, ky, v, t, w)
    raise ValueError(f"Unsupported model_type={model_type}")


def build_high_symmetry_path(nseg: int):
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
        a, b = nodes[i], nodes[i + 1]
        for j in range(nseg):
            u = j / nseg
            k_list.append((1.0 - u) * a + u * b)
        tick_indices.append(len(k_list))
    k_list.append(nodes[-1])

    kpts = np.array(k_list, dtype=float)
    xcoords = np.zeros(len(kpts), dtype=float)
    for i in range(1, len(kpts)):
        xcoords[i] = xcoords[i - 1] + np.linalg.norm(kpts[i] - kpts[i - 1])

    tick_positions = [xcoords[idx] for idx in tick_indices]
    return kpts, xcoords, tick_positions, labels


def plot_bulk_band(
    out_png: Path,
    model_type: str,
    v: float,
    t: float,
    w: float,
    nseg: int,
) -> None:
    kpts, xcoords, tick_positions, labels = build_high_symmetry_path(nseg=nseg)

    bands = []
    for k in kpts:
        evals = np.linalg.eigvalsh(hk(model_type, k[0], k[1], v, t, w))
        bands.append(np.real(evals))
    bands = np.array(bands, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for b in range(bands.shape[1]):
        ax.plot(xcoords, bands[:, b], linewidth=1.0, color="tab:blue")
    for xp in tick_positions:
        ax.axvline(xp, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)

    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy")
    ax.set_title(f"Bulk band ({model_type}), v={v:.2f}, t={t:.2f}, w={w:.2f}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def idx_1d(i: int, orb: int) -> int:
    return i * 4 + orb


def add_hop(H: np.ndarray, i: int, j: int, amp: complex) -> None:
    H[i, j] += amp
    H[j, i] += np.conjugate(amp)


def build_ribbon_x_open(
    ky: float, lx: int, model_type: str, v: float, t: float, w: float
) -> np.ndarray:
    # x-open / y-periodic
    H = np.zeros((4 * lx, 4 * lx), dtype=complex)
    t_same = 0.0 if model_type == "H_chiral" else t
    vky = v + w * np.exp(-1j * ky)

    for ix in range(lx):
        i0 = idx_1d(ix, 0)
        i1 = idx_1d(ix, 1)
        i2 = idx_1d(ix, 2)
        i3 = idx_1d(ix, 3)

        # intra-cell
        add_hop(H, i0, i1, t_same)
        add_hop(H, i0, i2, t)
        add_hop(H, i0, i3, v)
        add_hop(H, i1, i2, vky)
        add_hop(H, i1, i3, t)
        add_hop(H, i2, i3, t_same)

        # inter-cell along x
        if ix > 0:
            add_hop(H, i0, idx_1d(ix - 1, 3), w)

    return H


def build_ribbon_y_open(
    kx: float, ly: int, model_type: str, v: float, t: float, w: float
) -> np.ndarray:
    # y-open / x-periodic
    H = np.zeros((4 * ly, 4 * ly), dtype=complex)
    t_same = 0.0 if model_type == "H_chiral" else t
    vkx = v + w * np.exp(-1j * kx)

    for iy in range(ly):
        i0 = idx_1d(iy, 0)
        i1 = idx_1d(iy, 1)
        i2 = idx_1d(iy, 2)
        i3 = idx_1d(iy, 3)

        # intra-cell
        add_hop(H, i0, i1, t_same)
        add_hop(H, i0, i2, t)
        add_hop(H, i0, i3, vkx)
        add_hop(H, i1, i2, v)
        add_hop(H, i1, i3, t)
        add_hop(H, i2, i3, t_same)

        # inter-cell along y
        if iy > 0:
            add_hop(H, i1, idx_1d(iy - 1, 2), w)

    return H


def plot_ribbon_x_open(
    out_png: Path,
    model_type: str,
    v: float,
    t: float,
    w: float,
    lx: int,
    nk: int,
) -> None:
    ky_list = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    energies = []
    for ky in ky_list:
        H = build_ribbon_x_open(ky, lx, model_type, v, t, w)
        evals = np.linalg.eigvalsh(H)
        energies.append(np.real(evals))
    energies = np.array(energies, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.repeat(ky_list / np.pi, energies.shape[1])
    y = energies.reshape(-1)
    ax.scatter(x, y, s=4, color="tab:blue", alpha=0.75)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(
        f"Ribbon x-open ({model_type}), v={v:.2f}, t={t:.2f}, w={w:.2f}, Lx={lx}"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ribbon_y_open(
    out_png: Path,
    model_type: str,
    v: float,
    t: float,
    w: float,
    ly: int,
    nk: int,
) -> None:
    kx_list = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    energies = []
    for kx in kx_list:
        H = build_ribbon_y_open(kx, ly, model_type, v, t, w)
        evals = np.linalg.eigvalsh(H)
        energies.append(np.real(evals))
    energies = np.array(energies, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.repeat(kx_list / np.pi, energies.shape[1])
    y = energies.reshape(-1)
    ax.scatter(x, y, s=4, color="tab:orange", alpha=0.75)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_x/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(
        f"Ribbon y-open ({model_type}), v={v:.2f}, t={t:.2f}, w={w:.2f}, Ly={ly}"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot bulk and ribbon bands for 4x4 spinless model."
    )
    parser.add_argument("--model-type", default="H_original", choices=["H_original", "H_chiral"])
    parser.add_argument("--v-list", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--nseg", type=int, default=240)
    parser.add_argument("--Lx", type=int, default=50)
    parser.add_argument("--Ly", type=int, default=50)
    parser.add_argument("--Nk", type=int, default=201)
    parser.add_argument("--output-root", default="./outputs_manual_bands")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    v_list = parse_float_list(args.v_list)

    for v in v_list:
        bulk_png = out_root / f"bulk_{args.model_type}_v_{token(v)}.png"
        ribx_png = out_root / f"ribbon_xopen_{args.model_type}_v_{token(v)}.png"
        riby_png = out_root / f"ribbon_yopen_{args.model_type}_v_{token(v)}.png"

        plot_bulk_band(
            out_png=bulk_png,
            model_type=args.model_type,
            v=v,
            t=args.t,
            w=args.w,
            nseg=args.nseg,
        )
        plot_ribbon_x_open(
            out_png=ribx_png,
            model_type=args.model_type,
            v=v,
            t=args.t,
            w=args.w,
            lx=args.Lx,
            nk=args.Nk,
        )
        plot_ribbon_y_open(
            out_png=riby_png,
            model_type=args.model_type,
            v=v,
            t=args.t,
            w=args.w,
            ly=args.Ly,
            nk=args.Nk,
        )
        print(f"[ok] v={v:.2f}")
        print(f"     {bulk_png}")
        print(f"     {ribx_png}")
        print(f"     {riby_png}")


if __name__ == "__main__":
    main()
