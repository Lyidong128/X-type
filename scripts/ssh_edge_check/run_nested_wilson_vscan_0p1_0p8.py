#!/usr/bin/env python3
"""Nested Wilson-loop scan for original SOC model (v=0.1..0.8)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested Wilson-loop v-scan.")
    parser.add_argument("--output-root", default="/workspace/outputs/ssh_edge_check/nested_wilson_vscan_0p1_0p8")
    parser.add_argument("--v-start", type=float, default=0.1)
    parser.add_argument("--v-stop", type=float, default=0.8)
    parser.add_argument("--v-step", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--nk", type=int, default=81)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def token(v: float) -> str:
    return f"{v:.1f}".replace("-", "m").replace(".", "p")


def build_v_list(v_start: float, v_stop: float, v_step: float) -> list[float]:
    vals = []
    x = float(v_start)
    eps = 1e-12
    while x <= float(v_stop) + eps:
        vals.append(round(x, 10))
        x += float(v_step)
    return vals


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
    sector_bases: list[np.ndarray] = []
    min_sector_overlap = 1.0
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
        sec = occ0 @ ev[:, :2]
        sec, _ = np.linalg.qr(sec)
        sector_bases.append(sec)
        if j > 0:
            sv = np.linalg.svd(sector_bases[j - 1].conj().T @ sec, compute_uv=False)
            min_sector_overlap = min(min_sector_overlap, float(np.min(sv)))
    return ky_grid, centers, sector_bases, float(min_sector_overlap)


def wilson_y_data(v: float, t: float, w: float, lm: float, n_occ: int, nk: int):
    kx_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    centers = np.zeros((nk, n_occ), dtype=float)
    sector_bases: list[np.ndarray] = []
    min_sector_overlap = 1.0
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
            min_sector_overlap = min(min_sector_overlap, float(np.min(sv)))
    return kx_grid, centers, sector_bases, float(min_sector_overlap)


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
    gap = centers[:, 2] - centers[:, 1]
    return float(np.min(gap))


def plot_wannier(kvals: np.ndarray, centers: np.ndarray, out_png: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
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


def plot_qxy(vvals: np.ndarray, qxy: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(vvals, qxy, marker="o", linewidth=1.2)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("v")
    ax.set_ylabel("q_xy")
    ax.set_title("Nested Wilson q_xy vs v")
    ax.grid(alpha=0.25)
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

    nk = 41 if args.quick else int(args.nk)
    v_list = build_v_list(args.v_start, args.v_stop, args.v_step)
    rows: list[dict] = []

    for v in v_list:
        ky_x, centers_x, sec_x, ov_x = wilson_x_data(v=v, t=args.t, w=args.w, lm=args.lm, n_occ=args.n_occ, nk=nk)
        kx_y, centers_y, sec_y, ov_y = wilson_y_data(v=v, t=args.t, w=args.w, lm=args.lm, n_occ=args.n_occ, nk=nk)

        gap_x = wannier_split_gap(centers_x)
        gap_y = wannier_split_gap(centers_y)
        p_y_nux = nested_phase_from_sector(sec_x)
        p_x_nuy = nested_phase_from_sector(sec_y)
        q_xy = float(np.mod(0.5 * (p_y_nux + p_x_nuy), 1.0))

        reliability = "reliable"
        if gap_x < 0.03 or gap_y < 0.03 or min(ov_x, ov_y) < 0.20:
            reliability = "unreliable"

        rows.append(
            {
                "v": v,
                "p_y_nu_x": p_y_nux,
                "p_x_nu_y": p_x_nuy,
                "q_xy": q_xy,
                "wannier_gap_x": gap_x,
                "wannier_gap_y": gap_y,
                "min_sector_overlap_x": ov_x,
                "min_sector_overlap_y": ov_y,
                "reliability": reliability,
                "t": float(args.t),
                "w": float(args.w),
                "lm": float(args.lm),
            }
        )

        plot_wannier(
            ky_x,
            centers_x,
            out_png=out_root / f"wannier_x_v{token(v)}.png",
            title=f"v={v:.1f}: Wilson x-loop (scan ky)",
            xlabel=r"$k_y/\pi$",
        )
        plot_wannier(
            kx_y,
            centers_y,
            out_png=out_root / f"wannier_y_v{token(v)}.png",
            title=f"v={v:.1f}: Wilson y-loop (scan kx)",
            xlabel=r"$k_x/\pi$",
        )

    write_csv(
        out_root / "nested_wilson_summary_v0p1_to_0p8.csv",
        rows,
        [
            "v",
            "p_y_nu_x",
            "p_x_nu_y",
            "q_xy",
            "wannier_gap_x",
            "wannier_gap_y",
            "min_sector_overlap_x",
            "min_sector_overlap_y",
            "reliability",
            "t",
            "w",
            "lm",
        ],
    )

    vvals = np.array([float(r["v"]) for r in rows], dtype=float)
    qvals = np.array([float(r["q_xy"]) for r in rows], dtype=float)
    plot_qxy(vvals=vvals, qxy=qvals, out_png=out_root / "qxy_vs_v_v0p1_to_0p8.png")

    print(f"[ok] output_root={out_root}")
    print("[ok] generated nested Wilson summary and Wannier plots")


if __name__ == "__main__":
    main()
