#!/usr/bin/env python3
"""Wilson-loop v-scan (v=0.1..0.8) for original SOC model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Wilson-loop/Wannier flow for v scan.")
    parser.add_argument("--output-root", default="/workspace/outputs/ssh_edge_check/wilson_vscan_0p1_0p8")
    parser.add_argument("--v-start", type=float, default=0.1)
    parser.add_argument("--v-stop", type=float, default=0.8)
    parser.add_argument("--v-step", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--nk-scan", type=int, default=81)
    parser.add_argument("--nk-loop", type=int, default=121)
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


def wilson_centers_direction(
    v: float,
    t: float,
    w: float,
    lm: float,
    n_occ: int,
    nk_scan: int,
    nk_loop: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    k_scan = np.linspace(-np.pi, np.pi, nk_scan, endpoint=False)
    k_loop = np.linspace(-np.pi, np.pi, nk_loop, endpoint=False)
    centers = np.zeros((nk_scan, n_occ), dtype=float)

    for j, ks in enumerate(k_scan):
        wmat = np.eye(n_occ, dtype=complex)
        if direction == "x":
            kx0, ky0 = float(k_loop[0]), float(ks)
        else:
            kx0, ky0 = float(ks), float(k_loop[0])
        occ0 = occ_subspace(kx0, ky0, v=v, t=t, w=w, lm=lm, n_occ=n_occ)

        for i in range(nk_loop):
            if direction == "x":
                kx = float(k_loop[i])
                ky = float(ks)
                kx_n = float(k_loop[(i + 1) % nk_loop])
                ky_n = float(ks)
            else:
                kx = float(ks)
                ky = float(k_loop[i])
                kx_n = float(ks)
                ky_n = float(k_loop[(i + 1) % nk_loop])
            oa = occ_subspace(kx, ky, v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            ob = occ_subspace(kx_n, ky_n, v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            q = unitary_part(oa.conj().T @ ob)
            wmat = q @ wmat

        eigvals, eigvecs = np.linalg.eig(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]

        # Keep sector ordering smooth-ish by overlap with initial subspace.
        _ = occ0 @ eigvecs[:, order]

    return k_scan, centers


def crossing_parity(centers: np.ndarray, reference: float = 0.5) -> int:
    crossings = 0
    for i in range(centers.shape[0] - 1):
        a = centers[i, :] - reference
        b = centers[i + 1, :] - reference
        crossings += int(np.sum(a * b < 0.0))
    return int(crossings % 2)


def min_sector_gap(centers: np.ndarray) -> float:
    if centers.shape[1] < 4:
        return float("nan")
    gap = centers[:, 2] - centers[:, 1]
    return float(np.min(gap))


def plot_wilson(scan: np.ndarray, centers: np.ndarray, out_png: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for i in range(centers.shape[1]):
        ax.plot(scan / np.pi, centers[:, i], linewidth=1.0)
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


def plot_montage(
    v_list: list[float],
    data: dict[float, tuple[np.ndarray, np.ndarray]],
    out_png: Path,
    direction: str,
    summary_map: dict[tuple[float, str], dict],
) -> None:
    n = len(v_list)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for idx, v in enumerate(v_list):
        ax = axes[idx // cols][idx % cols]
        scan, centers = data[v]
        for i in range(centers.shape[1]):
            ax.plot(scan / np.pi, centers[:, i], linewidth=0.8)
        key = (v, direction)
        z2 = int(summary_map[key]["estimated_z2"])
        gap = float(summary_map[key]["wannier_sector_gap"])
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"v={v:.1f}, Z2~{z2}, gap={gap:.3f}", fontsize=8)
        ax.set_xlabel(r"$k/\pi$")
        ax.set_ylabel(r"$\nu$")
        ax.grid(alpha=0.2)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(f"Wilson-loop Wannier flow ({direction}-loop)", fontsize=12)
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

    nk_scan = 41 if args.quick else int(args.nk_scan)
    nk_loop = 61 if args.quick else int(args.nk_loop)
    v_list = build_v_list(args.v_start, args.v_stop, args.v_step)

    summary_rows: list[dict] = []
    centers_rows: list[dict] = []
    data_x: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    data_y: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for v in v_list:
        for direction in ["x", "y"]:
            scan, centers = wilson_centers_direction(
                v=v,
                t=float(args.t),
                w=float(args.w),
                lm=float(args.lm),
                n_occ=int(args.n_occ),
                nk_scan=nk_scan,
                nk_loop=nk_loop,
                direction=direction,
            )
            parity = crossing_parity(centers, reference=0.5)
            est_z2 = int(parity % 2)
            gap = min_sector_gap(centers)
            summary_rows.append(
                {
                    "v": v,
                    "direction": direction,
                    "estimated_z2": est_z2,
                    "crossing_parity": parity,
                    "wannier_sector_gap": gap,
                    "t": float(args.t),
                    "w": float(args.w),
                    "lm": float(args.lm),
                }
            )
            for i in range(centers.shape[0]):
                for j in range(centers.shape[1]):
                    centers_rows.append(
                        {
                            "v": v,
                            "direction": direction,
                            "scan_param": float(scan[i]),
                            "center_index": int(j),
                            "wannier_center": float(centers[i, j]),
                        }
                    )

            if direction == "x":
                data_x[v] = (scan, centers)
                plot_wilson(
                    scan=scan,
                    centers=centers,
                    out_png=out_root / f"wilson_v{token(v)}_xloop.png",
                    title=f"v={v:.1f}, x-loop (scan ky), t={args.t:.1f}, w={args.w:.1f}, lm={args.lm:.1f}",
                    xlabel=r"$k_y/\pi$",
                )
            else:
                data_y[v] = (scan, centers)
                plot_wilson(
                    scan=scan,
                    centers=centers,
                    out_png=out_root / f"wilson_v{token(v)}_yloop.png",
                    title=f"v={v:.1f}, y-loop (scan kx), t={args.t:.1f}, w={args.w:.1f}, lm={args.lm:.1f}",
                    xlabel=r"$k_x/\pi$",
                )

    summary_map = {(float(r["v"]), str(r["direction"])): r for r in summary_rows}
    plot_montage(
        v_list=v_list,
        data=data_x,
        out_png=out_root / "wilson_montage_xloop_v0p1_to_0p8.png",
        direction="x",
        summary_map=summary_map,
    )
    plot_montage(
        v_list=v_list,
        data=data_y,
        out_png=out_root / "wilson_montage_yloop_v0p1_to_0p8.png",
        direction="y",
        summary_map=summary_map,
    )

    write_csv(
        out_root / "wilson_summary_v0p1_to_0p8.csv",
        summary_rows,
        ["v", "direction", "estimated_z2", "crossing_parity", "wannier_sector_gap", "t", "w", "lm"],
    )
    write_csv(
        out_root / "wilson_centers_v0p1_to_0p8.csv",
        centers_rows,
        ["v", "direction", "scan_param", "center_index", "wannier_center"],
    )

    print(f"[ok] output_root={out_root}")
    print("[ok] generated Wilson-loop plots and CSVs for v=0.1..0.8")


if __name__ == "__main__":
    main()
