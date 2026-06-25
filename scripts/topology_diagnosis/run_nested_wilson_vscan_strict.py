#!/usr/bin/env python3
"""Strict nested Wilson-loop diagnostics with sector tracking and reliability checks."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.common_topology import ensure_dir, h8_k, unitary_part


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict nested Wilson-loop scan.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs")
    parser.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--nkx", type=int, default=81)
    parser.add_argument("--nky", type=int, default=81)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def token(v: float) -> str:
    return f"{v:.1f}".replace("-", "m").replace(".", "p")


def occ_subspace(kx: float, ky: float, v: float, t: float, w: float, lm: float, n_occ: int) -> np.ndarray:
    _, vecs = np.linalg.eigh(h8_k(kx, ky, v=v, t=t, w=w, lm=lm))
    return vecs[:, :n_occ]


def wilson_data_x(v: float, t: float, w: float, lm: float, n_occ: int, nkx: int, nky: int):
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nky, endpoint=False)
    centers = np.zeros((nky, n_occ), dtype=float)
    eigvec_occ: list[np.ndarray] = []
    occ_bases: list[np.ndarray] = []
    for j, ky in enumerate(ky_grid):
        wmat = np.eye(n_occ, dtype=complex)
        occ_base = occ_subspace(float(kx_grid[0]), float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
        for i, kx in enumerate(kx_grid):
            kx_n = float(kx_grid[(i + 1) % nkx])
            oa = occ_subspace(float(kx), float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            ob = occ_subspace(kx_n, float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            q = unitary_part(oa.conj().T @ ob)
            wmat = q @ wmat
        eigvals, eigvecs = np.linalg.eig(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        eigvec_occ.append(eigvecs[:, order])
        occ_bases.append(occ_base)
    return ky_grid, centers, eigvec_occ, occ_bases


def wilson_data_y(v: float, t: float, w: float, lm: float, n_occ: int, nkx: int, nky: int):
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nky, endpoint=False)
    centers = np.zeros((nkx, n_occ), dtype=float)
    eigvec_occ: list[np.ndarray] = []
    occ_bases: list[np.ndarray] = []
    for j, kx in enumerate(kx_grid):
        wmat = np.eye(n_occ, dtype=complex)
        occ_base = occ_subspace(float(kx), float(ky_grid[0]), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
        for i, ky in enumerate(ky_grid):
            ky_n = float(ky_grid[(i + 1) % nky])
            oa = occ_subspace(float(kx), float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            ob = occ_subspace(float(kx), ky_n, v=v, t=t, w=w, lm=lm, n_occ=n_occ)
            q = unitary_part(oa.conj().T @ ob)
            wmat = q @ wmat
        eigvals, eigvecs = np.linalg.eig(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        eigvec_occ.append(eigvecs[:, order])
        occ_bases.append(occ_base)
    return kx_grid, centers, eigvec_occ, occ_bases


def circular_middle_gap(centers: np.ndarray) -> float:
    # Robustly define 2+2 sector split gap on a periodic [0,1) axis.
    n = centers.shape[1]
    mid = n // 2
    gaps = []
    for row in centers:
        arr = np.sort(np.mod(row, 1.0))
        cyc = np.r_[arr, arr[0] + 1.0]
        cyc_gaps = np.diff(cyc)
        cut = int(np.argmax(cyc_gaps))
        start = arr[(cut + 1) % n]
        shifted = np.mod(arr - start, 1.0)
        shifted = np.sort(shifted)
        gaps.append(float(shifted[mid] - shifted[mid - 1]))
    return float(np.min(gaps))


def track_sector_frames(
    centers: np.ndarray,
    eigvec_occ: list[np.ndarray],
    n_occ: int,
) -> tuple[list[np.ndarray], float]:
    half = n_occ // 2
    combos = list(combinations(range(n_occ), half))

    # Start from lower sorted branch sector.
    init_idx = tuple(range(half))
    prev = eigvec_occ[0][:, init_idx]
    tracked = [prev]
    min_overlap = 1.0

    for j in range(1, len(centers)):
        vecs = eigvec_occ[j]
        best_combo = combos[0]
        best_score = -1.0
        best_min_sv = 0.0
        for combo in combos:
            cand = vecs[:, combo]
            sv = np.linalg.svd(prev.conj().T @ cand, compute_uv=False)
            score = float(np.min(sv))
            if score > best_score:
                best_score = score
                best_combo = combo
                best_min_sv = float(np.min(sv))
        chosen = vecs[:, best_combo]
        tracked.append(chosen)
        min_overlap = min(min_overlap, best_min_sv)
        prev = chosen

    close_sv = np.linalg.svd(tracked[-1].conj().T @ tracked[0], compute_uv=False)
    min_overlap = min(min_overlap, float(np.min(close_sv)))
    return tracked, float(min_overlap)


def nested_polarization(
    tracked_sector: list[np.ndarray],
    occ_bases: list[np.ndarray],
) -> float:
    half = tracked_sector[0].shape[1]
    wmat = np.eye(half, dtype=complex)
    n = len(tracked_sector)
    for j in range(n):
        jp = (j + 1) % n
        f = unitary_part(occ_bases[j].conj().T @ occ_bases[jp])
        g = tracked_sector[j].conj().T @ f @ tracked_sector[jp]
        q = unitary_part(g)
        wmat = q @ wmat
    detw = np.linalg.det(wmat)
    if abs(detw) < 1e-14:
        return 0.0
    return float(np.mod(np.angle(detw) / (2.0 * np.pi), 1.0))


def plot_wannier(scan: np.ndarray, centers: np.ndarray, out_png: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for i in range(centers.shape[1]):
        ax.plot(scan / np.pi, centers[:, i], linewidth=0.9)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Wannier center")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_reliable_series(
    x: np.ndarray,
    y_list: list[tuple[np.ndarray, str]],
    reliable: np.ndarray,
    out_png: Path,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for y, label in y_list:
        ax.plot(x, y, linewidth=1.1, alpha=0.8, label=label)
        ax.scatter(x[reliable], y[reliable], s=42, marker="o", facecolors="tab:blue", edgecolors="black", alpha=0.9)
        ax.scatter(x[~reliable], y[~reliable], s=42, marker="o", facecolors="none", edgecolors="gray", alpha=0.9)
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xlabel("v")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.24)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    v_list = parse_float_list(args.v_list)
    nkx = 41 if args.quick else int(args.nkx)
    nky = 41 if args.quick else int(args.nky)
    n_occ = int(args.n_occ)

    rows: list[dict] = []
    center_rows: list[dict] = []

    for v in v_list:
        ky, centers_x, eigvec_x, occ_base_x = wilson_data_x(v=v, t=args.t, w=args.w, lm=args.lm, n_occ=n_occ, nkx=nkx, nky=nky)
        kx, centers_y, eigvec_y, occ_base_y = wilson_data_y(v=v, t=args.t, w=args.w, lm=args.lm, n_occ=n_occ, nkx=nkx, nky=nky)

        gap_x = circular_middle_gap(centers_x)
        gap_y = circular_middle_gap(centers_y)

        track_x, min_ov_x = track_sector_frames(centers_x, eigvec_x, n_occ=n_occ)
        track_y, min_ov_y = track_sector_frames(centers_y, eigvec_y, n_occ=n_occ)

        p_y_nu_x = nested_polarization(track_x, occ_base_x)
        p_x_nu_y = nested_polarization(track_y, occ_base_y)
        qxy = float(np.mod(0.5 * (p_y_nu_x + p_x_nu_y), 1.0))

        reliable = bool(gap_x > 0.05 and gap_y > 0.05 and min_ov_x > 0.5 and min_ov_y > 0.5)
        rows.append(
            {
                "v": float(v),
                "nkx": int(nkx),
                "nky": int(nky),
                "n_occ": int(n_occ),
                "p_y_nu_x": float(p_y_nu_x),
                "p_x_nu_y": float(p_x_nu_y),
                "qxy": float(qxy),
                "wannier_gap_x": float(gap_x),
                "wannier_gap_y": float(gap_y),
                "min_sector_overlap_x": float(min_ov_x),
                "min_sector_overlap_y": float(min_ov_y),
                "reliable": int(reliable),
            }
        )

        for i in range(centers_x.shape[0]):
            for b in range(centers_x.shape[1]):
                center_rows.append(
                    {
                        "v": float(v),
                        "direction": "x",
                        "scan_param": float(ky[i]),
                        "center_index": int(b),
                        "wannier_center": float(centers_x[i, b]),
                    }
                )
        for i in range(centers_y.shape[0]):
            for b in range(centers_y.shape[1]):
                center_rows.append(
                    {
                        "v": float(v),
                        "direction": "y",
                        "scan_param": float(kx[i]),
                        "center_index": int(b),
                        "wannier_center": float(centers_y[i, b]),
                    }
                )

        plot_wannier(
            ky,
            centers_x,
            out_png=out_root / f"wannier_x_v{token(v)}_strict.png",
            title=f"strict Wannier x-loop v={v:.1f}",
            xlabel=r"$k_y/\pi$",
        )
        plot_wannier(
            kx,
            centers_y,
            out_png=out_root / f"wannier_y_v{token(v)}_strict.png",
            title=f"strict Wannier y-loop v={v:.1f}",
            xlabel=r"$k_x/\pi$",
        )
        print(
            f"[nested] v={v:.1f} qxy={qxy:.4f} p_y_nu_x={p_y_nu_x:.4f} p_x_nu_y={p_x_nu_y:.4f} "
            f"gap_x={gap_x:.4f} gap_y={gap_y:.4f} ov=({min_ov_x:.3f},{min_ov_y:.3f}) reliable={reliable}"
        )

    write_csv(
        out_root / "nested_wilson_strict_summary.csv",
        rows,
        [
            "v",
            "nkx",
            "nky",
            "n_occ",
            "p_y_nu_x",
            "p_x_nu_y",
            "qxy",
            "wannier_gap_x",
            "wannier_gap_y",
            "min_sector_overlap_x",
            "min_sector_overlap_y",
            "reliable",
        ],
    )
    write_csv(
        out_root / "wannier_centers_strict.csv",
        center_rows,
        ["v", "direction", "scan_param", "center_index", "wannier_center"],
    )

    v = np.array([float(r["v"]) for r in rows], dtype=float)
    qxy = np.array([float(r["qxy"]) for r in rows], dtype=float)
    py = np.array([float(r["p_y_nu_x"]) for r in rows], dtype=float)
    px = np.array([float(r["p_x_nu_y"]) for r in rows], dtype=float)
    gx = np.array([float(r["wannier_gap_x"]) for r in rows], dtype=float)
    gy = np.array([float(r["wannier_gap_y"]) for r in rows], dtype=float)
    ox = np.array([float(r["min_sector_overlap_x"]) for r in rows], dtype=float)
    oy = np.array([float(r["min_sector_overlap_y"]) for r in rows], dtype=float)
    reliable = np.array([int(r["reliable"]) == 1 for r in rows], dtype=bool)

    plot_reliable_series(
        v,
        [(qxy, "q_xy")],
        reliable=reliable,
        out_png=out_root / "qxy_vs_v_strict.png",
        ylabel="q_xy",
        title="Strict nested Wilson q_xy vs v",
    )
    plot_reliable_series(
        v,
        [(py, "p_y_nu_x"), (px, "p_x_nu_y")],
        reliable=reliable,
        out_png=out_root / "p_nested_vs_v_strict.png",
        ylabel="nested polarization",
        title="Strict nested polarizations vs v",
    )
    plot_reliable_series(
        v,
        [(gx, "wannier_gap_x"), (gy, "wannier_gap_y")],
        reliable=reliable,
        out_png=out_root / "wannier_gap_vs_v_strict.png",
        ylabel="gap",
        title="Wannier sector gaps vs v",
    )
    plot_reliable_series(
        v,
        [(ox, "min_sector_overlap_x"), (oy, "min_sector_overlap_y")],
        reliable=reliable,
        out_png=out_root / "sector_overlap_vs_v_strict.png",
        ylabel="overlap",
        title="Sector tracking overlap vs v",
    )
    print(f"[ok] strict nested summary saved at {out_root / 'nested_wilson_strict_summary.csv'}")


if __name__ == "__main__":
    main()
