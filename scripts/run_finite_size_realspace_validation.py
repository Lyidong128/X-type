from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import signal
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import build_obc_hamiltonian_sparse

matplotlib.use("Agg")


@dataclass
class Point:
    point_id: str
    v: float
    t: float
    lm: float
    w: float
    special_score: float
    special_rank: int


def parse_sizes(raw: str) -> list[int]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(v for v in vals if v >= 4))
    if not vals:
        raise ValueError("No valid sizes parsed.")
    return vals


def load_top_points(analysis_csv: Path, top_n: int) -> list[Point]:
    rows: list[Point] = []
    with analysis_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank_raw = row.get("special_rank", "").strip()
            if not rank_raw:
                continue
            rank = int(rank_raw)
            if rank > top_n:
                continue
            rows.append(
                Point(
                    point_id=row["point_id"],
                    v=float(row["v"]),
                    t=float(row["t"]),
                    lm=float(row["lm"]),
                    w=float(row["w"]),
                    special_score=float(row["special_score"]),
                    special_rank=rank,
                )
            )
    rows.sort(key=lambda p: p.special_rank)
    return rows


def robust_sparse_eigs_near_zero(ham_sparse, k: int, per_try_timeout: int = 20) -> np.ndarray:
    """Compute near-zero eigenvalues with sparse fallbacks."""

    def _timeout_handler(signum, frame):
        raise TimeoutError("eigsh timed out")

    dim = ham_sparse.shape[0]
    k = max(12, min(int(k), dim - 2))
    k_candidates = []
    for k_try in (k, k - 16, k + 16, 96, 80, 64, 48):
        kk = max(12, min(int(k_try), dim - 2))
        if kk not in k_candidates:
            k_candidates.append(kk)

    for kk in k_candidates:
        for kwargs in ({"sigma": 0.0, "which": "LM"}, {"which": "SM"}):
            old_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(max(1, int(per_try_timeout)))
                vals, _ = eigsh(ham_sparse, k=kk, maxiter=1200, tol=1e-7, **kwargs)
                signal.alarm(0)
                vals = np.real(vals)
                return np.sort(vals)
            except ArpackNoConvergence as exc:
                signal.alarm(0)
                vals = getattr(exc, "eigenvalues", None)
                if vals is not None and len(vals) >= 12:
                    return np.sort(np.real(vals))
            except Exception:
                signal.alarm(0)
                continue
            finally:
                signal.signal(signal.SIGALRM, old_handler)
    raise RuntimeError("failed_sparse_near_zero")


def near_zero_metrics(evals: np.ndarray) -> dict[str, float | int]:
    abs_e = np.abs(evals)
    neg = evals[evals < 0]
    pos = evals[evals > 0]
    if neg.size > 0 and pos.size > 0:
        gap = float(np.min(pos) - np.max(neg))
    else:
        gap = 0.0
    return {
        "min_abs_energy": float(np.min(abs_e)),
        "count_absE_le_0p02": int(np.sum(abs_e <= 0.02)),
        "count_absE_le_0p05": int(np.sum(abs_e <= 0.05)),
        "near_zero_gap_estimate": gap,
    }


def make_phase_vectors(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Build exp(i2pi x/Lx), exp(i2pi y/Ly) phase vectors on full Hilbert space."""
    dim = nx * ny * 8
    px = np.zeros(dim, dtype=complex)
    py = np.zeros(dim, dtype=complex)
    idx = 0
    for y in range(ny):
        for x in range(nx):
            phase_x = np.exp(1j * 2.0 * np.pi * (x / max(nx, 1)))
            phase_y = np.exp(1j * 2.0 * np.pi * (y / max(ny, 1)))
            for _ in range(8):
                px[idx] = phase_x
                py[idx] = phase_y
                idx += 1
    return px, py


def unitary_part(mat: np.ndarray) -> np.ndarray:
    """Return closest unitary matrix via polar decomposition M = U H."""
    u, _, vh = np.linalg.svd(mat, full_matrices=False)
    return u @ vh


def compute_bott_index_dense(ham_sparse, nx: int, ny: int, fermi: float = 0.0) -> float:
    """Compute Bott index using occupied projector in dense mode."""
    h = ham_sparse.toarray()
    evals, evecs = np.linalg.eigh(h)
    occ_mask = evals < fermi
    if not np.any(occ_mask):
        occ_mask = np.arange(evals.size) < (evals.size // 2)
    occ = evecs[:, occ_mask]

    px, py = make_phase_vectors(nx=nx, ny=ny)
    # projected phase operators in occupied subspace
    mx = occ.conj().T @ (px[:, None] * occ)
    my = occ.conj().T @ (py[:, None] * occ)
    ux = unitary_part(mx)
    uy = unitary_part(my)
    w = uy @ ux @ uy.conj().T @ ux.conj().T
    angles = np.angle(np.linalg.eigvals(w))
    bott = float(np.sum(angles) / (2.0 * np.pi))
    return float(np.round(bott))


def plot_scaling(metric_rows: list[dict[str, float | int | str]], metric_key: str, y_label: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["point_id"]), []).append(row)
    for pid, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r["size"]))
        x = [int(r["size"]) for r in rows]
        y = [float(r[metric_key]) for r in rows]
        ax.plot(x, y, marker="o", linewidth=1.1, markersize=3.5, alpha=0.85, label=pid)
    ax.set_xlabel("System size L (Lx=Ly=L)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Finite-size scaling: {metric_key}")
    ax.grid(alpha=0.25)
    if len(grouped) <= 12:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_bott_vs_score(bott_rows: list[dict[str, float | int | str]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    score = np.array([float(r["special_score"]) for r in bott_rows], dtype=float)
    bott = np.array([float(r["bott_index"]) for r in bott_rows], dtype=float)
    ranks = np.array([int(r["special_rank"]) for r in bott_rows], dtype=int)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sc = ax.scatter(score, bott, c=ranks, cmap="viridis_r", s=70, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("Heuristic special_score")
    ax.set_ylabel("Bott index (dense)")
    ax.set_title("Heuristic score vs Bott index")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax, label="special_rank")
    cbar.ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_bott_size_scaling(bott_rows: list[dict[str, float | int | str]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in bott_rows:
        if str(row.get("bott_status", "")) != "ok":
            continue
        grouped.setdefault(str(row["point_id"]), []).append(row)
    for pid, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r["bott_size"]))
        x = [int(r["bott_size"]) for r in rows]
        y = [float(r["bott_index"]) for r in rows]
        ax.plot(x, y, marker="o", linewidth=1.1, markersize=3.5, alpha=0.85, label=pid)
    ax.set_xlabel("System size L for Bott")
    ax.set_ylabel("Bott index")
    ax.set_title("Bott index finite-size scaling")
    ax.grid(alpha=0.25)
    if len(grouped) <= 12:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    analysis_csv = root / args.analysis_csv
    output_dir = root / args.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sizes = parse_sizes(args.sizes)
    bott_sizes = parse_sizes(args.bott_sizes)
    points = load_top_points(analysis_csv=analysis_csv, top_n=args.top_n)
    if not points:
        raise RuntimeError(f"No ranked points found in {analysis_csv}")

    print(f"selected_points={len(points)} sizes={sizes} bott_sizes={bott_sizes}")

    scaling_rows: list[dict[str, float | int | str]] = []
    bott_rows: list[dict[str, float | int | str]] = []

    total_jobs = len(points) * len(sizes)
    done = 0
    for p in points:
        for L in sizes:
            done += 1
            ham_sparse = build_obc_hamiltonian_sparse(
                v=p.v,
                t=p.t,
                lm=p.lm,
                w=p.w,
                j=0.0,
                nx=L,
                ny=L,
            )
            evals_nz = robust_sparse_eigs_near_zero(ham_sparse, k=args.sparse_k, per_try_timeout=args.per_try_timeout)
            m = near_zero_metrics(evals_nz)
            scaling_rows.append(
                {
                    "point_id": p.point_id,
                    "special_rank": p.special_rank,
                    "special_score": p.special_score,
                    "v": p.v,
                    "t": p.t,
                    "lm": p.lm,
                    "w": p.w,
                    "size": L,
                    **m,
                }
            )

            if L in bott_sizes:
                dim = int(ham_sparse.shape[0])
                if dim <= args.bott_max_dim:
                    bott = compute_bott_index_dense(ham_sparse=ham_sparse, nx=L, ny=L, fermi=args.fermi)
                    bott_rows.append(
                        {
                            "point_id": p.point_id,
                            "special_rank": p.special_rank,
                            "special_score": p.special_score,
                            "v": p.v,
                            "t": p.t,
                            "lm": p.lm,
                            "w": p.w,
                            "bott_size": L,
                            "bott_dim": dim,
                            "bott_index": bott,
                            "bott_status": "ok",
                            "bott_detail": "",
                        }
                    )
                else:
                    bott_rows.append(
                        {
                            "point_id": p.point_id,
                            "special_rank": p.special_rank,
                            "special_score": p.special_score,
                            "v": p.v,
                            "t": p.t,
                            "lm": p.lm,
                            "w": p.w,
                            "bott_size": L,
                            "bott_dim": dim,
                            "bott_index": "",
                            "bott_status": "skipped_dim_cap",
                            "bott_detail": f"dim={dim}>bott_max_dim={args.bott_max_dim}",
                        }
                    )

            if done % 10 == 0 or done == total_jobs:
                print(f"processed {done}/{total_jobs}")

    scaling_csv = output_dir / "finite_size_metrics.csv"
    scaling_fields = [
        "point_id",
        "special_rank",
        "special_score",
        "v",
        "t",
        "lm",
        "w",
        "size",
        "min_abs_energy",
        "count_absE_le_0p02",
        "count_absE_le_0p05",
        "near_zero_gap_estimate",
    ]
    with scaling_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scaling_fields)
        w.writeheader()
        w.writerows(scaling_rows)

    bott_csv = output_dir / "bott_size_scaling.csv"
    bott_fields = [
        "point_id",
        "special_rank",
        "special_score",
        "v",
        "t",
        "lm",
        "w",
        "bott_size",
        "bott_dim",
        "bott_index",
        "bott_status",
        "bott_detail",
    ]
    with bott_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bott_fields)
        w.writeheader()
        w.writerows(sorted(bott_rows, key=lambda r: (int(r["special_rank"]), int(r["bott_size"]))))

    plot_scaling(
        metric_rows=scaling_rows,
        metric_key="min_abs_energy",
        y_label="min |E| (near-zero sparse window)",
        save_path=figures_dir / "size_scaling_min_abs_energy.png",
    )
    plot_scaling(
        metric_rows=scaling_rows,
        metric_key="count_absE_le_0p02",
        y_label="N(|E|<=0.02) in near-zero sparse window",
        save_path=figures_dir / "size_scaling_count_absE_0p02.png",
    )
    if bott_rows:
        plot_bott_size_scaling(bott_rows=bott_rows, save_path=figures_dir / "bott_size_scaling.png")
        # use largest bott size with valid values for score-vs-bott comparison
        ok_rows = [r for r in bott_rows if str(r.get("bott_status", "")) == "ok"]
        if ok_rows:
            max_size = max(int(r["bott_size"]) for r in ok_rows)
            rows_max = [r for r in ok_rows if int(r["bott_size"]) == max_size]
            if rows_max:
                plot_bott_vs_score(bott_rows=rows_max, save_path=figures_dir / "bott_vs_special_score.png")

    summary_txt = output_dir / "summary.txt"
    lines = [
        "Finite-size scaling + real-space Bott validation",
        f"points_selected={len(points)}",
        f"sizes={sizes}",
        f"bott_sizes={bott_sizes}",
        f"bott_max_dim={args.bott_max_dim}",
        f"fermi={args.fermi}",
        f"sparse_k={args.sparse_k}",
        "",
        f"scaling_csv={scaling_csv}",
        f"bott_csv={bott_csv}",
        f"figures_dir={figures_dir}",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"done points={len(points)} jobs={total_jobs}")
    print(f"scaling_csv={scaling_csv}")
    print(f"bott_csv={bott_csv}")
    print(f"figures_dir={figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finite-size scaling and real-space Bott checks on top special points."
    )
    parser.add_argument(
        "--analysis-csv",
        default="outputs/first_stage_band_ribbon/obc_special_points_analysis.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/finite_size_realspace",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--sizes", default="12,16,20")
    parser.add_argument("--bott-sizes", default="12,16")
    parser.add_argument("--bott-max-dim", type=int, default=2200)
    parser.add_argument("--fermi", type=float, default=0.0)
    parser.add_argument("--sparse-k", type=int, default=96)
    parser.add_argument("--per-try-timeout", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
