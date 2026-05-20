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

from scripts.run_scan import (
    build_obc_hamiltonian_sparse,
    compute_bulk_gap,
    compute_chern_number,
    compute_dynamic_w,
    load_xtype_model,
    set_model_params,
)

matplotlib.use("Agg")


@dataclass
class CorridorCenter:
    rank: int
    point_id: str
    v: float
    t: float
    lm: float


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid rank list parsed.")
    return vals


def build_values(center: float, width: float, step: float, lo: float, hi: float) -> list[float]:
    start = max(lo, center - width)
    stop = min(hi, center + width)
    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    if vals and vals[-1] < stop - 1e-9:
        vals.append(round(stop, 10))
    return sorted(set(vals))


def load_centers(analysis_csv: Path, ranks: list[int]) -> list[CorridorCenter]:
    lookup: dict[int, CorridorCenter] = {}
    with analysis_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank_raw = row.get("special_rank", "").strip()
            if not rank_raw:
                continue
            rank = int(rank_raw)
            if rank not in ranks:
                continue
            lookup[rank] = CorridorCenter(
                rank=rank,
                point_id=row["point_id"],
                v=float(row["v"]),
                t=float(row["t"]),
                lm=float(row["lm"]),
            )
    out = [lookup[r] for r in ranks if r in lookup]
    return out


def robust_sparse_eigs_near_zero(ham_sparse, k: int, per_try_timeout: int = 20) -> np.ndarray:
    def _timeout_handler(signum, frame):
        raise TimeoutError("eigsh timed out")

    dim = ham_sparse.shape[0]
    k = max(12, min(int(k), dim - 2))
    candidates = []
    for kt in (k, k - 16, k + 16, 96, 80, 64):
        kk = max(12, min(int(kt), dim - 2))
        if kk not in candidates:
            candidates.append(kk)

    for kk in candidates:
        for kwargs in ({"sigma": 0.0, "which": "LM"}, {"which": "SM"}):
            old_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(max(1, int(per_try_timeout)))
                vals, _ = eigsh(ham_sparse, k=kk, maxiter=1200, tol=1e-7, **kwargs)
                signal.alarm(0)
                return np.sort(np.real(vals))
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


def near_zero_metrics(evals: np.ndarray) -> tuple[float, int]:
    abs_e = np.abs(evals)
    return float(np.min(abs_e)), int(np.sum(abs_e <= 0.02))


def rows_to_grid(rows: list[dict[str, float]], x_key: str, y_key: str, z_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = sorted({float(r[x_key]) for r in rows})
    ys = sorted({float(r[y_key]) for r in rows})
    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_j = {y: j for j, y in enumerate(ys)}
    z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    for r in rows:
        x = float(r[x_key])
        y = float(r[y_key])
        z[y_to_j[y], x_to_i[x]] = float(r[z_key])
    return np.array(xs), np.array(ys), z


def plot_heatmap(rows: list[dict[str, float]], z_key: str, title: str, save_path: Path, cmap: str = "viridis") -> None:
    xs, ys, z = rows_to_grid(rows, "v", "lm", z_key)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap=cmap,
    )
    ax.set_xlabel("v")
    ax.set_ylabel("lm")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=z_key)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    analysis_csv = root / args.analysis_csv
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_xtype_model(root / "models" / args.model_file)

    ranks = parse_int_list(args.corridor_ranks)
    centers = load_centers(analysis_csv=analysis_csv, ranks=ranks)
    if not centers:
        raise RuntimeError("No corridor centers found from analysis CSV.")

    summary_rows: list[dict[str, float | int | str]] = []
    for c in centers:
        corridor_dir = output_dir / f"rank_{c.rank:02d}_{c.point_id}"
        corridor_dir.mkdir(parents=True, exist_ok=True)
        v_values = build_values(c.v, args.v_width, args.v_step, lo=0.0, hi=2.0)
        lm_values = build_values(c.lm, args.lm_width, args.lm_step, lo=0.0, hi=0.5)
        rows: list[dict[str, float]] = []

        total = len(v_values) * len(lm_values)
        done = 0
        for v in v_values:
            for lm in lm_values:
                done += 1
                w = compute_dynamic_w(v)
                set_model_params(model, v=v, t=c.t, lm=lm, w=w, j=0.0)
                gap = compute_bulk_gap(model, nk=args.gap_nk, n_occ=4)
                chern = compute_chern_number(model, nk=args.chern_nk, n_occ=4)

                ham_sparse = build_obc_hamiltonian_sparse(
                    v=v,
                    t=c.t,
                    lm=lm,
                    w=w,
                    j=0.0,
                    nx=args.obc_size,
                    ny=args.obc_size,
                )
                evals_nz = robust_sparse_eigs_near_zero(ham_sparse, k=args.obc_sparse_k, per_try_timeout=args.per_try_timeout)
                min_abs, count_0p02 = near_zero_metrics(evals_nz)
                rows.append(
                    {
                        "center_rank": float(c.rank),
                        "center_point_id": c.point_id,
                        "v": float(v),
                        "t": float(c.t),
                        "lm": float(lm),
                        "w": float(w),
                        "gap": float(gap),
                        "chern": float(chern),
                        "obc_min_abs_energy": float(min_abs),
                        "obc_count_absE_le_0p02": float(count_0p02),
                    }
                )
                if done % 20 == 0 or done == total:
                    print(f"corridor rank={c.rank} processed {done}/{total}")

        csv_path = corridor_dir / "corridor_scan.csv"
        fields = [
            "center_rank",
            "center_point_id",
            "v",
            "t",
            "lm",
            "w",
            "gap",
            "chern",
            "obc_min_abs_energy",
            "obc_count_absE_le_0p02",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        plot_heatmap(
            rows=rows,
            z_key="gap",
            title=f"Corridor rank {c.rank}: bulk gap",
            save_path=corridor_dir / "gap_map.png",
            cmap="coolwarm",
        )
        plot_heatmap(
            rows=rows,
            z_key="chern",
            title=f"Corridor rank {c.rank}: Chern",
            save_path=corridor_dir / "chern_map.png",
            cmap="RdBu_r",
        )
        plot_heatmap(
            rows=rows,
            z_key="obc_min_abs_energy",
            title=f"Corridor rank {c.rank}: OBC min |E|",
            save_path=corridor_dir / "obc_min_abs_map.png",
            cmap="magma",
        )

        arr_gap = np.array([float(r["gap"]) for r in rows], dtype=float)
        arr_obc = np.array([float(r["obc_min_abs_energy"]) for r in rows], dtype=float)
        summary_rows.append(
            {
                "center_rank": c.rank,
                "center_point_id": c.point_id,
                "center_v": c.v,
                "center_t": c.t,
                "center_lm": c.lm,
                "grid_points": len(rows),
                "gap_min": float(np.min(arr_gap)),
                "gap_max": float(np.max(arr_gap)),
                "obc_min_abs_min": float(np.min(arr_obc)),
                "obc_min_abs_max": float(np.max(arr_obc)),
                "corridor_dir": str(corridor_dir.relative_to(root)),
            }
        )

    summary_csv = output_dir / "corridor_summary.csv"
    summary_fields = [
        "center_rank",
        "center_point_id",
        "center_v",
        "center_t",
        "center_lm",
        "grid_points",
        "gap_min",
        "gap_max",
        "obc_min_abs_min",
        "obc_min_abs_max",
        "corridor_dir",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    summary_txt = output_dir / "summary.txt"
    lines = [
        "Transition corridor local dense scan",
        f"corridor_ranks={ranks}",
        f"model_file={args.model_file}",
        f"v_width={args.v_width}, v_step={args.v_step}",
        f"lm_width={args.lm_width}, lm_step={args.lm_step}",
        f"gap_nk={args.gap_nk}, chern_nk={args.chern_nk}",
        f"obc_size={args.obc_size}, obc_sparse_k={args.obc_sparse_k}",
        "",
        f"summary_csv={summary_csv}",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"done corridors={len(summary_rows)}")
    print(f"summary_csv={summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local corridor scans around top special points.")
    parser.add_argument(
        "--analysis-csv",
        default="outputs/first_stage_band_ribbon/obc_special_points_analysis.csv",
    )
    parser.add_argument("--output-dir", default="outputs/transition_corridors")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--corridor-ranks", default="1,2")
    parser.add_argument("--v-width", type=float, default=0.2)
    parser.add_argument("--v-step", type=float, default=0.04)
    parser.add_argument("--lm-width", type=float, default=0.2)
    parser.add_argument("--lm-step", type=float, default=0.04)
    parser.add_argument("--gap-nk", type=int, default=11)
    parser.add_argument("--chern-nk", type=int, default=15)
    parser.add_argument("--obc-size", type=int, default=16)
    parser.add_argument("--obc-sparse-k", type=int, default=96)
    parser.add_argument("--per-try-timeout", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
