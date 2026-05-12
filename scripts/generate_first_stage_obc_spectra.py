from __future__ import annotations

import argparse
import csv
from pathlib import Path
import signal
import sys

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import build_obc_hamiltonian_sparse, plot_obc_spectrum


def robust_sparse_eigs(
    ham_sparse,
    base_k: int,
    min_candidate_states: int,
    per_try_timeout: int = 20,
) -> np.ndarray:
    """Compute near-zero sparse eigenvalues with fallbacks and timeout guards."""

    def _timeout_handler(signum, frame):
        raise TimeoutError("eigsh timed out")

    dim = ham_sparse.shape[0]
    k_candidates: list[int] = []
    for k_try in (base_k, base_k - 16, base_k - 32, 128, 96, 80, 64, 48):
        k = max(min_candidate_states, min(int(k_try), dim - 2))
        if k not in k_candidates:
            k_candidates.append(k)

    for k in k_candidates:
        for kwargs in ({"sigma": 0.0, "which": "LM"}, {"which": "SM"}):
            old_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(max(1, int(per_try_timeout)))
                vals, _ = eigsh(ham_sparse, k=k, maxiter=1200, tol=1e-7, **kwargs)
                signal.alarm(0)
                order = np.argsort(np.real(vals))
                return np.real(vals[order])
            except ArpackNoConvergence as exc:
                signal.alarm(0)
                vals = getattr(exc, "eigenvalues", None)
                if vals is not None and len(vals) >= min_candidate_states:
                    order = np.argsort(np.real(vals))
                    return np.real(vals[order])
            except Exception:
                signal.alarm(0)
                continue
            finally:
                signal.signal(signal.SIGALRM, old_handler)

    raise RuntimeError("Failed to compute sparse eigenvalues near zero.")


def load_points(summary_csv: Path) -> list[dict[str, float | str]]:
    """Load first-stage points from band_ribbon_summary.csv."""
    rows: list[dict[str, float | str]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            rows.append(
                {
                    "point_id": row["point_id"],
                    "v": float(row["v"]),
                    "t": float(row["t"]),
                    "lm": float(row["lm"]),
                    "w": float(row["w"]),
                }
            )
    return rows


def run(args: argparse.Namespace) -> None:
    project_root = Path("/workspace")
    summary_csv = project_root / args.summary_csv
    points_root = project_root / args.points_root
    log_path = project_root / args.log_path
    out_summary_csv = project_root / args.output_summary_csv

    points = load_points(summary_csv)
    if not points:
        raise RuntimeError(f"No valid points loaded from {summary_csv}")

    points_root.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    rows_out: list[dict[str, str | float | int]] = []

    for idx, p in enumerate(points, start=1):
        point_id = str(p["point_id"])
        v = float(p["v"])
        t = float(p["t"])
        lm = float(p["lm"])
        w = float(p["w"])
        point_dir = points_root / point_id
        point_dir.mkdir(parents=True, exist_ok=True)
        obc_path = point_dir / "obc_spectrum_e_vs_index.png"

        status = "ok"
        detail = ""
        eig_count = 0

        if obc_path.exists() and not args.force:
            status = "skipped_existing"
        else:
            try:
                ham_sparse = build_obc_hamiltonian_sparse(
                    v=v,
                    t=t,
                    lm=lm,
                    w=w,
                    j=0.0,
                    nx=args.obc_nx,
                    ny=args.obc_ny,
                )
                eigvals = robust_sparse_eigs(
                    ham_sparse=ham_sparse,
                    base_k=args.obc_k,
                    min_candidate_states=args.obc_min_k,
                    per_try_timeout=args.per_try_timeout,
                )
                eig_count = int(eigvals.size)
                plot_obc_spectrum(
                    eigvals=eigvals,
                    save_path=obc_path,
                    title=f"OBC E vs index (v={v:.2f}, t={t:.2f}, lm={lm:.2f}, {args.obc_nx}x{args.obc_ny})",
                )
            except Exception as exc:
                status = "failed"
                detail = repr(exc)
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(
                        f"point={point_id} v={v:.3f} t={t:.3f} lm={lm:.3f} "
                        f"status=failed error={detail}\n"
                    )

        rows_out.append(
            {
                "point_id": point_id,
                "v": v,
                "t": t,
                "lm": lm,
                "w": w,
                "obc_nx": int(args.obc_nx),
                "obc_ny": int(args.obc_ny),
                "eigvals_count": eig_count,
                "status": status,
                "detail": detail,
                "obc_spectrum_path": str(obc_path.relative_to(project_root)),
            }
        )

        if idx % 10 == 0 or idx == len(points):
            print(f"processed {idx}/{len(points)}")

    fieldnames = [
        "point_id",
        "v",
        "t",
        "lm",
        "w",
        "obc_nx",
        "obc_ny",
        "eigvals_count",
        "status",
        "detail",
        "obc_spectrum_path",
    ]
    with out_summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    ok_count = sum(1 for r in rows_out if r["status"] == "ok")
    skip_count = sum(1 for r in rows_out if r["status"] == "skipped_existing")
    fail_count = sum(1 for r in rows_out if r["status"] == "failed")
    print(f"done total={len(rows_out)} ok={ok_count} skipped={skip_count} failed={fail_count}")
    print(f"summary={out_summary_csv}")
    if log_path.exists():
        print(f"log={log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 20x20 OBC E-vs-index plots for first-stage points.")
    parser.add_argument("--summary-csv", default="outputs/first_stage_band_ribbon/band_ribbon_summary.csv")
    parser.add_argument("--points-root", default="outputs/first_stage_band_ribbon/points")
    parser.add_argument(
        "--output-summary-csv",
        default="outputs/first_stage_band_ribbon/obc_spectrum_generation_summary.csv",
    )
    parser.add_argument("--log-path", default="outputs/first_stage_band_ribbon/obc_spectrum_generation.log")
    parser.add_argument("--obc-nx", type=int, default=20)
    parser.add_argument("--obc-ny", type=int, default=20)
    parser.add_argument("--obc-k", type=int, default=96)
    parser.add_argument("--obc-min-k", type=int, default=16)
    parser.add_argument("--per-try-timeout", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
