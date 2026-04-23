from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import sys
import signal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import (
    build_obc_hamiltonian_sparse,
    compute_band_data,
    format_param_token,
    load_xtype_model,
    plot_band_structure,
    plot_obc_spectrum,
    plot_ribbon_spectrum,
    set_model_params,
)
from scripts.spatial_projectors import build_cell_probability_grid, build_region_projectors, compute_region_weights
from scripts.state_selection import choose_best_edge_state, detect_ribbon_window


matplotlib.use("Agg")


def robust_sparse_eigs(ham_sparse, base_k: int, min_candidate_states: int, per_try_timeout: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compute near-zero sparse eigenpairs with fallbacks."""
    def _timeout_handler(signum, frame):
        raise TimeoutError("eigsh timed out")

    dim = ham_sparse.shape[0]
    k_candidates = []
    for k_try in (base_k, base_k - 32, base_k - 64, 128, 96, 80, 64, 48):
        k = max(min_candidate_states, min(int(k_try), dim - 2))
        if k not in k_candidates:
            k_candidates.append(k)
    for k in k_candidates:
        for kwargs in ({"sigma": 0.0, "which": "LM"}, {"which": "SM"}):
            try:
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(max(1, int(per_try_timeout)))
                vals, vecs = eigsh(ham_sparse, k=k, maxiter=1200, tol=1e-7, **kwargs)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                order = np.argsort(np.real(vals))
                return np.real(vals[order]), vecs[:, order]
            except ArpackNoConvergence as exc:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                vals = getattr(exc, "eigenvalues", None)
                vecs = getattr(exc, "eigenvectors", None)
                if vals is not None and vecs is not None and len(vals) >= min_candidate_states:
                    order = np.argsort(np.real(vals))
                    return np.real(vals[order]), vecs[:, order]
            except Exception:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                continue
    raise RuntimeError("Failed to compute sparse eigenpairs near zero.")


def plot_filtered_wavefunction(grid: np.ndarray, energy: float, save_path: Path, title: str) -> None:
    """Plot selected filtered wavefunction density."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(grid, origin="lower", cmap="magma")
    peak = np.unravel_index(np.argmax(grid), grid.shape)
    ax.scatter([peak[1]], [peak[0]], marker="x", c="cyan", s=40, linewidths=1.2)
    ax.set_title(f"{title}\nselected E={energy:.3e}")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    fig.colorbar(im, ax=ax, label="Probability density")
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def load_points(results_csv: Path) -> list[dict[str, float]]:
    """Load all scanned parameter points with invariants from results.csv."""
    rows = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "v": float(row["v"]),
                    "t": float(row["t"]),
                    "lm": float(row["lm"]),
                    "gap": float(row["gap"]),
                    "chern": float(row["chern"]),
                    "wilson": float(row["Wilson_loop"]),
                    "z2": int(float(row["Z2_topology"])),
                }
            )
    return rows


def point_name(v: float, t: float, lm: float) -> str:
    return f"v_{format_param_token(v)}_t_{format_param_token(t)}_lm_{format_param_token(lm)}"


def run(args: argparse.Namespace) -> None:
    project_root = Path("/workspace")
    model = load_xtype_model(project_root / "models" / args.model_file)
    rows = load_points(project_root / args.results_csv)
    if args.point_limit > 0:
        rows = rows[: args.point_limit]

    output_root = project_root / args.output_dir
    points_dir = output_root / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_root / "run.log"
    summary_path = output_root / "all_points_summary.csv"

    projectors = build_region_projectors(
        nx=args.obc_nx,
        ny=args.obc_ny,
        edge_width=args.edge_width,
        corner_size=args.corner_size,
    )

    summary_rows: list[dict[str, float | int | str]] = []
    required_names = [
        "band.png",
        "ribbon.png",
        "obc_spectrum_e_vs_index.png",
        "obc_wavefunction_filtered.png",
        "point_summary.json",
    ]

    for idx, row in enumerate(rows, start=1):
        v, t, lm = row["v"], row["t"], row["lm"]
        pname = point_name(v, t, lm)
        pdir = points_dir / pname
        pdir.mkdir(parents=True, exist_ok=True)

        band_path = pdir / "band.png"
        ribbon_path = pdir / "ribbon.png"
        obc_spec_path = pdir / "obc_spectrum_e_vs_index.png"
        wf_filtered_path = pdir / "obc_wavefunction_filtered.png"
        info_path = pdir / "point_summary.json"

        if all((pdir / nm).exists() for nm in required_names):
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            summary_rows.append(info["summary_row"])
            if idx % 25 == 0:
                print(f"skip {idx}/{len(rows)}")
            continue

        try:
            # Bulk band
            set_model_params(model, v=v, t=t, lm=lm, w=1.0, j=0.0)
            band_data = compute_band_data(model)
            plot_band_structure(
                eigvals=band_data,
                save_path=band_path,
                title=f"Band (v={v:.2f}, t={t:.2f}, lm={lm:.2f})",
            )

            # Ribbon and ribbon-guided window
            ribbon_window = detect_ribbon_window(
                v=v,
                t=t,
                lm=lm,
                nx=args.ribbon_nx,
                nk=args.ribbon_nk,
                edge_cells=args.ribbon_edge_cells,
                edge_threshold=args.ribbon_edge_threshold,
            )
            plot_ribbon_spectrum(
                ky_values=ribbon_window.ky_values,
                eigvals=ribbon_window.eigvals,
                save_path=ribbon_path,
                title=f"Ribbon (v={v:.2f}, t={t:.2f}, lm={lm:.2f})",
            )

            # OBC and E vs index
            ham_sparse = build_obc_hamiltonian_sparse(
                v=v,
                t=t,
                lm=lm,
                w=1.0,
                j=0.0,
                nx=args.obc_nx,
                ny=args.obc_ny,
            )
            eigvals, eigvecs = robust_sparse_eigs(
                ham_sparse=ham_sparse,
                base_k=args.obc_k,
                min_candidate_states=args.obc_min_k,
            )
            plot_obc_spectrum(
                eigvals=eigvals,
                save_path=obc_spec_path,
                title=f"OBC E vs index (v={v:.2f}, t={t:.2f}, lm={lm:.2f})",
            )

            # Filtered wavefunction distribution: best edge-localized state near ribbon-guided center
            half_width = max(args.filtered_half_width_min, 0.5 * abs(ribbon_window.high - ribbon_window.low))
            sel = choose_best_edge_state(
                eigvals=eigvals,
                eigvecs=eigvecs,
                projectors=projectors,
                center=ribbon_window.center,
                half_width=half_width,
                fallback_half_width=args.filtered_fallback_half_width,
            )
            prob_grid = build_cell_probability_grid(sel.vector, nx=args.obc_nx, ny=args.obc_ny)
            weights = compute_region_weights(prob_grid, projectors)
            plot_filtered_wavefunction(
                grid=prob_grid,
                energy=sel.energy,
                save_path=wf_filtered_path,
                title=f"Filtered wavefunction (best edge state)",
            )

            summary_row = {
                "point_id": pname,
                "v": v,
                "t": t,
                "lm": lm,
                "gap": row["gap"],
                "chern": row["chern"],
                "z2": row["z2"],
                "wilson": row["wilson"],
                "selected_energy": float(sel.energy),
                "selected_index": int(sel.index),
                "selected_edge_weight": float(weights["W_edge_total"]),
                "selected_corner_weight": float(weights["W_corner"]),
                "selected_bulk_weight": float(weights["W_bulk"]),
                "ribbon_window_low": float(ribbon_window.low),
                "ribbon_window_high": float(ribbon_window.high),
                "ribbon_window_center": float(ribbon_window.center),
            }

            info = {
                "summary_row": summary_row,
                "invariants": {
                    "gap": row["gap"],
                    "chern": row["chern"],
                    "z2": row["z2"],
                    "wilson": row["wilson"],
                },
                "selected_state": {
                    "method": "best_edge_state_near_ribbon_window",
                    "energy": float(sel.energy),
                    "index": int(sel.index),
                    "window_low": float(sel.window_low),
                    "window_high": float(sel.window_high),
                    "weights": weights,
                },
                "artifacts": {
                    "band": "band.png",
                    "ribbon": "ribbon.png",
                    "obc_spectrum": "obc_spectrum_e_vs_index.png",
                    "obc_wavefunction_filtered": "obc_wavefunction_filtered.png",
                },
            }
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_rows.append(summary_row)
        except Exception as exc:
            with logs_path.open("a", encoding="utf-8") as lf:
                lf.write(f"failed point={pname} error={exc!r}\n")
        if idx % 25 == 0:
            print(f"processed {idx}/{len(rows)}")

    # Summary table
    fieldnames = [
        "point_id",
        "v",
        "t",
        "lm",
        "gap",
        "chern",
        "z2",
        "wilson",
        "selected_energy",
        "selected_index",
        "selected_edge_weight",
        "selected_corner_weight",
        "selected_bulk_weight",
        "ribbon_window_low",
        "ribbon_window_high",
        "ribbon_window_center",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    zip_base = project_root / args.zip_base
    zip_base.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=str(output_root.parent),
        base_dir=output_root.name,
    )
    print(f"done rows={len(summary_rows)}")
    print(f"output_dir={output_root}")
    print(f"zip={zip_base}.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full per-point package for all scanned points.")
    parser.add_argument("--results-csv", default="outputs/results.csv")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-dir", default="outputs/all_points_full_package")
    parser.add_argument("--zip-base", default="outputs/all_points_full_package")
    parser.add_argument("--point-limit", type=int, default=0)
    parser.add_argument("--obc-nx", type=int, default=20)
    parser.add_argument("--obc-ny", type=int, default=20)
    parser.add_argument("--obc-k", type=int, default=160)
    parser.add_argument("--obc-min-k", type=int, default=24)
    parser.add_argument("--ribbon-nx", type=int, default=20)
    parser.add_argument("--ribbon-nk", type=int, default=101)
    parser.add_argument("--ribbon-edge-cells", type=int, default=2)
    parser.add_argument("--ribbon-edge-threshold", type=float, default=0.45)
    parser.add_argument("--edge-width", type=int, default=2)
    parser.add_argument("--corner-size", type=int, default=3)
    parser.add_argument("--filtered-half-width-min", type=float, default=0.06)
    parser.add_argument("--filtered-fallback-half-width", type=float, default=0.12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
