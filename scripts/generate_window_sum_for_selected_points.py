from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_all_points_full_package import robust_sparse_eigs
from scripts.run_scan import build_obc_hamiltonian_sparse
from scripts.spatial_projectors import build_cell_probability_grid
from scripts.state_selection import detect_ribbon_window

matplotlib.use("Agg")


def load_selected_points(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_window_sum(prob_grid: np.ndarray, point_id: str, low: float, high: float, n_states: int, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(prob_grid, origin="lower", cmap="magma")
    peak = np.unravel_index(int(np.argmax(prob_grid)), prob_grid.shape)
    ax.scatter([peak[1]], [peak[0]], marker="x", c="cyan", s=40, linewidths=1.2)
    ax.set_title(
        f"{point_id}\nWindow-summed |psi|^2, [{low:.3e}, {high:.3e}], N={n_states}",
        fontsize=9,
    )
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    fig.colorbar(im, ax=ax, label="Summed probability density")
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    package_dir = root / args.package_dir
    points_dir = package_dir / "points"
    selected_csv = package_dir / "analysis" / "prl_special_points_selected.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(f"Selected CSV not found: {selected_csv}")
    if not points_dir.exists():
        raise FileNotFoundError(f"Point folder not found: {points_dir}")

    rows = load_selected_points(selected_csv)
    summary_rows: list[dict[str, str | int | float]] = []

    for idx, row in enumerate(rows, start=1):
        point_id = row["point_id"]
        v = float(row["v"])
        t = float(row["t"])
        lm = float(row["lm"])
        pdir = points_dir / point_id
        if not pdir.exists():
            continue

        ribbon = detect_ribbon_window(
            v=v,
            t=t,
            lm=lm,
            nx=args.ribbon_nx,
            nk=args.ribbon_nk,
            edge_cells=args.ribbon_edge_cells,
            edge_threshold=args.ribbon_edge_threshold,
        )
        requested_low = float(ribbon.low)
        requested_high = float(ribbon.high)

        ham = build_obc_hamiltonian_sparse(
            v=v,
            t=t,
            lm=lm,
            w=1.0,
            j=0.0,
            nx=args.obc_nx,
            ny=args.obc_ny,
        )
        eigvals, eigvecs = robust_sparse_eigs(ham_sparse=ham, base_k=args.obc_k, min_candidate_states=args.obc_min_k)
        eigvals = np.real(eigvals)

        used_low = max(requested_low, float(np.min(eigvals)))
        used_high = min(requested_high, float(np.max(eigvals)))
        sel_idx = np.where((eigvals >= used_low) & (eigvals <= used_high))[0]

        # Fallback: if ribbon window is not covered by near-zero sparse set, use a symmetric local window around ribbon center.
        if sel_idx.size == 0:
            center = float(ribbon.center)
            half = float(args.fallback_half_width)
            used_low = center - half
            used_high = center + half
            sel_idx = np.where((eigvals >= used_low) & (eigvals <= used_high))[0]

        if sel_idx.size == 0:
            sel_idx = np.array([int(np.argmin(np.abs(eigvals - float(ribbon.center))))], dtype=int)
            used_low = float(eigvals[sel_idx[0]])
            used_high = float(eigvals[sel_idx[0]])

        accum = np.zeros((args.obc_ny, args.obc_nx), dtype=float)
        for k in sel_idx:
            accum += build_cell_probability_grid(eigvecs[:, int(k)], nx=args.obc_nx, ny=args.obc_ny)
        total = float(np.sum(accum))
        if total > 0:
            accum /= total

        plot_path = pdir / "obc_wavefunction_window_sum.png"
        plot_window_sum(
            prob_grid=accum,
            point_id=point_id,
            low=used_low,
            high=used_high,
            n_states=int(sel_idx.size),
            save_path=plot_path,
        )

        selection_path = pdir / "window_sum_selection.txt"
        selection_path.write_text(
            "\n".join(
                [
                    f"point_id={point_id}",
                    f"requested_window_low={requested_low:.16e}",
                    f"requested_window_high={requested_high:.16e}",
                    f"used_window_low={used_low:.16e}",
                    f"used_window_high={used_high:.16e}",
                    f"selected_state_count={int(sel_idx.size)}",
                    "selected_energies=" + ",".join(f"{float(eigvals[i]):.10e}" for i in sel_idx),
                ]
            ),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "point_id": point_id,
                "v": v,
                "t": t,
                "lm": lm,
                "requested_window_low": requested_low,
                "requested_window_high": requested_high,
                "used_window_low": used_low,
                "used_window_high": used_high,
                "selected_state_count": int(sel_idx.size),
                "min_selected_energy": float(np.min(eigvals[sel_idx])),
                "max_selected_energy": float(np.max(eigvals[sel_idx])),
            }
        )
        if idx % 10 == 0:
            print(f"processed {idx}/{len(rows)}")

    summary_csv = package_dir / "analysis" / "window_sum_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "point_id",
        "v",
        "t",
        "lm",
        "requested_window_low",
        "requested_window_high",
        "used_window_low",
        "used_window_high",
        "selected_state_count",
        "min_selected_energy",
        "max_selected_energy",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"done points={len(summary_rows)}")
    print(f"summary={summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate window-summed wavefunction maps for PRL selected points.")
    parser.add_argument("--package-dir", default="outputs/prl_special_points/selected_points_package")
    parser.add_argument("--obc-nx", type=int, default=20)
    parser.add_argument("--obc-ny", type=int, default=20)
    parser.add_argument("--obc-k", type=int, default=160)
    parser.add_argument("--obc-min-k", type=int, default=24)
    parser.add_argument("--ribbon-nx", type=int, default=20)
    parser.add_argument("--ribbon-nk", type=int, default=101)
    parser.add_argument("--ribbon-edge-cells", type=int, default=2)
    parser.add_argument("--ribbon-edge-threshold", type=float, default=0.45)
    parser.add_argument("--fallback-half-width", type=float, default=0.12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
