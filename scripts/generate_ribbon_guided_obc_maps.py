from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import build_obc_hamiltonian_sparse, build_ribbon_hamiltonian


@dataclass
class RibbonGuidedWindow:
    bulk_valence_max: float
    bulk_conduction_min: float
    edge_energy_min: float
    edge_energy_max: float
    window_low: float
    window_high: float
    edge_states_in_gap: int
    edge_states_all: int
    edge_threshold_used: float


def _robust_eigs_near_zero_sparse(ham_sparse, target_k: int = 128) -> tuple[np.ndarray, np.ndarray]:
    for k in (target_k, 112, 96, 80, 64, 48):
        k = min(max(16, k), ham_sparse.shape[0] - 2)
        for kwargs in ({"sigma": 0.0, "which": "LM"}, {"which": "SM"}):
            try:
                vals, vecs = eigsh(ham_sparse, k=k, **kwargs)
                order = np.argsort(np.real(vals))
                return np.real(vals[order]), vecs[:, order]
            except Exception:
                continue
    raise RuntimeError("Failed to compute OBC eigenpairs near zero.")


def _cell_probability_from_eigvecs(vec: np.ndarray, nx: int, ny: int) -> np.ndarray:
    cell_prob = np.zeros(nx * ny, dtype=float)
    for cell_id in range(nx * ny):
        start = cell_id * 8
        cell_prob[cell_id] = float(np.sum(np.abs(vec[start : start + 8]) ** 2))
    grid = cell_prob.reshape((ny, nx))
    total = float(np.sum(grid))
    return grid / total if total > 0 else grid


def _boundary_localization_score(grid: np.ndarray, boundary_cells: int = 2) -> float:
    ny, nx = grid.shape
    bc = max(1, min(boundary_cells, ny // 2, nx // 2))
    mask = np.zeros_like(grid, dtype=bool)
    mask[:bc, :] = True
    mask[-bc:, :] = True
    mask[:, :bc] = True
    mask[:, -bc:] = True
    return float(np.sum(grid[mask]))


def _infer_ribbon_edge_window(
    v: float,
    t: float,
    lm: float,
    nx: int = 20,
    nk: int = 101,
    edge_cells: int = 2,
    edge_threshold: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, RibbonGuidedWindow]:
    ky_values = np.linspace(-np.pi, np.pi, nk)
    ribbon_eigs: list[np.ndarray] = []
    ky_data: list[tuple[np.ndarray, np.ndarray]] = []
    all_energies: list[float] = []

    for ky in ky_values:
        ham = build_ribbon_hamiltonian(v=v, t=t, lm=lm, ky=ky, w=1.0, j=0.0, nx=nx)
        evals, evecs = np.linalg.eigh(ham)
        evals = np.real(evals)
        ribbon_eigs.append(evals)
        all_energies.extend(evals.tolist())

        # evecs columns are eigenvectors.
        probs = np.abs(evecs) ** 2  # (dim, dim)
        dim = probs.shape[0]
        cell_prob = probs.reshape(nx, 8, dim).sum(axis=1)  # (nx, dim)
        edge_weight = cell_prob[:edge_cells, :].sum(axis=0) + cell_prob[-edge_cells:, :].sum(axis=0)
        ky_data.append((evals, edge_weight))

    ribbon_eigs_arr = np.array(ribbon_eigs)

    def collect_with_threshold(threshold: float) -> tuple[float, float, list[float], list[float]]:
        valence_candidates: list[float] = []
        conduction_candidates: list[float] = []
        edge_energies_all: list[float] = []
        for evals, edge_weight in ky_data:
            edge_mask = edge_weight >= threshold
            bulk_mask = ~edge_mask
            edge_energies_all.extend(np.real(evals[edge_mask]).tolist())
            bulk_evals = np.real(evals[bulk_mask])
            bulk_neg = bulk_evals[bulk_evals < 0]
            bulk_pos = bulk_evals[bulk_evals > 0]
            if bulk_neg.size > 0:
                valence_candidates.append(float(np.max(bulk_neg)))
            if bulk_pos.size > 0:
                conduction_candidates.append(float(np.min(bulk_pos)))

        if valence_candidates and conduction_candidates:
            bulk_valence_max = float(np.max(valence_candidates))
            bulk_conduction_min = float(np.min(conduction_candidates))
        else:
            # Fallback to a narrow central window if bulk-only bands cannot be robustly separated.
            bulk_valence_max = float(np.percentile(all_energies, 49))
            bulk_conduction_min = float(np.percentile(all_energies, 51))

        if bulk_conduction_min <= bulk_valence_max:
            # Keep a sane central interval around zero.
            bulk_valence_max = min(bulk_valence_max, -0.02)
            bulk_conduction_min = max(bulk_conduction_min, 0.02)

        edge_energies_in_gap = [
            float(e)
            for e in edge_energies_all
            if bulk_valence_max <= float(e) <= bulk_conduction_min
        ]
        return bulk_valence_max, bulk_conduction_min, edge_energies_all, edge_energies_in_gap

    bulk_valence_max, bulk_conduction_min, edge_energies_all, edge_energies_in_gap = collect_with_threshold(
        edge_threshold
    )

    # If no in-gap edge branch is detected at strict threshold, relax threshold progressively.
    if not edge_energies_in_gap:
        for relaxed_threshold in (0.40, 0.35, 0.30, 0.25):
            if relaxed_threshold >= edge_threshold:
                continue
            (
                relaxed_bulk_valence_max,
                relaxed_bulk_conduction_min,
                relaxed_all,
                relaxed_in_gap,
            ) = collect_with_threshold(relaxed_threshold)
            if relaxed_in_gap:
                bulk_valence_max = relaxed_bulk_valence_max
                bulk_conduction_min = relaxed_bulk_conduction_min
                edge_energies_all = relaxed_all
                edge_energies_in_gap = relaxed_in_gap
                edge_threshold = relaxed_threshold
                break

    if edge_energies_in_gap:
        edge_energy_min = float(np.min(edge_energies_in_gap))
        edge_energy_max = float(np.max(edge_energies_in_gap))
    elif edge_energies_all:
        edge_sorted = sorted(edge_energies_all, key=lambda e: abs(e))
        sample = np.array(edge_sorted[: min(12, len(edge_sorted))], dtype=float)
        edge_energy_min = float(np.min(sample))
        edge_energy_max = float(np.max(sample))
    else:
        edge_energy_min = bulk_valence_max
        edge_energy_max = bulk_conduction_min

    span = max(1e-6, edge_energy_max - edge_energy_min)
    pad = max(0.01, 0.15 * span)
    window_low = max(bulk_valence_max, edge_energy_min - pad)
    window_high = min(bulk_conduction_min, edge_energy_max + pad)
    if window_high <= window_low:
        mid = 0.5 * (bulk_valence_max + bulk_conduction_min)
        half = 0.5 * max(0.02, bulk_conduction_min - bulk_valence_max)
        window_low, window_high = mid - half, mid + half

    info = RibbonGuidedWindow(
        bulk_valence_max=bulk_valence_max,
        bulk_conduction_min=bulk_conduction_min,
        edge_energy_min=edge_energy_min,
        edge_energy_max=edge_energy_max,
        window_low=window_low,
        window_high=window_high,
        edge_states_in_gap=len(edge_energies_in_gap),
        edge_states_all=len(edge_energies_all),
        edge_threshold_used=edge_threshold,
    )
    return ky_values, ribbon_eigs_arr, info


def _write_selection_text(
    path: Path,
    row: dict[str, str],
    info: RibbonGuidedWindow,
    selected_energies: np.ndarray,
    raw_count: int,
    score_min: float,
    score_max: float,
) -> None:
    path.write_text(
        "Ribbon-guided OBC wavefunction selection\n"
        f"params: v={float(row['v']):.2f}, t={float(row['t']):.2f}, lm={float(row['lm']):.2f}\n"
        "ribbon inference:\n"
        f"- bulk_valence_max={info.bulk_valence_max:.6e}\n"
        f"- bulk_conduction_min={info.bulk_conduction_min:.6e}\n"
        f"- edge_energy_min={info.edge_energy_min:.6e}\n"
        f"- edge_energy_max={info.edge_energy_max:.6e}\n"
        f"- selected_window=[{info.window_low:.6e}, {info.window_high:.6e}]\n"
        f"- edge_states_in_gap={info.edge_states_in_gap}\n"
        f"- edge_states_all={info.edge_states_all}\n"
        f"- edge_threshold_used={info.edge_threshold_used:.2f}\n"
        f"obc_window_candidate_count={raw_count}\n"
        f"obc_selected_count={selected_energies.size}\n"
        f"obc_selected_energy_min={float(np.min(selected_energies)):.6e}\n"
        f"obc_selected_energy_max={float(np.max(selected_energies)):.6e}\n"
        f"obc_boundary_score_min={score_min:.6f}\n"
        f"obc_boundary_score_max={score_max:.6f}\n",
        encoding="utf-8",
    )


def _plot_ribbon_guided_map(
    save_path: Path,
    ky_values: np.ndarray,
    ribbon_eigs: np.ndarray,
    info: RibbonGuidedWindow,
    obc_grid: np.ndarray,
    row: dict[str, str],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Ribbon panel
    ax0 = axes[0]
    ky_norm = ky_values / np.pi
    for band_idx in range(ribbon_eigs.shape[1]):
        ax0.plot(ky_norm, ribbon_eigs[:, band_idx], color="tab:blue", linewidth=0.5, alpha=0.85)
    ax0.axhspan(info.window_low, info.window_high, color="gold", alpha=0.2, label="selected edge window")
    ax0.axhline(info.bulk_valence_max, color="black", linestyle="--", linewidth=0.7)
    ax0.axhline(info.bulk_conduction_min, color="black", linestyle="--", linewidth=0.7)
    ax0.set_xlabel(r"$k_y/\pi$")
    ax0.set_ylabel("Energy")
    ax0.set_title("Ribbon spectrum with inferred edge window")
    ax0.grid(alpha=0.2)
    ax0.legend(loc="upper right", fontsize=7)

    # OBC panel
    ax1 = axes[1]
    im = ax1.imshow(obc_grid, origin="lower", cmap="magma")
    peak = np.unravel_index(np.argmax(obc_grid), obc_grid.shape)
    ax1.scatter([peak[1]], [peak[0]], c="cyan", marker="x", s=35, linewidths=1.1)
    ax1.set_xlabel("x cell")
    ax1.set_ylabel("y cell")
    ax1.set_title("OBC subspace wavefunction\n(ribbon-guided window)")
    fig.colorbar(im, ax=ax1, label="Probability density")

    fig.suptitle(f"v={float(row['v']):.2f}, t={float(row['t']):.2f}, lm={float(row['lm']):.2f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main() -> None:
    package_dir = Path("/workspace/outputs/physical_special_points_package")
    manifest_path = package_dir / "manifest.csv"
    points_dir = package_dir / "points"

    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    summary_rows = []

    for idx, row in enumerate(rows, start=1):
        folder = row["folder"]
        pdir = points_dir / folder
        out_png = pdir / "obc_wavefunction_ribbon_guided.png"
        out_txt = pdir / "ribbon_guided_selection.txt"

        v = float(row["v"])
        t = float(row["t"])
        lm = float(row["lm"])

        ky_values, ribbon_eigs, info = _infer_ribbon_edge_window(v=v, t=t, lm=lm, nx=20, nk=101)

        ham_sparse = build_obc_hamiltonian_sparse(v=v, t=t, lm=lm, w=1.0, j=0.0, nx=20, ny=20)
        evals, evecs = _robust_eigs_near_zero_sparse(ham_sparse, target_k=128)

        sel_mask = (evals >= info.window_low) & (evals <= info.window_high)
        selected_idx = np.where(sel_mask)[0]
        if selected_idx.size < 2:
            center = 0.5 * (info.window_low + info.window_high)
            order = np.argsort(np.abs(evals - center))
            selected_idx = order[: min(8, evals.size)]

        candidate_energies = evals[selected_idx]
        candidate_grids = np.stack(
            [_cell_probability_from_eigvecs(evecs[:, i], nx=20, ny=20) for i in selected_idx],
            axis=0,
        )
        candidate_scores = np.array([_boundary_localization_score(g, boundary_cells=2) for g in candidate_grids])
        raw_count = int(candidate_scores.size)
        if candidate_scores.size > 2:
            keep_count = min(8, max(2, candidate_scores.size // 2))
            keep_order = np.argsort(-candidate_scores)[:keep_count]
        else:
            keep_order = np.arange(candidate_scores.size)
        selected_energies = candidate_energies[keep_order]
        selected_grids = candidate_grids[keep_order]
        selected_scores = candidate_scores[keep_order]
        weights = np.square(selected_scores) + 1e-12
        weights = weights / np.sum(weights)
        obc_grid = np.tensordot(weights, selected_grids, axes=(0, 0))
        obc_grid = obc_grid / max(float(np.sum(obc_grid)), 1e-16)

        _plot_ribbon_guided_map(out_png, ky_values, ribbon_eigs, info, obc_grid, row)
        _write_selection_text(
            out_txt,
            row,
            info,
            selected_energies,
            raw_count=raw_count,
            score_min=float(np.min(candidate_scores)),
            score_max=float(np.max(candidate_scores)),
        )

        summary_rows.append(
            {
                "folder": folder,
                "v": row["v"],
                "t": row["t"],
                "lm": row["lm"],
                "bulk_valence_max": f"{info.bulk_valence_max:.6e}",
                "bulk_conduction_min": f"{info.bulk_conduction_min:.6e}",
                "window_low": f"{info.window_low:.6e}",
                "window_high": f"{info.window_high:.6e}",
                "edge_states_in_gap": str(info.edge_states_in_gap),
                "edge_threshold_used": f"{info.edge_threshold_used:.2f}",
                "obc_selected_count": str(selected_energies.size),
                "obc_selected_e_min": f"{float(np.min(selected_energies)):.6e}",
                "obc_selected_e_max": f"{float(np.max(selected_energies)):.6e}",
            }
        )

        if idx % 5 == 0:
            print(f"processed {idx}/{len(rows)}")

    summary_csv = package_dir / "ribbon_guided_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "v",
                "t",
                "lm",
                "bulk_valence_max",
                "bulk_conduction_min",
                "window_low",
                "window_high",
                "edge_states_in_gap",
                "edge_threshold_used",
                "obc_selected_count",
                "obc_selected_e_min",
                "obc_selected_e_max",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    readme = package_dir / "README.txt"
    text = readme.read_text(encoding="utf-8")
    if "obc_wavefunction_ribbon_guided.png" not in text:
        text += "\nribbon-guided OBC files (new):\n"
        text += "- obc_wavefunction_ribbon_guided.png: ribbon-based edge-window + OBC subspace wavefunction map\n"
        text += "- ribbon_guided_selection.txt: inferred bulk gap edges, selected window, and selected OBC energies\n"
        text += "- ribbon_guided_summary.csv: package-level summary of ribbon-guided windows\n"
        readme.write_text(text, encoding="utf-8")

    zip_path = shutil.make_archive("/workspace/outputs/physical_special_points_package", "zip", root_dir=str(package_dir))
    print(f"summary={summary_csv}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
