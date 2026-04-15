from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.degeneracy_rotation import find_degenerate_groups, rotate_max_localized_state
from scripts.report_builder import build_report_pdf
from scripts.spatial_projectors import (
    RegionProjectors,
    build_cell_probability_grid,
    build_region_projectors,
    compute_region_weights,
    compute_side_weights,
    spatial_ldos_from_indices,
)
from scripts.state_selection import (
    MethodSelection,
    choose_best_corner_state,
    choose_best_edge_state,
    choose_best_side_states,
    choose_old_mode,
    detect_ribbon_window,
)
from scripts.run_scan import build_obc_hamiltonian_sparse


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def robust_sparse_eigs(ham_sparse, base_k: int, min_candidate_states: int) -> tuple[np.ndarray, np.ndarray]:
    dim = ham_sparse.shape[0]
    k_list = []
    for k_try in (base_k, base_k - 32, base_k - 64, base_k - 96, 128, 96, 80, 64, 48):
        if k_try is None:
            continue
        k = max(min_candidate_states, min(int(k_try), dim - 2))
        if k not in k_list:
            k_list.append(k)
    for k in k_list:
        for kwargs in ({"sigma": 0.0, "which": "LM"}, {"which": "SM"}):
            try:
                from scipy.sparse.linalg import eigsh

                vals, vecs = eigsh(ham_sparse, k=k, **kwargs)
                order = np.argsort(np.real(vals))
                return np.real(vals[order]), vecs[:, order]
            except Exception:
                continue
    raise RuntimeError("Unable to obtain near-zero sparse eigenpairs")


def save_single_map(path: Path, grid: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(grid, origin="lower", cmap="magma")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_ribbon_reference(path: Path, ky_values: np.ndarray, eigvals: np.ndarray, low: float, high: float, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for b in range(eigvals.shape[1]):
        ax.plot(ky_values / np.pi, eigvals[:, b], color="tab:blue", linewidth=0.45, alpha=0.9)
    ax.axhspan(low, high, color="gold", alpha=0.2, label="ribbon window")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_method_compare(path: Path, maps: list[tuple[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, (name, grid) in zip(axes, maps):
        im = ax.imshow(grid, origin="lower", cmap="magma")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def point_id(row: dict[str, str]) -> str:
    return row["folder"]


def role_from_weights(edge_weight: float, corner_weight: float, bulk_weight: float) -> str:
    if corner_weight >= 0.32:
        return "corner_dominated"
    if edge_weight >= 0.45:
        return "edge_dominated"
    if bulk_weight >= 0.55:
        return "bulk_mixed"
    return "intermediate_mixed"


def best_method_name(old_w: float, best_edge_w: float, best_corner_w: float) -> str:
    candidates = {
        "old_mode": old_w,
        "best_edge_state": best_edge_w,
        "best_corner_state": best_corner_w,
    }
    return max(candidates, key=candidates.get)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_transition_figures(output_root: Path, summary_rows: list[dict]) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def to_arr(key: str) -> np.ndarray:
        return np.array([float(r[key]) for r in summary_rows], dtype=float)

    v = to_arr("v")
    t = to_arr("t")
    lm = to_arr("lm")
    delta_edge = to_arr("edge_gain_vs_old")
    delta_corner = to_arr("corner_gain_vs_old")
    z2 = np.array([int(r["Z2"]) for r in summary_rows], dtype=int)
    cat = np.array([r["category"] for r in summary_rows], dtype=object)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sc = ax.scatter(v, t, c=delta_edge, cmap="RdBu_r", s=55, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("v")
    ax.set_ylabel("t")
    ax.set_title("transition_corridor_A: edge gain vs old")
    fig.colorbar(sc, ax=ax, label="best_edge_weight - old_edge_weight")
    fig.tight_layout()
    fig.savefig(fig_dir / "transition_corridor_A.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sc = ax.scatter(t, lm, c=delta_corner, cmap="coolwarm", s=55, edgecolors="black", linewidths=0.3)
    ax.set_xlabel("t")
    ax.set_ylabel("lm")
    ax.set_title("transition_corridor_B: corner gain vs old")
    fig.colorbar(sc, ax=ax, label="best_corner_weight - old_corner_weight")
    fig.tight_layout()
    fig.savefig(fig_dir / "transition_corridor_B.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for label, marker in (("robust_topological", "o"), ("near_transition", "^")):
        m = cat == label
        ax.scatter(to_arr("old_edge_weight")[m], to_arr("best_edge_weight")[m], marker=marker, s=58, alpha=0.85, label=label)
    lim = [min(to_arr("old_edge_weight").min(), to_arr("best_edge_weight").min()), max(to_arr("old_edge_weight").max(), to_arr("best_edge_weight").max())]
    ax.plot(lim, lim, "--", color="black", linewidth=0.8)
    ax.set_xlabel("old_edge_weight")
    ax.set_ylabel("best_edge_weight")
    ax.set_title("robust_vs_transition_compare")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "robust_vs_transition_compare.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    methods = ["old_mode", "best_edge_state", "best_corner_state", "rotated_edge_state", "rotated_corner_state"]
    values = []
    for m in methods:
        if m == "old_mode":
            values.append(float(np.mean(to_arr("old_edge_weight"))))
        elif m == "best_edge_state":
            values.append(float(np.mean(to_arr("best_edge_weight"))))
        elif m == "best_corner_state":
            values.append(float(np.mean(to_arr("best_corner_weight"))))
        elif m == "rotated_edge_state":
            values.append(float(np.mean(to_arr("rotated_edge_weight"))))
        else:
            values.append(float(np.mean(to_arr("rotated_corner_weight"))))
    ax.bar(np.arange(len(methods)), values, color=["#999999", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"])
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("mean localization weight")
    ax.set_title("state_selection_method_compare")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "state_selection_method_compare.png", dpi=180)
    plt.close(fig)


def process_point(row: dict[str, str], cfg: dict, package_dir: Path, output_root: Path) -> tuple[dict, list[dict]]:
    point = point_id(row)
    per_dir = output_root / "per_point" / point
    per_dir.mkdir(parents=True, exist_ok=True)

    v = float(row["v"])
    t = float(row["t"])
    lm = float(row["lm"])
    category = row["category"]
    gap = float(row["gap"])
    z2 = int(float(row["z2"]))
    chern = float(row["chern"])
    wilson = float(row["wilson"])

    nx = int(cfg["obc"]["nx"])
    ny = int(cfg["obc"]["ny"])
    base_k = int(cfg["obc"]["base_k"])
    min_candidate_states = int(cfg["obc"]["min_candidate_states"])
    edge_width = int(cfg["geometry"]["edge_width"])
    corner_size = int(cfg["geometry"]["corner_size"])
    abs_delta_e = float(cfg["degeneracy"]["abs_delta_e"])
    rel_delta_e = float(cfg["degeneracy"]["relative_delta_e"])

    ham_sparse = build_obc_hamiltonian_sparse(v=v, t=t, lm=lm, w=1.0, j=0.0, nx=nx, ny=ny)
    eigvals, eigvecs = robust_sparse_eigs(ham_sparse, base_k=base_k, min_candidate_states=min_candidate_states)

    projectors = build_region_projectors(nx=nx, ny=ny, edge_width=edge_width, corner_size=corner_size)
    side_projectors = projectors.side_projectors()

    # Method windows.
    zero_center = float(cfg["energy_windows"]["zero_center"])
    zero_half_width = float(cfg["energy_windows"]["zero_half_width"])
    fallback_half_width = float(cfg["energy_windows"]["fallback_half_width"])
    ldos_half_width = float(cfg["energy_windows"]["ldos_half_width"])

    old_sel = choose_old_mode(eigvals, eigvecs)
    best_edge = choose_best_edge_state(
        eigvals=eigvals,
        eigvecs=eigvecs,
        projectors=projectors,
        center=zero_center,
        half_width=zero_half_width,
        fallback_half_width=fallback_half_width,
    )
    best_corner = choose_best_corner_state(
        eigvals=eigvals,
        eigvecs=eigvecs,
        projectors=projectors,
        center=zero_center,
        half_width=zero_half_width,
        fallback_half_width=fallback_half_width,
    )
    side_states = choose_best_side_states(
        eigvals=eigvals,
        eigvecs=eigvecs,
        side_projectors=side_projectors,
        center=zero_center,
        half_width=zero_half_width,
        fallback_half_width=fallback_half_width,
    )

    ribbon_window = detect_ribbon_window(
        v=v,
        t=t,
        lm=lm,
        nx=int(cfg["ribbon"]["nx"]),
        nk=int(cfg["ribbon"]["nk"]),
        edge_cells=int(cfg["ribbon"]["edge_cells"]),
        edge_threshold=float(cfg["ribbon"]["edge_threshold"]),
    )

    # Degeneracy rotation.
    deg_groups = find_degenerate_groups(eigvals=eigvals, abs_tol=abs_delta_e, rel_tol=rel_delta_e)
    rot_edge_sel = None
    rot_corner_sel = None
    deg_rows: list[dict] = []

    for gidx, g in enumerate(deg_groups):
        if len(g) < 2:
            continue
        rot_edge, edge_eigs = rotate_max_localized_state(eigvecs=eigvecs, indices=g, projector=projectors.edge)
        rot_corner, corner_eigs = rotate_max_localized_state(eigvecs=eigvecs, indices=g, projector=projectors.corner)
        center_e = float(np.mean(eigvals[g]))
        if ribbon_window.low <= center_e <= ribbon_window.high:
            if (rot_edge_sel is None) or (rot_edge_sel.weight < edge_eigs.max()):
                rot_edge_sel = MethodSelection(
                    method_name="rotated_edge_state",
                    index=int(g[0]),
                    energy=center_e,
                    vector=rot_edge,
                    weight=float(edge_eigs.max()),
                    window_low=float(min(eigvals[g])),
                    window_high=float(max(eigvals[g])),
                )
            if (rot_corner_sel is None) or (rot_corner_sel.weight < corner_eigs.max()):
                rot_corner_sel = MethodSelection(
                    method_name="rotated_corner_state",
                    index=int(g[0]),
                    energy=center_e,
                    vector=rot_corner,
                    weight=float(corner_eigs.max()),
                    window_low=float(min(eigvals[g])),
                    window_high=float(max(eigvals[g])),
                )
        deg_rows.append(
            {
                "point_id": point,
                "group_id": gidx,
                "dim": len(g),
                "E_min": float(np.min(eigvals[g])),
                "E_max": float(np.max(eigvals[g])),
                "delta_E": float(np.max(eigvals[g]) - np.min(eigvals[g])),
                "best_edge_projection_after": float(edge_eigs.max()),
                "best_corner_projection_after": float(corner_eigs.max()),
            }
        )

    # Fallback when no rotated state inside ribbon window.
    if rot_edge_sel is None:
        g = [best_edge.index]
        vec = eigvecs[:, best_edge.index]
        rot_edge_sel = MethodSelection(
            method_name="rotated_edge_state",
            index=best_edge.index,
            energy=best_edge.energy,
            vector=vec,
            weight=best_edge.weight,
            window_low=best_edge.window_low,
            window_high=best_edge.window_high,
        )
    if rot_corner_sel is None:
        vec = eigvecs[:, best_corner.index]
        rot_corner_sel = MethodSelection(
            method_name="rotated_corner_state",
            index=best_corner.index,
            energy=best_corner.energy,
            vector=vec,
            weight=best_corner.weight,
            window_low=best_corner.window_low,
            window_high=best_corner.window_high,
        )

    old_grid = build_cell_probability_grid(old_sel.vector, nx=nx, ny=ny)
    edge_grid = build_cell_probability_grid(best_edge.vector, nx=nx, ny=ny)
    corner_grid = build_cell_probability_grid(best_corner.vector, nx=nx, ny=ny)
    rot_edge_grid = build_cell_probability_grid(rot_edge_sel.vector, nx=nx, ny=ny)
    rot_corner_grid = build_cell_probability_grid(rot_corner_sel.vector, nx=nx, ny=ny)

    # LDOS around ribbon center and old-mode center for comparison.
    ldos_ribbon = spatial_ldos_from_indices(
        eigvals=eigvals,
        eigvecs=eigvecs,
        nx=nx,
        ny=ny,
        center_energy=ribbon_window.center,
        half_width=ldos_half_width,
    )
    ldos_zero = spatial_ldos_from_indices(
        eigvals=eigvals,
        eigvecs=eigvecs,
        nx=nx,
        ny=ny,
        center_energy=old_sel.energy,
        half_width=ldos_half_width,
    )

    # Save figures.
    save_single_map(per_dir / "old_selection.png", old_grid, f"old_mode E={old_sel.energy:.3e}")
    save_single_map(per_dir / "best_edge_state.png", edge_grid, f"best_edge E={best_edge.energy:.3e}")
    save_single_map(per_dir / "best_corner_state.png", corner_grid, f"best_corner E={best_corner.energy:.3e}")
    save_single_map(per_dir / "rotated_edge_state.png", rot_edge_grid, f"rotated_edge E={rot_edge_sel.energy:.3e}")
    save_single_map(per_dir / "rotated_corner_state.png", rot_corner_grid, f"rotated_corner E={rot_corner_sel.energy:.3e}")
    save_method_compare(
        per_dir / "ldos_map.png",
        [
            (f"LDOS ribbon center {ribbon_window.center:.3e}", ldos_ribbon),
            (f"LDOS old center {old_sel.energy:.3e}", ldos_zero),
        ],
    )
    save_ribbon_reference(
        per_dir / "ribbon_reference.png",
        ky_values=ribbon_window.ky_values,
        eigvals=ribbon_window.eigvals,
        low=ribbon_window.low,
        high=ribbon_window.high,
        title=f"ribbon reference: {point}",
    )
    save_method_compare(
        per_dir / "state_selection_method_compare.png",
        [
            ("old", old_grid),
            ("best_edge", edge_grid),
            ("best_corner", corner_grid),
            ("rot_edge", rot_edge_grid),
        ],
    )

    old_weights = compute_region_weights(old_grid, projectors)
    best_edge_weights = compute_region_weights(edge_grid, projectors)
    best_corner_weights = compute_region_weights(corner_grid, projectors)
    rot_edge_weights = compute_region_weights(rot_edge_grid, projectors)
    rot_corner_weights = compute_region_weights(rot_corner_grid, projectors)

    side_summary = {}
    for side_name, sel in side_states.items():
        sgrid = build_cell_probability_grid(sel.vector, nx=nx, ny=ny)
        side_w = compute_side_weights(sgrid, side_projectors)
        side_summary[side_name] = {
            "energy": sel.energy,
            "side_weight": side_w[side_name],
            "all_side_weights": side_w,
        }
        save_single_map(
            per_dir / f"optional_best_{side_name}_state.png",
            sgrid,
            f"optional {side_name} E={sel.energy:.3e}",
        )

    comparison = {
        "point_id": point,
        "params": {"v": v, "t": t, "lm": lm},
        "category": category,
        "invariants": {"gap": gap, "z2": z2, "chern": chern, "wilson": wilson},
        "window": {
            "old_center": old_sel.energy,
            "zero_window": [zero_center - zero_half_width, zero_center + zero_half_width],
            "ribbon_window": [ribbon_window.low, ribbon_window.high],
            "ribbon_center": ribbon_window.center,
        },
        "weights": {
            "old": old_weights,
            "best_edge": best_edge_weights,
            "best_corner": best_corner_weights,
            "rotated_edge": rot_edge_weights,
            "rotated_corner": rot_corner_weights,
        },
        "side_states": side_summary,
        "methods": {
            "old_mode_energy": old_sel.energy,
            "best_edge_energy": best_edge.energy,
            "best_corner_energy": best_corner.energy,
            "rotated_edge_energy": rot_edge_sel.energy,
            "rotated_corner_energy": rot_corner_sel.energy,
        },
        "degenerate_groups": deg_rows,
    }
    (per_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "point_id": point,
        "v": v,
        "t": t,
        "lm": lm,
        "category": category,
        "gap": gap,
        "Z2": z2,
        "Chern": chern,
        "Wilson": wilson,
        "old_selected_mode": old_sel.index,
        "old_energy": old_sel.energy,
        "old_edge_weight": old_weights["W_edge_total"],
        "old_corner_weight": old_weights["W_corner"],
        "best_edge_energy": best_edge.energy,
        "best_edge_weight": best_edge_weights["W_edge_total"],
        "best_corner_energy": best_corner.energy,
        "best_corner_weight": best_corner_weights["W_corner"],
        "rotated_edge_weight": rot_edge_weights["W_edge_total"],
        "rotated_corner_weight": rot_corner_weights["W_corner"],
        "whether_selection_changed": int(best_edge.index != old_sel.index or best_corner.index != old_sel.index),
        "recommended_physical_role": role_from_weights(
            edge_weight=rot_edge_weights["W_edge_total"],
            corner_weight=rot_corner_weights["W_corner"],
            bulk_weight=rot_edge_weights["W_bulk"],
        ),
        "edge_gain_vs_old": best_edge_weights["W_edge_total"] - old_weights["W_edge_total"],
        "corner_gain_vs_old": best_corner_weights["W_corner"] - old_weights["W_corner"],
        "best_method": best_method_name(
            old_w=old_weights["W_edge_total"],
            best_edge_w=best_edge_weights["W_edge_total"],
            best_corner_w=best_corner_weights["W_corner"],
        ),
    }
    return summary, deg_rows


def build_key_points(summary_rows: list[dict]) -> list[dict]:
    rows = summary_rows

    def pick_max(key: str, filt=None):
        cand = rows if filt is None else [r for r in rows if filt(r)]
        if not cand:
            return None
        return max(cand, key=lambda r: float(r[key]))

    def pick_min(key: str, filt=None):
        cand = rows if filt is None else [r for r in rows if filt(r)]
        if not cand:
            return None
        return min(cand, key=lambda r: float(r[key]))

    key_defs = [
        ("strongest edge precursor", pick_max("best_edge_weight", lambda r: r["category"] == "near_transition")),
        ("strongest corner precursor", pick_max("best_corner_weight", lambda r: r["category"] == "near_transition")),
        ("most critical transition core", pick_min("gap", lambda r: r["category"] == "near_transition")),
        ("strongest Chern-anomalous point", pick_max("best_edge_weight", lambda r: abs(float(r["Chern"])) >= 0.5)),
        ("reopened robust edge-dominated point", pick_max("rotated_edge_weight", lambda r: r["category"] == "robust_topological")),
        ("reopened robust but bulk-mixed point", pick_max("gap", lambda r: r["category"] == "robust_topological" and r["recommended_physical_role"] == "bulk_mixed")),
    ]

    out = []
    for role, row in key_defs:
        if row is None:
            continue
        out.append(
            {
                "key_role": role,
                "point_id": row["point_id"],
                "v": row["v"],
                "t": row["t"],
                "lm": row["lm"],
                "category": row["category"],
                "gap": row["gap"],
                "Z2": row["Z2"],
                "Chern": row["Chern"],
                "Wilson": row["Wilson"],
                "best_edge_weight": row["best_edge_weight"],
                "best_corner_weight": row["best_corner_weight"],
                "rotated_edge_weight": row["rotated_edge_weight"],
                "rotated_corner_weight": row["rotated_corner_weight"],
                "recommended_physical_role": row["recommended_physical_role"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reselected special-point analysis")
    parser.add_argument("--package-dir", type=Path, default=Path("/workspace/outputs/physical_special_points_package"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/results_reselected"))
    parser.add_argument("--config", type=Path, default=Path("/workspace/scripts/reselected_analysis_config.json"))
    parser.add_argument("--point-limit", type=int, default=0, help="If >0, only process first N points for dry-run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = load_manifest(args.package_dir / "manifest.csv")
    if args.point_limit > 0:
        manifest = manifest[: args.point_limit]

    summary_rows = []
    deg_rows = []
    for idx, row in enumerate(manifest, start=1):
        summary, point_deg = process_point(row=row, cfg=cfg, package_dir=args.package_dir, output_root=args.output_dir)
        summary_rows.append(summary)
        deg_rows.extend(point_deg)
        if idx % 5 == 0:
            print(f"processed {idx}/{len(manifest)}")

    summary_fields = [
        "point_id",
        "v",
        "t",
        "lm",
        "category",
        "gap",
        "Z2",
        "Chern",
        "Wilson",
        "old_selected_mode",
        "old_energy",
        "old_edge_weight",
        "old_corner_weight",
        "best_edge_energy",
        "best_edge_weight",
        "best_corner_energy",
        "best_corner_weight",
        "rotated_edge_weight",
        "rotated_corner_weight",
        "whether_selection_changed",
        "recommended_physical_role",
        "edge_gain_vs_old",
        "corner_gain_vs_old",
        "best_method",
    ]
    write_csv(args.output_dir / "summary_tables" / "point_summary.csv", summary_rows, summary_fields)

    deg_fields = [
        "point_id",
        "group_id",
        "dim",
        "E_min",
        "E_max",
        "delta_E",
        "best_edge_projection_after",
        "best_corner_projection_after",
    ]
    write_csv(args.output_dir / "summary_tables" / "degeneracy_subspace_summary.csv", deg_rows, deg_fields)

    key_rows = build_key_points(summary_rows)
    key_fields = [
        "key_role",
        "point_id",
        "v",
        "t",
        "lm",
        "category",
        "gap",
        "Z2",
        "Chern",
        "Wilson",
        "best_edge_weight",
        "best_corner_weight",
        "rotated_edge_weight",
        "rotated_corner_weight",
        "recommended_physical_role",
    ]
    write_csv(args.output_dir / "summary_tables" / "report_key_points.csv", key_rows, key_fields)

    generate_transition_figures(args.output_dir, summary_rows)
    build_report_pdf(output_root=args.output_dir, config=cfg)
    print(f"done output={args.output_dir}")


if __name__ == "__main__":
    main()
