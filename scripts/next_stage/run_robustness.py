#!/usr/bin/env python3
"""Disorder and boundary perturbation robustness checks for selected points."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from scripts.next_stage.common import (
    auto_select_points,
    cell_probability,
    classify_distribution,
    infer_spin_operator,
    load_model,
    robust_sparse_near_zero,
    summarize_points,
    write_csv,
    write_text,
)
from scripts.run_scan import build_obc_hamiltonian_sparse


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    return vals


def edge_cell_ids(Lx: int, Ly: int, edge_width: int = 2) -> np.ndarray:
    mask = np.zeros((Ly, Lx), dtype=bool)
    ew = max(1, min(edge_width, Lx // 2, Ly // 2))
    mask[:, :ew] = True
    mask[:, Lx - ew :] = True
    mask[:ew, :] = True
    mask[Ly - ew :, :] = True
    return np.where(mask.reshape(-1))[0]


def add_onsite_disorder(ham: sparse.csr_matrix, W: float, rng: np.random.Generator) -> sparse.csr_matrix:
    if W <= 0:
        return ham
    dim = ham.shape[0]
    disorder = rng.uniform(-W, W, size=dim)
    return ham + sparse.diags(disorder, offsets=0, shape=(dim, dim), format="csr")


def add_edge_potential(ham: sparse.csr_matrix, Lx: int, Ly: int, V_edge: float, edge_width: int = 2) -> sparse.csr_matrix:
    if abs(V_edge) <= 0:
        return ham
    ids = edge_cell_ids(Lx=Lx, Ly=Ly, edge_width=edge_width)
    dim = ham.shape[0]
    diag = np.zeros(dim, dtype=float)
    for cid in ids:
        diag[cid * 8 : cid * 8 + 8] = V_edge
    return ham + sparse.diags(diag, offsets=0, shape=(dim, dim), format="csr")


def add_trs_breaking_mass(ham: sparse.csr_matrix, Lx: int, Ly: int, mass: float, sz8: np.ndarray | None) -> sparse.csr_matrix:
    if sz8 is None or abs(mass) <= 0:
        return ham
    dim = ham.shape[0]
    addon = sparse.lil_matrix((dim, dim), dtype=complex)
    for cid in range(Lx * Ly):
        start = cid * 8
        addon[start : start + 8, start : start + 8] = mass * sz8
    return (ham + addon.tocsr()).tocsr()


def metrics_from_modes(vals: np.ndarray, vecs: np.ndarray, L: int) -> dict[str, float]:
    abs_vals = np.abs(vals)
    min_abs = float(np.min(abs_vals))
    n002 = int(np.sum(abs_vals <= 0.02))
    near_mask = abs_vals <= 0.02
    if np.sum(near_mask) == 0:
        near_mask[int(np.argmin(abs_vals))] = True
    idxs = np.where(near_mask)[0]
    grid = np.zeros((L, L), dtype=float)
    for i in idxs:
        grid += cell_probability(vecs[:, i], Lx=L, Ly=L)
    total = float(np.sum(grid))
    if total > 0:
        grid /= total
    dist = classify_distribution(grid, edge_width=2, corner_size=3)
    return {
        "min_abs_E": min_abs,
        "nearzero_count_0p02": float(n002),
        "edge_weight": float(dist["edge_weight"]),
        "corner_weight": float(dist["corner_weight"]),
    }


def errorbar_plot(path: Path, x: np.ndarray, mean: np.ndarray, std: np.ndarray, title: str, xlab: str, ylab: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.errorbar(x, mean, yerr=std, marker="o", linewidth=1.2, capsize=3)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness check: disorder / edge perturbation / TRS breaking.")
    parser.add_argument("--model-file", type=Path, default=Path("/workspace/models/xtype_model.py"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/outputs/next_stage_analysis/robustness"))
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--near-k", type=int, default=96)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--disorder-list", default="0.0,0.02,0.05,0.10,0.20,0.30")
    parser.add_argument("--boundary-list", default="0.0,0.05,0.10,0.20,0.30")
    parser.add_argument("--trs-mass-list", default="0.0,0.02,0.05,0.10,0.20")
    parser.add_argument("--realizations", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    W_list = parse_float_list("0.0,0.05,0.10" if args.quick else args.disorder_list)
    V_edge_list = parse_float_list("0.0,0.10,0.20" if args.quick else args.boundary_list)
    m_list = parse_float_list("0.0,0.05,0.10" if args.quick else args.trs_mass_list)
    realizations = 6 if args.quick else args.realizations

    points, notes = auto_select_points(model_path=args.model_file)
    pmap = {p.label: p for p in points}
    model = load_model(args.model_file)
    sz8, sz_reason = infer_spin_operator(model)
    labels = [lb for lb in ("A", "B", "C") if lb in pmap]
    run_logs = [f"seed={args.seed}", f"TRS-mass operator: {sz_reason}"]

    disorder_rows = []
    boundary_rows = []
    trs_rows = []

    for label in labels:
        p = pmap[label]
        base = build_obc_hamiltonian_sparse(v=p.v, t=p.t, lm=p.lm, w=p.w, j=0.0, nx=args.L, ny=args.L)

        # Disorder averaging
        for W in W_list:
            per_real = []
            for ridx in range(realizations):
                ham = add_onsite_disorder(base, W=W, rng=rng)
                try:
                    vals, vecs = robust_sparse_near_zero(ham, base_k=args.near_k)
                    met = metrics_from_modes(vals, vecs, L=args.L)
                    per_real.append(met)
                except Exception as exc:
                    run_logs.append(f"[warn] disorder fail {label} W={W} r={ridx}: {exc!r}")
            if not per_real:
                continue
            for key in ("min_abs_E", "nearzero_count_0p02", "edge_weight", "corner_weight"):
                arr = np.array([m[key] for m in per_real], dtype=float)
                disorder_rows.append(
                    {
                        "point_label": label,
                        "v": p.v,
                        "lm": p.lm,
                        "t": p.t,
                        "w": p.w,
                        "W_disorder": W,
                        "metric": key,
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "realizations": len(per_real),
                    }
                )

        # Boundary perturbation
        for Ve in V_edge_list:
            ham = add_edge_potential(base, Lx=args.L, Ly=args.L, V_edge=Ve, edge_width=2)
            try:
                vals, vecs = robust_sparse_near_zero(ham, base_k=args.near_k)
                met = metrics_from_modes(vals, vecs, L=args.L)
                boundary_rows.append(
                    {
                        "point_label": label,
                        "v": p.v,
                        "lm": p.lm,
                        "t": p.t,
                        "w": p.w,
                        "V_edge": Ve,
                        **met,
                    }
                )
            except Exception as exc:
                run_logs.append(f"[warn] boundary fail {label} Ve={Ve}: {exc!r}")

        # TRS breaking mass
        if sz8 is None:
            run_logs.append(f"[skip] trs-breaking for {label}: no spin operator")
        else:
            for m in m_list:
                ham = add_trs_breaking_mass(base, Lx=args.L, Ly=args.L, mass=m, sz8=sz8)
                try:
                    vals, vecs = robust_sparse_near_zero(ham, base_k=args.near_k)
                    met = metrics_from_modes(vals, vecs, L=args.L)
                    trs_rows.append(
                        {
                            "point_label": label,
                            "v": p.v,
                            "lm": p.lm,
                            "t": p.t,
                            "w": p.w,
                            "m_trs_break": m,
                            **met,
                            "operator_note": sz_reason,
                        }
                    )
                except Exception as exc:
                    run_logs.append(f"[warn] trs-break fail {label} m={m}: {exc!r}")

    write_csv(
        out_root / "disorder_summary.csv",
        disorder_rows,
        ["point_label", "v", "lm", "t", "w", "W_disorder", "metric", "mean", "std", "realizations"],
    )
    write_csv(
        out_root / "boundary_perturbation_summary.csv",
        boundary_rows,
        ["point_label", "v", "lm", "t", "w", "V_edge", "min_abs_E", "nearzero_count_0p02", "edge_weight", "corner_weight"],
    )
    write_csv(
        out_root / "trs_breaking_summary.csv",
        trs_rows,
        ["point_label", "v", "lm", "t", "w", "m_trs_break", "min_abs_E", "nearzero_count_0p02", "edge_weight", "corner_weight", "operator_note"],
    )

    # required plots for A; also generate for B/C when present
    for label in labels:
        drows = [r for r in disorder_rows if r["point_label"] == label]
        if drows:
            for metric, ylab, fname in (
                ("min_abs_E", "min |E|", f"{label}_disorder_min_abs_E_vs_W.png"),
                ("nearzero_count_0p02", "N(|E|<0.02)", f"{label}_disorder_nearzero_count_vs_W.png"),
            ):
                rows_m = [r for r in drows if r["metric"] == metric]
                x = np.array([float(r["W_disorder"]) for r in rows_m], dtype=float)
                y = np.array([float(r["mean"]) for r in rows_m], dtype=float)
                e = np.array([float(r["std"]) for r in rows_m], dtype=float)
                order = np.argsort(x)
                errorbar_plot(
                    out_root / fname,
                    x[order],
                    y[order],
                    e[order],
                    title=f"{label}: disorder robustness",
                    xlab="W disorder",
                    ylab=ylab,
                )

        brows = [r for r in boundary_rows if r["point_label"] == label]
        if brows:
            x = np.array([float(r["V_edge"]) for r in brows], dtype=float)
            y = np.array([float(r["min_abs_E"]) for r in brows], dtype=float)
            order = np.argsort(x)
            errorbar_plot(
                out_root / f"{label}_boundary_min_abs_E_vs_Vedge.png",
                x[order],
                y[order],
                np.zeros_like(y[order]),
                title=f"{label}: boundary perturbation",
                xlab="V_edge",
                ylab="min |E|",
            )

        trows = [r for r in trs_rows if r["point_label"] == label]
        if trows:
            x = np.array([float(r["m_trs_break"]) for r in trows], dtype=float)
            y = np.array([float(r["min_abs_E"]) for r in trows], dtype=float)
            order = np.argsort(x)
            errorbar_plot(
                out_root / f"{label}_trs_breaking_min_abs_E_vs_m.png",
                x[order],
                y[order],
                np.zeros_like(y[order]),
                title=f"{label}: TRS-breaking mass",
                xlab="m_trs_break",
                ylab="min |E|",
            )

    write_text(
        out_root / "selection_notes.txt",
        summarize_points(points, notes)
        + (
            f"\n\nL={args.L}, realizations={realizations}, W_list={W_list}, "
            f"V_edge_list={V_edge_list}, m_list={m_list}\n"
        ),
    )
    write_text(out_root / "run_log.txt", "\n".join(run_logs) + "\n")
    print(f"[ok] robustness outputs at {out_root}")


if __name__ == "__main__":
    main()
