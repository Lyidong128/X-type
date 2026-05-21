"""Shared utilities for next-stage topology analysis scripts."""

from __future__ import annotations

import csv
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from scripts.run_scan import (
    build_obc_hamiltonian,
    build_obc_hamiltonian_sparse,
    build_ribbon_hamiltonian,
    compute_bulk_gap,
    compute_chern_number,
    compute_dynamic_w,
    compute_wilson_loop_and_z2,
    load_xtype_model,
    set_model_params,
)


PROJECT_ROOT = Path("/workspace")
DEFAULT_MODEL_FILE = PROJECT_ROOT / "models" / "xtype_model.py"
DEFAULT_TOPOLOGY_CSV = PROJECT_ROOT / "outputs" / "first_stage_band_ribbon" / "topology_summary.csv"
DEFAULT_RIBBON_CSV = (
    PROJECT_ROOT / "outputs" / "first_stage_band_ribbon" / "ribbon_gap_state_classification_fermi45.csv"
)
DEFAULT_OBC_CSV = PROJECT_ROOT / "outputs" / "first_stage_band_ribbon" / "obc_special_points_analysis.csv"
DEFAULT_TRANSITION_ROOT = PROJECT_ROOT / "outputs" / "transition_corridors"


@dataclass
class AnalysisPoint:
    """Named analysis point with parameter set."""

    label: str
    v: float
    lm: float
    t: float = 0.5
    w: float = 0.0
    source: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "v": float(self.v),
            "lm": float(self.lm),
            "t": float(self.t),
            "w": float(self.w),
            "source": self.source,
            "note": self.note,
        }


def make_params(v: float, lm: float, t: float = 0.5) -> dict[str, float]:
    """Compat helper requested by user: return complete model parameter dict."""
    w = compute_dynamic_w(v)
    return {"v": float(v), "lm": float(lm), "t": float(t), "w": float(w), "J": 0.0}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV with fixed field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, data: Any) -> None:
    """Write JSON with utf-8 and indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write plain text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV as list of dict rows."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float parsing."""
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    """Best-effort int parsing."""
    try:
        return int(float(value))
    except Exception:
        return default


def load_model(model_path: Path = DEFAULT_MODEL_FILE):
    """Load Hamiltonian model module from path."""
    return load_xtype_model(model_path)


def set_params(model: Any, v: float, lm: float, t: float = 0.5, j: float = 0.0) -> dict[str, float]:
    """Set model global parameters and return the actual parameter dict."""
    p = make_params(v=v, lm=lm, t=t)
    set_model_params(model, v=p["v"], t=p["t"], lm=p["lm"], w=p["w"], j=j)
    return p


def robust_sparse_near_zero(
    ham_sparse: sparse.csr_matrix,
    base_k: int = 96,
    min_k: int = 12,
    timeout_hint: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust near-zero sparse eigensolver with fallbacks.

    `timeout_hint` kept for CLI compatibility; eigsh timeout is not hard-enforced here.
    """
    dim = ham_sparse.shape[0]
    candidate_k: list[int] = []
    for k_try in (base_k, base_k - 16, base_k + 16, 128, 80, 64, 48, 32):
        kk = max(min_k, min(int(k_try), dim - 2))
        if kk not in candidate_k:
            candidate_k.append(kk)

    for kk in candidate_k:
        for kwargs in ({"sigma": 0.0, "which": "LM"}, {"which": "SM"}):
            try:
                vals, vecs = eigsh(ham_sparse, k=kk, maxiter=1800, tol=1e-7, **kwargs)
                order = np.argsort(np.real(vals))
                return np.real(vals[order]), vecs[:, order]
            except ArpackNoConvergence as exc:
                vals = getattr(exc, "eigenvalues", None)
                vecs = getattr(exc, "eigenvectors", None)
                if vals is not None and vecs is not None and len(vals) >= min_k:
                    order = np.argsort(np.real(vals))
                    return np.real(vals[order]), vecs[:, order]
            except Exception:
                continue
    raise RuntimeError("Failed to solve sparse near-zero eigenproblem.")


def solve_obc_modes(
    v: float,
    lm: float,
    t: float = 0.5,
    Lx: int = 20,
    Ly: int = 20,
    near_k: int = 96,
    dense_max_dim: int = 2200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve OBC spectrum/eigenvectors with dense-or-sparse fallback."""
    w = compute_dynamic_w(v)
    dim = Lx * Ly * 8
    if dim <= dense_max_dim:
        ham = build_obc_hamiltonian(v=v, t=t, lm=lm, w=w, j=0.0, nx=Lx, ny=Ly)
        vals, vecs = np.linalg.eigh(ham)
        return np.real(vals), vecs
    ham_sp = build_obc_hamiltonian_sparse(v=v, t=t, lm=lm, w=w, j=0.0, nx=Lx, ny=Ly)
    return robust_sparse_near_zero(ham_sp, base_k=near_k)


def cell_probability(vec: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    """Convert eigenvector to normalized cell probability map."""
    n_cells = Lx * Ly
    out = np.zeros(n_cells, dtype=float)
    for cid in range(n_cells):
        start = cid * 8
        out[cid] = float(np.sum(np.abs(vec[start : start + 8]) ** 2))
    grid = out.reshape((Ly, Lx))
    total = float(np.sum(grid))
    return grid / total if total > 0 else grid


def classify_distribution(
    grid: np.ndarray,
    edge_width: int = 2,
    corner_size: int = 3,
    edge_like_thr: float = 0.45,
    corner_like_thr: float = 0.28,
) -> dict[str, Any]:
    """Classify a probability map as bulk/edge/corner/mixed."""
    ny, nx = grid.shape
    ew = max(1, min(edge_width, nx // 2, ny // 2))
    cs = max(1, min(corner_size, nx // 2, ny // 2))
    edge_mask = np.zeros_like(grid, dtype=bool)
    edge_mask[:, :ew] = True
    edge_mask[:, nx - ew :] = True
    edge_mask[:ew, :] = True
    edge_mask[ny - ew :, :] = True
    corner_mask = np.zeros_like(grid, dtype=bool)
    corner_mask[:cs, :cs] = True
    corner_mask[:cs, nx - cs :] = True
    corner_mask[ny - cs :, :cs] = True
    corner_mask[ny - cs :, nx - cs :] = True
    bulk_mask = ~(edge_mask | corner_mask)

    edge_weight = float(np.sum(grid[edge_mask]))
    corner_weight = float(np.sum(grid[corner_mask]))
    bulk_weight = float(np.sum(grid[bulk_mask]))

    if corner_weight >= corner_like_thr:
        label = "corner-like"
    elif edge_weight >= edge_like_thr:
        label = "edge-like"
    elif edge_weight <= 0.30 and bulk_weight >= 0.55:
        label = "bulk-like"
    else:
        label = "mixed"

    return {
        "classification": label,
        "edge_weight": edge_weight,
        "corner_weight": corner_weight,
        "bulk_weight": bulk_weight,
        "edge_width": ew,
        "corner_size": cs,
    }


def compute_ipr(vec: np.ndarray) -> float:
    """Inverse participation ratio in full basis."""
    p = np.abs(vec) ** 2
    denom = float(np.sum(p) ** 2)
    if denom <= 0:
        return 0.0
    return float(np.sum(p**2) / denom)


def find_transition_candidate_from_existing(
    center_v: float,
    center_lm: float,
    transition_root: Path = DEFAULT_TRANSITION_ROOT,
    radius: float = 0.25,
) -> dict[str, Any] | None:
    """Choose the minimum-gap candidate near (center_v, center_lm) from existing corridor scans."""
    if not transition_root.exists():
        return None

    best: dict[str, Any] | None = None
    for path in sorted(transition_root.glob("rank_*/corridor_scan.csv")):
        rows = read_csv(path)
        for r in rows:
            v = to_float(r.get("v"))
            lm = to_float(r.get("lm"))
            dist = math.hypot(v - center_v, lm - center_lm)
            if dist > radius:
                continue
            gap = abs(to_float(r.get("gap"), default=np.inf))
            key = (gap, dist, to_float(r.get("obc_min_abs_energy"), default=np.inf))
            if best is None or key < best["_sort_key"]:
                best = {
                    "v": v,
                    "lm": lm,
                    "t": to_float(r.get("t"), default=0.5),
                    "w": to_float(r.get("w"), default=compute_dynamic_w(v)),
                    "gap": to_float(r.get("gap"), default=np.nan),
                    "chern": to_float(r.get("chern"), default=np.nan),
                    "obc_min_abs_energy": to_float(r.get("obc_min_abs_energy"), default=np.nan),
                    "source_csv": str(path),
                    "_sort_key": key,
                }
    if best is None:
        return None
    best.pop("_sort_key", None)
    return best


def find_transition_candidate_fallback(
    model: Any,
    center_v: float,
    center_lm: float,
    t: float = 0.5,
    width_v: float = 0.08,
    width_lm: float = 0.08,
    step: float = 0.02,
) -> dict[str, Any]:
    """Fallback local search for minimum absolute bulk gap."""
    v_vals = np.arange(max(0.0, center_v - width_v), min(2.0, center_v + width_v) + 1e-9, step)
    lm_vals = np.arange(max(0.0, center_lm - width_lm), min(0.5, center_lm + width_lm) + 1e-9, step)
    best = {
        "v": center_v,
        "lm": center_lm,
        "t": t,
        "w": compute_dynamic_w(center_v),
        "gap": np.inf,
        "chern": np.nan,
        "source_csv": "fallback_local_scan",
    }
    for v in v_vals:
        for lm in lm_vals:
            p = set_params(model, v=float(v), lm=float(lm), t=t)
            gap = compute_bulk_gap(model, nk=11, n_occ=4)
            if abs(gap) < abs(float(best["gap"])):
                best = {
                    "v": float(v),
                    "lm": float(lm),
                    "t": t,
                    "w": p["w"],
                    "gap": float(gap),
                    "chern": float(compute_chern_number(model, nk=15, n_occ=4)),
                    "source_csv": "fallback_local_scan",
                }
    return best


def auto_select_points(
    topology_csv: Path = DEFAULT_TOPOLOGY_CSV,
    ribbon_csv: Path = DEFAULT_RIBBON_CSV,
    obc_csv: Path = DEFAULT_OBC_CSV,
    transition_root: Path = DEFAULT_TRANSITION_ROOT,
    model_path: Path = DEFAULT_MODEL_FILE,
) -> tuple[list[AnalysisPoint], list[str]]:
    """Build A/B/C/D/E point set from existing outputs + fallback logic."""
    notes: list[str] = []
    topo_rows = read_csv(topology_csv)
    rib_rows = read_csv(ribbon_csv)
    obc_rows = read_csv(obc_csv)
    rib_map = {r["point_id"]: r for r in rib_rows}
    obc_map = {r["point_id"]: r for r in obc_rows}

    points: list[AnalysisPoint] = [
        AnalysisPoint(label="A", v=0.90, lm=0.20, t=0.5, w=compute_dynamic_w(0.90), source="fixed_user"),
        AnalysisPoint(label="B", v=1.10, lm=0.20, t=0.5, w=compute_dynamic_w(1.10), source="fixed_user"),
    ]

    # C: trivial reference point.
    candidates_c = []
    for r in topo_rows:
        pid = r["point_id"]
        z2 = to_int(r.get("z2"), default=0)
        ch = abs(to_float(r.get("chern"), default=0.0))
        rb = rib_map.get(pid, {})
        connecting = to_int(rb.get("has_connecting_edge_branch"), default=0)
        if z2 == 0 and ch <= 0.1 and connecting == 0:
            score = to_float(rb.get("band_window_width"), default=0.0)
            candidates_c.append((score, r))
    if candidates_c:
        _, best = sorted(candidates_c, key=lambda x: x[0], reverse=True)[0]
        c_v = to_float(best["v"])
        c_lm = to_float(best["lm"])
        points.append(
            AnalysisPoint(
                label="C",
                v=c_v,
                lm=c_lm,
                t=to_float(best.get("t"), default=0.5),
                w=to_float(best.get("w"), default=compute_dynamic_w(c_v)),
                source="auto_from_topology_summary",
                note="z2=0, chern≈0, ribbon no connecting branch",
            )
        )
    else:
        points.append(
            AnalysisPoint(
                label="C",
                v=0.00,
                lm=0.00,
                t=0.5,
                w=compute_dynamic_w(0.00),
                source="fallback_default",
                note="Could not auto-detect trivial C; fallback to default point.",
            )
        )
        notes.append("C point auto-detection failed; used fallback C=(0.00,0.00).")

    # D: z2=1 but OBC near-zero not obvious.
    candidates_d = []
    for r in topo_rows:
        pid = r["point_id"]
        z2 = to_int(r.get("z2"), default=0)
        if z2 != 1:
            continue
        obc = obc_map.get(pid)
        if obc is None:
            continue
        min_abs = to_float(obc.get("min_abs_energy"), default=np.inf)
        n02 = to_int(obc.get("count_absE_le_0p02"), default=9999)
        if n02 <= 4 and min_abs >= 0.01:
            candidates_d.append((min_abs, r, obc))
    if candidates_d:
        _, best, ob = sorted(candidates_d, key=lambda x: x[0], reverse=True)[0]
        d_v = to_float(best["v"])
        d_lm = to_float(best["lm"])
        points.append(
            AnalysisPoint(
                label="D",
                v=d_v,
                lm=d_lm,
                t=to_float(best.get("t"), default=0.5),
                w=to_float(best.get("w"), default=compute_dynamic_w(d_v)),
                source="auto_from_topology+obc",
                note=f"z2=1 but weak near-zero: min|E|={to_float(ob['min_abs_energy']):.4f}, N0.02={to_int(ob['count_absE_le_0p02'])}",
            )
        )
    else:
        notes.append("D point not found: no z2=1 point with weak OBC near-zero signature under current criteria.")

    # E points near A and B.
    model = load_model(model_path)
    for label, center in (("E_A", points[0]), ("E_B", points[1])):
        candidate = find_transition_candidate_from_existing(
            center_v=center.v,
            center_lm=center.lm,
            transition_root=transition_root,
            radius=0.3,
        )
        if candidate is None:
            candidate = find_transition_candidate_fallback(
                model=model,
                center_v=center.v,
                center_lm=center.lm,
                t=center.t,
            )
            source = "fallback_local_scan"
        else:
            source = "existing_transition_corridors"
        points.append(
            AnalysisPoint(
                label=label,
                v=float(candidate["v"]),
                lm=float(candidate["lm"]),
                t=float(candidate.get("t", 0.5)),
                w=float(candidate.get("w", compute_dynamic_w(float(candidate["v"])))),
                source=source,
                note=f"gap={to_float(candidate.get('gap')):.3e}, from={candidate.get('source_csv','')}",
            )
        )
    return points, notes


def get_points_dict(points: list[AnalysisPoint]) -> dict[str, AnalysisPoint]:
    """Map label -> point for quick lookup."""
    return {p.label: p for p in points}


def summarize_points(points: list[AnalysisPoint], notes: list[str]) -> str:
    """Human-readable points summary."""
    lines = ["Selected analysis points:"]
    for p in points:
        lines.append(
            f"- {p.label}: v={p.v:.3f}, lm={p.lm:.3f}, t={p.t:.3f}, w={p.w:.3f} "
            f"(source={p.source}; note={p.note})"
        )
    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
    return "\n".join(lines)


def infer_spin_operator(model: Any) -> tuple[np.ndarray | None, str]:
    """Try to infer a spin/psuedospin operator Sz in 8x8 space.

    Returns (operator_or_None, reason).
    """
    # Case 1: model has 2x2 sz and uses 4 orbital x spin structure.
    if hasattr(model, "sz"):
        sz = np.array(getattr(model, "sz"))
        if sz.shape == (2, 2):
            return np.kron(np.eye(4, dtype=complex), sz), "from model.sz with kron(I4, sz)"
        if sz.shape == (8, 8):
            return sz.astype(complex), "from model.sz directly (8x8)"

    # Case 2: explicit symbol names.
    for name in ("S_z", "Sz", "spin_z", "sigma_z"):
        if hasattr(model, name):
            op = np.array(getattr(model, name))
            if op.shape == (8, 8):
                return op.astype(complex), f"from model.{name}"
            if op.shape == (2, 2):
                return np.kron(np.eye(4, dtype=complex), op.astype(complex)), f"from model.{name} as 2x2"

    return None, "No explicit spin/psuedospin operator found in current model symbols."


def estimate_z2_from_wannier_centers(wannier_centers: np.ndarray) -> int:
    """Heuristic Z2 estimator from Wannier flow crossings around 0.5."""
    if wannier_centers.ndim != 2 or wannier_centers.shape[0] < 2:
        return 0
    crossings = 0
    for band in range(wannier_centers.shape[1]):
        y = wannier_centers[:, band]
        side = np.sign(y - 0.5)
        side[side == 0] = 1.0
        changes = np.where(side[1:] * side[:-1] < 0)[0]
        crossings += int(len(changes))
    return crossings % 2


def wilson_loop_flow(
    model: Any,
    direction: str = "kx",
    nk_loop: int = 61,
    nk_scan: int = 61,
    n_occ: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Wilson-loop Wannier center flow.

    Returns (scan_axis_values in [0,1), centers shape [nk_scan, n_occ] in [0,1)).
    """
    scan_vals = np.linspace(0.0, 1.0, nk_scan, endpoint=False)
    centers = np.zeros((nk_scan, n_occ), dtype=float)

    for j, scan in enumerate(scan_vals):
        wilson = np.eye(n_occ, dtype=complex)
        # initialize occupied subspace
        if direction == "kx":
            k0 = 0.0 * model.b1 + scan * model.b2
            step_vec = model.b1
        else:
            k0 = scan * model.b1 + 0.0 * model.b2
            step_vec = model.b2
        _, vec_prev = np.linalg.eigh(model.Hxtype(k0))
        occ_prev = vec_prev[:, :n_occ]

        for i in range(1, nk_loop + 1):
            u = (i % nk_loop) / nk_loop
            if direction == "kx":
                k = u * model.b1 + scan * model.b2
            else:
                k = scan * model.b1 + u * model.b2
            _, vec_curr = np.linalg.eigh(model.Hxtype(k))
            occ_curr = vec_curr[:, :n_occ]
            overlap = occ_prev.conj().T @ occ_curr
            # Unitary projection improves numerical stability
            uu, _, vh = np.linalg.svd(overlap, full_matrices=False)
            wilson = wilson @ (uu @ vh)
            occ_prev = occ_curr

        evals = np.linalg.eigvals(wilson)
        phase = np.angle(evals)
        # Map to [0,1)
        x = np.mod(phase / (2.0 * np.pi), 1.0)
        centers[j, :] = np.sort(x)
    return scan_vals, centers


def fit_exponential(xs: np.ndarray, ys: np.ndarray) -> dict[str, float] | None:
    """Fit y ~ exp(a + b x), return parameters + R2."""
    mask = (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    if np.sum(mask) < 3:
        return None
    x = xs[mask]
    y = np.log(ys[mask])
    b, a = np.polyfit(x, y, 1)
    y_hat = a + b * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    xi = -1.0 / b if abs(b) > 1e-12 else np.inf
    return {"a": float(a), "b": float(b), "xi": float(xi), "r2": float(r2)}


def fit_powerlaw(xs: np.ndarray, ys: np.ndarray) -> dict[str, float] | None:
    """Fit y ~ c * x^(-alpha), return parameters + R2."""
    mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    if np.sum(mask) < 3:
        return None
    lx = np.log(xs[mask])
    ly = np.log(ys[mask])
    m, c = np.polyfit(lx, ly, 1)
    y_hat = c + m * lx
    ss_res = float(np.sum((ly - y_hat) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    alpha = -m
    return {"c": float(c), "alpha": float(alpha), "r2": float(r2)}


def compute_bulk_gap_kmin(
    model: Any,
    nk: int = 31,
    n_occ: int = 4,
) -> tuple[float, float, float]:
    """Return minimum direct gap and the (kx,ky) fractional coordinates."""
    min_gap = np.inf
    best_kx = 0.0
    best_ky = 0.0
    for ix in range(nk):
        for iy in range(nk):
            ux = ix / (nk - 1)
            uy = iy / (nk - 1)
            k = ux * model.b1 + uy * model.b2
            evals = np.linalg.eigvalsh(model.Hxtype(k))
            gap = float(np.real(evals[n_occ] - evals[n_occ - 1]))
            if gap < min_gap:
                min_gap = gap
                best_kx = ux
                best_ky = uy
    return float(min_gap), float(best_kx), float(best_ky)

