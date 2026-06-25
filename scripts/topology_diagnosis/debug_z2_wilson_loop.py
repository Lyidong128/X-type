#!/usr/bin/env python3
"""Recompute Wilson-loop Z2 with robust branch tracking and diagnostics."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.common_topology import (
    chern_fukui,
    commutator_norm_h_sz,
    compute_bulk_gap,
    ensure_dir,
    h8_k,
    h_spin_block,
    unitary_part,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Wilson-loop Z2 inconsistency.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs/z2_debug")
    parser.add_argument("--v-list", default="0.5,0.8,1.0")
    parser.add_argument("--nkx", type=int, default=101)
    parser.add_argument("--nky", type=int, default=101)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def token(v: float) -> str:
    return f"{v:.1f}".replace("-", "m").replace(".", "p")


def circular_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def occ_subspace(kx: float, ky: float, v: float, t: float, w: float, lm: float, n_occ: int) -> np.ndarray:
    _, vecs = np.linalg.eigh(h8_k(kx, ky, v=v, t=t, w=w, lm=lm))
    return vecs[:, :n_occ]


def wilson_x_per_ky(
    ky: float,
    v: float,
    t: float,
    w: float,
    lm: float,
    n_occ: int,
    kx_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    wmat = np.eye(n_occ, dtype=complex)
    for i, kx in enumerate(kx_grid):
        kx_next = float(kx_grid[(i + 1) % kx_grid.size])
        oa = occ_subspace(float(kx), float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
        ob = occ_subspace(kx_next, float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
        f = oa.conj().T @ ob
        q = unitary_part(f)
        wmat = q @ wmat
    err = float(np.linalg.norm(wmat.conj().T @ wmat - np.eye(n_occ)))
    eigvals, eigvecs = np.linalg.eig(wmat)
    centers = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
    order = np.argsort(centers)
    centers = centers[order]
    eigvecs = eigvecs[:, order]
    return centers, eigvecs, err


def compute_wilson_centers(
    v: float,
    t: float,
    w: float,
    lm: float,
    n_occ: int,
    nkx: int,
    ky_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    centers = np.zeros((ky_grid.size, n_occ), dtype=float)
    eigvecs = np.zeros((ky_grid.size, n_occ, n_occ), dtype=complex)
    max_err = 0.0
    for j, ky in enumerate(ky_grid):
        c, u, err = wilson_x_per_ky(ky=float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ, kx_grid=kx_grid)
        centers[j, :] = c
        eigvecs[j, :, :] = u
        max_err = max(max_err, err)
    return centers, eigvecs, float(max_err)


def write_centers_csv(path: Path, ky_grid: np.ndarray, centers: np.ndarray) -> None:
    fields = ["ky"] + [f"nu_{i+1}" for i in range(centers.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(ky_grid.size):
            row = {"ky": float(ky_grid[i])}
            for b in range(centers.shape[1]):
                row[f"nu_{b+1}"] = float(centers[i, b])
            writer.writerow(row)


def track_branches(centers_sorted: np.ndarray, eigvecs_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nky, nband = centers_sorted.shape
    tracked_mod = np.zeros_like(centers_sorted)
    tracked_vecs = np.zeros_like(eigvecs_sorted, dtype=complex)
    tracked_mod[0, :] = centers_sorted[0, :]
    tracked_vecs[0, :, :] = eigvecs_sorted[0, :, :]
    match_quality = np.ones(nky, dtype=float)

    perms = list(itertools.permutations(range(nband)))
    for j in range(1, nky):
        prev = tracked_mod[j - 1, :]
        prev_vec = tracked_vecs[j - 1, :, :]
        curr = centers_sorted[j, :]
        curr_vec = eigvecs_sorted[j, :, :]
        best_perm = perms[0]
        best_cost = float("inf")
        for p in perms:
            c = np.array([curr[k] for k in p], dtype=float)
            v = curr_vec[:, p]
            dist_cost = float(np.sum([circular_distance(prev[b], c[b]) for b in range(nband)]))
            overlap = np.array([abs(np.vdot(prev_vec[:, b], v[:, b])) for b in range(nband)], dtype=float)
            overlap_cost = float(np.sum(1.0 - np.clip(overlap, 0.0, 1.0)))
            cost = overlap_cost + 0.2 * dist_cost
            if cost < best_cost:
                best_cost = cost
                best_perm = p
        tracked_mod[j, :] = np.array([curr[k] for k in best_perm], dtype=float)
        tracked_vecs[j, :, :] = curr_vec[:, best_perm]
        ov = np.array([abs(np.vdot(prev_vec[:, b], tracked_vecs[j, :, b])) for b in range(nband)], dtype=float)
        match_quality[j] = float(np.mean(np.clip(ov, 0.0, 1.0)))

    tracked_unwrapped = np.zeros_like(tracked_mod)
    tracked_unwrapped[0, :] = tracked_mod[0, :]
    for j in range(1, nky):
        delta = tracked_mod[j, :] - tracked_mod[j - 1, :]
        delta = (delta + 0.5) % 1.0 - 0.5
        tracked_unwrapped[j, :] = tracked_unwrapped[j - 1, :] + delta
    return tracked_mod, tracked_unwrapped, match_quality


def crossing_count_half_bz(
    ky_half: np.ndarray, branches_unwrapped_half: np.ndarray, ref: float, eps_touch: float = 5e-3
) -> tuple[int, int]:
    count = 0
    touch_count = 0
    nseg = ky_half.size - 1
    for b in range(branches_unwrapped_half.shape[1]):
        y = branches_unwrapped_half[:, b]
        for i in range(nseg):
            a = float(y[i])
            c = float(y[i + 1])
            nmin = int(np.floor(min(a, c) - ref)) - 1
            nmax = int(np.ceil(max(a, c) - ref)) + 1
            for n in range(nmin, nmax + 1):
                r = ref + n
                fa = a - r
                fb = c - r
                if abs(fa) < eps_touch or abs(fb) < eps_touch:
                    touch_count += 1
                    continue
                if fa * fb < 0:
                    count += 1
    return int(count), int(touch_count)


def largest_gap_center(arr_sorted: np.ndarray) -> float:
    cyc = np.r_[arr_sorted, arr_sorted[0] + 1.0]
    gaps = np.diff(cyc)
    idx = int(np.argmax(gaps))
    start = cyc[idx]
    center = (start + 0.5 * gaps[idx]) % 1.0
    return float(center)


def gap_centers_and_sizes(arr_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cyc = np.r_[arr_sorted, arr_sorted[0] + 1.0]
    gaps = np.diff(cyc)
    centers = np.array([(cyc[i] + 0.5 * gaps[i]) % 1.0 for i in range(gaps.size)], dtype=float)
    return centers, gaps


def largest_gap_method(centers_half_sorted: np.ndarray, min_margin_thresh: float = 5e-4) -> tuple[int, float, float, bool]:
    nky = centers_half_sorted.shape[0]
    chosen_centers = np.zeros(nky, dtype=float)
    margins = np.zeros(nky, dtype=float)

    c0, g0 = gap_centers_and_sizes(centers_half_sorted[0])
    i0 = int(np.argmax(g0))
    chosen_centers[0] = c0[i0]
    sg0 = np.sort(g0)
    margins[0] = float(sg0[-1] - sg0[-2]) if sg0.size >= 2 else float(sg0[-1])

    for j in range(1, nky):
        cands, sizes = gap_centers_and_sizes(centers_half_sorted[j])
        prev = chosen_centers[j - 1]
        gmax = float(np.max(sizes))
        best_i = 0
        best_cost = float("inf")
        for i in range(cands.size):
            dist = circular_distance(prev, float(cands[i]))
            size_penalty = (gmax - float(sizes[i])) / (gmax + 1e-12)
            cost = dist + 0.15 * size_penalty
            if cost < best_cost:
                best_cost = cost
                best_i = i
        chosen_centers[j] = float(cands[best_i])
        ssorted = np.sort(sizes)
        margins[j] = float(ssorted[-1] - ssorted[-2]) if ssorted.size >= 2 else float(ssorted[-1])

    unwrapped = np.zeros_like(chosen_centers)
    unwrapped[0] = chosen_centers[0]
    for i in range(1, chosen_centers.size):
        d = chosen_centers[i] - chosen_centers[i - 1]
        d = (d + 0.5) % 1.0 - 0.5
        unwrapped[i] = unwrapped[i - 1] + d
    winding_full = float(2.0 * (unwrapped[-1] - unwrapped[0]))
    z2 = int(np.round(winding_full)) % 2
    min_margin = float(np.min(np.array(margins, dtype=float)))
    return z2, winding_full, min_margin, bool(min_margin > min_margin_thresh)


def choose_adaptive_reference(centers_half_sorted: np.ndarray, ngrid: int = 2001) -> tuple[float, float]:
    refs = np.linspace(0.0, 1.0, ngrid, endpoint=False)
    flat = np.mod(centers_half_sorted.reshape(-1), 1.0)
    best_ref = 0.5
    best_clearance = -1.0
    for r in refs:
        d = np.abs(flat - r)
        circ = np.minimum(d, 1.0 - d)
        clearance = float(np.min(circ))
        if clearance > best_clearance:
            best_clearance = clearance
            best_ref = float(r)
    return best_ref, best_clearance


def plot_raw(
    ky_half: np.ndarray,
    centers_half: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    for b in range(centers_half.shape[1]):
        ax.plot(ky_half, centers_half[:, b], linewidth=0.9)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel("Wannier center")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_tracked(
    ky_half: np.ndarray,
    tracked_half: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    for b in range(tracked_half.shape[1]):
        ax.plot(ky_half, tracked_half[:, b], linewidth=1.0)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlim(0.0, np.pi)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel("Wannier center (tracked)")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_tracked_csv(path: Path, ky_half: np.ndarray, tracked_half: np.ndarray) -> None:
    fields = ["ky"] + [f"branch_{i+1}" for i in range(tracked_half.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(ky_half.size):
            row = {"ky": float(ky_half[i])}
            for b in range(tracked_half.shape[1]):
                row[f"branch_{b+1}"] = float(tracked_half[i, b])
            writer.writerow(row)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_montage(out_png: Path, v_list: list[float], out_root: Path) -> None:
    n = len(v_list)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for i, v in enumerate(v_list):
        ax = axes[0][i]
        p = out_root / f"z2_wilson_tracked_v{token(v)}.png"
        if p.exists():
            ax.imshow(plt.imread(p))
            ax.axis("off")
            ax.set_title(f"v={v:.1f}")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
    fig.suptitle("Tracked Wilson flows (Z2 debug)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    v_list = parse_float_list(args.v_list)
    nkx = 51 if args.quick else int(args.nkx)
    nky = 51 if args.quick else int(args.nky)
    n_occ = int(args.n_occ)

    summary_rows: list[dict] = []
    gap_rows: list[dict] = []

    for v in v_list:
        # Gap check over full BZ.
        kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
        ky_grid = np.linspace(-np.pi, np.pi, nky, endpoint=False)
        min_direct_gap = float("inf")
        gap_kx = 0.0
        gap_ky = 0.0
        for kx in kx_grid:
            for ky in ky_grid:
                evals = np.real(np.linalg.eigvalsh(h8_k(float(kx), float(ky), v=v, t=args.t, w=args.w, lm=args.lm)))
                g = float(evals[n_occ] - evals[n_occ - 1])
                if g < min_direct_gap:
                    min_direct_gap = g
                    gap_kx = float(kx)
                    gap_ky = float(ky)
        reliable_gap = bool(min_direct_gap > 1e-4)
        gap_rows.append(
            {
                "v": float(v),
                "n_occ": int(n_occ),
                "min_direct_gap": float(min_direct_gap),
                "gap_kx": float(gap_kx),
                "gap_ky": float(gap_ky),
                "reliable_gap": int(reliable_gap),
            }
        )

        # Wilson centers on full BZ and on strict half BZ including ky=0, pi.
        ky_full = np.linspace(-np.pi, np.pi, nky, endpoint=False)
        nky_half = (nky + 1) // 2
        ky_half = np.linspace(0.0, np.pi, nky_half, endpoint=True)
        centers_full, _, unit_err_full = compute_wilson_centers(
            v=v, t=args.t, w=args.w, lm=args.lm, n_occ=n_occ, nkx=nkx, ky_grid=ky_full
        )
        centers_half, vecs_half, unit_err_half = compute_wilson_centers(
            v=v, t=args.t, w=args.w, lm=args.lm, n_occ=n_occ, nkx=nkx, ky_grid=ky_half
        )
        unit_err = max(unit_err_full, unit_err_half)
        write_centers_csv(out_root / f"wilson_centers_v{token(v)}.csv", ky_full, centers_full)
        if unit_err > 1e-6:
            print(f"[warning] v={v:.1f} Wilson unitarity error={unit_err:.3e} (>1e-6)")

        tracked_half_mod, tracked_half_unwrapped, match_quality = track_branches(centers_half, vecs_half)

        write_tracked_csv(out_root / f"wilson_branches_tracked_v{token(v)}.csv", ky_half, tracked_half_unwrapped)

        # Z2 method A: crossings at multiple references with touch protection.
        adaptive_ref, adaptive_clearance = choose_adaptive_reference(centers_half)
        c45, t45 = crossing_count_half_bz(ky_half, tracked_half_unwrapped, 0.45)
        c50, t50 = crossing_count_half_bz(ky_half, tracked_half_unwrapped, 0.50)
        c55, t55 = crossing_count_half_bz(ky_half, tracked_half_unwrapped, 0.55)
        c_adapt, t_adapt = crossing_count_half_bz(ky_half, tracked_half_unwrapped, adaptive_ref)
        p45, p50, p55 = c45 % 2, c50 % 2, c55 % 2
        stable_cross = bool(p45 == p50 == p55)
        touches_ok = bool(t50 <= 1 and t_adapt <= 1)
        z2_crossing = int(p50) if (stable_cross and touches_ok) else -1

        # Z2 method B: largest gap center.
        z2_largest, largest_winding, largest_gap_margin, largest_reliable = largest_gap_method(centers_half)
        if not largest_reliable:
            z2_largest = -1

        # Spin Chern parity.
        comm_norm = commutator_norm_h_sz(v=v, t=args.t, w=args.w, lm=args.lm, nk=13 if args.quick else 21)
        spin_conserved = bool(comm_norm < 1e-8)
        if spin_conserved:
            cup = chern_fukui(
                lambda kx, ky: h_spin_block(kx, ky, v=v, t=args.t, w=args.w, lm=args.lm, spin_sign=1),
                n_occ=2,
                nk=41 if args.quick else 61,
            )
            cdn = chern_fukui(
                lambda kx, ky: h_spin_block(kx, ky, v=v, t=args.t, w=args.w, lm=args.lm, spin_sign=-1),
                n_occ=2,
                nk=41 if args.quick else 61,
            )
            cspin = 0.5 * (round(cup) - round(cdn))
            z2_expected = int(int(round(cspin)) % 2)
        else:
            cup = np.nan
            cdn = np.nan
            cspin = np.nan
            z2_expected = -1

        consistent_crossing = int(z2_crossing in [0, 1] and z2_crossing == z2_expected)
        consistent_largest = int(z2_largest == z2_expected if z2_expected >= 0 else 0)

        warning_parts: list[str] = []
        if not stable_cross:
            warning_parts.append("reference-line parity is unstable across 0.45/0.50/0.55")
        if not touches_ok:
            warning_parts.append("crossing count touched reference line too closely; parity may be unstable")
        if float(np.min(match_quality)) < 0.5:
            warning_parts.append("low Wilson-eigenvector overlap in branch tracking; possible branch mis-match")
        if not largest_reliable:
            warning_parts.append("largest-gap center is not unique (small max-gap margin); winding may be unstable")
        if z2_crossing != z2_largest and z2_crossing in [0, 1]:
            warning_parts.append("WARNING: crossing and largest-gap Z2 methods disagree.")
        if not reliable_gap:
            warning_parts.append("occupied-gap check: min_direct_gap <= 1e-4; Z2 may be unreliable")

        if z2_crossing in [0, 1] and z2_largest in [0, 1] and z2_crossing == z2_largest and touches_ok and largest_reliable:
            final_z2: str | int = int(z2_crossing)
        else:
            final_z2 = "unresolved"
            if "WARNING: crossing and largest-gap Z2 methods disagree." not in warning_parts:
                warning_parts.append("WARNING: crossing and largest-gap Z2 methods disagree.")

        if isinstance(final_z2, int) and z2_expected >= 0:
            final_consistent = int(final_z2 == z2_expected)
        else:
            final_consistent = 0

        if isinstance(final_z2, int) and z2_expected >= 0 and final_z2 != z2_expected:
            warning_parts.append(
                "WARNING: final Z2 is inconsistent with spin Chern parity. Check Wilson loop branch tracking, occupied bands, or BZ path."
            )

        title_suffix = (
            f"Z2_cross={z2_crossing}, Z2_gap={z2_largest}, "
            f"C_spin_mod2={z2_expected if z2_expected >= 0 else 'NA'}"
        )
        if z2_expected >= 0 and isinstance(final_z2, int) and final_z2 != z2_expected:
            title_suffix += " [inconsistent]"
        plot_raw(
            ky_half,
            centers_half,
            out_root / f"z2_wilson_raw_v{token(v)}.png",
            title=f"v={v:.1f} raw Wannier centers\n{title_suffix}",
        )
        plot_tracked(
            ky_half,
            tracked_half_unwrapped,
            out_root / f"z2_wilson_tracked_v{token(v)}.png",
            title=f"v={v:.1f} tracked Wannier branches\n{title_suffix}",
        )

        summary_rows.append(
            {
                "v": float(v),
                "bulk_gap": float(compute_bulk_gap(v=v, t=args.t, w=args.w, lm=args.lm, nk=41 if args.quick else 61)),
                "C_up": float(cup) if not np.isnan(cup) else "nan",
                "C_down": float(cdn) if not np.isnan(cdn) else "nan",
                "C_spin": float(cspin) if not np.isnan(cspin) else "nan",
                "Z2_expected_from_spin_chern": int(z2_expected) if z2_expected >= 0 else "nan",
                "Z2_crossing": int(z2_crossing),
                "Z2_largest_gap": int(z2_largest),
                "crossing_number_ref_0p45": int(c45),
                "crossing_number_ref_0p50": int(c50),
                "crossing_number_ref_0p55": int(c55),
                "crossing_touch_ref_0p45": int(t45),
                "crossing_touch_ref_0p50": int(t50),
                "crossing_touch_ref_0p55": int(t55),
                "crossing_number_ref_adaptive": int(c_adapt),
                "crossing_touch_ref_adaptive": int(t_adapt),
                "adaptive_ref": float(adaptive_ref),
                "adaptive_ref_clearance": float(adaptive_clearance),
                "largest_gap_winding": float(largest_winding),
                "largest_gap_margin": float(largest_gap_margin),
                "largest_gap_reliable": int(largest_reliable),
                "unitarity_error_max": float(unit_err),
                "min_branch_overlap": float(np.min(match_quality)),
                "consistent_crossing": int(consistent_crossing),
                "consistent_largest_gap": int(consistent_largest),
                "final_Z2": final_z2,
                "final_consistent": int(final_consistent),
                "warning": "; ".join(warning_parts),
            }
        )
        print(
            f"[z2-debug] v={v:.1f} unit_err={unit_err:.2e} "
            f"cross=({c45},{c50},{c55}; adapt={c_adapt}) z2_cross={z2_crossing} "
            f"z2_gap={z2_largest} final={final_z2} expected={z2_expected}"
        )

    write_csv(
        out_root / "occupied_gap_check.csv",
        gap_rows,
        ["v", "n_occ", "min_direct_gap", "gap_kx", "gap_ky", "reliable_gap"],
    )
    write_csv(
        out_root / "z2_debug_summary.csv",
        summary_rows,
        [
            "v",
            "bulk_gap",
            "C_up",
            "C_down",
            "C_spin",
            "Z2_expected_from_spin_chern",
            "Z2_crossing",
            "Z2_largest_gap",
            "crossing_number_ref_0p45",
            "crossing_number_ref_0p50",
            "crossing_number_ref_0p55",
            "crossing_touch_ref_0p45",
            "crossing_touch_ref_0p50",
            "crossing_touch_ref_0p55",
            "crossing_number_ref_adaptive",
            "crossing_touch_ref_adaptive",
            "adaptive_ref",
            "adaptive_ref_clearance",
            "largest_gap_winding",
            "largest_gap_margin",
            "largest_gap_reliable",
            "unitarity_error_max",
            "min_branch_overlap",
            "consistent_crossing",
            "consistent_largest_gap",
            "final_Z2",
            "final_consistent",
            "warning",
        ],
    )

    make_montage(out_root / "z2_wilson_debug_montage.png", v_list, out_root)

    # report
    by_v = {float(r["v"]): r for r in summary_rows}
    spin_cons = all(
        commutator_norm_h_sz(v=v, t=args.t, w=args.w, lm=args.lm, nk=11 if args.quick else 15) < 1e-8 for v in v_list
    )
    v08 = by_v.get(0.8)
    if v08 is not None:
        cspin08 = v08["C_spin"]
        z2cross08 = int(v08["Z2_crossing"])
        z2gap08 = int(v08["Z2_largest_gap"])
        final08 = v08["final_Z2"]
        cons08 = int(v08["final_consistent"])
    else:
        cspin08, z2cross08, z2gap08, final08, cons08 = "nan", -1, -1, "unresolved", 0

    if isinstance(final08, int) and final08 == 1 and str(cspin08) != "nan" and int(round(float(cspin08))) == 1 and cons08 == 1:
        verdict08 = "v=0.8 supports QSH phase."
    elif str(cspin08) != "nan" and int(round(float(cspin08))) == 1 and final08 == 0:
        verdict08 = "QSH diagnosis remains inconsistent. Since spin is conserved and C_spin=1, the Wilson-loop Z2 implementation should be further checked."
    elif str(cspin08) == "nan":
        verdict08 = "QSH diagnosis is inconclusive because spin Chern number is not converged."
    else:
        verdict08 = "QSH diagnosis remains inconclusive."

    report_lines = [
        "Z2 Wilson-loop debug report",
        "===========================",
        "",
        "1. Hamiltonian spin conservation",
        f"- [H, s_z] normalized commutator check indicates spin_conserved={spin_cons}.",
        "",
        "2. C_spin stability at v=0.8",
        f"- C_spin(v=0.8)={cspin08} (from spin block Chern).",
        "",
        "3. Wilson-center odd crossing at v=0.8",
        f"- crossing counts (ref 0.45/0.50/0.55) = "
        f"{v08['crossing_number_ref_0p45'] if v08 else 'NA'}/"
        f"{v08['crossing_number_ref_0p50'] if v08 else 'NA'}/"
        f"{v08['crossing_number_ref_0p55'] if v08 else 'NA'}.",
        f"- Z2_crossing(v=0.8)={z2cross08}.",
        "",
        "4. crossing vs largest-gap method consistency",
        f"- Z2_largest_gap(v=0.8)={z2gap08}.",
        f"- final_Z2(v=0.8)={final08}.",
        "",
        "5. final_Z2 parity vs C_spin mod 2",
        f"- final_consistent(v=0.8)={cons08}.",
        "",
        "6. most likely inconsistency source",
        "- If mismatch persists, likely branch-tracking/crossing counting conventions (half-BZ counting and reference-line stability) rather than Hamiltonian spin structure.",
        "",
        "7. QSH verdict at v=0.8",
        f"- {verdict08}",
    ]
    (out_root / "z2_debug_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[ok] z2 summary: {out_root / 'z2_debug_summary.csv'}")
    print(f"[ok] z2 report: {out_root / 'z2_debug_report.txt'}")


if __name__ == "__main__":
    main()
