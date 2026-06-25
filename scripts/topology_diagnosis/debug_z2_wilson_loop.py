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
) -> tuple[np.ndarray, float]:
    wmat = np.eye(n_occ, dtype=complex)
    for i, kx in enumerate(kx_grid):
        kx_next = float(kx_grid[(i + 1) % kx_grid.size])
        oa = occ_subspace(float(kx), float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
        ob = occ_subspace(kx_next, float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ)
        f = oa.conj().T @ ob
        q = unitary_part(f)
        wmat = q @ wmat
    err = float(np.linalg.norm(wmat.conj().T @ wmat - np.eye(n_occ)))
    eigvals = np.linalg.eigvals(wmat)
    centers = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
    centers = np.sort(centers)
    return centers, err


def compute_wilson_centers(
    v: float,
    t: float,
    w: float,
    lm: float,
    n_occ: int,
    nkx: int,
    nky: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    ky_grid = np.linspace(-np.pi, np.pi, nky, endpoint=False)
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    centers = np.zeros((nky, n_occ), dtype=float)
    max_err = 0.0
    for j, ky in enumerate(ky_grid):
        c, err = wilson_x_per_ky(ky=float(ky), v=v, t=t, w=w, lm=lm, n_occ=n_occ, kx_grid=kx_grid)
        centers[j, :] = c
        max_err = max(max_err, err)
    return ky_grid, centers, float(max_err)


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


def track_branches(centers_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nky, nband = centers_sorted.shape
    tracked_mod = np.zeros_like(centers_sorted)
    tracked_mod[0, :] = centers_sorted[0, :]

    perms = list(itertools.permutations(range(nband)))
    for j in range(1, nky):
        prev = tracked_mod[j - 1, :]
        curr = centers_sorted[j, :]
        best_perm = perms[0]
        best_cost = float("inf")
        for p in perms:
            c = np.array([curr[k] for k in p], dtype=float)
            cost = float(np.sum([circular_distance(prev[b], c[b]) for b in range(nband)]))
            if cost < best_cost:
                best_cost = cost
                best_perm = p
        tracked_mod[j, :] = np.array([curr[k] for k in best_perm], dtype=float)

    tracked_unwrapped = np.zeros_like(tracked_mod)
    tracked_unwrapped[0, :] = tracked_mod[0, :]
    for j in range(1, nky):
        delta = tracked_mod[j, :] - tracked_mod[j - 1, :]
        delta = (delta + 0.5) % 1.0 - 0.5
        tracked_unwrapped[j, :] = tracked_unwrapped[j - 1, :] + delta
    return tracked_mod, tracked_unwrapped


def crossing_count_half_bz(ky_half: np.ndarray, branches_unwrapped_half: np.ndarray, ref: float) -> int:
    count = 0
    nseg = ky_half.size - 1
    eps = 1e-10
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
                if abs(fa) < eps:
                    fa = eps
                if abs(fb) < eps:
                    fb = -eps
                if fa * fb < 0:
                    count += 1
    return int(count)


def largest_gap_center(arr_sorted: np.ndarray) -> float:
    cyc = np.r_[arr_sorted, arr_sorted[0] + 1.0]
    gaps = np.diff(cyc)
    idx = int(np.argmax(gaps))
    start = cyc[idx]
    center = (start + 0.5 * gaps[idx]) % 1.0
    return float(center)


def largest_gap_method(centers_half_sorted: np.ndarray) -> tuple[int, float]:
    centers = np.array([largest_gap_center(row) for row in centers_half_sorted], dtype=float)
    unwrapped = np.zeros_like(centers)
    unwrapped[0] = centers[0]
    for i in range(1, centers.size):
        d = centers[i] - centers[i - 1]
        d = (d + 0.5) % 1.0 - 0.5
        unwrapped[i] = unwrapped[i - 1] + d
    winding_full = float(2.0 * (unwrapped[-1] - unwrapped[0]))
    z2 = int(np.round(winding_full)) % 2
    return z2, winding_full


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

        # Wilson centers.
        ky, centers, unit_err = compute_wilson_centers(v=v, t=args.t, w=args.w, lm=args.lm, n_occ=n_occ, nkx=nkx, nky=nky)
        write_centers_csv(out_root / f"wilson_centers_v{token(v)}.csv", ky, centers)
        if unit_err > 1e-6:
            print(f"[warning] v={v:.1f} Wilson unitarity error={unit_err:.3e} (>1e-6)")

        tracked_mod, tracked_unwrapped = track_branches(centers)
        idx_half = np.where((ky >= 0.0) & (ky <= np.pi))[0]
        ky_half = ky[idx_half]
        centers_half = centers[idx_half, :]
        tracked_half_mod = tracked_mod[idx_half, :]
        tracked_half_unwrapped = tracked_unwrapped[idx_half, :]

        write_tracked_csv(out_root / f"wilson_branches_tracked_v{token(v)}.csv", ky_half, tracked_half_unwrapped)

        # Z2 method A: crossings at multiple references.
        c45 = crossing_count_half_bz(ky_half, tracked_half_unwrapped, 0.45)
        c50 = crossing_count_half_bz(ky_half, tracked_half_unwrapped, 0.50)
        c55 = crossing_count_half_bz(ky_half, tracked_half_unwrapped, 0.55)
        p45, p50, p55 = c45 % 2, c50 % 2, c55 % 2
        stable_cross = bool(p45 == p50 == p55)
        z2_crossing = int(p50) if stable_cross else -1

        # Z2 method B: largest gap center.
        z2_largest, largest_winding = largest_gap_method(centers_half)

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
        if z2_crossing != z2_largest and z2_crossing in [0, 1]:
            warning_parts.append("WARNING: crossing and largest-gap Z2 methods disagree.")
        if not reliable_gap:
            warning_parts.append("occupied-gap check: min_direct_gap <= 1e-4; Z2 may be unreliable")

        if z2_crossing in [0, 1] and z2_crossing == z2_largest:
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
                "largest_gap_winding": float(largest_winding),
                "unitarity_error_max": float(unit_err),
                "consistent_crossing": int(consistent_crossing),
                "consistent_largest_gap": int(consistent_largest),
                "final_Z2": final_z2,
                "final_consistent": int(final_consistent),
                "warning": "; ".join(warning_parts),
            }
        )
        print(
            f"[z2-debug] v={v:.1f} unit_err={unit_err:.2e} "
            f"cross=({c45},{c50},{c55}) z2_cross={z2_crossing} "
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
            "largest_gap_winding",
            "unitarity_error_max",
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
