#!/usr/bin/env python3
"""Fix and diagnose largest-gap Z2 criterion using tracked Wilson branches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.common_topology import ensure_dir
from scripts.topology_diagnosis.debug_z2_wilson_loop import (
    compute_wilson_centers,
    token,
    track_branches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix largest-gap Z2 from Wilson branches.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs/z2_debug/largest_gap_fix")
    parser.add_argument("--z2-root", default="/workspace/topology_diagnosis_outputs/z2_debug")
    parser.add_argument("--v-list", default="0.5,0.8,1.0")
    parser.add_argument("--nkx", type=int, default=101)
    parser.add_argument("--nky", type=int, default=101)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_table(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_tracked_branches(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_table(path)
    if not rows:
        raise RuntimeError(f"tracked branch file is empty: {path}")
    bcols = [c for c in rows[0].keys() if c.startswith("branch_")]
    ky = np.array([float(r["ky"]) for r in rows], dtype=float)
    branches = np.array([[float(r[c]) for c in bcols] for r in rows], dtype=float)
    return ky, branches


def write_tracked_branches(path: Path, ky: np.ndarray, branches: np.ndarray) -> None:
    fields = ["ky"] + [f"branch_{i+1}" for i in range(branches.shape[1])]
    rows = []
    for i, ky_i in enumerate(ky):
        row = {"ky": float(ky_i)}
        for j in range(branches.shape[1]):
            row[f"branch_{j+1}"] = float(branches[i, j])
        rows.append(row)
    write_table(path, rows, fields)


def ensure_tracked_branches(
    v: float,
    z2_root: Path,
    nkx: int,
    nky: int,
    n_occ: int,
    t: float,
    w: float,
    lm: float,
) -> tuple[np.ndarray, np.ndarray]:
    path = z2_root / f"wilson_branches_tracked_v{token(v)}.csv"
    if path.exists():
        return read_tracked_branches(path)

    nky_half = (nky + 1) // 2
    ky_half = np.linspace(0.0, np.pi, nky_half, endpoint=True)
    centers_half, vecs_half, _ = compute_wilson_centers(
        v=v, t=t, w=w, lm=lm, n_occ=n_occ, nkx=nkx, ky_grid=ky_half
    )
    _, tracked_unwrapped, _ = track_branches(centers_half, vecs_half)
    write_tracked_branches(path, ky_half, tracked_unwrapped)
    return ky_half, tracked_unwrapped


def largest_gap_from_centers(centers_unwrapped_row: np.ndarray) -> tuple[float, float, int, np.ndarray]:
    centers_mod = np.mod(centers_unwrapped_row, 1.0)
    centers_sorted = np.sort(centers_mod)
    cyc = np.r_[centers_sorted, centers_sorted[0] + 1.0]
    gaps = np.diff(cyc)
    idx = int(np.argmax(gaps))
    gap_size = float(gaps[idx])

    if idx < centers_sorted.size - 1:
        gap_center = 0.5 * (centers_sorted[idx] + centers_sorted[idx + 1])
    else:
        gap_center = 0.5 * (centers_sorted[-1] + centers_sorted[0] + 1.0)
    gap_center_mod = float(gap_center % 1.0)
    return gap_size, gap_center_mod, idx, centers_sorted


def unwrap_modular_curve(values_mod: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values_mod, dtype=float)
    out[0] = float(values_mod[0])
    for i in range(1, values_mod.size):
        delta = float(values_mod[i] - values_mod[i - 1])
        if delta > 0.5:
            delta -= 1.0
        if delta < -0.5:
            delta += 1.0
        out[i] = out[i - 1] + delta
    return out


def build_spin_converged_map(path: Path) -> dict[float, bool]:
    if not path.exists():
        return {}
    rows = read_table(path)
    by_v: dict[float, list[dict[str, str]]] = {}
    for r in rows:
        v = float(r["v"])
        by_v.setdefault(v, []).append(r)
    out: dict[float, bool] = {}
    for v, group in by_v.items():
        cup_round = {int(float(x["C_up_rounded"])) for x in group}
        cdn_round = {int(float(x["C_down_rounded"])) for x in group}
        err_ok = all(
            abs(float(x["integer_error_up"])) < 1e-6 and abs(float(x["integer_error_down"])) < 1e-6 for x in group
        )
        out[v] = len(cup_round) == 1 and len(cdn_round) == 1 and err_ok
    return out


def plot_raw_curve(ky: np.ndarray, center_mod: np.ndarray, out_png: Path, v: float) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ky, center_mod, color="#1f77b4", linewidth=1.2, marker="o", markersize=2.2)
    ax.axvspan(0.0, np.pi, color="#b3cde3", alpha=0.18, label=r"$k_y \in [0,\pi]$")
    ax.set_xlim(float(np.min(ky)), float(np.max(ky)))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel("largest gap center (mod 1)")
    ax.set_title(f"v={v:.1f} largest-gap center (raw mod-1)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_tracked_curve(
    ky: np.ndarray,
    center_unwrapped: np.ndarray,
    out_png: Path,
    v: float,
    delta_g: float,
    winding: int,
    z2_fixed: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ky, center_unwrapped, color="#d62728", linewidth=1.2)
    ax.set_xlim(0.0, np.pi)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel("largest gap center (unwrapped)")
    ax.set_title(
        f"v={v:.1f} tracked largest-gap center\n"
        f"delta_g={delta_g:.6f}, winding={winding}, Z2_fixed={z2_fixed}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_compare(
    ky: np.ndarray,
    branches_unwrapped: np.ndarray,
    center_mod: np.ndarray,
    out_png: Path,
    v: float,
    z2_crossing: int,
    z2_fixed: int,
    z2_expected: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for b in range(branches_unwrapped.shape[1]):
        ax.plot(ky, np.mod(branches_unwrapped[:, b], 1.0), color="#7f7f7f", linewidth=0.9, alpha=0.85)
    ax.plot(ky, center_mod, color="#d62728", linewidth=1.5, label="largest-gap center")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9, label=r"$\nu_{\mathrm{ref}}=0.5$")
    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel("Wannier center / largest-gap center")
    ax.set_title(
        f"v={v:.1f}: Z2_crossing={z2_crossing}, Z2_largest_gap_fixed={z2_fixed}, C_spin mod2={z2_expected}"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def make_montage(v_list: list[float], out_root: Path, out_png: Path) -> None:
    n = len(v_list)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.2), squeeze=False)
    for i, v in enumerate(v_list):
        ax = axes[0][i]
        p = out_root / f"z2_largest_gap_compare_v{token(v)}.png"
        if p.exists():
            ax.imshow(plt.imread(p))
            ax.axis("off")
            ax.set_title(f"v={v:.1f}")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
    fig.suptitle("Largest-gap fix comparison montage", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    z2_root = Path(args.z2_root)
    v_list = parse_float_list(args.v_list)

    old_summary_path = z2_root / "z2_debug_summary.csv"
    old_rows = read_table(old_summary_path)
    old_by_v = {float(r["v"]): r for r in old_rows}

    spin_conv_map = build_spin_converged_map(z2_root / "spin_chern_convergence.csv")

    fix_rows: list[dict] = []
    qsh_rows: list[dict] = []
    report_lines = [
        "Largest-gap Z2 fix report",
        "=========================",
        "",
        "1. ky range used for largest-gap winding",
        "- All largest-gap winding evaluations use half BZ ky in [0, pi].",
        "",
        "2. largest-gap center unwrap",
        "- The largest-gap center is first computed in mod-1 and then unwrapped continuously using +/-1 correction when jump exceeds 0.5.",
        "",
    ]

    for v in v_list:
        if v not in old_by_v:
            raise RuntimeError(f"v={v:.3f} not found in existing z2 summary: {old_summary_path}")

        ky, branches_unwrapped = ensure_tracked_branches(
            v=v,
            z2_root=z2_root,
            nkx=int(args.nkx),
            nky=int(args.nky),
            n_occ=int(args.n_occ),
            t=float(args.t),
            w=float(args.w),
            lm=float(args.lm),
        )

        if not np.all((ky >= -1e-12) & (ky <= np.pi + 1e-12)):
            raise RuntimeError("tracked ky grid is not limited to half BZ [0, pi]")

        gap_sizes = np.zeros_like(ky, dtype=float)
        gap_centers_mod = np.zeros_like(ky, dtype=float)
        gap_indices = np.zeros_like(ky, dtype=int)
        centers_sorted_list: list[np.ndarray] = []

        for i in range(ky.size):
            gsz, gctr, gidx, csorted = largest_gap_from_centers(branches_unwrapped[i, :])
            gap_sizes[i] = gsz
            gap_centers_mod[i] = gctr
            gap_indices[i] = gidx
            centers_sorted_list.append(csorted)

        gap_unwrapped = unwrap_modular_curve(gap_centers_mod)
        delta_g = float(gap_unwrapped[-1] - gap_unwrapped[0])
        winding = int(np.round(delta_g))
        z2_fixed = int(winding % 2)
        winding_reliable = bool(abs(delta_g - round(delta_g)) < 0.25)

        raw_rows = []
        tracked_rows = []
        for i, ky_i in enumerate(ky):
            raw_rows.append(
                {
                    "ky": float(ky_i),
                    "largest_gap_size": float(gap_sizes[i]),
                    "largest_gap_center_mod": float(gap_centers_mod[i]),
                    "largest_gap_index": int(gap_indices[i]),
                    "centers_sorted": ";".join(f"{x:.16f}" for x in centers_sorted_list[i]),
                }
            )
            tracked_rows.append(
                {
                    "ky": float(ky_i),
                    "gap_center_mod": float(gap_centers_mod[i]),
                    "gap_center_unwrapped": float(gap_unwrapped[i]),
                    "largest_gap_size": float(gap_sizes[i]),
                    "largest_gap_index": int(gap_indices[i]),
                }
            )

        write_table(
            out_root / f"largest_gap_center_raw_v{token(v)}.csv",
            raw_rows,
            ["ky", "largest_gap_size", "largest_gap_center_mod", "largest_gap_index", "centers_sorted"],
        )
        write_table(
            out_root / f"largest_gap_center_tracked_v{token(v)}.csv",
            tracked_rows,
            ["ky", "gap_center_mod", "gap_center_unwrapped", "largest_gap_size", "largest_gap_index"],
        )

        old = old_by_v[v]
        cspin = float(old["C_spin"])
        z2_expected = int(float(old["Z2_expected_from_spin_chern"]))
        z2_crossing = int(float(old["Z2_crossing"]))
        z2_largest_old = int(float(old["Z2_largest_gap"]))
        bulk_gap = float(old["bulk_gap"])
        spin_conv = spin_conv_map.get(v, abs(cspin - round(cspin)) < 1e-6)

        warning_parts: list[str] = []
        if not winding_reliable:
            warning_parts.append(
                "WARNING: largest gap center winding is not close to an integer. The largest-gap method is unreliable."
            )
        if z2_fixed != z2_crossing:
            warning_parts.append("WARNING: largest-gap fixed Z2 disagrees with stable reference-line crossing Z2.")
        if v in (0.8, 1.0) and z2_fixed == 0 and z2_crossing == 1 and z2_expected == 1:
            warning_parts.append(
                "WARNING: largest-gap method still disagrees with both crossing Z2 and spin Chern parity. The largest-gap implementation or gap-center tracking is likely unreliable."
            )

        consistent_crossing = bool(z2_fixed == z2_crossing)
        consistent_spin = bool(z2_fixed == z2_expected)

        fix_rows.append(
            {
                "v": float(v),
                "C_spin": float(cspin),
                "Z2_expected_from_spin_chern": int(z2_expected),
                "Z2_crossing": int(z2_crossing),
                "Z2_largest_gap_old": int(z2_largest_old),
                "Z2_largest_gap_fixed": int(z2_fixed),
                "delta_g": float(delta_g),
                "largest_gap_winding": int(winding),
                "winding_reliable": int(winding_reliable),
                "consistent_with_crossing": int(consistent_crossing),
                "consistent_with_spin_chern": int(consistent_spin),
                "warning": "; ".join(warning_parts),
            }
        )

        if z2_crossing == z2_fixed:
            z2_final: int | str = int(z2_crossing)
            qsh_comment = "crossing and largest-gap fixed results agree"
        elif z2_crossing != z2_fixed and z2_expected == z2_crossing and spin_conv:
            z2_final = int(z2_crossing)
            qsh_comment = (
                "largest-gap method inconsistent; final Z2 follows spin Chern parity and stable crossing result"
            )
        else:
            z2_final = "unresolved"
            qsh_comment = "crossing/largest-gap/spin-Chern parity are not jointly consistent"

        qsh_supported = bool(int(round(cspin)) == 1 and z2_final == 1)
        qsh_rows.append(
            {
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "C_spin": float(cspin),
                "Z2_crossing": int(z2_crossing),
                "Z2_largest_gap_fixed": int(z2_fixed),
                "Z2_final": z2_final,
                "QSH_supported": int(qsh_supported),
                "comment": qsh_comment,
            }
        )

        plot_raw_curve(ky, gap_centers_mod, out_root / f"largest_gap_center_raw_v{token(v)}.png", v=v)
        plot_tracked_curve(
            ky,
            gap_unwrapped,
            out_root / f"largest_gap_center_tracked_v{token(v)}.png",
            v=v,
            delta_g=delta_g,
            winding=winding,
            z2_fixed=z2_fixed,
        )
        plot_compare(
            ky=ky,
            branches_unwrapped=branches_unwrapped,
            center_mod=gap_centers_mod,
            out_png=out_root / f"z2_largest_gap_compare_v{token(v)}.png",
            v=v,
            z2_crossing=z2_crossing,
            z2_fixed=z2_fixed,
            z2_expected=z2_expected,
        )

        verdict = "inconclusive"
        if abs(v - 0.5) < 1e-9 and int(round(cspin)) == 0 and z2_crossing == 0 and z2_fixed == 0:
            verdict = "v=0.5 is Z2-trivial and does not belong to the QSH phase."
        if abs(v - 0.8) < 1e-9:
            if int(round(cspin)) == 1 and z2_crossing == 1 and z2_fixed == 1:
                verdict = "The QSH diagnosis at v=0.8 is fully consistent. The system supports a QSH phase."
            elif int(round(cspin)) == 1 and z2_crossing == 1 and z2_fixed == 0:
                verdict = (
                    "The QSH diagnosis at v=0.8 is supported by spin Chern number and reference-line Wilson-loop crossing. "
                    "The largest-gap method remains inconsistent and should be regarded as unreliable in the current implementation."
                )
        if abs(v - 1.0) < 1e-9 and int(round(cspin)) == 1 and z2_crossing == 1 and z2_fixed == 0:
            verdict = (
                "v=1.0 shows the same pattern as v=0.8: crossing and spin Chern parity support QSH, "
                "while largest-gap remains inconsistent."
            )
        if abs(v - 1.0) < 1e-9 and int(round(cspin)) == 1 and z2_crossing == 1 and z2_fixed == 1:
            verdict = "v=1.0 is also fully consistent with QSH indicators (spin Chern parity, crossing, and fixed largest-gap)."

        report_lines.extend(
            [
                f"{3 if abs(v-0.5)<1e-9 else (4 if abs(v-0.8)<1e-9 else 5)}. v={v:.1f} summary",
                f"- C_spin={cspin:.6f}",
                f"- Z2_crossing={z2_crossing}",
                f"- Z2_largest_gap_fixed={z2_fixed}",
                f"- delta_g={delta_g:.6f}, largest_gap_winding={winding}, winding_reliable={winding_reliable}",
                f"- Conclusion: {verdict}",
                "",
            ]
        )

    make_montage(v_list=v_list, out_root=out_root, out_png=out_root / "largest_gap_fix_montage.png")

    write_table(
        out_root / "largest_gap_fix_summary.csv",
        fix_rows,
        [
            "v",
            "C_spin",
            "Z2_expected_from_spin_chern",
            "Z2_crossing",
            "Z2_largest_gap_old",
            "Z2_largest_gap_fixed",
            "delta_g",
            "largest_gap_winding",
            "winding_reliable",
            "consistent_with_crossing",
            "consistent_with_spin_chern",
            "warning",
        ],
    )
    write_table(
        out_root / "qsh_invariant_summary_fixed.csv",
        qsh_rows,
        [
            "v",
            "bulk_gap",
            "C_spin",
            "Z2_crossing",
            "Z2_largest_gap_fixed",
            "Z2_final",
            "QSH_supported",
            "comment",
        ],
    )

    solved = all(int(r["consistent_with_crossing"]) == 1 for r in fix_rows if float(r["v"]) > 0.6)
    solved_all = all(int(r["consistent_with_crossing"]) == 1 for r in fix_rows)
    report_lines.extend(
        [
            "6. Whether largest-gap/crossing contradiction is resolved",
            f"- Resolved for target QSH region v>0.6: {solved}.",
            f"- Resolved across all tested v values: {solved_all}.",
            "",
        ]
    )
    if solved:
        report_lines.extend(
            [
                "7. Most likely cause of previous mismatch",
                "- Previous largest-gap implementation used a fragile center-branch selection that did not enforce the strict per-ky largest-gap definition before unwrap.",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "7. Most likely reason if mismatch persists",
                "- Even with corrected per-ky largest-gap extraction and unwrap, the largest-gap label can switch between competing near-degenerate gaps, making this method unstable for this dataset.",
                "",
            ]
        )

    v08_qsh = [r for r in qsh_rows if abs(float(r["v"]) - 0.8) < 1e-9]
    v10_qsh = [r for r in qsh_rows if abs(float(r["v"]) - 1.0) < 1e-9]
    qsh_supported = bool(v08_qsh and int(v08_qsh[0]["QSH_supported"]) == 1)
    qsh_supported = qsh_supported or bool(v10_qsh and int(v10_qsh[0]["QSH_supported"]) == 1)
    report_lines.extend(
        [
            "8. Final judgment for v>0.6 QSH",
            (
                "- QSH is supported for v>0.6 by converged spin Chern parity and stable crossing Z2; "
                "largest-gap should be treated as a secondary check and marked unreliable when contradictory."
                if qsh_supported
                else "- QSH remains unresolved in this fixed-largest-gap run."
            ),
            "",
        ]
    )

    (out_root / "largest_gap_fix_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[ok] largest-gap fix summary: {out_root / 'largest_gap_fix_summary.csv'}")
    print(f"[ok] largest-gap fix report: {out_root / 'largest_gap_fix_report.txt'}")
    print(f"[ok] qsh fixed summary: {out_root / 'qsh_invariant_summary_fixed.csv'}")


if __name__ == "__main__":
    main()
