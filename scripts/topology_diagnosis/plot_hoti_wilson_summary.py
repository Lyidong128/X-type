#!/usr/bin/env python3
"""Aggregate Wilson/nested/open-open diagnostics for HOTI decision."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and summarize HOTI Wilson diagnostics.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs/wilson_polarization_check")
    parser.add_argument("--open-open-summary", default="")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(row: dict[str, str] | None, key: str, default: float = float("nan")) -> float:
    if row is None:
        return default
    val = row.get(key, "")
    if val in ("", "nan", "None"):
        return default
    return float(val)


def safe_int(row: dict[str, str] | None, key: str, default: int = 0) -> int:
    if row is None:
        return default
    val = row.get(key, "")
    if val in ("", "nan", "None"):
        return default
    return int(float(val))


def find_open_open_summary(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    candidates = [
        Path("/workspace/topology_diagnosis_outputs/open_open_corner_summary.csv"),
        Path("/workspace/topology_diagnosis_outputs/open_open_corner_check/open_open_corner_summary.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("open_open_corner_summary.csv not found.")


def nearest_row(rows: list[dict[str, str]], v: float, tol: float = 1e-8) -> dict[str, str] | None:
    matched = [r for r in rows if abs(float(r["v"]) - v) <= tol]
    if not matched:
        return None
    matched.sort(
        key=lambda r: (
            abs(float(r.get("abs_energy", "nan"))),
            -float(r.get("Lx", "0")) * float(r.get("Ly", "0")),
        )
    )
    return matched[0]


def make_hoti_diagnosis(
    pol_rows: list[dict[str, str]],
    nested_rows: list[dict[str, str]],
    open_rows: list[dict[str, str]],
    qsh_rows: list[dict[str, str]],
) -> list[dict]:
    nested_by_v = {float(r["v"]): r for r in nested_rows}
    qsh_by_v = {float(r["v"]): r for r in qsh_rows}
    out: list[dict] = []
    for r in pol_rows:
        v = float(r["v"])
        px = float(r["p_x"])
        py = float(r["p_y"])
        px_cls = r["p_x_class"]
        py_cls = r["p_y_class"]
        ptype = r["polarization_type"]
        gap_x = float(r["wannier_gap_x"])
        gap_y = float(r["wannier_gap_y"])

        nr = nested_by_v.get(v)
        nested_reliable = bool(safe_int(nr, "nested_reliable", 0) == 1)
        qxy = safe_float(nr, "qxy")
        nested_supports = bool(safe_int(nr, "nested_supports_hoti", 0) == 1)

        orow = nearest_row(open_rows, v=v)
        w_corner = safe_float(orow, "W_corner")
        w_edge = safe_float(orow, "W_edge")
        w_bulk = safe_float(orow, "W_bulk")
        ocls = (orow or {}).get("classification", "no_open_open_data")

        cspin = safe_float(qsh_by_v.get(v), "C_spin")
        z2_crossing = safe_int(qsh_by_v.get(v), "Z2_crossing", -1)

        final = "outside_target_region"
        comment = "Primary HOTI/SSH-like diagnosis target is v < 0.6."

        both_half = px_cls == "nontrivial_0p5" and py_cls == "nontrivial_0p5"
        one_half = (px_cls == "nontrivial_0p5") or (py_cls == "nontrivial_0p5")
        not_quantized = px_cls == "not_quantized" and py_cls == "not_quantized"
        edge_dom = np.isfinite(w_edge) and np.isfinite(w_corner) and (w_edge > w_corner)
        corner_dom = np.isfinite(w_corner) and (w_corner > 0.6)

        if v >= 0.6 and np.isfinite(cspin) and abs(cspin - 1.0) < 0.1 and z2_crossing == 1:
            final = "QSH_region_not_HOTI_target"
            comment = "This point is consistent with QSH (C_spin≈1, Z2 crossing=1), outside the v<0.6 HOTI/SSH target regime."
        elif both_half and nested_reliable and np.isfinite(qxy) and abs(qxy - 0.5) < 0.05 and corner_dom:
            final = "HOTI_supported"
            comment = (
                "Bulk polarization, nested Wilson loop, and corner-localized open-boundary states consistently support HOTI."
            )
        elif v < 0.6 and one_half and (not nested_reliable) and edge_dom:
            final = "SSH_like_edge_phase_not_HOTI"
            comment = (
                "Wilson loop indicates SSH-like polarization, but nested Wilson loop is unreliable and open-open states are edge-localized rather than corner-localized."
            )
        elif v < 0.6 and not_quantized and edge_dom:
            final = "edge_localized_trivial_or_termination_induced"
            comment = "No quantized bulk polarization or corner-state evidence is found."
        elif v < 0.6 and edge_dom:
            final = "SSH_like_edge_phase_not_HOTI"
            comment = "Z2-trivial SSH-like edge-localized phase."

        out.append(
            {
                "v": float(v),
                "bulk_gap": float(r["bulk_gap"]),
                "p_x": float(px),
                "p_y": float(py),
                "polarization_type": ptype,
                "wannier_gap_x": float(gap_x),
                "wannier_gap_y": float(gap_y),
                "nested_reliable": int(nested_reliable),
                "qxy": float(qxy) if np.isfinite(qxy) else "nan",
                "nested_supports_hoti": int(nested_supports),
                "W_corner": float(w_corner) if np.isfinite(w_corner) else "nan",
                "W_edge": float(w_edge) if np.isfinite(w_edge) else "nan",
                "W_bulk": float(w_bulk) if np.isfinite(w_bulk) else "nan",
                "open_open_classification": ocls,
                "final_hoti_diagnosis": final,
                "comment": comment,
            }
        )
    return out


def make_plots(
    diag_rows: list[dict],
    nested_rows: list[dict[str, str]],
    out_root: Path,
) -> None:
    v = np.array([float(r["v"]) for r in diag_rows], dtype=float)
    bulk_gap = np.array([float(r["bulk_gap"]) for r in diag_rows], dtype=float)
    px = np.array([float(r["p_x"]) for r in diag_rows], dtype=float)
    py = np.array([float(r["p_y"]) for r in diag_rows], dtype=float)
    gx = np.array([float(r["wannier_gap_x"]) for r in diag_rows], dtype=float)
    gy = np.array([float(r["wannier_gap_y"]) for r in diag_rows], dtype=float)
    wc = np.array([safe_float(r, "W_corner") for r in diag_rows], dtype=float)
    we = np.array([safe_float(r, "W_edge") for r in diag_rows], dtype=float)
    wb = np.array([safe_float(r, "W_bulk") for r in diag_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    ax = axes[0, 0]
    ax.plot(v, bulk_gap, marker="o", linewidth=1.2)
    ax.axvline(0.6, color="gray", linestyle="--", linewidth=0.9)
    ax.set_title("Panel 1: bulk_gap vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("bulk_gap")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(v, px, marker="o", linewidth=1.2, label="p_x")
    ax.plot(v, py, marker="s", linewidth=1.2, label="p_y")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    ax.set_title("Panel 2: p_x, p_y vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("polarization")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(v, gx, marker="o", linewidth=1.2, label="wannier_gap_x")
    ax.plot(v, gy, marker="s", linewidth=1.2, label="wannier_gap_y")
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.9)
    ax.set_title("Panel 3: Wannier gaps vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("gap")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    mask = np.isfinite(wc) & np.isfinite(we) & np.isfinite(wb)
    if np.any(mask):
        ax.plot(v[mask], wc[mask], marker="o", linewidth=1.2, label="W_corner")
        ax.plot(v[mask], we[mask], marker="s", linewidth=1.2, label="W_edge")
        ax.plot(v[mask], wb[mask], marker="^", linewidth=1.2, label="W_bulk")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.9)
    ax.set_title("Panel 4: W_corner/W_edge/W_bulk vs v")
    ax.set_xlabel("v")
    ax.set_ylabel("weight")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_root / "hoti_wilson_summary.png", dpi=180)
    plt.close(fig)

    if nested_rows:
        nv = np.array([float(r["v"]) for r in nested_rows], dtype=float)
        qxy = np.array([safe_float(r, "qxy") for r in nested_rows], dtype=float)
        rel = np.array([safe_int(r, "nested_reliable", 0) == 1 for r in nested_rows], dtype=bool)

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        if np.any(rel):
            ax.scatter(nv[rel], qxy[rel], s=46, facecolors="tab:blue", edgecolors="black", label="reliable")
        if np.any(~rel):
            ax.scatter(nv[~rel], qxy[~rel], s=46, facecolors="none", edgecolors="gray", label="unreliable")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("v")
        ax.set_ylabel("qxy")
        ax.set_title("Nested qxy reliability check")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_root / "qxy_hoti_check.png", dpi=180)
        plt.close(fig)


def build_report(
    diag_rows: list[dict],
    nested_rows: list[dict[str, str]],
    out_root: Path,
) -> None:
    by_v = {float(r["v"]): r for r in diag_rows}
    subset_lt06 = [r for r in diag_rows if float(r["v"]) < 0.6]
    half_quant_v = [float(r["v"]) for r in subset_lt06 if r["p_x_class"] == "nontrivial_0p5" or r["p_y_class"] == "nontrivial_0p5"] if False else []
    # explicit quantization checks from stored classifications
    quant_lines = []
    for r in subset_lt06:
        v = float(r["v"])
        ptype = r["polarization_type"]
        quant_lines.append(f"  v={v:.1f}: {ptype} (p_x={float(r['p_x']):.4f}, p_y={float(r['p_y']):.4f})")

    nested_by_v = {float(r["v"]): r for r in nested_rows}
    reliable_vs = [float(v) for v, r in nested_by_v.items() if safe_int(r, "nested_reliable", 0) == 1]
    qxy_half_vs = [float(v) for v, r in nested_by_v.items() if safe_int(r, "nested_reliable", 0) == 1 and abs(safe_float(r, "qxy") - 0.5) < 0.05]

    v05 = by_v.get(0.5)
    if v05 is None:
        raise RuntimeError("v=0.5 is missing in hoti diagnosis summary.")
    w_corner_05 = safe_float(v05, "W_corner")
    w_edge_05 = safe_float(v05, "W_edge")
    w_bulk_05 = safe_float(v05, "W_bulk")
    cls_05 = v05.get("open_open_classification", "unknown")
    final_05 = v05.get("final_hoti_diagnosis", "unknown")

    z2_path = Path("/workspace/topology_diagnosis_outputs/z2_debug/z2_debug_summary.csv")
    cspin05 = "nan"
    z2cross05 = "nan"
    if z2_path.exists():
        z2_rows = read_csv(z2_path)
        for r in z2_rows:
            if abs(float(r["v"]) - 0.5) < 1e-9:
                cspin05 = r.get("C_spin", "nan")
                z2cross05 = r.get("Z2_crossing", "nan")
                break

    lines = [
        "HOTI Wilson/polarization diagnosis report",
        "=======================================",
        "",
        "1) Does v < 0.6 show quantized Wilson-loop polarization p_x, p_y?",
        *quant_lines,
        "",
        "2) If p_x or p_y is near 0.5, what does it imply?",
        "- It indicates SSH-like boundary polarization tendency (dipole-like boundary topology), not a standalone HOTI proof.",
        "",
        "3) Are Wannier-sector gaps open?",
        "- Use `wannier_gap_x`, `wannier_gap_y` with threshold 0.05 from `wilson_polarization_summary.csv`.",
        "",
        "4) Is nested Wilson loop reliable?",
        f"- reliable v values: {reliable_vs if reliable_vs else 'none'}",
        "",
        "5) Does nested Wilson support qxy = 0.5?",
        f"- reliable points with qxy≈0.5: {qxy_half_vs if qxy_half_vs else 'none'}",
        "",
        "6) Do open-open finite-size results show corner states?",
        f"- v=0.5 nearest |E| state: W_corner={w_corner_05:.6f}, W_edge={w_edge_05:.6f}, W_bulk={w_bulk_05:.6f}, classification={cls_05}.",
        "",
        "7) What is v=0.5 finally?",
        (
            "- v = 0.5 should be identified as a Z2-trivial SSH-like edge-localized phase rather than a confirmed higher-order topological insulator."
            if (w_edge_05 > w_corner_05 and final_05 != "HOTI_supported")
            else "- v=0.5 does not show dominant corner localization required for HOTI."
        ),
        f"- supporting indicators: C_spin(v=0.5)={cspin05}, Z2_crossing(v=0.5)={z2cross05}.",
        "",
        "8) Why ordinary Wilson loop alone cannot prove HOTI?",
        "- Ordinary Wilson-loop polarization diagnoses dipole-like boundary topology, but HOTI requires consistent higher-order evidence: reliable nested Wilson (quadrupole-like indicator) and dominant corner-localized open-boundary states.",
        "",
        "9) Recommended naming for v < 0.6 in report/paper",
        (
            "- The quantized Wilson-loop polarization suggests SSH-like boundary topology, but the absence of dominant corner localization prevents a HOTI assignment."
            if (v05.get("polarization_type") in ("2D_SSH_like_polarized", "1D_SSH_like_polarized") and w_edge_05 > w_corner_05)
            else "- No bulk Wilson-loop evidence for SSH-like polarization is found, and the observed boundary states are likely termination-induced."
        ),
        "",
        f"Final label at v=0.5: {final_05}.",
    ]

    (out_root / "hoti_wilson_diagnosis_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))

    pol_path = out_root / "wilson_polarization_summary.csv"
    nested_path = out_root / "nested_hoti_check_summary.csv"
    if not pol_path.exists():
        raise FileNotFoundError(f"{pol_path} not found. Please run run_wilson_polarization_check.py first.")
    if not nested_path.exists():
        raise FileNotFoundError(f"{nested_path} not found. Please run run_nested_wilson_for_hoti_check.py first.")

    open_path = find_open_open_summary(args.open_open_summary)
    pol_rows = read_csv(pol_path)
    nested_rows = read_csv(nested_path)
    open_rows = read_csv(open_path)
    z2_path = Path("/workspace/topology_diagnosis_outputs/z2_debug/z2_debug_summary.csv")
    qsh_rows = read_csv(z2_path) if z2_path.exists() else []

    diag_rows = make_hoti_diagnosis(pol_rows, nested_rows, open_rows, qsh_rows=qsh_rows)
    write_csv(
        out_root / "hoti_diagnosis_summary.csv",
        diag_rows,
        [
            "v",
            "bulk_gap",
            "p_x",
            "p_y",
            "polarization_type",
            "wannier_gap_x",
            "wannier_gap_y",
            "nested_reliable",
            "qxy",
            "nested_supports_hoti",
            "W_corner",
            "W_edge",
            "W_bulk",
            "open_open_classification",
            "final_hoti_diagnosis",
            "comment",
        ],
    )

    make_plots(diag_rows, nested_rows, out_root=out_root)
    build_report(diag_rows, nested_rows, out_root=out_root)
    print(f"[ok] saved {out_root / 'hoti_diagnosis_summary.csv'}")
    print(f"[ok] saved {out_root / 'hoti_wilson_summary.png'}")
    if (out_root / "qxy_hoti_check.png").exists():
        print(f"[ok] saved {out_root / 'qxy_hoti_check.png'}")
    print(f"[ok] saved {out_root / 'hoti_wilson_diagnosis_report.txt'}")


if __name__ == "__main__":
    main()
