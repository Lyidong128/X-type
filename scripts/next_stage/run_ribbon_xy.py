#!/usr/bin/env python3
"""Compute ribbon spectra for x-open and y-open geometries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.next_stage.common import auto_select_points, load_model, set_params, summarize_points, write_csv, write_text
from scripts.run_scan import build_onsite_cell_matrix_ribbon, build_ribbon_hamiltonian


def build_ribbon_hamiltonian_yopen(v: float, t: float, lm: float, kx: float, w: float, j: float, ny: int) -> np.ndarray:
    """Ribbon Hamiltonian with y-open / x-periodic boundary condition."""
    dim = 8 * ny
    ham = np.zeros((dim, dim), dtype=complex)

    # onsite with x-periodic hopping on (0,3), y-internal uses only v on (1,2).
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v + w * np.exp(-1j * kx)
    h0[1, 0] = t
    h0[1, 2] = v
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v
    h0[2, 3] = t
    h0[3, 0] = v + w * np.exp(1j * kx)
    h0[3, 1] = t
    h0[3, 2] = t

    h1 = np.zeros((4, 4), dtype=complex)
    h1[0, 1] = 1j * lm
    h1[0, 2] = -1j * lm
    h1[2, 3] = -1j * lm
    h1[3, 2] = 1j * lm
    h1[3, 1] = -1j * lm
    h1[1, 0] = -1j * lm
    h1[2, 0] = 1j * lm
    h1[1, 3] = 1j * lm
    h3 = np.diag([j, -j, j, -j]).astype(complex)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    onsite = np.kron(h0, s0) + np.kron(h1, sz) + np.kron(h3, sz)

    ty = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        ty[1 * 2 + spin, 2 * 2 + spin] = w

    for y in range(ny):
        start = y * 8
        ham[start : start + 8, start : start + 8] += onsite
        if y > 0:
            prev = (y - 1) * 8
            ham[start : start + 8, prev : prev + 8] += ty
            ham[prev : prev + 8, start : start + 8] += ty.conj().T
    return ham


def spectrum_xopen(v: float, t: float, lm: float, w: float, nx: int, nk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x-open / y-periodic spectrum and edge weights."""
    ky = np.linspace(-np.pi, np.pi, nk)
    all_e = []
    all_w = []
    edge_cells = 2
    for k in ky:
        ham = build_ribbon_hamiltonian(v=v, t=t, lm=lm, ky=k, w=w, j=0.0, nx=nx)
        e, vec = np.linalg.eigh(ham)
        e = np.real(e)
        prob = np.abs(vec) ** 2
        prob_cell = prob.reshape(nx, 8, prob.shape[1]).sum(axis=1)
        edge_weight = prob_cell[:edge_cells, :].sum(axis=0) + prob_cell[-edge_cells:, :].sum(axis=0)
        all_e.append(e)
        all_w.append(edge_weight)
    return ky, np.array(all_e), np.array(all_w)


def spectrum_yopen(v: float, t: float, lm: float, w: float, ny: int, nk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y-open / x-periodic spectrum and edge weights."""
    kx = np.linspace(-np.pi, np.pi, nk)
    all_e = []
    all_w = []
    edge_cells = 2
    for k in kx:
        ham = build_ribbon_hamiltonian_yopen(v=v, t=t, lm=lm, kx=k, w=w, j=0.0, ny=ny)
        e, vec = np.linalg.eigh(ham)
        e = np.real(e)
        prob = np.abs(vec) ** 2
        prob_cell = prob.reshape(ny, 8, prob.shape[1]).sum(axis=1)
        edge_weight = prob_cell[:edge_cells, :].sum(axis=0) + prob_cell[-edge_cells:, :].sum(axis=0)
        all_e.append(e)
        all_w.append(edge_weight)
    return kx, np.array(all_e), np.array(all_w)


def classify_gap_state(energies: np.ndarray, edge_weights: np.ndarray, edge_thr: float = 0.55) -> dict[str, float | int | str]:
    """Classify ribbon in-gap state morphology."""
    flat_e = energies.reshape(-1)
    flat_w = edge_weights.reshape(-1)
    neg = flat_e[flat_e < 0]
    pos = flat_e[flat_e > 0]
    if neg.size == 0 or pos.size == 0:
        return {
            "classification": "uncertain",
            "lower_edge": np.nan,
            "upper_edge": np.nan,
            "gap_width": 0.0,
            "in_gap_edge_count": 0,
            "span_ratio": 0.0,
            "touches_lower": 0,
            "touches_upper": 0,
            "comment": "unable_to_identify_signed_gap",
        }
    lower = float(np.max(neg))
    upper = float(np.min(pos))
    width = upper - lower
    if width <= 1e-10:
        return {
            "classification": "uncertain",
            "lower_edge": lower,
            "upper_edge": upper,
            "gap_width": width,
            "in_gap_edge_count": 0,
            "span_ratio": 0.0,
            "touches_lower": 0,
            "touches_upper": 0,
            "comment": "collapsed_gap",
        }

    mask_in_gap = (flat_e > lower) & (flat_e < upper)
    mask_edge = flat_w >= edge_thr
    pick = np.where(mask_in_gap & mask_edge)[0]
    if pick.size == 0:
        return {
            "classification": "empty_gap",
            "lower_edge": lower,
            "upper_edge": upper,
            "gap_width": width,
            "in_gap_edge_count": 0,
            "span_ratio": 0.0,
            "touches_lower": 0,
            "touches_upper": 0,
            "comment": "no_edge_state_in_gap",
        }

    edge_e = flat_e[pick]
    span = float(np.max(edge_e) - np.min(edge_e))
    span_ratio = span / max(width, 1e-12)
    touch_tol = 0.08 * width
    touches_lower = int(np.min(np.abs(edge_e - lower)) <= touch_tol)
    touches_upper = int(np.min(np.abs(edge_e - upper)) <= touch_tol)
    if touches_lower and touches_upper and span_ratio >= 0.7:
        cls = "connecting_edge_branch"
    elif span_ratio < 0.25:
        cls = "isolated_midgap_state"
    else:
        cls = "uncertain"
    return {
        "classification": cls,
        "lower_edge": lower,
        "upper_edge": upper,
        "gap_width": width,
        "in_gap_edge_count": int(pick.size),
        "span_ratio": float(span_ratio),
        "touches_lower": touches_lower,
        "touches_upper": touches_upper,
        "comment": "auto_rule",
    }


def save_ribbon_plot(
    k: np.ndarray,
    energies: np.ndarray,
    edge_weights: np.ndarray,
    out_png: Path,
    title: str,
    k_label: str,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    x = np.repeat(k / np.pi, energies.shape[1])
    y = energies.reshape(-1)
    c = edge_weights.reshape(-1)
    sc = ax.scatter(x, y, c=c, s=6, cmap="viridis", edgecolors="none", vmin=0.0, vmax=1.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(k_label)
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    cb = fig.colorbar(sc, ax=ax, label="edge weight")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def dump_ribbon_csv(path: Path, k: np.ndarray, energies: np.ndarray, edge_weights: np.ndarray, k_name: str) -> None:
    rows = []
    for i in range(energies.shape[0]):
        for b in range(energies.shape[1]):
            rows.append(
                {
                    k_name: float(k[i]),
                    "band_index": int(b),
                    "energy": float(energies[i, b]),
                    "edge_weight": float(edge_weights[i, b]),
                }
            )
    write_csv(path, rows, [k_name, "band_index", "energy", "edge_weight"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ribbon spectra for x-open and y-open directions.")
    parser.add_argument("--model-file", type=Path, default=Path("/workspace/models/xtype_model.py"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/outputs/next_stage_analysis/ribbon_xy"))
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--nk", type=int, default=201)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    width = 60 if args.quick else args.width
    nk = 81 if args.quick else args.nk
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_file)
    points, notes = auto_select_points(model_path=args.model_file)
    point_map = {p.label: p for p in points}
    labels = ["A", "B", "C", "D", "E_A", "E_B"]
    class_rows = []

    for label in labels:
        p = point_map.get(label)
        if p is None:
            class_rows.append(
                {
                    "point_label": label,
                    "direction": "",
                    "classification": "uncertain",
                    "comment": "point_not_available",
                }
            )
            continue

        set_params(model, v=p.v, lm=p.lm, t=p.t)
        # x-open
        ky, e_x, w_x = spectrum_xopen(v=p.v, t=p.t, lm=p.lm, w=p.w, nx=width, nk=nk)
        save_ribbon_plot(
            ky,
            e_x,
            w_x,
            out_root / f"{label}_ribbon_xopen.png",
            title=f"{label} x-open (v={p.v:.3f}, lm={p.lm:.3f}, t={p.t:.3f}, w={p.w:.3f})",
            k_label=r"$k_y/\pi$",
        )
        dump_ribbon_csv(out_root / f"{label}_ribbon_xopen.csv", ky, e_x, w_x, "ky")
        cls_x = classify_gap_state(e_x, w_x)
        class_rows.append(
            {
                "point_label": label,
                "direction": "x_open",
                **cls_x,
                "v": p.v,
                "lm": p.lm,
                "t": p.t,
                "w": p.w,
            }
        )

        # y-open
        kx, e_y, w_y = spectrum_yopen(v=p.v, t=p.t, lm=p.lm, w=p.w, ny=width, nk=nk)
        save_ribbon_plot(
            kx,
            e_y,
            w_y,
            out_root / f"{label}_ribbon_yopen.png",
            title=f"{label} y-open (v={p.v:.3f}, lm={p.lm:.3f}, t={p.t:.3f}, w={p.w:.3f})",
            k_label=r"$k_x/\pi$",
        )
        dump_ribbon_csv(out_root / f"{label}_ribbon_yopen.csv", kx, e_y, w_y, "kx")
        cls_y = classify_gap_state(e_y, w_y)
        class_rows.append(
            {
                "point_label": label,
                "direction": "y_open",
                **cls_y,
                "v": p.v,
                "lm": p.lm,
                "t": p.t,
                "w": p.w,
            }
        )

    write_csv(
        out_root / "ribbon_xy_classification.csv",
        class_rows,
        [
            "point_label",
            "direction",
            "v",
            "lm",
            "t",
            "w",
            "classification",
            "lower_edge",
            "upper_edge",
            "gap_width",
            "in_gap_edge_count",
            "span_ratio",
            "touches_lower",
            "touches_upper",
            "comment",
        ],
    )
    write_text(
        out_root / "selection_notes.txt",
        summarize_points(points, notes) + f"\nribbon_width={width}, nk={nk}\n",
    )
    print(f"[ok] ribbon xy outputs at {out_root}")


if __name__ == "__main__":
    main()
