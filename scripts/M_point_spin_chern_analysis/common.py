#!/usr/bin/env python3
"""Shared utilities for M-point spin-Chern focused analysis."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import (  # reuse validated utilities
    build_scan_values,
    chern_number_general,
    load_model,
    set_params,
    split_spin_indices,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def m_point_from_model(model) -> np.ndarray:
    # b1=(2pi,0), b2=(0,2pi) for the current model, so M=(pi,pi)=0.5*(b1+b2)
    return 0.5 * (model.b1 + model.b2)


def m_point_cartesian() -> tuple[float, float]:
    return float(np.pi), float(np.pi)


def direct_gap_at_k(model, kvec: np.ndarray, n_occ: int = 4) -> tuple[float, float, float]:
    evals = np.linalg.eigvalsh(model.Hxtype(kvec))
    val = float(np.real(evals[n_occ - 1]))
    cond = float(np.real(evals[n_occ]))
    return val, cond, cond - val


def direct_gap_global(
    model,
    nk: int = 61,
    n_occ: int = 4,
) -> tuple[float, float, float]:
    min_gap = np.inf
    kx_min = 0.0
    ky_min = 0.0
    denom = max(1, nk - 1)
    for i, j in product(range(nk), range(nk)):
        u = i / denom
        v = j / denom
        k = u * model.b1 + v * model.b2
        _, _, gap = direct_gap_at_k(model, k, n_occ=n_occ)
        if gap < min_gap:
            min_gap = float(gap)
            kx_min = float(np.real(k[0]))
            ky_min = float(np.real(k[1]))
    return float(min_gap), kx_min, ky_min


def spin_block_hamiltonians(model):
    h0 = np.array(model.Hxtype(np.zeros(3)), dtype=complex)
    up, dn = split_spin_indices(h0.shape[0])

    def h_up(kvec):
        hk = np.array(model.Hxtype(kvec), dtype=complex)
        return hk[np.ix_(up, up)]

    def h_dn(kvec):
        hk = np.array(model.Hxtype(kvec), dtype=complex)
        return hk[np.ix_(dn, dn)]

    return h_up, h_dn


def compute_spin_chern_pair(model, nk: int) -> tuple[float, float]:
    h_up, h_dn = spin_block_hamiltonians(model)
    cup = chern_number_general(h_up, model.b1, model.b2, nk=nk, n_occ=2)
    cdn = chern_number_general(h_dn, model.b1, model.b2, nk=nk, n_occ=2)
    return float(cup), float(cdn)


def nearest_value(value: float, choices: list[float]) -> float:
    return min(choices, key=lambda x: abs(float(x) - float(value)))


def token(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def plot_two_lines(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    xlab: str,
    ylab: str,
    title: str,
    label1: str,
    label2: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, y1, linewidth=1.2, label=label1)
    ax.plot(x, y2, linewidth=1.2, label=label2)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
