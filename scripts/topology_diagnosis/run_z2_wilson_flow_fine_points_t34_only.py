#!/usr/bin/env python3
"""Wilson flow plots and crossing summary at representative fine points."""

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

from scripts.topology_diagnosis.t34_only_common import (  # noqa: E402
    ModelParams,
    crossing_count,
    h_spin_block_t34,
    track_wilson_branches,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wilson flow representative points (t34-only fine scan).")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    p.add_argument("--v-points", default="0.80,0.90,0.95,1.00,1.05,1.10")
    p.add_argument("--nkx", type=int, default=101)
    p.add_argument("--nky", type=int, default=101)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def vtok2(v: float) -> str:
    return f"{v:.2f}".replace("-", "m").replace(".", "p")


def load_qsh_map(path: Path) -> dict[float, dict]:
    out: dict[float, dict] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out[round(float(row["v"]), 10)] = row
    return out


def build_h8_from_spin_blocks(kx: float, ky: float, v: float, params: ModelParams) -> np.ndarray:
    pup = np.array([[1, 0], [0, 0]], dtype=complex)
    pdn = np.array([[0, 0], [0, 1]], dtype=complex)
    return np.kron(h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=+1), pup) + np.kron(
        h_spin_block_t34(kx, ky, v=v, params=params, spin_sign=-1),
        pdn,
    )


def tracked_branches(v: float, params: ModelParams, nkx: int, nky: int) -> tuple[np.ndarray, np.ndarray]:
    ky_list = np.linspace(0.0, np.pi, nky)
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    centers = np.zeros((nky, 4), dtype=float)
    eigvecs = []
    for j, ky in enumerate(ky_list):
        wmat = np.eye(4, dtype=complex)
        for i, kx in enumerate(kx_grid):
            kx2 = float(kx_grid[(i + 1) % nkx])
            _, va = np.linalg.eigh(build_h8_from_spin_blocks(float(kx), float(ky), v=v, params=params))
            _, vb = np.linalg.eigh(build_h8_from_spin_blocks(float(kx2), float(ky), v=v, params=params))
            oa = va[:, :4]
            ob = vb[:, :4]
            u, _, vh = np.linalg.svd(oa.conj().T @ ob)
            q = u @ vh
            wmat = q @ wmat
        ew, ev = np.linalg.eig(wmat)
        nu = np.mod(np.angle(ew) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        eigvecs.append(ev[:, order])
    return ky_list, track_wilson_branches(centers, eigvecs)


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    out = base / "fine_scan_0p8_1p2" / "04_wilson_flow"
    out.mkdir(parents=True, exist_ok=True)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_points = parse_float_list(args.v_points)
    qsh_map = load_qsh_map(base / "fine_scan_0p8_1p2" / "qsh_fine_scan_t34_only.csv")

    rows = []
    panel_paths = []
    for v in v_points:
        ky, tr = tracked_branches(v=v, params=params, nkx=args.nkx, nky=args.nky)
        c45 = crossing_count(tr, 0.45)
        c50 = crossing_count(tr, 0.50)
        c55 = crossing_count(tr, 0.55)
        z2_raw = int(c50 % 2)
        qrow = qsh_map.get(round(v, 10), {})
        cspin = float(qrow.get("C_spin", "nan"))
        z2_sc = int(qrow.get("Z2_from_spin_chern", z2_raw))
        z2_final = int(qrow.get("Z2_final", z2_sc))
        consistent = int(z2_raw == z2_sc)
        warning = (
            ""
            if consistent
            else (
                "WARNING: Wilson crossing Z2 disagrees with spin Chern parity. "
                "Since spin is conserved, final Z2 follows C_spin mod 2."
            )
        )
        rows.append(
            {
                "v": float(v),
                "crossing_number_ref_0p45": int(c45),
                "crossing_number_ref_0p50": int(c50),
                "crossing_number_ref_0p55": int(c55),
                "Z2_crossing_raw": int(z2_raw),
                "Z2_from_spin_chern": int(z2_sc),
                "Z2_final": int(z2_final),
                "consistent": int(consistent),
                "warning": warning,
            }
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        for b in range(tr.shape[1]):
            ax.plot(ky / np.pi, tr[:, b], linewidth=1.0)
        for ref, color in [(0.45, "gray"), (0.50, "black"), (0.55, "gray")]:
            ax.axhline(ref, color=color, linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$k_y/\pi$")
        ax.set_ylabel("tracked Wannier center")
        ax.set_title(
            f"v={v:.2f}, C_spin={cspin:.3f}, Z2(raw/sc/final)={z2_raw}/{z2_sc}/{z2_final}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        png = out / f"z2_wilson_flow_v{vtok2(v)}_t34_only.png"
        fig.savefig(png, dpi=180)
        fig.savefig(base / "fine_scan_0p8_1p2" / png.name, dpi=180)
        plt.close(fig)
        panel_paths.append(png)
        print(f"[wilson-flow-fine-t34] v={v:.2f} crossings=({c45},{c50},{c55}) Z2_final={z2_final}")

    csv_path = out / "z2_wilson_flow_fine_summary_t34_only.csv"
    write_csv(
        csv_path,
        rows,
        [
            "v",
            "crossing_number_ref_0p45",
            "crossing_number_ref_0p50",
            "crossing_number_ref_0p55",
            "Z2_crossing_raw",
            "Z2_from_spin_chern",
            "Z2_final",
            "consistent",
            "warning",
        ],
    )
    (base / "fine_scan_0p8_1p2" / "z2_wilson_flow_fine_summary_t34_only.csv").write_bytes(csv_path.read_bytes())

    cols = 3
    n = len(panel_paths)
    rr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rr, cols, figsize=(5.2 * cols, 3.9 * rr))
    axes = np.array(axes).reshape(rr, cols)
    for i in range(rr * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(plt.imread(panel_paths[i]))
            ax.axis("off")
            ax.set_title(panel_paths[i].stem)
        else:
            ax.axis("off")
    fig.suptitle("Z2 Wilson-flow representative points (t34-only)", fontsize=12)
    fig.tight_layout()
    montage = out / "z2_wilson_flow_fine_montage_t34_only.png"
    fig.savefig(montage, dpi=180)
    fig.savefig(base / "fine_scan_0p8_1p2" / montage.name, dpi=180)
    plt.close(fig)
    print(f"[ok] wilson-flow fine summary: {csv_path}")


if __name__ == "__main__":
    main()
