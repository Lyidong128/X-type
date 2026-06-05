#!/usr/bin/env python3
"""Analyze M-point band inversion around the v~0.6 critical region."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.M_point_spin_chern_analysis.common import ensure_dir, load_model, m_point_from_model, set_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze M-point band inversion.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/M_point_spin_chern_analysis")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _weights(vec: np.ndarray) -> np.ndarray:
    p = np.abs(vec) ** 2
    p = p / max(float(np.sum(p)), 1e-16)
    return p


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    m_csv = out_root / "M_gap_scan.csv"
    if not m_csv.exists():
        raise FileNotFoundError(f"Missing prerequisite file: {m_csv}")
    m_rows = list(csv.DictReader(m_csv.open("r", encoding="utf-8")))
    if not m_rows:
        raise RuntimeError("M_gap_scan.csv is empty.")

    # prioritize first interval [0.55, 0.65] for the main transition around v~0.6
    rows_mid = [r for r in m_rows if 0.55 <= float(r["v"]) <= 0.65]
    target_rows = rows_mid if rows_mid else m_rows
    vc = float(min(target_rows, key=lambda r: float(r["Delta_M"]))["v"])
    v_values = sorted({float(r["v"]) for r in m_rows})
    v_before = min(v_values, key=lambda x: abs(x - (vc - 0.01)))
    v_after = min(v_values, key=lambda x: abs(x - (vc + 0.01)))

    model = load_model(args.model_file)
    mpt = m_point_from_model(model)
    stages = [("before", v_before), ("critical", vc), ("after", v_after)]

    all_rows = []
    stage_data = {}
    for label, v in stages:
        set_params(model, v=v, lm=args.lm, t=args.t, w=args.w, j=0.0)
        evals, evecs = np.linalg.eigh(model.Hxtype(mpt))
        val_idx = 3
        con_idx = 4
        val_w = _weights(evecs[:, val_idx])
        con_w = _weights(evecs[:, con_idx])
        val_up = float(np.sum(val_w[0::2]))
        val_dn = float(np.sum(val_w[1::2]))
        con_up = float(np.sum(con_w[0::2]))
        con_dn = float(np.sum(con_w[1::2]))

        row = {
            "stage": label,
            "v": float(v),
            "lm": float(args.lm),
            "t": float(args.t),
            "w": float(args.w),
            "E_valence_M": float(np.real(evals[val_idx])),
            "E_conduction_M": float(np.real(evals[con_idx])),
            "spin_up_weight_valence": val_up,
            "spin_down_weight_valence": val_dn,
            "spin_up_weight_conduction": con_up,
            "spin_down_weight_conduction": con_dn,
            "major_component_valence": int(np.argmax(val_w)),
            "major_component_conduction": int(np.argmax(con_w)),
        }
        for i in range(len(val_w)):
            row[f"component_{i}_weight_valence"] = float(val_w[i])
            row[f"component_{i}_weight_conduction"] = float(con_w[i])
        all_rows.append(row)
        stage_data[label] = row

    # 판단: before/after에서 주성분 교환 여부
    before = stage_data["before"]
    after = stage_data["after"]
    inversion = (
        int(before["major_component_valence"]) == int(after["major_component_conduction"])
        and int(before["major_component_conduction"]) == int(after["major_component_valence"])
    )
    all_rows.append(
        {
            "stage": "summary",
            "v": float(vc),
            "lm": float(args.lm),
            "t": float(args.t),
            "w": float(args.w),
            "E_valence_M": np.nan,
            "E_conduction_M": np.nan,
            "spin_up_weight_valence": np.nan,
            "spin_down_weight_valence": np.nan,
            "spin_up_weight_conduction": np.nan,
            "spin_down_weight_conduction": np.nan,
            "major_component_valence": int(before["major_component_valence"]),
            "major_component_conduction": int(after["major_component_conduction"]),
            "band_inversion": "detected" if inversion else "not_clearly_detected",
        }
    )

    # Write CSV.
    fieldnames = [
        "stage",
        "v",
        "lm",
        "t",
        "w",
        "E_valence_M",
        "E_conduction_M",
        "spin_up_weight_valence",
        "spin_down_weight_valence",
        "spin_up_weight_conduction",
        "spin_down_weight_conduction",
        "major_component_valence",
        "major_component_conduction",
        "band_inversion",
    ]
    for i in range(8):
        fieldnames.append(f"component_{i}_weight_valence")
        fieldnames.append(f"component_{i}_weight_conduction")

    out_csv = out_root / "M_band_inversion_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    # Plot component weights.
    x = np.arange(8)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, key, title in (
        (axes[0], "valence", "Valence band component weights at M"),
        (axes[1], "conduction", "Conduction band component weights at M"),
    ):
        for label, marker in (("before", "o"), ("critical", "s"), ("after", "^")):
            row = stage_data[label]
            y = [float(row[f"component_{i}_weight_{key}"]) for i in range(8)]
            ax.plot(x, y, marker=marker, linewidth=1.1, label=f"{label} (v={row['v']:.4f})")
        ax.set_xticks(x)
        ax.set_xlabel("component index")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("weight")
    axes[0].legend(fontsize=8)
    fig.suptitle("M-point component evolution before/critical/after")
    fig.tight_layout()
    fig.savefig(out_root / "M_component_weights_before_critical_after.png", dpi=180)
    plt.close(fig)

    # Plot local band around M along qx with qy=0.
    nq = 41 if args.quick else 81
    qvals = np.linspace(-0.08, 0.08, nq)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for label, v in stages:
        set_params(model, v=v, lm=args.lm, t=args.t, w=args.w, j=0.0)
        val_line = []
        con_line = []
        for q in qvals:
            k = mpt + np.array([q, 0.0, 0.0], dtype=float)
            evals = np.linalg.eigvalsh(model.Hxtype(k))
            val_line.append(float(np.real(evals[3])))
            con_line.append(float(np.real(evals[4])))
        ax.plot(qvals, val_line, linewidth=1.1, label=f"{label} val (v={v:.4f})")
        ax.plot(qvals, con_line, linewidth=1.1, linestyle="--", label=f"{label} con (v={v:.4f})")
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$q_x$, with $k=M+(q_x,0)$")
    ax.set_ylabel("Energy")
    ax.set_title("Local valence/conduction bands around M")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_root / "M_local_band_before_after.png", dpi=180)
    plt.close(fig)

    summary_txt = out_root / "M_band_inversion_summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"vc={vc:.6f}",
                f"v_before={v_before:.6f}",
                f"v_after={v_after:.6f}",
                f"band_inversion={'detected' if inversion else 'not_clearly_detected'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] csv={out_csv}")
    print(f"[ok] vc={vc:.6f} inversion={inversion}")


if __name__ == "__main__":
    main()
