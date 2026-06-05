#!/usr/bin/env python3
"""Analyze band inversion signatures around transition candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import ensure_dir, load_model, set_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Band inversion analysis near transitions.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    return parser.parse_args()


def _closest_row(rows: list[dict[str, str]], lm: float, v_target: float, require_gapped: bool = False) -> dict[str, str]:
    cand = [r for r in rows if abs(float(r["lm"]) - lm) < 1e-12]
    if require_gapped:
        gapped = [r for r in cand if float(r["direct_gap"]) > 1e-4]
        if gapped:
            cand = gapped
    return min(cand, key=lambda r: abs(float(r["v"]) - v_target))


def _component_weights(vec: np.ndarray) -> np.ndarray:
    prob = np.abs(vec) ** 2
    total = float(np.sum(prob))
    if total > 0:
        prob = prob / total
    return prob


def _spin_weights(prob: np.ndarray) -> tuple[float, float]:
    up = float(np.sum(prob[0::2]))
    dn = float(np.sum(prob[1::2]))
    return up, dn


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "band_inversion")
    gap_csv = Path(args.output_root) / "gap_z2_scan" / "gap_z2_scan.csv"
    tr_csv = Path(args.output_root) / "gap_closing" / "transition_candidates.csv"
    if not gap_csv.exists():
        raise FileNotFoundError(f"Missing prerequisite file: {gap_csv}")
    if not tr_csv.exists():
        skip = out_dir / "skip_reason.txt"
        skip.write_text("transition_candidates.csv is missing; run analyze_gap_closing.py first.\n", encoding="utf-8")
        print(f"[skip] {skip}")
        return

    scan_rows = list(csv.DictReader(gap_csv.open("r", encoding="utf-8")))
    transitions = list(csv.DictReader(tr_csv.open("r", encoding="utf-8")))
    transitions = [t for t in transitions if t["transition_type"] in {"z2_topological_transition", "ordinary_gap_reopening"}]
    if not transitions:
        skip = out_dir / "skip_reason.txt"
        skip.write_text("No suitable transition candidates for band inversion analysis.\n", encoding="utf-8")
        print(f"[skip] {skip}")
        return

    model = load_model(args.model_file)
    summary_rows = []
    delta = 0.02 if args.quick else 0.03
    nq = 31 if args.quick else 61

    for idx, tr in enumerate(transitions, start=1):
        lm = float(tr["lm"])
        vc = float(tr["estimated_vc"])
        kx0 = float(tr["kx_min"])
        ky0 = float(tr["ky_min"])

        before_row = _closest_row(scan_rows, lm=lm, v_target=vc - delta, require_gapped=True)
        crit_row = _closest_row(scan_rows, lm=lm, v_target=vc, require_gapped=False)
        after_row = _closest_row(scan_rows, lm=lm, v_target=vc + delta, require_gapped=True)
        stages = [("before", before_row), ("critical", crit_row), ("after", after_row)]

        comp_data = {}
        for stage_name, row in stages:
            v = float(row["v"])
            set_params(model, v=v, lm=lm, t=args.t, w=args.w, j=0.0)
            k = np.array([kx0, ky0, 0.0], dtype=float)
            evals, evecs = np.linalg.eigh(model.Hxtype(k))
            n_occ = 4
            val_idx = n_occ - 1
            con_idx = n_occ
            val_prob = _component_weights(evecs[:, val_idx])
            con_prob = _component_weights(evecs[:, con_idx])
            val_up, val_dn = _spin_weights(val_prob)
            con_up, con_dn = _spin_weights(con_prob)
            comp_data[stage_name] = {
                "v": v,
                "val_prob": val_prob,
                "con_prob": con_prob,
                "val_major_component": int(np.argmax(val_prob)),
                "con_major_component": int(np.argmax(con_prob)),
                "val_up": val_up,
                "val_dn": val_dn,
                "con_up": con_up,
                "con_dn": con_dn,
            }

            summary_rows.append(
                {
                    "transition_id": tr["transition_id"],
                    "stage": stage_name,
                    "lm": lm,
                    "v": v,
                    "kx": kx0,
                    "ky": ky0,
                    "val_major_component": int(np.argmax(val_prob)),
                    "con_major_component": int(np.argmax(con_prob)),
                    "val_spin_up_weight": float(val_up),
                    "val_spin_down_weight": float(val_dn),
                    "con_spin_up_weight": float(con_up),
                    "con_spin_down_weight": float(con_dn),
                    "transition_type": tr["transition_type"],
                }
            )

        before = comp_data["before"]
        after = comp_data["after"]
        inversion_detected = (
            before["val_major_component"] == after["con_major_component"]
            and before["con_major_component"] == after["val_major_component"]
        )
        summary_rows.append(
            {
                "transition_id": tr["transition_id"],
                "stage": "summary",
                "lm": lm,
                "v": vc,
                "kx": kx0,
                "ky": ky0,
                "val_major_component": int(before["val_major_component"]),
                "con_major_component": int(after["con_major_component"]),
                "val_spin_up_weight": float(before["val_up"]),
                "val_spin_down_weight": float(before["val_dn"]),
                "con_spin_up_weight": float(after["con_up"]),
                "con_spin_down_weight": float(after["con_dn"]),
                "transition_type": "band_inversion_detected" if inversion_detected else "no_clear_band_inversion",
            }
        )

        # Local band around kx0.
        qvals = np.linspace(-0.08, 0.08, nq)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for stage_name, row in stages:
            v = float(row["v"])
            set_params(model, v=v, lm=lm, t=args.t, w=args.w, j=0.0)
            val_band = []
            con_band = []
            for q in qvals:
                k = np.array([kx0 + q, ky0, 0.0], dtype=float)
                evals = np.linalg.eigvalsh(model.Hxtype(k))
                val_band.append(float(np.real(evals[3])))
                con_band.append(float(np.real(evals[4])))
            ax.plot(qvals, val_band, linewidth=1.1, label=f"{stage_name} val")
            ax.plot(qvals, con_band, linewidth=1.1, linestyle="--", label=f"{stage_name} con")
        ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"$\Delta k_x$")
        ax.set_ylabel("Energy")
        ax.set_title(f"{tr['transition_id']} local band near k-min")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"transition_{idx}_local_band.png", dpi=180)
        plt.close(fig)

        # Component weights (before/after, val/con).
        labels = [f"c{i}" for i in range(8)]
        x = np.arange(8)
        width = 0.18
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.bar(x - 1.5 * width, before["val_prob"], width=width, label="before val")
        ax.bar(x - 0.5 * width, before["con_prob"], width=width, label="before con")
        ax.bar(x + 0.5 * width, after["val_prob"], width=width, label="after val")
        ax.bar(x + 1.5 * width, after["con_prob"], width=width, label="after con")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("Component index")
        ax.set_ylabel("Weight")
        ax.set_title(f"{tr['transition_id']} component weights at k-min")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"transition_{idx}_component_weights.png", dpi=180)
        plt.close(fig)

    csv_path = out_dir / "band_inversion_summary.csv"
    fields = [
        "transition_id",
        "stage",
        "lm",
        "v",
        "kx",
        "ky",
        "val_major_component",
        "con_major_component",
        "val_spin_up_weight",
        "val_spin_down_weight",
        "con_spin_up_weight",
        "con_spin_down_weight",
        "transition_type",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[ok] csv={csv_path}")
    print(f"[ok] out_dir={out_dir}")


if __name__ == "__main__":
    main()
