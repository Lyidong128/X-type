from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Thresholds:
    robust_gap: float
    critical_gap: float
    chern_quantized: float
    chern_jump: float


def load_rows(csv_path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "v": float(row["v"]),
                    "t": float(row["t"]),
                    "lm": float(row["lm"]),
                    "gap": float(row["gap"]),
                    "chern": float(row["chern"]),
                    "edge_state": int(float(row["edge_state"])),
                    "corner_state": int(float(row["corner_state"])),
                    "Wilson_loop": float(row["Wilson_loop"]),
                    "z2": int(float(row["Z2_topology"])),
                }
            )
    return rows


def key_of(row: dict[str, float | int]) -> tuple[float, float, float]:
    return (
        round(float(row["v"]), 10),
        round(float(row["t"]), 10),
        round(float(row["lm"]), 10),
    )


def format_token(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "p")


def point_id(v: float, t: float, lm: float) -> str:
    return f"v_{format_token(v)}_t_{format_token(t)}_lm_{format_token(lm)}"


def sorted_unique(rows: list[dict[str, float | int]], field: str) -> list[float]:
    return sorted({float(r[field]) for r in rows})


def local_features(
    row: dict[str, float | int],
    idx: dict[tuple[float, float, float], dict[str, float | int]],
    all_v: list[float],
    all_t: list[float],
    all_lm: list[float],
    chern_jump_threshold: float,
) -> dict[str, float | int]:
    v = float(row["v"])
    t = float(row["t"])
    lm = float(row["lm"])
    chern = float(row["chern"])
    z2 = int(row["z2"])

    def neighbors_along_axis(values: list[float], current: float) -> list[float]:
        i = values.index(current)
        out = []
        if i - 1 >= 0:
            out.append(values[i - 1])
        if i + 1 < len(values):
            out.append(values[i + 1])
        return out

    z2_flip_axes = 0
    max_chern_delta = 0.0
    min_neighbor_abs_gap = math.inf
    for axis, values in (("v", all_v), ("t", all_t), ("lm", all_lm)):
        axis_has_flip = False
        for nv in neighbors_along_axis(values, {"v": v, "t": t, "lm": lm}[axis]):
            if axis == "v":
                k = (round(nv, 10), round(t, 10), round(lm, 10))
            elif axis == "t":
                k = (round(v, 10), round(nv, 10), round(lm, 10))
            else:
                k = (round(v, 10), round(t, 10), round(nv, 10))
            if k not in idx:
                continue
            nr = idx[k]
            n_z2 = int(nr["z2"])
            n_ch = float(nr["chern"])
            n_gap = abs(float(nr["gap"]))
            if n_z2 != z2:
                axis_has_flip = True
            max_chern_delta = max(max_chern_delta, abs(chern - n_ch))
            min_neighbor_abs_gap = min(min_neighbor_abs_gap, n_gap)
        if axis_has_flip:
            z2_flip_axes += 1

    if math.isinf(min_neighbor_abs_gap):
        min_neighbor_abs_gap = abs(float(row["gap"]))
    return {
        "z2_flip_axes": z2_flip_axes,
        "max_chern_delta": max_chern_delta,
        "min_neighbor_abs_gap": min_neighbor_abs_gap,
        "is_chern_jump": int(max_chern_delta >= chern_jump_threshold),
    }


def phase_family(row: dict[str, float | int], feats: dict[str, float | int], th: Thresholds) -> str:
    gap = float(row["gap"])
    abs_gap = abs(gap)
    ch = abs(float(row["chern"]))
    z2 = int(row["z2"])
    edge = int(row["edge_state"])
    corner = int(row["corner_state"])
    z2_flip_axes = int(feats["z2_flip_axes"])
    chern_jump = int(feats["is_chern_jump"])

    if gap > th.robust_gap:
        if corner == 1:
            return "R3_robust_hoti_corner"
        if ch >= th.chern_quantized:
            return "R2_robust_chern"
        if z2 == 1:
            return "R1_robust_z2"
        return "R0_robust_trivial"

    if abs_gap <= th.critical_gap:
        if z2_flip_axes > 0 or chern_jump == 1:
            if corner == 1:
                return "C2_critical_corner_reconstruction"
            return "C1_critical_inversion_core"
        if edge == 1 and z2 == 0 and corner == 0:
            return "C3_critical_boundary_metal_like"
        return "C0_critical_generic"

    if (z2 == 1 or ch >= th.chern_quantized or edge == 1 or corner == 1):
        return "P1_precursor_topological"
    return "P0_precursor_trivial"


def special_score(row: dict[str, float | int], feats: dict[str, float | int], th: Thresholds) -> float:
    # Higher means more physically interesting for publication emphasis.
    gap = abs(float(row["gap"]))
    ch = abs(float(row["chern"]))
    z2 = int(row["z2"])
    edge = int(row["edge_state"])
    corner = int(row["corner_state"])
    z2_flip_axes = int(feats["z2_flip_axes"])
    chern_jump = int(feats["is_chern_jump"])

    score = 0.0
    score += max(0.0, (th.critical_gap - gap) / max(th.critical_gap, 1e-12)) * 4.0
    score += z2_flip_axes * 1.4
    score += chern_jump * 1.8
    score += edge * 1.0
    score += corner * 2.0
    score += (1.0 if z2 == 1 else 0.0) * 0.6
    score += min(ch / max(th.chern_quantized, 1e-12), 2.0) * 0.6
    return score


def choose_candidates(
    classified_rows: list[dict[str, float | int | str]],
    core_n: int,
    robust_n: int,
    anomaly_n: int,
) -> list[dict[str, float | int | str]]:
    core = [r for r in classified_rows if str(r["phase_family"]).startswith("C1_") or str(r["phase_family"]).startswith("C2_")]
    robust = [r for r in classified_rows if str(r["phase_family"]) in {"R1_robust_z2", "R2_robust_chern", "R3_robust_hoti_corner"}]
    anomaly = [r for r in classified_rows if str(r["phase_family"]) == "C3_critical_boundary_metal_like"]

    core_sorted = sorted(core, key=lambda r: float(r["special_score"]), reverse=True)[:core_n]
    robust_sorted = sorted(
        robust,
        key=lambda r: (
            float(r["gap"]),
            int(r["corner_state"]),
            abs(float(r["chern"])),
            abs(float(r["Wilson_loop"])),
        ),
        reverse=True,
    )[:robust_n]
    anomaly_sorted = sorted(
        anomaly,
        key=lambda r: (
            float(r["special_score"]),
            -abs(float(r["gap"])),
        ),
        reverse=True,
    )[:anomaly_n]

    merged: dict[str, dict[str, float | int | str]] = {}
    for r in core_sorted + robust_sorted + anomaly_sorted:
        merged[str(r["point_id"])] = r
    return list(merged.values())


def build_transition_corridors(
    rows: list[dict[str, float | int]],
    critical_gap: float,
) -> list[dict[str, float | int | str]]:
    by_vt: dict[tuple[float, float], list[dict[str, float | int]]] = defaultdict(list)
    for r in rows:
        by_vt[(float(r["v"]), float(r["t"]))].append(r)

    corridors: list[dict[str, float | int | str]] = []
    for (v, t), seq in by_vt.items():
        seq = sorted(seq, key=lambda x: float(x["lm"]))
        z2_seq = [int(x["z2"]) for x in seq]
        gap_seq = [abs(float(x["gap"])) for x in seq]
        edge_seq = [int(x["edge_state"]) for x in seq]
        corner_seq = [int(x["corner_state"]) for x in seq]
        chern_seq = [float(x["chern"]) for x in seq]
        z2_flips = sum(1 for a, b in zip(z2_seq[:-1], z2_seq[1:]) if a != b)
        max_chern_jump = max((abs(a - b) for a, b in zip(chern_seq[:-1], chern_seq[1:])), default=0.0)
        min_abs_gap = min(gap_seq) if gap_seq else 1.0
        if z2_flips == 0 and min_abs_gap > critical_gap:
            continue
        corridors.append(
            {
                "v": v,
                "t": t,
                "z2_flips": z2_flips,
                "max_chern_jump": max_chern_jump,
                "min_abs_gap": min_abs_gap,
                "has_edge": int(max(edge_seq) if edge_seq else 0),
                "has_corner": int(max(corner_seq) if corner_seq else 0),
                "z2_pattern": "-".join(map(str, z2_seq)),
            }
        )
    corridors.sort(key=lambda r: (-int(r["z2_flips"]), float(r["min_abs_gap"])))
    return corridors


def write_csv(path: Path, rows: list[dict[str, float | int | str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    results_csv = root / args.results_csv
    src_points = root / args.source_points_dir
    output_dir = root / args.output_dir
    selected_dir = output_dir / "selected_points_package"
    selected_points_dir = selected_dir / "points"
    analysis_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not src_points.exists():
        raise FileNotFoundError(f"Per-point folder not found: {src_points}")

    th = Thresholds(
        robust_gap=float(args.robust_gap),
        critical_gap=float(args.critical_gap),
        chern_quantized=float(args.chern_quantized),
        chern_jump=float(args.chern_jump),
    )

    rows = load_rows(results_csv)
    idx = {key_of(r): r for r in rows}
    all_v = sorted_unique(rows, "v")
    all_t = sorted_unique(rows, "t")
    all_lm = sorted_unique(rows, "lm")

    classified_rows: list[dict[str, float | int | str]] = []
    for row in rows:
        feats = local_features(row, idx, all_v, all_t, all_lm, chern_jump_threshold=th.chern_jump)
        fam = phase_family(row, feats, th)
        pid = point_id(float(row["v"]), float(row["t"]), float(row["lm"]))
        score = special_score(row, feats, th)
        rec: dict[str, float | int | str] = {
            "point_id": pid,
            **row,
            **feats,
            "phase_family": fam,
            "special_score": score,
        }
        classified_rows.append(rec)

    full_fields = [
        "point_id",
        "v",
        "t",
        "lm",
        "gap",
        "chern",
        "z2",
        "edge_state",
        "corner_state",
        "Wilson_loop",
        "z2_flip_axes",
        "max_chern_delta",
        "min_neighbor_abs_gap",
        "is_chern_jump",
        "phase_family",
        "special_score",
    ]
    write_csv(analysis_dir / "prl_classification_full.csv", classified_rows, full_fields)

    selected = choose_candidates(
        classified_rows,
        core_n=int(args.core_n),
        robust_n=int(args.robust_n),
        anomaly_n=int(args.anomaly_n),
    )
    selected = sorted(selected, key=lambda r: float(r["special_score"]), reverse=True)
    write_csv(analysis_dir / "prl_special_points_selected.csv", selected, full_fields)

    corridors = build_transition_corridors(rows, critical_gap=th.critical_gap)
    corridor_fields = [
        "v",
        "t",
        "z2_flips",
        "max_chern_jump",
        "min_abs_gap",
        "has_edge",
        "has_corner",
        "z2_pattern",
    ]
    write_csv(analysis_dir / "prl_transition_corridors.csv", corridors, corridor_fields)

    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_points_dir.mkdir(parents=True, exist_ok=True)

    # Ensure analysis artifacts are included inside the packaged directory.
    selected_analysis_dir = selected_dir / "analysis"
    selected_analysis_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for r in selected:
        pid = str(r["point_id"])
        src = src_points / pid
        dst = selected_points_dir / pid
        if src.exists():
            shutil.copytree(src, dst)
        else:
            missing.append(pid)
    (selected_dir / "missing_points.txt").write_text("\n".join(missing), encoding="utf-8")

    phase_counts = Counter(str(r["phase_family"]) for r in selected)
    lines = [
        "# PRL-style topological classification logic",
        "",
        "## Hierarchical logic",
        "1) Bulk robustness axis: gap (robust/critical/precursor).",
        "2) Topological invariant axis: Z2, Chern, Wilson consistency.",
        "3) Boundary response axis: edge_state / corner_state.",
        "4) Local continuity axis: Z2 flips and Chern jumps against nearest neighbors.",
        "",
        "## Thresholds",
        f"- robust_gap = {th.robust_gap}",
        f"- critical_gap = {th.critical_gap}",
        f"- chern_quantized = {th.chern_quantized}",
        f"- chern_jump = {th.chern_jump}",
        "",
        "## Selection policy (publication-oriented)",
        f"- transition cores: top {args.core_n} from C1/C2 by special_score",
        f"- robust anchors: top {args.robust_n} from R1/R2/R3 by gap+invariant strength",
        f"- anomalous boundary states: top {args.anomaly_n} from C3",
        "",
        f"selected_total = {len(selected)}",
        f"missing_artifacts = {len(missing)}",
        f"selected_phase_counts = {dict(phase_counts)}",
        f"transition_corridor_count = {len(corridors)}",
        "",
        "## Key transition corridors (top 10 by z2_flips and min_abs_gap)",
    ]
    for c in corridors[:10]:
        lines.append(
            f"- (v={c['v']:.2f}, t={c['t']:.2f}): z2_flips={c['z2_flips']}, "
            f"min_abs_gap={c['min_abs_gap']:.4e}, max_chern_jump={c['max_chern_jump']:.4f}, "
            f"edge={c['has_edge']}, corner={c['has_corner']}"
        )
    logic_path = analysis_dir / "prl_classification_logic.md"
    logic_path.write_text("\n".join(lines), encoding="utf-8")

    # Mirror key analysis outputs into selected package directory.
    shutil.copy2(analysis_dir / "prl_classification_full.csv", selected_analysis_dir / "prl_classification_full.csv")
    shutil.copy2(analysis_dir / "prl_special_points_selected.csv", selected_analysis_dir / "prl_special_points_selected.csv")
    shutil.copy2(analysis_dir / "prl_transition_corridors.csv", selected_analysis_dir / "prl_transition_corridors.csv")
    shutil.copy2(logic_path, selected_analysis_dir / "prl_classification_logic.md")

    zip_base = root / args.zip_base
    zip_base.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=str(selected_dir.parent),
        base_dir=selected_dir.name,
    )
    print(f"classified_total={len(classified_rows)}")
    print(f"selected_total={len(selected)}")
    print(f"zip={zip_base}.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRL-style classification and selection for topological special points.")
    parser.add_argument("--results-csv", default="outputs/results.csv")
    parser.add_argument("--source-points-dir", default="outputs/all_points_full_package/points")
    parser.add_argument("--output-dir", default="outputs/prl_special_points")
    parser.add_argument("--zip-base", default="outputs/prl_special_points/selected_points_package")
    parser.add_argument("--robust-gap", type=float, default=0.10)
    parser.add_argument("--critical-gap", type=float, default=0.05)
    parser.add_argument("--chern-quantized", type=float, default=0.5)
    parser.add_argument("--chern-jump", type=float, default=0.5)
    parser.add_argument("--core-n", type=int, default=40)
    parser.add_argument("--robust-n", type=int, default=24)
    parser.add_argument("--anomaly-n", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
