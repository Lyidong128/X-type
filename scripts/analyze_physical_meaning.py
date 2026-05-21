#!/usr/bin/env python3
"""Build physically interpretable classification from existing scan outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_csv_dict(path: Path, key_field: str) -> dict[str, dict[str, str]]:
    """Load a CSV file into a dict keyed by `key_field`."""
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row[key_field]] = row
    return out


def to_float(value: str, default: float = 0.0) -> float:
    """Best-effort float conversion."""
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: str, default: int = 0) -> int:
    """Best-effort int conversion."""
    try:
        return int(float(value))
    except Exception:
        return default


def classify_point(
    rec: dict[str, object],
    robust_gap_min: float,
    transition_gap_max: float,
    transition_min_abs_max: float,
) -> tuple[str, str]:
    """Return (category, rationale) for a merged point record."""
    z2 = int(rec["z2"])
    connecting = int(rec["has_connecting_edge_branch"])
    isolated = int(rec["has_isolated_state"])
    band_gap = float(rec["band_window_width"])
    min_abs = float(rec["min_abs_energy"])
    near_zero_n = int(rec["count_absE_le_0p02"])
    special_rank = int(rec["special_rank"]) if rec["special_rank"] is not None else 9999

    transition_like = (
        band_gap <= transition_gap_max
        or min_abs <= transition_min_abs_max
        or special_rank <= 10
    )
    robust_helical = z2 == 1 and connecting == 1 and band_gap >= robust_gap_min and near_zero_n >= 4
    bulk_mixed_nontrivial = z2 == 1 and connecting == 0 and isolated == 1 and band_gap >= robust_gap_min
    trivial_gapped = z2 == 0 and connecting == 0 and isolated == 0 and band_gap >= robust_gap_min and near_zero_n == 0

    if robust_helical:
        return "robust_edge_topology", "Z2=1 且 ribbon(4/5) 连接边界分支清晰，且能隙足够打开"
    if transition_like:
        return "transition_core_or_precursor", "小隙/近零能聚集/高 special_rank，属于相变核心或前驱"
    if bulk_mixed_nontrivial:
        return "bulk_mixed_nontrivial", "Z2 非平庸但 ribbon 未形成连接边界分支，边界性被弱化"
    if trivial_gapped:
        return "trivial_gapped", "边界分支缺失且近零态缺失，平庸绝缘体特征明显"
    return "ambiguous_need_review", "指标互相竞争，需结合波函数和更大尺寸复核"


def physical_score(rec: dict[str, object]) -> float:
    """Compute a bounded physical interpretability score in [0, 1]."""
    z2 = int(rec["z2"])
    connecting = int(rec["has_connecting_edge_branch"])
    band_gap = max(0.0, float(rec["band_window_width"]))
    edge_count = max(0.0, float(rec["in_window_edge_state_count"]))
    near_zero_n = max(0.0, float(rec["count_absE_le_0p02"]))
    span_ratio = max(0.0, float(rec["span_ratio"]))

    gap_norm = min(band_gap / 1.0, 1.0)
    edge_norm = min(edge_count / 320.0, 1.0)
    near_zero_norm = min(near_zero_n / 32.0, 1.0)
    score = (
        0.30 * float(connecting)
        + 0.20 * float(z2)
        + 0.20 * gap_norm
        + 0.20 * edge_norm
        + 0.10 * near_zero_norm
    )
    score *= (0.6 + 0.4 * min(span_ratio, 1.0))
    return max(0.0, min(score, 1.0))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_category_map(rows: list[dict[str, object]], out_png: Path) -> None:
    """Plot v-lm scatter colored by physical category."""
    category_style = {
        "robust_edge_topology": ("#0072B2", "robust_edge_topology"),
        "transition_core_or_precursor": ("#D55E00", "transition_core_or_precursor"),
        "bulk_mixed_nontrivial": ("#009E73", "bulk_mixed_nontrivial"),
        "trivial_gapped": ("#999999", "trivial_gapped"),
        "ambiguous_need_review": ("#CC79A7", "ambiguous_need_review"),
    }

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=170)
    for cat, (color, _) in category_style.items():
        xs = [float(r["v"]) for r in rows if r["category"] == cat]
        ys = [float(r["lm"]) for r in rows if r["category"] == cat]
        if xs:
            ax.scatter(xs, ys, s=28, c=color, alpha=0.90, edgecolors="none", label=cat)
    ax.set_xlabel("v")
    ax.set_ylabel("lm")
    ax.set_title("Physically Interpretable Category Map (t=0.5)")
    ax.grid(alpha=0.20, linewidth=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_score_map(rows: list[dict[str, object]], out_png: Path) -> None:
    """Plot v-lm map with marker size based on physical score."""
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=170)
    xs = [float(r["v"]) for r in rows]
    ys = [float(r["lm"]) for r in rows]
    scores = [float(r["physical_score"]) for r in rows]
    sizes = [18.0 + 120.0 * s for s in scores]

    sc = ax.scatter(xs, ys, c=scores, s=sizes, cmap="viridis", edgecolors="none", alpha=0.95)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("physical_score")
    ax.set_xlabel("v")
    ax.set_ylabel("lm")
    ax.set_title("Physical Score Map (larger marker = stronger physical signature)")
    ax.grid(alpha=0.20, linewidth=0.4)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def select_key_points(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Extract concise key points with explicit physical roles."""
    by_cat: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_cat[str(row["category"])].append(row)

    def pick(cat: str, sort_key: str, reverse: bool = True) -> dict[str, object] | None:
        group = by_cat.get(cat, [])
        if not group:
            return None
        return sorted(group, key=lambda r: float(r[sort_key]), reverse=reverse)[0]

    key_points: list[dict[str, object]] = []
    p = pick("robust_edge_topology", "physical_score", True)
    if p:
        key_points.append({**p, "key_role": "strongest_robust_edge_candidate"})
    p = pick("transition_core_or_precursor", "special_score", True)
    if p:
        key_points.append({**p, "key_role": "strongest_transition_core"})
    p = pick("bulk_mixed_nontrivial", "physical_score", True)
    if p:
        key_points.append({**p, "key_role": "strongest_bulk_mixed_nontrivial"})
    p = pick("trivial_gapped", "band_window_width", True)
    if p:
        key_points.append({**p, "key_role": "representative_trivial_gapped"})

    # Fallback: top global score.
    if not key_points and rows:
        p = sorted(rows, key=lambda r: float(r["physical_score"]), reverse=True)[0]
        key_points.append({**p, "key_role": "global_best_physical_score"})
    return key_points


def build_corridor_candidates(transition_root: Path, gap_eps: float, obc_eps: float) -> list[dict[str, object]]:
    """Find likely transition cores from local dense corridor scans."""
    if not transition_root.exists():
        return []

    rows: list[dict[str, object]] = []
    for csv_path in sorted(transition_root.glob("rank_*/corridor_scan.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for rec in csv.DictReader(fh):
                gap = to_float(rec.get("gap", "nan"), default=math.nan)
                obc_min = to_float(rec.get("obc_min_abs_energy", "nan"), default=math.nan)
                if math.isnan(gap) or math.isnan(obc_min):
                    continue
                if abs(gap) <= gap_eps or obc_min <= obc_eps:
                    rows.append(
                        {
                            "corridor": csv_path.parent.name,
                            "v": to_float(rec["v"]),
                            "t": to_float(rec["t"]),
                            "lm": to_float(rec["lm"]),
                            "w": to_float(rec["w"]),
                            "gap": gap,
                            "chern": to_float(rec.get("chern", "nan"), default=math.nan),
                            "obc_min_abs_energy": obc_min,
                            "trigger": f"|gap|<={gap_eps} or obc_min<={obc_eps}",
                        }
                    )
    rows.sort(key=lambda r: (abs(float(r["gap"])), float(r["obc_min_abs_energy"])))
    return rows


def write_markdown_summary(
    out_md: Path,
    rows: list[dict[str, object]],
    counts: Counter[str],
    key_points: list[dict[str, object]],
    corridor_candidates: list[dict[str, object]],
    cfg: dict[str, float],
) -> None:
    """Write a concise physical-logic report in Chinese."""
    total = len(rows)
    underestimated = sum(
        1
        for r in rows
        if r["category"] in {"robust_edge_topology", "transition_core_or_precursor"}
        and float(r["count_absE_le_0p02"]) >= 6
        and int(r["has_connecting_edge_branch"]) == 1
    )
    lines = [
        "# 物理意义重分析总结",
        "",
        "## 1) 判据（物理可解释）",
        f"- 稳健边界相候选：`Z2=1` 且 `ribbon(4/5)连接分支=1` 且 `band_window_width >= {cfg['robust_gap_min']}`。",
        f"- 相变核心/前驱：`band_window_width <= {cfg['transition_gap_max']}` 或 `OBC min|E| <= {cfg['transition_min_abs_max']}` 或 `special_rank <= 10`。",
        "- 平庸绝缘体：`Z2=0` 且 ribbon 无连接/孤立态，且近零能态不显著。",
        "- 其余归入 bulk-mixed / ambiguous，提示需要更大尺寸与波函数复核。",
        "",
        "## 2) 总体统计",
        f"- 总点数：{total}",
    ]
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    lines += [
        f"- 具有明显边界/临界特征的点数（连接分支+近零态聚集）：{underestimated}",
        "",
        "## 3) 关键点（建议优先看）",
    ]
    if key_points:
        for p in key_points:
            lines.append(
                f"- {p['key_role']}: {p['point_id']} "
                f"(v={p['v']:.2f}, lm={p['lm']:.2f}, cat={p['category']}, score={p['physical_score']:.3f})"
            )
    else:
        lines.append("- 无（请检查输入数据完整性）")

    lines += ["", "## 4) 局域走廊的相变候选点"]
    if corridor_candidates:
        lines.append(f"- 命中数：{len(corridor_candidates)}")
        for p in corridor_candidates[:10]:
            lines.append(
                f"- {p['corridor']}: (v={p['v']:.3f}, lm={p['lm']:.3f}), "
                f"gap={p['gap']:.3e}, obc_min={p['obc_min_abs_energy']:.3e}"
            )
    else:
        lines.append("- 当前走廊数据中未检出满足阈值的候选点")

    lines += [
        "",
        "## 5) 结论",
        "- 这版结果不再只看“是否特殊”，而是强制结合 bulk 指标 + ribbon 边界连通 + OBC 近零模式三重证据。",
        "- 可以直接用于区分：稳健边界相、相变前驱、平庸相、以及 bulk-mixed 的可疑点。",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """Run physical meaning analysis and write outputs."""
    out_dir = Path(args.out_dir).resolve()
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    topo = load_csv_dict(Path(args.topology_csv), "point_id")
    ribbon = load_csv_dict(Path(args.ribbon_csv), "point_id")
    obc = load_csv_dict(Path(args.obc_csv), "point_id")

    merged_rows: list[dict[str, object]] = []
    missing = []
    for pid, trow in topo.items():
        if pid not in ribbon or pid not in obc:
            missing.append(pid)
            continue
        rrow = ribbon[pid]
        orow = obc[pid]
        rec: dict[str, object] = {
            "point_id": pid,
            "v": to_float(trow["v"]),
            "t": to_float(trow["t"]),
            "lm": to_float(trow["lm"]),
            "w": to_float(trow["w"]),
            "z2": to_int(trow["z2"]),
            "chern": to_float(trow["chern"]),
            "wilson": to_float(trow["wilson"]),
            "ribbon_classification": rrow["classification"],
            "has_connecting_edge_branch": to_int(rrow["has_connecting_edge_branch"]),
            "has_isolated_state": to_int(rrow["has_isolated_state"]),
            "band_window_width": to_float(rrow["band_window_width"]),
            "in_window_edge_state_count": to_int(rrow["in_window_edge_state_count"]),
            "span_ratio": to_float(rrow["span_ratio"]),
            "min_abs_energy": to_float(orow["min_abs_energy"]),
            "obc_gap": to_float(orow["obc_gap"]),
            "count_absE_le_0p02": to_int(orow["count_absE_le_0p02"]),
            "special_score": to_float(orow["special_score"]),
            "special_rank": to_int(orow["special_rank"], default=9999) if orow.get("special_rank") else 9999,
        }
        category, rationale = classify_point(
            rec,
            robust_gap_min=args.robust_gap_min,
            transition_gap_max=args.transition_gap_max,
            transition_min_abs_max=args.transition_min_abs_max,
        )
        rec["category"] = category
        rec["rationale"] = rationale
        rec["physical_score"] = round(physical_score(rec), 6)
        merged_rows.append(rec)

    merged_rows.sort(key=lambda r: (float(r["v"]), float(r["lm"])))
    counts = Counter(str(r["category"]) for r in merged_rows)

    key_points = select_key_points(merged_rows)
    corridor_candidates = build_corridor_candidates(
        Path(args.transition_root),
        gap_eps=args.corridor_gap_eps,
        obc_eps=args.corridor_obc_eps,
    )

    write_csv(
        out_dir / "physical_point_classification.csv",
        merged_rows,
        [
            "point_id",
            "v",
            "t",
            "lm",
            "w",
            "z2",
            "chern",
            "wilson",
            "ribbon_classification",
            "has_connecting_edge_branch",
            "has_isolated_state",
            "band_window_width",
            "in_window_edge_state_count",
            "span_ratio",
            "min_abs_energy",
            "obc_gap",
            "count_absE_le_0p02",
            "special_score",
            "special_rank",
            "category",
            "physical_score",
            "rationale",
        ],
    )
    write_csv(
        out_dir / "phase_counts.csv",
        [{"category": k, "count": v} for k, v in sorted(counts.items())],
        ["category", "count"],
    )
    write_csv(
        out_dir / "key_physical_points.csv",
        key_points,
        [
            "key_role",
            "point_id",
            "v",
            "t",
            "lm",
            "w",
            "category",
            "physical_score",
            "z2",
            "chern",
            "wilson",
            "band_window_width",
            "count_absE_le_0p02",
            "ribbon_classification",
            "rationale",
        ],
    )
    write_csv(
        out_dir / "corridor_transition_candidates.csv",
        corridor_candidates,
        ["corridor", "v", "t", "lm", "w", "gap", "chern", "obc_min_abs_energy", "trigger"],
    )

    plot_category_map(merged_rows, figure_dir / "physical_category_map.png")
    plot_score_map(merged_rows, figure_dir / "physical_score_map.png")

    cfg = {
        "robust_gap_min": args.robust_gap_min,
        "transition_gap_max": args.transition_gap_max,
        "transition_min_abs_max": args.transition_min_abs_max,
    }
    write_markdown_summary(
        out_md=out_dir / "physical_interpretation.md",
        rows=merged_rows,
        counts=counts,
        key_points=key_points,
        corridor_candidates=corridor_candidates,
        cfg=cfg,
    )
    (out_dir / "analysis_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    with zipfile.ZipFile(out_dir / "physical_meaning_package.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p.name != "physical_meaning_package.zip":
                zf.write(p, p.relative_to(out_dir))

    if missing:
        (out_dir / "missing_points.txt").write_text("\n".join(missing), encoding="utf-8")

    print(f"[ok] merged_points={len(merged_rows)}")
    print(f"[ok] categories={dict(counts)}")
    print(f"[ok] key_points={len(key_points)}")
    print(f"[ok] corridor_candidates={len(corridor_candidates)}")
    print(f"[ok] out_dir={out_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    p = argparse.ArgumentParser(description="Physically interpretable post-analysis")
    p.add_argument(
        "--topology-csv",
        type=Path,
        default=Path("/workspace/outputs/first_stage_band_ribbon/topology_summary.csv"),
    )
    p.add_argument(
        "--ribbon-csv",
        type=Path,
        default=Path("/workspace/outputs/first_stage_band_ribbon/ribbon_gap_state_classification_fermi45.csv"),
    )
    p.add_argument(
        "--obc-csv",
        type=Path,
        default=Path("/workspace/outputs/first_stage_band_ribbon/obc_special_points_analysis.csv"),
    )
    p.add_argument(
        "--transition-root",
        type=Path,
        default=Path("/workspace/outputs/transition_corridors"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("/workspace/outputs/physical_meaning"))
    p.add_argument("--robust-gap-min", type=float, default=0.20)
    p.add_argument("--transition-gap-max", type=float, default=0.08)
    p.add_argument("--transition-min-abs-max", type=float, default=0.002)
    p.add_argument("--corridor-gap-eps", type=float, default=0.02)
    p.add_argument("--corridor-obc-eps", type=float, default=0.001)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
