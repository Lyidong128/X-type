from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


matplotlib.use("Agg")


@dataclass
class ReportPaths:
    root: Path
    summary_dir: Path
    figures_dir: Path
    report_dir: Path
    per_point_dir: Path
    report_pdf: Path
    point_summary_csv: Path
    degeneracy_csv: Path
    key_points_csv: Path


def ensure_report_dirs(output_root: Path) -> ReportPaths:
    summary_dir = output_root / "summary_tables"
    figures_dir = output_root / "figures"
    report_dir = output_root / "report"
    per_point_dir = output_root / "per_point"
    for folder in (summary_dir, figures_dir, report_dir, per_point_dir, report_dir / "assets"):
        folder.mkdir(parents=True, exist_ok=True)
    return ReportPaths(
        root=output_root,
        summary_dir=summary_dir,
        figures_dir=figures_dir,
        report_dir=report_dir,
        per_point_dir=per_point_dir,
        report_pdf=report_dir / "special_points_reselected_report.pdf",
        point_summary_csv=summary_dir / "point_summary.csv",
        degeneracy_csv=summary_dir / "degeneracy_subspace_summary.csv",
        key_points_csv=summary_dir / "report_key_points.csv",
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _series(values: Iterable[float]) -> np.ndarray:
    return np.array(list(values), dtype=float)


def make_summary_figures(point_rows: list[dict], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    changed = _series(row["whether_selection_changed"] for row in point_rows)
    cat = np.array([row["category"] for row in point_rows], dtype=object)
    edge_old = _series(row["old_edge_weight"] for row in point_rows)
    edge_new = _series(row["best_edge_weight"] for row in point_rows)
    corner_old = _series(row["old_corner_weight"] for row in point_rows)
    corner_new = _series(row["best_corner_weight"] for row in point_rows)

    robust_mask = cat == "robust_topological"
    trans_mask = cat == "near_transition"

    fig1 = out_dir / "state_selection_method_compare.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].scatter(edge_old, edge_new, c=changed, cmap="coolwarm", s=28, alpha=0.85)
    mn = min(float(np.min(edge_old)), float(np.min(edge_new)))
    mx = max(float(np.max(edge_old)), float(np.max(edge_new)))
    axes[0].plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
    axes[0].set_xlabel("old edge weight")
    axes[0].set_ylabel("best_edge_state weight")
    axes[0].set_title("Edge weight: old vs new")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(corner_old, corner_new, c=changed, cmap="coolwarm", s=28, alpha=0.85)
    mn = min(float(np.min(corner_old)), float(np.min(corner_new)))
    mx = max(float(np.max(corner_old)), float(np.max(corner_new)))
    axes[1].plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
    axes[1].set_xlabel("old corner weight")
    axes[1].set_ylabel("best_corner_state weight")
    axes[1].set_title("Corner weight: old vs new")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig1, dpi=170)
    plt.close(fig)
    paths["state_selection_method_compare"] = fig1

    fig2 = out_dir / "robust_vs_transition_compare.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for mask, label, color in (
        (robust_mask, "robust_topological", "tab:blue"),
        (trans_mask, "near_transition", "tab:orange"),
    ):
        axes[0].hist(edge_new[mask] - edge_old[mask], bins=12, alpha=0.55, label=label, color=color)
        axes[1].hist(corner_new[mask] - corner_old[mask], bins=12, alpha=0.55, label=label, color=color)
    axes[0].set_title("Edge gain distribution")
    axes[1].set_title("Corner gain distribution")
    axes[0].set_xlabel("best_edge - old_edge")
    axes[1].set_xlabel("best_corner - old_corner")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig2, dpi=170)
    plt.close(fig)
    paths["robust_vs_transition_compare"] = fig2

    corr_a = out_dir / "transition_corridor_A.png"
    corr_b = out_dir / "transition_corridor_B.png"
    _make_transition_corridor(point_rows, corr_a, ("v", "t"), fixed_key="lm")
    _make_transition_corridor(point_rows, corr_b, ("t", "lm"), fixed_key="v")
    paths["transition_corridor_A"] = corr_a
    paths["transition_corridor_B"] = corr_b
    return paths


def _make_transition_corridor(rows: list[dict], out_path: Path, xk_yk: tuple[str, str], fixed_key: str) -> None:
    xk, yk = xk_yk
    x = _series(row[xk] for row in rows)
    y = _series(row[yk] for row in rows)
    gain = _series(float(row["best_edge_weight"]) - float(row["old_edge_weight"]) for row in rows)
    fixed = _series(row[fixed_key] for row in rows)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sc = ax.scatter(x, y, c=gain, s=90, cmap="RdBu_r", edgecolors="black", linewidths=0.4)
    for xi, yi, fi in zip(x, y, fixed):
        ax.text(xi + 0.005, yi + 0.005, f"{fixed_key}={fi:.2f}", fontsize=6, alpha=0.8)
    ax.set_xlabel(xk)
    ax.set_ylabel(yk)
    ax.set_title(f"Transition corridor map: gain in edge weight")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="best_edge - old_edge")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_pdf_report(
    report_path: Path,
    point_rows: list[dict],
    degeneracy_rows: list[dict],
    key_rows: list[dict],
    figure_paths: dict[str, Path],
    per_point_dirs: dict[str, Path],
    config_text: str,
) -> None:
    with PdfPages(report_path) as pdf:
        _page_title(pdf)
        _page_motivation(pdf)
        _page_methods(pdf, config_text=config_text)
        _page_overall_conclusion(pdf, point_rows=point_rows)
        _page_main_transition(pdf, figure_paths=figure_paths)
        _page_physical_judgement(pdf, point_rows=point_rows, key_rows=key_rows)
        _page_limitations(pdf)
        _appendix_points(pdf, point_rows=point_rows, per_point_dirs=per_point_dirs)
        _appendix_methods(pdf, config_text=config_text, degeneracy_rows=degeneracy_rows)


def _page_title(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    ax.text(
        0.0,
        0.95,
        "交叉链特殊点的重选态分析报告：\n边界态是否被当前选态方式低估？",
        fontsize=19,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        0.75,
        "核心问题：稳健区看起来不够边界化，究竟是物理上确实如此，还是选态方式没有抓到代表性的边界态。",
        fontsize=12,
        va="top",
    )
    ax.text(0.0, 0.67, "本报告基于 special points 数据包，采用 old/new/degeneracy-rotation 三层对比。", fontsize=11, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _page_motivation(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    text = (
        "第1节 研究动机\n\n"
        "旧报告里常用 near-zero mode 或子空间加权模式。对于稳健拓扑区，"
        "这可能偏向混合态，而不是最边界化态。\n\n"
        "因此本次重分析把“态选取”本身当作可检验对象：\n"
        "1) 逐态计算边/角/体投影；\n"
        "2) 给出 old 与 best_edge/best_corner 的单态对照；\n"
        "3) 在近简并子空间内做投影旋转，寻找极大边界化线性组合；\n"
        "4) 对照 ribbon 引导窗口与零能窗口。"
    )
    ax.text(0.0, 0.95, text, fontsize=12, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _page_methods(pdf: PdfPages, config_text: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    text = (
        "第2节 方法改进\n\n"
        "old_mode: 旧逻辑，接近零能 + 子空间混合。\n"
        "best_edge_state: 在目标窗口内选择 W_edge_total 最大单态。\n"
        "best_corner_state: 在目标窗口内选择 W_corner 最大单态。\n"
        "optional_best_side_state: left/right/top/bottom 最大单态。\n\n"
        "近简并旋转：对 delta_E 足够小的子空间，分别对 P_edge 与 P_corner 做再对角化，\n"
        "得到 maximally edge/corner-localized rotated state。"
    )
    ax.text(0.0, 0.95, text, fontsize=11.4, va="top")
    ax.text(0.0, 0.55, "附：参数配置（截断）", fontsize=11, fontweight="bold")
    ax.text(0.0, 0.52, config_text[:2200], family="monospace", fontsize=8.6, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _page_overall_conclusion(pdf: PdfPages, point_rows: list[dict]) -> None:
    total = len(point_rows)
    changed = sum(int(row["whether_selection_changed"]) for row in point_rows)
    gain_edge = float(
        np.mean([float(row["best_edge_weight"]) - float(row["old_edge_weight"]) for row in point_rows])
    )
    gain_corner = float(
        np.mean([float(row["best_corner_weight"]) - float(row["old_corner_weight"]) for row in point_rows])
    )
    robust = [r for r in point_rows if r["category"] == "robust_topological"]
    trans = [r for r in point_rows if r["category"] == "near_transition"]
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    text = (
        "第3节 总体结论\n\n"
        f"- 新选态后有 {changed}/{total} 个点发生主选态变化。\n"
        f"- 平均边界权重提升（best_edge-old）: {gain_edge:.4f}\n"
        f"- 平均角区权重提升（best_corner-old）: {gain_corner:.4f}\n\n"
        f"- robust_topological 点数: {len(robust)}\n"
        f"- near_transition 点数: {len(trans)}\n\n"
        "回答核心问题：\n"
        "1) 部分稳健区点存在明显“旧方法低估边界性”；\n"
        "2) 也存在重选态后仍 bulk/mixed 的稳健点；\n"
        "3) 因此“选态偏差 + 物理本征混合”两者并存。"
    )
    ax.text(0.0, 0.95, text, fontsize=12, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _page_main_transition(pdf: PdfPages, figure_paths: dict[str, Path]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    ax.text(0.0, 0.96, "第4节 主相变过程（两条走廊）", fontsize=13, fontweight="bold", va="top")
    a = plt.imread(figure_paths["transition_corridor_A"])
    b = plt.imread(figure_paths["transition_corridor_B"])
    c = plt.imread(figure_paths["robust_vs_transition_compare"])
    ax1 = fig.add_axes([0.08, 0.59, 0.84, 0.26])
    ax1.imshow(a)
    ax1.axis("off")
    ax2 = fig.add_axes([0.08, 0.31, 0.84, 0.26])
    ax2.imshow(b)
    ax2.axis("off")
    ax3 = fig.add_axes([0.08, 0.03, 0.84, 0.26])
    ax3.imshow(c)
    ax3.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _page_physical_judgement(pdf: PdfPages, point_rows: list[dict], key_rows: list[dict]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    stronger = [
        r
        for r in point_rows
        if float(r["best_edge_weight"]) > float(r["old_edge_weight"]) + 0.08
    ]
    still_mixed = [
        r
        for r in point_rows
        if float(r["best_edge_weight"]) < 0.35 and r["category"] == "robust_topological"
    ]
    lines = [
        "第5节 最关键的物理判断",
        "",
        f"旧方法低估边界性的点（best_edge-old > 0.08）: {len(stronger)}",
        f"重选态后仍偏 bulk/mixed 的稳健点: {len(still_mixed)}",
        "",
        "结论：不是单一原因。部分点主要是选态问题，部分点确实体现强混合。",
        "",
        "报告关键点：",
    ]
    for row in key_rows:
        lines.append(
            f"- {row['key_role']}: {row['point_id']} "
            f"(edge={float(row['best_edge_weight']):.3f}, corner={float(row['best_corner_weight']):.3f}, role={row['recommended_physical_role']})"
        )
    ax.text(0.0, 0.95, "\n".join(lines), fontsize=11.6, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _page_limitations(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    text = (
        "第6节 方法局限性\n\n"
        "1) 有限尺寸效应仍在：20x20 flake 不代表热力学极限。\n"
        "2) ribbon 与 flake 的能量映射不是严格一一对应。\n"
        "3) Z2 与边界图像不必逐点同步。\n"
        "4) 单态图和窗口积分 LDOS 可能给出不同视觉判断。"
    )
    ax.text(0.0, 0.95, text, fontsize=12, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _appendix_points(pdf: PdfPages, point_rows: list[dict], per_point_dirs: dict[str, Path]) -> None:
    for row in point_rows:
        point_id = row["point_id"]
        point_dir = per_point_dirs[point_id]
        fig = plt.figure(figsize=(8.27, 11.69))
        title_ax = fig.add_axes([0.06, 0.92, 0.88, 0.06])
        title_ax.axis("off")
        title_ax.text(
            0.0,
            0.6,
            (
                f"附录A - {point_id}\n"
                f"v={float(row['v']):.2f}, t={float(row['t']):.2f}, lm={float(row['lm']):.2f}, "
                f"gap={float(row['gap']):.3e}, Z2={row['Z2']}, Chern={float(row['Chern']):.3f}, Wilson={float(row['Wilson']):.3f}"
            ),
            fontsize=9.8,
            va="center",
        )
        panels = [
            ("old_selection.png", [0.06, 0.59, 0.42, 0.29], "old"),
            ("best_edge_state.png", [0.52, 0.59, 0.42, 0.29], "best_edge"),
            ("rotated_edge_state.png", [0.06, 0.25, 0.42, 0.29], "rot_edge"),
            ("ldos_map.png", [0.52, 0.25, 0.42, 0.29], "LDOS"),
        ]
        for filename, pos, label in panels:
            ax = fig.add_axes(pos)
            path = point_dir / filename
            if path.exists():
                ax.imshow(plt.imread(path))
            ax.axis("off")
            ax.set_title(label, fontsize=8)
        note_ax = fig.add_axes([0.06, 0.05, 0.88, 0.14])
        note_ax.axis("off")
        note = (
            f"old_edge={float(row['old_edge_weight']):.3f}, best_edge={float(row['best_edge_weight']):.3f}, "
            f"rot_edge={float(row['rotated_edge_weight']):.3f}; role={row['recommended_physical_role']}"
        )
        note_ax.text(0.0, 0.8, note, fontsize=8.8, va="top")
        pdf.savefig(fig)
        plt.close(fig)


def _appendix_methods(pdf: PdfPages, config_text: str, degeneracy_rows: list[dict]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    ax.text(0.0, 0.95, "附录B 方法参数", fontsize=13, fontweight="bold", va="top")
    ax.text(0.0, 0.90, config_text[:2600], family="monospace", fontsize=8.5, va="top")
    if degeneracy_rows:
        lines = ["", "near-degeneracy 统计（前20条）"]
        for row in degeneracy_rows[:20]:
            dim = row.get("subspace_dim", row.get("dim", "NA"))
            edge_gain = row.get("edge_gain", row.get("best_edge_projection_after", "NA"))
            corner_gain = row.get("corner_gain", row.get("best_corner_projection_after", "NA"))
            lines.append(
                (
                    f"{row['point_id']} dim={dim} "
                    f"edge_gain={float(edge_gain):.4f} corner_gain={float(corner_gain):.4f}"
                )
            )
        ax.text(0.0, 0.38, "\n".join(lines), fontsize=8.7, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def build_report_pdf(output_root: Path, config: dict) -> Path:
    """Build the complete reselected-analysis PDF report from result tables."""
    output_root = Path(output_root)
    paths = ensure_report_dirs(output_root)
    point_rows = list(csv.DictReader(paths.point_summary_csv.open("r", encoding="utf-8")))
    deg_rows = list(csv.DictReader(paths.degeneracy_csv.open("r", encoding="utf-8")))
    key_rows = list(csv.DictReader(paths.key_points_csv.open("r", encoding="utf-8")))
    figure_paths = make_summary_figures(point_rows, paths.figures_dir)
    per_point_dirs = {row["point_id"]: paths.per_point_dir / row["point_id"] for row in point_rows}
    render_pdf_report(
        report_path=paths.report_pdf,
        point_rows=point_rows,
        degeneracy_rows=deg_rows,
        key_rows=key_rows,
        figure_paths=figure_paths,
        per_point_dirs=per_point_dirs,
        config_text=json.dumps(config, ensure_ascii=False, indent=2),
    )
    return paths.report_pdf
