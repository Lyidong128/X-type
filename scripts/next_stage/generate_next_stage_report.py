#!/usr/bin/env python3
"""Generate consolidated markdown report for next-stage analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(v: str | float | int, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def generate_report(root: Path) -> str:
    sym_csv = root / "symmetry" / "symmetry_summary.csv"
    wilson_csv = root / "wilson_loop" / "wilson_loop_summary.csv"
    ribbon_csv = root / "ribbon_xy" / "ribbon_xy_classification.csv"
    obc_csv = root / "obc_wavefunction" / "obc_wavefunction_summary.csv"
    finite_csv = root / "finite_size_ipr" / "finite_size_summary.csv"
    fit_json = root / "finite_size_ipr" / "finite_size_fit_summary.json"
    trans_csv = root / "transition_refinement" / "transition_candidates.csv"
    rob_dis_csv = root / "robustness" / "disorder_summary.csv"
    rob_bd_csv = root / "robustness" / "boundary_perturbation_summary.csv"
    rob_trs_csv = root / "robustness" / "trs_breaking_summary.csv"
    spin_csv = root / "spin_topology_optional" / "spin_topology_summary.csv"
    spin_skip = root / "spin_topology_optional" / "skip_reason.txt"

    sym_rows = read_csv(sym_csv)
    wil_rows = read_csv(wilson_csv)
    rib_rows = read_csv(ribbon_csv)
    obc_rows = read_csv(obc_csv)
    fs_rows = read_csv(finite_csv)
    trans_rows = read_csv(trans_csv)
    dis_rows = read_csv(rob_dis_csv)
    bd_rows = read_csv(rob_bd_csv)
    trs_rows = read_csv(rob_trs_csv)
    spin_rows = read_csv(spin_csv)
    fit_data = json.loads(fit_json.read_text(encoding="utf-8")) if fit_json.exists() else {}

    lines: list[str] = []
    lines.append("# NEXT_STAGE_REPORT")
    lines.append("")
    lines.append("## 1. 模型与参数说明")
    lines.append("- 模型：二维 `Hxtype(k)`，参数约束使用 `t=0.5` 与 `w=2-v`。")
    lines.append("- 重点点位：A(0.90,0.20), B(1.10,0.20) 及自动选择的 C/D/E。")
    lines.append("- 本报告仅总结计算输出，不把候选对称性直接当成已证明保护机制。")
    lines.append("")

    lines.append("## 2. 对称性检查结果")
    if sym_rows:
        by_status = {}
        for r in sym_rows:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        lines.append(f"- 已输出 `symmetry_summary.csv`，记录 A/B 上 TRS/INV/PHS/CHIRAL 的最优候选残差。")
        lines.append(f"- 状态统计：{by_status}。")
        lines.append("- 若 `comment=candidate_symmetry`，表示仅为候选矩阵测试，不等同于物理上已确认保护对称性。")
    else:
        lines.append("- 未检测到对称性输出文件。")
    lines.append("")

    lines.append("## 3. Wilson loop / Wannier flow")
    if wil_rows:
        lines.append(f"- 已输出 `wilson_loop_summary.csv`，共 {len(wil_rows)} 条方向记录（kx-loop 与 ky-loop）。")
        z2_vals = [int(fnum(r.get("estimated_z2", 0))) for r in wil_rows if r.get("estimated_z2", "") != ""]
        if z2_vals:
            lines.append(f"- estimated_z2 统计：0={z2_vals.count(0)}, 1={z2_vals.count(1)}。")
        lines.append("- 该估计来自 Wannier center 交叉计数，属于数值判据，建议与对称性和边界谱联合解读。")
    else:
        lines.append("- 未检测到 Wilson loop 输出文件。")
    lines.append("")

    lines.append("## 4. x/y 双方向 ribbon 对比")
    if rib_rows:
        cls_count = {}
        for r in rib_rows:
            key = r.get("classification", "unknown")
            cls_count[key] = cls_count.get(key, 0) + 1
        lines.append(f"- 已输出 `ribbon_xy_classification.csv`，分类统计：{cls_count}。")
        lines.append("- 通过 x-open 与 y-open 对比，可以检查边界谱各向异性与连接分支方向选择性。")
    else:
        lines.append("- 未检测到 ribbon_xy 分类输出。")
    lines.append("")

    lines.append("## 5. OBC 近零能态空间分布")
    if obc_rows:
        cls = {}
        for r in obc_rows:
            key = r.get("classification", "unknown")
            cls[key] = cls.get(key, 0) + 1
        lines.append(f"- 已输出 `obc_wavefunction_summary.csv`，状态分类统计：{cls}。")
        corner_like = cls.get("corner-like", 0)
        edge_like = cls.get("edge-like", 0)
        if corner_like <= edge_like:
            lines.append("- 当前证据以 edge-like 为主；若缺少稳定 corner-like 证据，不应宣称已证明高阶拓扑角态。")
        else:
            lines.append("- 存在一定 corner-like 分布，但仍需更大尺寸与扰动鲁棒性检验确认。")
    else:
        lines.append("- 未检测到 OBC 波函数输出。")
    lines.append("")

    lines.append("## 6. IPR 与有限尺寸标度")
    if fs_rows:
        lines.append(f"- 已输出 `finite_size_summary.csv`，记录 {len(fs_rows)} 条 (point,L) 数据。")
        if fit_data:
            model_count = {}
            for _, item in fit_data.items():
                best = item.get("best_model", "undetermined")
                model_count[best] = model_count.get(best, 0) + 1
            lines.append(f"- `finite_size_fit_summary.json` 最优拟合类型统计：{model_count}。")
        lines.append("- min|E| 的指数/幂律拟合仅反映有限尺寸趋势，不单独构成拓扑相证明。")
    else:
        lines.append("- 未检测到有限尺寸输出。")
    lines.append("")

    lines.append("## 7. transition refinement")
    if trans_rows:
        lines.append(f"- 已输出 `transition_candidates.csv`，列出 A/B 走廊内 gap 最小候选点。")
        top = trans_rows[: min(6, len(trans_rows))]
        for r in top:
            lines.append(
                f"  - {r['center_label']} rank{r['rank']}: v={float(r['v']):.4f}, lm={float(r['lm']):.4f}, "
                f"gap={float(r['gap']):.3e}, z2={int(float(r['z2']))}, chern={float(r['chern']):.3f}"
            )
    else:
        lines.append("- 未检测到 transition refinement 输出。")
    lines.append("")

    lines.append("## 8. 扰动鲁棒性")
    if dis_rows or bd_rows or trs_rows:
        lines.append(f"- disorder 记录数：{len(dis_rows)}；boundary 记录数：{len(bd_rows)}；TRS-breaking 记录数：{len(trs_rows)}。")
        lines.append("- 随机无序结果包含均值与标准差，避免单次 realization 偶然性。")
    else:
        lines.append("- 未检测到 robustness 输出。")
    lines.append("")

    lines.append("## 9. spin topology 可计算性")
    if spin_skip.exists():
        lines.append(f"- 本轮跳过 spin Chern/spin Bott：{spin_skip.read_text(encoding='utf-8').strip()}")
    elif spin_rows:
        lines.append(f"- 已输出 `spin_topology_summary.csv`，共 {len(spin_rows)} 条点位记录。")
        lines.append("- 该结果依赖可识别的自旋/赝自旋算符定义。")
    else:
        lines.append("- 未检测到 spin topology 输出。")
    lines.append("")

    lines.append("## 10. 谨慎结论")
    lines.append("- 若 Wilson loop 给出非平庸指示且 ribbon 出现连接边界分支，可表述为“支持 Z2-like 非平庸相”。")
    lines.append("- 若 Bott≈0，应解释为“不支持 Chern 型拓扑相”，不等价于否定 Z2-like 机制。")
    lines.append("- 若近零态主要 edge-like，可表述为“近零能态主要表现为边界局域态”。")
    lines.append("- 若对称性仍为 candidate 级别，应明确“保护对称性仍需进一步确认”。")
    lines.append("")

    lines.append("## 11. 仍需补充的问题")
    lines.append("- 更大尺寸 OBC 下 corner-like 分布是否稳定。")
    lines.append("- 候选保护对称性是否能由模型构造显式推导。")
    lines.append("- transition corridor 中 Z2 变化与 gap closing 动量的对应关系是否连续。")
    lines.append("- 在可定义自旋算符时，spin Chern/spin Bott 与 Wilson 结论一致性。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate next stage markdown report.")
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/outputs/next_stage_analysis"))
    args = parser.parse_args()
    report = generate_report(args.output_root)
    out_path = args.output_root / "NEXT_STAGE_REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[ok] report generated: {out_path}")


if __name__ == "__main__":
    main()
