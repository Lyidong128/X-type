#!/usr/bin/env python3
"""Generate consolidated Kane-Mele SOC topology report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KM SOC topology markdown report.")
    parser.add_argument("--output-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--quick", action="store_true", help="Accepted for CLI consistency; report generation is lightweight.")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def _fmt_float(x: float) -> str:
    return f"{x:.6e}"


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    report_path = root / "KM_SOC_TOPOLOGY_REPORT.md"

    gap_rows = _read_csv(root / "gap_z2_scan" / "gap_z2_scan.csv")
    spin_rows = _read_csv(root / "spin_chern" / "spin_chern_summary.csv")
    wilson_rows = _read_csv(root / "wilson_loop" / "wilson_loop_summary.csv")
    trans_rows = _read_csv(root / "gap_closing" / "transition_candidates.csv")
    inv_rows = _read_csv(root / "band_inversion" / "band_inversion_summary.csv")
    chiral_rows = _read_csv(root / "chiral_symmetry" / "chiral_breaking_summary.csv")
    fu_kane_rows = _read_csv(root / "fu_kane" / "fu_kane_summary.csv")

    lines = []
    lines.append("# Kane-Mele SOC Topology Analysis Report")
    lines.append("")
    lines.append("## 1. Model statement")
    lines.append("- 本模型中的 SOC 项呈现自旋依赖虚数跃迁结构，符合 Kane-Mele 型自旋分辨拓扑机制的候选特征。")
    lines.append("- 本报告重点检验 `C_total≈0` 条件下的 `Z2-like` 相变，不将近零态直接解释为角态。")
    lines.append("")

    lines.append("## 2. Gap / Chern / Z2 scan summary")
    if gap_rows:
        gap_arr = np.array([float(r["direct_gap"]) for r in gap_rows], dtype=float)
        ch_arr = np.array([float(r["chern_total"]) for r in gap_rows], dtype=float)
        reli = np.array([int(r["z2_reliable"]) for r in gap_rows], dtype=int)
        if np.any(reli == 1):
            ch_reli = np.abs(ch_arr[reli == 1])
            lines.append(f"- 在可靠开隙点上，`|C_total|` 最大值约为 `{np.max(ch_reli):.3e}`，整体接近零。")
        lines.append(f"- 扫描区间最小 direct gap: `{np.min(gap_arr):.3e}`。")
        for lm in (0.0, 0.1):
            subset = [r for r in gap_rows if abs(float(r["lm"]) - lm) < 1e-12]
            zvals = [int(r["z2"]) for r in subset if int(r["z2_reliable"]) == 1]
            if zvals:
                lines.append(f"- `lm={lm:.1f}` 可靠点的 Z2 取值集合: `{sorted(set(zvals))}`。")
            else:
                lines.append(f"- `lm={lm:.1f}` 无可靠 Z2 点（gap 过小或不稳定）。")
    else:
        lines.append("- 缺少 gap_z2_scan 数据。")
    lines.append("")

    lines.append("## 3. Transition classification")
    if trans_rows:
        reliable = [r for r in trans_rows if r["reliability"] in {"high", "medium"}]
        lines.append(f"- 候选相变总数: `{len(trans_rows)}`；中高可靠候选: `{len(reliable)}`。")
        for r in reliable[:6]:
            lines.append(
                f"- `{r['transition_id']}`: lm={r['lm']}, v~[{r['v_left']},{r['v_right']}], "
                f"type=`{r['transition_type']}`, gap_min={float(r['gap_min']):.3e}."
            )
    else:
        lines.append("- 未检测到 transition_candidates.csv。")
    lines.append("")

    lines.append("## 4. Spin Chern evidence")
    if spin_rows:
        reli = [r for r in spin_rows if r["reliability"] in {"reliable", "semi_reliable"}]
        lines.append(f"- 计算点数: `{len(spin_rows)}`；可用点数: `{len(reli)}`。")
        if reli:
            ctot = np.array([float(r["C_total"]) for r in reli], dtype=float)
            cspin = np.array([float(r["C_spin"]) for r in reli], dtype=float)
            lines.append(f"- 可用点中 `|C_total|` 最大约 `{np.max(np.abs(ctot)):.3e}`。")
            lines.append(f"- 可用点中 `|C_spin|` 最大约 `{np.max(np.abs(cspin)):.3e}`。")
            lines.append("- 若同时出现 `C_up≈-C_down` 与非零 `C_spin`，支持 Kane-Mele 型自旋分辨拓扑机制。")
    else:
        lines.append("- 未生成 spin_chern_summary.csv。")
    lines.append("")

    lines.append("## 5. Wilson loop flow and Z2")
    if wilson_rows:
        uncertain = [r for r in wilson_rows if r["reliability"] != "reliable"]
        lines.append(f"- Wilson flow 条目: `{len(wilson_rows)}`，其中 uncertain: `{len(uncertain)}`。")
        points = sorted(set(r["point_label"] for r in wilson_rows))
        lines.append(f"- 覆盖点标签: `{points}`。")
        lines.append("- Wannier center crossing parity 用于估计 Z2，gap 过小时标记为 uncertain。")
    else:
        lines.append("- 未生成 wilson_loop_summary.csv。")
    lines.append("")

    lines.append("## 6. Band inversion analysis")
    if inv_rows:
        inv_summary = [r for r in inv_rows if r["stage"] == "summary"]
        if inv_summary:
            for r in inv_summary:
                lines.append(f"- {r['transition_id']}: `{r['transition_type']}`。")
        else:
            lines.append("- 已生成局部能带/分量图，但 summary 行缺失。")
    else:
        lines.append("- 未生成 band_inversion_summary.csv（可能无可用相变点）。")
    lines.append("")

    lines.append("## 7. Chiral-symmetry breaking by SOC")
    if chiral_rows:
        lm0 = [float(r["r_break_mean"]) for r in chiral_rows if abs(float(r["lm"]) - 0.0) < 1e-12]
        lm1 = [float(r["r_break_mean"]) for r in chiral_rows if abs(float(r["lm"]) - 0.1) < 1e-12]
        if lm0 and lm1:
            lines.append(f"- 平均 `r_break_mean(lm=0.0)` ≈ `{np.mean(lm0):.3e}`。")
            lines.append(f"- 平均 `r_break_mean(lm=0.1)` ≈ `{np.mean(lm1):.3e}`。")
            if np.mean(lm1) > np.mean(lm0):
                lines.append("- SOC 增强了候选手征破缺量，支持“零能角态不再被手征对称性钉扎”的解释。")
            else:
                lines.append("- SOC 下手征破缺量未明显增加，需谨慎解释角态缺失机制。")
    else:
        lines.append("- 未生成 chiral_breaking_summary.csv。")
    lines.append("")

    lines.append("## 8. Optional Fu-Kane parity")
    if fu_kane_rows:
        agree = [r for r in fu_kane_rows if int(r["z2_fu_kane"]) == int(r["z2_wilson_scan"]) and int(r["z2_wilson_scan"]) >= 0]
        lines.append(f"- Fu-Kane 可用点: `{len(fu_kane_rows)}`，与 Wilson Z2 一致点: `{len(agree)}`。")
    else:
        skip_reason = root / "fu_kane" / "skip_reason.txt"
        if skip_reason.exists():
            lines.append(f"- Fu-Kane parity 跳过：`{skip_reason.read_text(encoding='utf-8').strip()}`")
        else:
            lines.append("- 未生成 Fu-Kane 结果。")
    lines.append("")

    lines.append("## 9. Main physical conclusions (cautious)")
    lines.append("- 建议使用表述：`零 Chern 数条件下的 Kane-Mele SOC 调控 Z2-like 拓扑相变`。")
    lines.append("- 若 `C_up≈-C_down` 且 `C_total≈0`，更符合量子自旋霍尔/自旋分辨拓扑机制，而非 Chern 绝缘体。")
    lines.append("- 即使存在 finite-energy 边界态，也不应直接将其等同于已证明的高阶零能角态。")
    lines.append("")

    lines.append("## 10. Recommended next checks")
    lines.append("- 进一步细化相变附近 `v` 步长与 `k` 分辨率。")
    lines.append("- 对 x/y ribbon 做 helical 分支一致性比对。")
    lines.append("- 加入 disorder robustness 与 spin Bott 交叉验证。")
    lines.append("- 若继续研究角态，单独做 edge-gap / corner-weight / nested Wilson chain。")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] report={report_path}")


if __name__ == "__main__":
    main()
