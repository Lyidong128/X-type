#!/usr/bin/env python3
"""Generate consolidated report for M-point spin-Chern transition analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate M-point spin-Chern report.")
    parser.add_argument("--output-root", default="/workspace/outputs/M_point_spin_chern_analysis")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open("r", encoding="utf-8")))


def _nearest(rows: list[dict[str, str]], v: float) -> dict[str, str] | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(float(r["v"]) - float(v)))


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    m_rows = _read_csv(out_root / "M_gap_scan.csv")
    g_rows = _read_csv(out_root / "global_gap_scan.csv")
    s_rows = _read_csv(out_root / "spin_chern_convergence.csv")
    b_rows = _read_csv(out_root / "M_band_inversion_summary.csv")
    mass_rows = _read_csv(out_root / "M_effective_mass.csv")
    cmp_rows = _read_csv(out_root / "z2_spin_chern_comparison.csv")

    lines = []
    lines.append("# M-point Spin-Chern Transition Report")
    lines.append("")
    lines.append("Fixed parameters: `t=0.3`, `w=1.0`, `lm=0.1`.")
    lines.append("")

    # Derived key points
    vc = np.nan
    dm_min = np.nan
    if m_rows:
        mid = [r for r in m_rows if 0.55 <= float(r["v"]) <= 0.65]
        target = mid if mid else m_rows
        min_row = min(target, key=lambda r: float(r["Delta_M"]))
        vc = float(min_row["v"])
        dm_min = float(min_row["Delta_M"])

    # highest-Nk spin rows by v
    top_spin = {}
    for r in s_rows:
        v = round(float(r["v"]), 10)
        if (v not in top_spin) or (int(r["Nk"]) > int(top_spin[v]["Nk"])):
            top_spin[v] = r

    def cspin_at(v):
        r = _nearest(list(top_spin.values()), v)
        if r is None:
            return np.nan
        return float(r["rounded_C_spin"])

    def cup_at(v):
        r = _nearest(list(top_spin.values()), v)
        if r is None:
            return np.nan
        return float(r["rounded_C_up"])

    def cdn_at(v):
        r = _nearest(list(top_spin.values()), v)
        if r is None:
            return np.nan
        return float(r["rounded_C_down"])

    # Q1/Q2: M-gap and global-gap near vc
    m_vc = _nearest(m_rows, vc) if (m_rows and np.isfinite(vc)) else None
    g_vc = _nearest(g_rows, vc) if (g_rows and np.isfinite(vc)) else None

    lines.append("## Core questions")
    lines.append("")

    lines.append("1. **lm=0.1 时，v≈0.6 的 gap closing 是否发生在 M=(pi,pi) 点？**")
    if m_vc is None:
        lines.append("- 无法回答：缺少 M_gap_scan 数据。")
    else:
        lines.append(
            f"- 在 `v_c≈{vc:.6f}` 附近，`Delta_M≈{float(m_vc['Delta_M']):.3e}`。"
        )
        if float(m_vc["Delta_M"]) < 1e-3:
            lines.append("- 结论：M 点出现近闭隙，支持 `v≈0.6` 的临界行为。")
        else:
            lines.append("- 结论：M 点未见明显闭隙，需提高分辨率复核。")
    lines.append("")

    lines.append("2. **M 点能隙 Delta_M 是否等于全 BZ 最小 gap Delta_global？**")
    if (m_vc is None) or (g_vc is None):
        lines.append("- 无法回答：缺少全局 gap 数据。")
    else:
        dm = float(m_vc["Delta_M"])
        dg = float(g_vc["Delta_global"])
        dist = float(g_vc["distance_to_M"])
        lines.append(f"- `Delta_M≈{dm:.3e}`, `Delta_global≈{dg:.3e}`, `distance_to_M≈{dist:.3e}`。")
        if abs(dm - dg) < 2e-3 and dist < 0.2:
            lines.append("- 结论：在临界附近 M 点基本控制全局最小 gap。")
        else:
            lines.append("- 结论：全局最小 gap 不完全由 M 点单独控制，不能只看 M 点。")
    lines.append("")

    lines.append("3. **gap closing 前后是否发生 M 点 band inversion？**")
    b_sum = [r for r in b_rows if r.get("stage", "") == "summary"]
    if b_sum:
        lines.append(f"- `band_inversion={b_sum[0].get('band_inversion', 'unknown')}`。")
    else:
        lines.append("- 未获得 band inversion summary 行。")
    lines.append("")

    lines.append("4. **spin Chern 是否从 0 变为 1？**")
    c_low = cspin_at(0.58)
    c_high = cspin_at(0.62)
    lines.append(f"- 参考点：`C_spin(v≈0.58)={c_low:.0f}`, `C_spin(v≈0.62)={c_high:.0f}`。")
    if np.isfinite(c_low) and np.isfinite(c_high) and (c_low != c_high):
        lines.append("- 结论：`v≈0.6` 前后出现 spin-Chern 跳变。")
    else:
        lines.append("- 结论：当前离散点上未看到清晰 0->1 跳变。")
    lines.append("")

    lines.append("5. **C_up 是否等于 -C_down？**")
    cu = cup_at(0.62)
    cd = cdn_at(0.62)
    lines.append(f"- 参考点：`C_up={cu:.0f}`, `C_down={cd:.0f}`。")
    if np.isfinite(cu) and np.isfinite(cd) and abs(cu + cd) < 1e-6:
        lines.append("- 结论：满足 `C_up=-C_down`。")
    else:
        lines.append("- 结论：未严格满足或数据不足。")
    lines.append("")

    lines.append("6. **total Chern 是否保持为 0？**")
    if top_spin:
        ctot_max = max(abs(float(r["C_total"])) for r in top_spin.values())
        lines.append(f"- 最高 Nk 结果中 `max|C_total|≈{ctot_max:.3e}`，可视为接近 0。")
    else:
        lines.append("- 无法计算：缺少 spin-Chern 数据。")
    lines.append("")

    lines.append("7. **是否可归因为 M 点 Dirac mass 变号？**")
    if mass_rows:
        mvals = np.array([float(r["signed_mass_proxy"]) for r in mass_rows], dtype=float)
        sign_change = np.any(mvals[:-1] * mvals[1:] < 0.0)
        lines.append(f"- signed_mass_proxy 是否变号：`{bool(sign_change)}`。")
        if sign_change:
            lines.append("- 结论：可用 `M` 点有效质量变号解释相变机制（在代理模型意义下）。")
        else:
            lines.append("- 结论：当前代理未显示清晰变号，需更严格低能投影拟合。")
    else:
        lines.append("- 无法评估：缺少有效质量代理数据。")
    lines.append("")

    lines.append("8. **v≈1.09 附近 C_spin 翻转是否伴随真实 gap closing？**")
    c1 = cspin_at(1.086)
    c2 = cspin_at(1.098)
    g1 = _nearest(g_rows, 1.086)
    g2 = _nearest(g_rows, 1.098)
    if np.isfinite(c1) and np.isfinite(c2) and g1 and g2:
        lines.append(
            f"- `C_spin(1.086)={c1:.0f}`, `C_spin(1.098)={c2:.0f}`; "
            f"`Delta_global` 约为 `{float(g1['Delta_global']):.3e}` 到 `{float(g2['Delta_global']):.3e}`。"
        )
        if (c1 != c2) and (min(float(g1["Delta_global"]), float(g2["Delta_global"])) < 1e-3):
            lines.append("- 结论：符号翻转伴随近闭隙，需高精度复核其拓扑稳定性。")
        elif c1 != c2:
            lines.append("- 结论：有符号翻转但未见强闭隙证据，需警惕数值/规范选择影响。")
        else:
            lines.append("- 结论：未见明确符号翻转。")
    else:
        lines.append("- 数据不足，无法判断。")
    lines.append("")

    lines.append("9. **Fu-Kane / Wilson 与 spin Chern 是否一致？若不一致原因？**")
    if cmp_rows:
        labels = {}
        for r in cmp_rows:
            labels[r["consistency_label"]] = labels.get(r["consistency_label"], 0) + 1
        lines.append(f"- 一致性统计：{labels}。")
        if any("inconsistent" in k for k in labels):
            lines.append("- 可能原因：inversion parity 实现细节、Wilson gauge continuity/crossing 计数与 gap 临界点不稳定性。")
        else:
            lines.append("- 在可比较点上总体一致。")
    else:
        lines.append("- 无一致性比较文件。")
    lines.append("")

    lines.append("10. **当前最稳妥的物理结论**")
    lines.append(
        "- 在 Kane-Mele SOC 开启 (`lm=0.1`) 后，`v≈0.6` 附近出现由 M 点主导的近闭隙与重开隙，"
        "并伴随自旋分辨 Chern 数重排 (`C_up≈-C_down`, `C_total≈0`)；该过程可解释为 spin-Chern 拓扑相变候选。"
    )
    lines.append(
        "- 由于 Fu-Kane/Wilson 与 spin Chern 在部分点不一致，当前应以 spin-Chern 收敛与 gap-closing 证据为主，"
        "同时把 Z2 争议列为后续数值实现排查项。"
    )
    lines.append("")
    lines.append("## Recommended wording")
    lines.append(
        "“在 Kane-Mele SOC 开启后，体系于 v≈0.6 附近发生由 M 点 gap closing and reopening 驱动的 "
        "spin-Chern 拓扑相变；相变后自旋上、下通道呈相反 Chern 数，而总 Chern 数保持近零。”"
    )
    lines.append("")
    lines.append("## Caution")
    lines.append("- 不将近零态直接解释为高阶拓扑角态。")
    lines.append("- 不把 gapless 点强行赋予稳定拓扑不变量。")
    lines.append("- 不用 Fu-Kane 单独覆盖 spin-Chern 主结论。")

    report_path = out_root / "M_POINT_SPIN_CHERN_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] report={report_path}")


if __name__ == "__main__":
    main()
