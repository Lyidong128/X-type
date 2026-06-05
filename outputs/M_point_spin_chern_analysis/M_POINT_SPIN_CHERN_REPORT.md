# M-point Spin-Chern Transition Report

Fixed parameters: `t=0.3`, `w=1.0`, `lm=0.1`.

## Core questions

1. **lm=0.1 时，v≈0.6 的 gap closing 是否发生在 M=(pi,pi) 点？**
- 在 `v_c≈0.600000` 附近，`Delta_M≈1.388e-16`。
- 结论：M 点出现近闭隙，支持 `v≈0.6` 的临界行为。

2. **M 点能隙 Delta_M 是否等于全 BZ 最小 gap Delta_global？**
- `Delta_M≈1.388e-16`, `Delta_global≈1.388e-16`, `distance_to_M≈0.000e+00`。
- 结论：在临界附近 M 点基本控制全局最小 gap。

3. **gap closing 前后是否发生 M 点 band inversion？**
- `band_inversion=detected`。

4. **spin Chern 是否从 0 变为 1？**
- 参考点：`C_spin(v≈0.58)=0`, `C_spin(v≈0.62)=1`。
- 结论：`v≈0.6` 前后出现 spin-Chern 跳变。

5. **C_up 是否等于 -C_down？**
- 参考点：`C_up=1`, `C_down=-1`。
- 结论：满足 `C_up=-C_down`。

6. **total Chern 是否保持为 0？**
- 最高 Nk 结果中 `max|C_total|≈6.008e-16`，可视为接近 0。

7. **是否可归因为 M 点 Dirac mass 变号？**
- signed_mass_proxy 是否变号：`True`。
- 结论：可用 `M` 点有效质量变号解释相变机制（在代理模型意义下）。

8. **v≈1.09 附近 C_spin 翻转是否伴随真实 gap closing？**
- 次级窗口最小全局 gap 在 `v≈1.093000`，`Delta_global≈5.718e-06`，`k_min≈(6.283, 3.142)`。
- 近邻点 `C_spin(v_left, v_c, v_right)=(1, -1, -1)`；`Nk` 横向取值集合（在 v_c）为 `[-1]`。
- 结论：若左右点符号不同且闭隙显著，可视为候选翻转；当前仍建议加密 Nk/步长复核。

9. **Fu-Kane / Wilson 与 spin Chern 是否一致？若不一致原因？**
- 一致性统计：{'all_consistent': 3, 'spin_chern_vs_fu_kane_inconsistent': 8, 'z2_unavailable': 1}。
- 可能原因：inversion parity 实现细节、Wilson gauge continuity/crossing 计数与 gap 临界点不稳定性。

10. **当前最稳妥的物理结论**
- 在 Kane-Mele SOC 开启 (`lm=0.1`) 后，`v≈0.6` 附近出现由 M 点主导的近闭隙与重开隙，并伴随自旋分辨 Chern 数重排 (`C_up≈-C_down`, `C_total≈0`)；该过程可解释为 spin-Chern 拓扑相变候选。
- 由于 Fu-Kane/Wilson 与 spin Chern 在部分点不一致，当前应以 spin-Chern 收敛与 gap-closing 证据为主，同时把 Z2 争议列为后续数值实现排查项。

## Recommended wording
“在 Kane-Mele SOC 开启后，体系于 v≈0.6 附近发生由 M 点 gap closing and reopening 驱动的 spin-Chern 拓扑相变；相变后自旋上、下通道呈相反 Chern 数，而总 Chern 数保持近零。”

## Caution
- 不将近零态直接解释为高阶拓扑角态。
- 不把 gapless 点强行赋予稳定拓扑不变量。
- 不用 Fu-Kane 单独覆盖 spin-Chern 主结论。
