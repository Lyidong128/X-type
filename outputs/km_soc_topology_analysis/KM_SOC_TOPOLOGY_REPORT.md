# Kane-Mele SOC Topology Analysis Report

## 1. Model statement
- 本模型中的 SOC 项呈现自旋依赖虚数跃迁结构，符合 Kane-Mele 型自旋分辨拓扑机制的候选特征。
- 本报告重点检验 `C_total≈0` 条件下的 `Z2-like` 相变，不将近零态直接解释为角态。

## 2. Gap / Chern / Z2 scan summary
- 在可靠开隙点上，`|C_total|` 最大值约为 `8.902e-16`，整体接近零。
- 扫描区间最小 direct gap: `0.000e+00`。
- `lm=0.0` 可靠点的 Z2 取值集合: `[0]`。
- `lm=0.1` 可靠点的 Z2 取值集合: `[0]`。

## 3. Transition classification
- 候选相变总数: `3`；中高可靠候选: `2`。
- `T1`: lm=0.0, v~[0.698,1.1], type=`gapless_region`, gap_min=0.000e+00.
- `T2`: lm=0.1, v~[0.598,0.602], type=`gapless_region`, gap_min=1.388e-16.

## 4. Spin Chern evidence
- 计算点数: `9`；可用点数: `9`。
- 可用点中 `|C_total|` 最大约 `4.594e-16`。
- 可用点中 `|C_spin|` 最大约 `1.000e+00`。
- 若同时出现 `C_up≈-C_down` 与非零 `C_spin`，支持 Kane-Mele 型自旋分辨拓扑机制。

## 5. Wilson loop flow and Z2
- Wilson flow 条目: `14`，其中 uncertain: `2`。
- 覆盖点标签: `['A', 'B', 'C', 'D', 'E', 'F', 'G']`。
- Wannier center crossing parity 用于估计 Z2，gap 过小时标记为 uncertain。

## 6. Band inversion analysis
- T3: `no_clear_band_inversion`。

## 7. Chiral-symmetry breaking by SOC
- 平均 `r_break_mean(lm=0.0)` ≈ `2.565e-01`。
- 平均 `r_break_mean(lm=0.1)` ≈ `2.680e-01`。
- SOC 增强了候选手征破缺量，支持“零能角态不再被手征对称性钉扎”的解释。

## 8. Optional Fu-Kane parity
- Fu-Kane 可用点: `12`，与 Wilson Z2 一致点: `11`。

## 9. Main physical conclusions (cautious)
- 建议使用表述：`零 Chern 数条件下的 Kane-Mele SOC 调控 Z2-like 拓扑相变`。
- 若 `C_up≈-C_down` 且 `C_total≈0`，更符合量子自旋霍尔/自旋分辨拓扑机制，而非 Chern 绝缘体。
- 即使存在 finite-energy 边界态，也不应直接将其等同于已证明的高阶零能角态。

## 10. Recommended next checks
- 进一步细化相变附近 `v` 步长与 `k` 分辨率。
- 对 x/y ribbon 做 helical 分支一致性比对。
- 加入 disorder robustness 与 spin Bott 交叉验证。
- 若继续研究角态，单独做 edge-gap / corner-weight / nested Wilson chain。
