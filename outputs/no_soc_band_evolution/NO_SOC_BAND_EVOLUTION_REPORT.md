# NO_SOC_BAND_EVOLUTION_REPORT

## 1. 分析目的
- 在无 SOC (`lm=0.0`) 条件下，观察 `v` 穿过 `v_c=w-t=0.700` 时的能带演化。
- 固定参数：`t=0.3, w=1.0, lm=0.0`。

## 2. 高对称路径能带结果
- 已计算 `Gamma->X->M->Y->Gamma` 上 `v=[0.60,0.65,0.68,0.70,0.72,0.75,0.80]` 的完整能带图。
- 随 `v` 增大，M 点附近中间两条带逐步靠近，在 `v≈0.70` 闭合，随后重新打开。
- 对应图：`band_v_*.png` 与 `band_evolution_montage.png`。

## 3. M 点能隙与质量项
- 数值最小 M 点能隙出现在 `v≈0.700`，`Delta_M≈4.996e-16`。
- 在 `v=0.68` 时 `Delta_M≈4.000e-02`，在 `v=0.72` 时 `Delta_M≈4.000e-02`，表现为闭合后重开。
- `Delta_M(v)` 与理论 `2|m_M|=2|v+t-w|` 在扫描中一致（见 `M_gap_vs_v.png`）。

## 4. 带成分交换（M 点）
- 基于 up-spin 4x4 block 的 valence/conduction 本征矢分量权重，比较了 `v=0.68` 与 `v=0.72`。
- 组件交换判据：`cross_score=0.141803`, `direct_score=0.613461`, `inversion_by_components=True`。
- 若 `cross_score < direct_score`，说明前后 valence/conduction 的分量更接近互换关系，可解释为 M 点 band inversion。

## 5. 物理解释
- 无 SOC 下，M 点能带重构由 `m_M(v,t,w)=v+t-w` 变号控制。
- 当 `v` 穿过 `v_c=w-t`，两条低能态 `E_a=v-w+2t` 与 `E_b=w-v` 在 M 点交叉，导致局域闭隙与重开。
- 由于没有 Kane-Mele SOC，本过程不能直接称为 spin-Chern 或 Z2 拓扑相变。

## 6. 推荐表述
“无 SOC 条件下，随着 v 增大，M 点有效质量项 m_M=v+t-w 在 v_c=w-t=0.7 处变号，导致 M 点能隙闭合并重新打开，价带和导带成分发生交换，表明体系发生 M 点 band inversion / 能带重构。”
