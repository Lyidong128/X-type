# NEXT_STAGE_REPORT

## 1. 模型与参数说明
- 模型：二维 `Hxtype(k)`，参数约束使用 `t=0.5` 与 `w=2-v`。
- 重点点位：A(0.90,0.20), B(1.10,0.20) 及自动选择的 C/D/E。
- 本报告仅总结计算输出，不把候选对称性直接当成已证明保护机制。

## 2. 对称性检查结果
- 已输出 `symmetry_summary.csv`，记录 A/B 上 TRS/INV/PHS/CHIRAL 的最优候选残差。
- 状态统计：{'reliably_satisfied': 4, 'not_satisfied': 4}。
- 若 `comment=candidate_symmetry`，表示仅为候选矩阵测试，不等同于物理上已确认保护对称性。

## 3. Wilson loop / Wannier flow
- 已输出 `wilson_loop_summary.csv`，共 12 条方向记录（kx-loop 与 ky-loop）。
- estimated_z2 统计：0=9, 1=3。
- 该估计来自 Wannier center 交叉计数，属于数值判据，建议与对称性和边界谱联合解读。

## 4. x/y 双方向 ribbon 对比
- 已输出 `ribbon_xy_classification.csv`，分类统计：{'empty_gap': 12}。
- 通过 x-open 与 y-open 对比，可以检查边界谱各向异性与连接分支方向选择性。

## 5. OBC 近零能态空间分布
- 已输出 `obc_wavefunction_summary.csv`，状态分类统计：{'corner-like': 20, 'mixed': 8, 'bulk-like': 12, 'edge-like': 8}。
- 存在一定 corner-like 分布，但仍需更大尺寸与扰动鲁棒性检验确认。

## 6. IPR 与有限尺寸标度
- 已输出 `finite_size_summary.csv`，记录 18 条 (point,L) 数据。
- `finite_size_fit_summary.json` 最优拟合类型统计：{'exponential': 2, 'powerlaw': 4}。
- min|E| 的指数/幂律拟合仅反映有限尺寸趋势，不单独构成拓扑相证明。

## 7. transition refinement
- 已输出 `transition_candidates.csv`，列出 A/B 走廊内 gap 最小候选点。
  - A rank1: v=1.0000, lm=0.1000, gap=2.198e-01, z2=1, chern=0.000
  - A rank2: v=0.9900, lm=0.1000, gap=2.196e-01, z2=1, chern=-0.000
  - A rank3: v=0.9800, lm=0.1000, gap=2.194e-01, z2=0, chern=0.000
  - A rank4: v=0.9700, lm=0.1000, gap=2.192e-01, z2=1, chern=-0.000
  - A rank5: v=0.9600, lm=0.1000, gap=2.191e-01, z2=0, chern=-0.000
  - A rank6: v=0.9500, lm=0.1000, gap=2.189e-01, z2=0, chern=0.000

## 8. 扰动鲁棒性
- disorder 记录数：36；boundary 记录数：9；TRS-breaking 记录数：9。
- 随机无序结果包含均值与标准差，避免单次 realization 偶然性。

## 9. spin topology 可计算性
- 已输出 `spin_topology_summary.csv`，共 4 条点位记录。
- 该结果依赖可识别的自旋/赝自旋算符定义。

## 10. 谨慎结论
- 若 Wilson loop 给出非平庸指示且 ribbon 出现连接边界分支，可表述为“支持 Z2-like 非平庸相”。
- 若 Bott≈0，应解释为“不支持 Chern 型拓扑相”，不等价于否定 Z2-like 机制。
- 若近零态主要 edge-like，可表述为“近零能态主要表现为边界局域态”。
- 若对称性仍为 candidate 级别，应明确“保护对称性仍需进一步确认”。

## 11. 仍需补充的问题
- 更大尺寸 OBC 下 corner-like 分布是否稳定。
- 候选保护对称性是否能由模型构造显式推导。
- transition corridor 中 Z2 变化与 gap closing 动量的对应关系是否连续。
- 在可定义自旋算符时，spin Chern/spin Bott 与 Wilson 结论一致性。
