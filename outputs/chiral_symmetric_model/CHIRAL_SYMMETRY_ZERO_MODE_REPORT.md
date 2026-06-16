# CHIRAL_SYMMETRY_ZERO_MODE_REPORT

## 1) 原始 H0 的手征对称性检查
- 使用 Gamma=diag(+1,+1,-1,-1)，每个 v 在 BZ (101x101) 网格上计算。
- H0 最大误差（跨全部 v 取最大）: 1.279204e+00
- H0 平均误差（对每个 v 的 mean 再取平均）: 5.154375e-01

## 2) 跃迁项手征分类
- 允许 A-B: (0,2), (0,3), (1,2), (1,3)
- 破缺同子格: (0,1), (2,3)
- 详见 `hopping_chiral_classification.csv`。

## 3) 手征对称化 H_chiral 验证
- H_chiral 最大误差（跨全部 v 取最大）: 0.000000e+00
- H_chiral 平均误差（对每个 v 的 mean 再取平均）: 0.000000e+00
- block-off-diagonal 重构误差 (max/mean): 0.000000e+00 / 0.000000e+00

## 4) Q(k) block 输出
- 已输出: `Q_block_expression.md`

## 5) 5x5 OBC 谱比较设置
- Lx=5, Ly=5, t=0.300, w=1.000
- v_list=[0.5, 0.6, 0.7, 0.8]
- alpha_list=[0.0, 0.25, 0.5, 0.75, 1.0]
- 零能阈值: |E| <= 1.0e-08
- 近零窗口: |E| <= 1.0e-03

## 6) 关键数值结论（每个 v）

### 6.1 Chiral error per v
- v=0.50: H0(max/mean)=(9.150e-01, 5.313e-01), H_chiral(max/mean)=(0.000e+00, 0.000e+00)
- v=0.60: H0(max/mean)=(1.029e+00, 5.220e-01), H_chiral(max/mean)=(0.000e+00, 0.000e+00)
- v=0.70: H0(max/mean)=(1.155e+00, 5.109e-01), H_chiral(max/mean)=(0.000e+00, 0.000e+00)
- v=0.80: H0(max/mean)=(1.279e+00, 4.975e-01), H_chiral(max/mean)=(0.000e+00, 0.000e+00)

### 6.2 OBC near-zero summary per v
- v=0.50: alpha=0 -> min|E|=2.228288e-06, n_zero=0, n_near=4; alpha=1 -> min|E|=1.647572e-02, n_zero=0, n_near=0
- v=0.60: alpha=0 -> min|E|=8.393944e-05, n_zero=0, n_near=2; alpha=1 -> min|E|=5.536271e-03, n_zero=0, n_near=0
- v=0.70: alpha=0 -> min|E|=1.017139e-03, n_zero=0, n_near=0; alpha=1 -> min|E|=7.008396e-03, n_zero=0, n_near=0
- v=0.80: alpha=0 -> min|E|=6.291007e-03, n_zero=0, n_near=0; alpha=1 -> min|E|=1.815533e-03, n_zero=0, n_near=0

## 7) 近零态局域性（alpha=0）
- v=0.50: <W_corner, W_edge, W_bulk> = (0.720, 0.277, 0.003) -> corner-dominant
- v=0.60: <W_corner, W_edge, W_bulk> = (0.630, 0.353, 0.016) -> corner-dominant
- v=0.70: <W_corner, W_edge, W_bulk> = (0.522, 0.429, 0.049) -> corner-dominant
- v=0.80: <W_corner, W_edge, W_bulk> = (0.415, 0.488, 0.097) -> edge-dominant

## 8) 对你提出的四个判断
1. alpha=0 是否有零能态：严格阈值下 无（n_zero_tol>0）。
   - 但在近零窗口内，v=[0.5, 0.6] 出现准零模（|E|<= 1.0e-03）。
2. 这些近零态是否角局域：
   - 角主导 v 点: [0.5, 0.6, 0.7]（依据 alpha=0 下最靠近零能的 4 个态平均权重）。
3. alpha 从 0 到 1 是否偏离零能：
   - 在 4 个 v 点中，min|E| 有 3 个上升、1 个下降（见 `min_abs_energy_vs_alpha.png`）。
4. 对零能钉扎的结论：
   - 整体上 H_break（同子格跃迁）使准零模脱离零能，支持“手征破缺破坏零能钉扎”；
   - 个别 v 点会因有限尺寸与态混合出现非单调偏移。

## 9) 输出文件索引
- `hopping_chiral_classification.csv`
- `chiral_error_map.csv`, `chiral_symmetry_error_scan.csv`
- `Q_block_expression.md`
- `obc_eigenvalues_all.csv`, `zero_mode_summary.csv`, `near_zero_state_metrics.csv`
- `spectra/*.png`, `wavefunctions/*.png`, `min_abs_energy_vs_alpha.png`, `near_zero_corner_weight_vs_alpha.png`

## Appendix: hopping classification rows
- (0,1) t -> breaks_chiral_same_sublattice
- (0,2) t -> allowed_AB
- (0,3) v + w*exp(-i kx) -> allowed_AB
- (1,2) v + w*exp(-i ky) -> allowed_AB
- (1,3) t -> allowed_AB
- (2,3) t -> breaks_chiral_same_sublattice
