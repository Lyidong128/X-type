# 物理意义重分析总结

## 1) 判据（物理可解释）
- 稳健边界相候选：`Z2=1` 且 `ribbon(4/5)连接分支=1` 且 `band_window_width >= 0.2`。
- 相变核心/前驱：`band_window_width <= 0.08` 或 `OBC min|E| <= 0.002` 或 `special_rank <= 10`。
- 平庸绝缘体：`Z2=0` 且 ribbon 无连接/孤立态，且近零能态不显著。
- 其余归入 bulk-mixed / ambiguous，提示需要更大尺寸与波函数复核。

## 2) 总体统计
- 总点数：126
- ambiguous_need_review: 48
- bulk_mixed_nontrivial: 18
- transition_core_or_precursor: 28
- robust_edge_topology: 15
- trivial_gapped: 17
- 具有明显边界/临界特征的点数（连接分支+近零态聚集）：12

## 3) 关键点（建议优先看）
- strongest_robust_edge_candidate: v_0p80_t_0p50_lm_0p40 (v=0.80, lm=0.40, cat=robust_edge_topology, score=0.907)
- strongest_transition_core: v_0p90_t_0p50_lm_0p20 (v=0.90, lm=0.20, cat=transition_core_or_precursor, score=0.531)
- strongest_bulk_mixed_nontrivial: v_0p20_t_0p50_lm_0p50 (v=0.20, lm=0.50, cat=bulk_mixed_nontrivial, score=0.551)
- representative_trivial_gapped: v_1p90_t_0p50_lm_0p00 (v=1.90, lm=0.00, cat=trivial_gapped, score=0.120)

## 4) 局域走廊的相变候选点
- 命中数：52
- rank_02_v_1p10_t_0p50_lm_0p20: (v=0.900, lm=0.000), gap=-3.608e-16, obc_min=2.392e-04
- rank_01_v_0p90_t_0p50_lm_0p20: (v=0.900, lm=0.000), gap=-3.608e-16, obc_min=2.392e-04
- rank_02_v_1p10_t_0p50_lm_0p20: (v=1.020, lm=0.000), gap=-5.204e-16, obc_min=3.042e-03
- rank_01_v_0p90_t_0p50_lm_0p20: (v=1.020, lm=0.000), gap=-5.204e-16, obc_min=3.042e-03
- rank_01_v_0p90_t_0p50_lm_0p20: (v=1.100, lm=0.000), gap=-5.274e-16, obc_min=5.997e-03
- rank_02_v_1p10_t_0p50_lm_0p20: (v=1.100, lm=0.000), gap=-5.274e-16, obc_min=5.997e-03
- rank_02_v_1p10_t_0p50_lm_0p20: (v=0.940, lm=0.000), gap=-6.523e-16, obc_min=1.681e-02
- rank_01_v_0p90_t_0p50_lm_0p20: (v=0.940, lm=0.000), gap=-6.523e-16, obc_min=1.681e-02
- rank_02_v_1p10_t_0p50_lm_0p20: (v=1.180, lm=0.000), gap=-6.661e-16, obc_min=1.655e-03
- rank_01_v_0p90_t_0p50_lm_0p20: (v=0.780, lm=0.000), gap=-6.661e-16, obc_min=5.681e-03

## 5) 结论
- 这版结果不再只看“是否特殊”，而是强制结合 bulk 指标 + ribbon 边界连通 + OBC 近零模式三重证据。
- 可以直接用于区分：稳健边界相、相变前驱、平庸相、以及 bulk-mixed 的可疑点。
