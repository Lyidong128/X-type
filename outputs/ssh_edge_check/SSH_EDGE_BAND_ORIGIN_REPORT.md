# SSH_EDGE_BAND_ORIGIN_REPORT

## Setup
- Model: H8(k)=H0(k)⊗s0 + HSOC⊗sz, with w=1.0, t=0.3, lm=0.1.
- Focus: v=0.5, x-open / y-periodic ribbon, Nx=40.
- v scan: 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2.

## Key numeric observations
- v=0.5: avg W_edge (spin-up)=0.996, avg W_edge (spin-down)=0.996.
- v<w band-existence ratio=1.000, v>=w ratio=1.000.
- Termination A/B: exists=1/1, merged=0/0, ribbon_gap=2.3518e-03/8.0528e-04.
- lambda_R scan: lambda=0 max_edge=1.000, lambda=0.10 max_edge=1.000.
- C_spin(v=0.5)=0, C_spin(v=0.8)=1.

## Answers
1) v=0.5 中间孤立条带是否由 spin-up/down 两套边界带组成: 是.
2) 波函数是否局域在 ribbon 两侧边界: 是 (W_edge_up=0.996, W_edge_dn=0.996).
3) 该带是否主要存在于 v<w 区域: 证据不足/否 (ratio<v<w=1.000, ratio>=w=1.000).
4) 2D Zak phase 是否支持单自旋/单block SSH-like 极化: noSOC(Px,Py)=(0.500,0.500), up=(0.500,0.500), dn=(0.500,0.500).
5) full spinful 总 Zak phase 是否平庸: 是 (Px,Py)=(0.000,0.000).
6) termination 改变后中间带是否改变: 变化较小.
7) 加入自旋混合扰动后该带是否不稳定: 有退化但未完全消失.
8) 该带应解释为何种边界态: 更接近 spin-resolved SSH-like edge-localized band，不应直接称为 full-spinful 强保护 QSH helical edge state 或高阶角态。
9) 如何区分 v=0.5 SSH-like 与 v>0.6 QSH-like: 看 (i) 对 termination 敏感性、(ii) 对 spin-mixing (Rashba-like) 敏感性、(iii) full spinful 总极化/拓扑指标是否非平庸、(iv) spin Chern 是否进入非平庸区。

## Final classification language
- spin-resolved SSH-like edge band: yes (for v=0.5 focus, with strong edge localization and termination/spin-mixing sensitivity checks).
- full spinful total Zak phase: use 2d_zak_phase_summary.csv values as the primary criterion.
- QSH/spin-Chern helical edge state: compare against v=0.8 reference and spin-Chern marker.
- higher-order corner state: not supported by ribbon-edge evidence alone.
