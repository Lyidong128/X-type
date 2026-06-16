# HOTI_INVARIANT_REPORT

## Setup
- model_types=['H_chiral', 'H_original']
- v_list=[0.5, 0.6, 0.7, 0.8]
- parameters: t=0.300, w=1.000, lm=0.0
- Wilson grid nk=81; ribbon L=40, nk=101; corner L_list=[10, 20, 30]

## 1) Ordinary Wilson loop / Wannier spectrum
- `wannier_spectrum_x.png` and `wannier_spectrum_y.png` show nu_x(ky), nu_y(kx) for both models and all v.
- `nested_wilson_summary.csv` gives q_xy and Wannier-gap based reliability.

## 2) Nested Wilson / quadrupole proxy
### H_chiral
- v=0.50: q_xy=0.236452, wannier_gap=0.3278, reliability=high, verdict=not-near-1/2
- v=0.60: q_xy=0.056749, wannier_gap=0.3679, reliability=high, verdict=not-near-1/2
- v=0.70: q_xy=0.165632, wannier_gap=0.3546, reliability=medium_sector_tracking, verdict=not-near-1/2
- v=0.80: q_xy=0.689328, wannier_gap=0.1542, reliability=medium_sector_tracking, verdict=not-near-1/2
### H_original
- v=0.50: q_xy=0.429850, wannier_gap=0.3668, reliability=high, verdict=near-1/2
- v=0.60: q_xy=0.446275, wannier_gap=0.3658, reliability=high, verdict=near-1/2
- v=0.70: q_xy=0.546779, wannier_gap=0.3297, reliability=medium_sector_tracking, verdict=near-1/2
- v=0.80: q_xy=0.304768, wannier_gap=0.1958, reliability=high, verdict=not-near-1/2

## 3) Edge polarization (ribbon proxy)
- `edge_polarization_summary.csv` reports P_x_edge / P_y_edge from edge excess-charge asymmetry proxies.
### H_chiral
- v=0.50: P_x_edge=0.001241, P_y_edge=-0.000837
- v=0.60: P_x_edge=-0.000001, P_y_edge=0.000001
- v=0.70: P_x_edge=0.000000, P_y_edge=0.000000
- v=0.80: P_x_edge=-0.000000, P_y_edge=0.000000
### H_original
- v=0.50: P_x_edge=-0.000756, P_y_edge=-0.096453
- v=0.60: P_x_edge=0.000022, P_y_edge=-0.000011
- v=0.70: P_x_edge=0.000000, P_y_edge=-0.000000
- v=0.80: P_x_edge=-0.000000, P_y_edge=-0.000000

## 4) Corner charge scaling (OBC)
- `corner_charge_summary.csv` gives corner charge in corner patches for L=10,20,30.
### H_chiral
- v=0.50, L=10: Q_corner_mean=-0.000000, Q_corner_abs_mean=0.000093
- v=0.50, L=20: Q_corner_mean=-0.058176, Q_corner_abs_mean=0.434605
- v=0.50, L=30: Q_corner_mean=0.147288, Q_corner_abs_mean=0.147288
- v=0.60, L=10: Q_corner_mean=-0.000000, Q_corner_abs_mean=0.000000
- v=0.60, L=20: Q_corner_mean=0.021458, Q_corner_abs_mean=0.125007
- v=0.60, L=30: Q_corner_mean=0.043325, Q_corner_abs_mean=0.175204
- v=0.70, L=10: Q_corner_mean=-0.000000, Q_corner_abs_mean=0.000000
- v=0.70, L=20: Q_corner_mean=-0.000000, Q_corner_abs_mean=0.025661
- v=0.70, L=30: Q_corner_mean=-0.094553, Q_corner_abs_mean=0.101760
- v=0.80, L=10: Q_corner_mean=0.000000, Q_corner_abs_mean=0.000000
- v=0.80, L=20: Q_corner_mean=0.000000, Q_corner_abs_mean=0.000000
- v=0.80, L=30: Q_corner_mean=0.025147, Q_corner_abs_mean=0.188410
### H_original
- v=0.50, L=10: Q_corner_mean=0.222728, Q_corner_abs_mean=0.222728
- v=0.50, L=20: Q_corner_mean=0.422467, Q_corner_abs_mean=0.422467
- v=0.50, L=30: Q_corner_mean=0.543059, Q_corner_abs_mean=0.543059
- v=0.60, L=10: Q_corner_mean=0.266017, Q_corner_abs_mean=0.266017
- v=0.60, L=20: Q_corner_mean=0.511365, Q_corner_abs_mean=0.511365
- v=0.60, L=30: Q_corner_mean=0.660520, Q_corner_abs_mean=0.660520
- v=0.70, L=10: Q_corner_mean=0.293673, Q_corner_abs_mean=0.293673
- v=0.70, L=20: Q_corner_mean=0.490155, Q_corner_abs_mean=0.490155
- v=0.70, L=30: Q_corner_mean=0.668958, Q_corner_abs_mean=0.668958
- v=0.80, L=10: Q_corner_mean=0.203413, Q_corner_abs_mean=0.203413
- v=0.80, L=20: Q_corner_mean=0.476821, Q_corner_abs_mean=0.476821
- v=0.80, L=30: Q_corner_mean=0.614802, Q_corner_abs_mean=0.614802

## 5) Compare with existing near-zero corner weights
- Existing reference from `outputs/chiral_symmetric_model/near_zero_state_metrics.csv` (rank_by_absE<4).
### H_chiral
- v=0.50: <W_corner,W_edge,W_bulk>=(0.720,0.277,0.003)
- v=0.60: <W_corner,W_edge,W_bulk>=(0.630,0.353,0.016)
- v=0.70: <W_corner,W_edge,W_bulk>=(0.522,0.429,0.049)
- v=0.80: <W_corner,W_edge,W_bulk>=(0.415,0.488,0.097)
### H_original
- v=0.50: <W_corner,W_edge,W_bulk>=(0.239,0.595,0.167)
- v=0.60: <W_corner,W_edge,W_bulk>=(0.194,0.597,0.209)
- v=0.70: <W_corner,W_edge,W_bulk>=(0.233,0.547,0.220)
- v=0.80: <W_corner,W_edge,W_bulk>=(0.090,0.544,0.367)

## Final judgments
1. H_chiral 是否量子化 q_xy=1/2：0/4 个 v 点接近 1/2（按 |q_xy-0.5|<0.1 判据）。
2. H_original 是否保留二阶拓扑特征：3/4 个 v 点接近 1/2；需结合 reliability 与 corner charge 尺度行为判断稳健性。
3. 原始模型近零态是否 edge-corner mixed：在 v=[0.5, 0.6, 0.7, 0.8] 上 W_edge>=W_corner，显示明显混合。
4. 是否可称为高阶拓扑角态：现有证据更偏向“近零角局域/边角混合态”，不足以强称高阶拓扑角态。
