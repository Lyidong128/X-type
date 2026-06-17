# HOTI_V_LT_0P6_REPORT

## Bulk gap summary
- v=0.3: global_bulk_gap=6.000000e-01, M_point_gap=6.000000e-01, band_min_k=(-3.141593,-3.141593), bulk_gapped=True
- v=0.4: global_bulk_gap=4.000000e-01, M_point_gap=4.000000e-01, band_min_k=(-3.141593,-3.141593), bulk_gapped=True
- v=0.5: global_bulk_gap=2.000000e-01, M_point_gap=2.000000e-01, band_min_k=(-3.141593,-3.141593), bulk_gapped=True

## Edge/ribbon summary
- v=0.3: edge_gap_x=4.973597e-03, edge_gap_y=4.973597e-03, gapless_edges=1
- v=0.4: edge_gap_x=4.581594e-03, edge_gap_y=4.581594e-03, gapless_edges=1
- v=0.5: edge_gap_x=2.351752e-03, edge_gap_y=2.351752e-03, gapless_edges=1

## OBC corner-like summary (from in-gap candidate states)
- v=0.3, L=20: corner_like_count=0, max_W_corner=0.262, max_W_edge=0.737, E@max_corner=9.594e-04
- v=0.3, L=30: corner_like_count=0, max_W_corner=0.191, max_W_edge=0.808, E@max_corner=-3.242e-04
- v=0.3, L=40: corner_like_count=0, max_W_corner=0.171, max_W_edge=0.828, E@max_corner=1.251e-03
- v=0.4, L=20: corner_like_count=0, max_W_corner=0.304, max_W_edge=0.693, E@max_corner=3.103e-03
- v=0.4, L=30: corner_like_count=0, max_W_corner=0.203, max_W_edge=0.781, E@max_corner=6.320e-04
- v=0.4, L=40: corner_like_count=0, max_W_corner=0.184, max_W_edge=0.815, E@max_corner=1.400e-03
- v=0.5, L=20: corner_like_count=0, max_W_corner=0.297, max_W_edge=0.697, E@max_corner=-8.486e-03
- v=0.5, L=30: corner_like_count=0, max_W_corner=0.197, max_W_edge=0.790, E@max_corner=3.585e-03
- v=0.5, L=40: corner_like_count=0, max_W_corner=0.191, max_W_edge=0.806, E@max_corner=-3.937e-05

## Nested Wilson summary
- v=0.3: wannier_gap_x=0.1153, wannier_gap_y=0.1130, q_xy=0.514998, reliability=unreliable
- v=0.4: wannier_gap_x=0.1165, wannier_gap_y=0.0129, q_xy=0.239865, reliability=unreliable
- v=0.5: wannier_gap_x=0.1425, wannier_gap_y=0.0931, q_xy=0.254798, reliability=unreliable

## Corner charge summary
- v=0.3, L=20: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition
- v=0.3, L=30: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition
- v=0.3, L=40: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition
- v=0.4, L=20: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition
- v=0.4, L=30: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition
- v=0.4, L=40: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition
- v=0.5, L=20: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition
- v=0.5, L=30: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition
- v=0.5, L=40: Q_corner_mean=nan, Q_corner_abs_mean=nan, status=skipped_no_edge_gapped_cornerlike_condition

## Answers to requested questions
1) v<0.6 区域是否 bulk gapped: 是.
2) v<0.6 区域是否 edge gapped: 否.
3) 是否存在穿过 gap 的一阶 helical edge states: 是 (基于 edge-localized crossing 判据).
4) OBC 中是否存在稳定 corner-localized in-gap states: v=0.3:否, v=0.4:否, v=0.5:否.
5) corner-like 态是否随 L=20,30,40 稳定: v=0.3:不稳定, v=0.4:不稳定, v=0.5:不稳定.
6) nested Wilson 的 q_xy 是否量子化: v=0.3:非0.5/不可靠, v=0.4:非0.5/不可靠, v=0.5:非0.5/不可靠.
7) corner charge 是否接近量子化值: v=0.3:否/未满足条件, v=0.4:否/未满足条件, v=0.5:否/未满足条件.
8) 是否可认为 v<0.6 存在二阶拓扑角态: 见下方逐 v 分类。
9) 若不能，类型更可能为何: 见下方理由。

## Final classification (A/B/C)
- v=0.3: C. 不确定. reason: ribbon edge-localized states cross near zero; more consistent with first-order edge-state phase than HOTI corner phase.
- v=0.4: C. 不确定. reason: ribbon edge-localized states cross near zero; more consistent with first-order edge-state phase than HOTI corner phase.
- v=0.5: C. 不确定. reason: ribbon edge-localized states cross near zero; more consistent with first-order edge-state phase than HOTI corner phase.
