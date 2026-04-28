# PRL-style topological classification logic

## Hierarchical logic
1) Bulk robustness axis: gap (robust/critical/precursor).
2) Topological invariant axis: Z2, Chern, Wilson consistency.
3) Boundary response axis: edge_state / corner_state.
4) Local continuity axis: Z2 flips and Chern jumps against nearest neighbors.

## Thresholds
- robust_gap = 0.1
- critical_gap = 0.05
- chern_quantized = 0.5
- chern_jump = 0.5

## Selection policy (publication-oriented)
- transition cores: top 40 from C1/C2 by special_score
- robust anchors: top 24 from R1/R2/R3 by gap+invariant strength
- anomalous boundary states: top 16 from C3

selected_total = 65
missing_artifacts = 0
selected_phase_counts = {'C2_critical_corner_reconstruction': 6, 'C1_critical_inversion_core': 34, 'C3_critical_boundary_metal_like': 1, 'R1_robust_z2': 24}
transition_corridor_count = 121

## Key transition corridors (top 10 by z2_flips and min_abs_gap)
- (v=0.80, t=1.00): z2_flips=9, min_abs_gap=4.1633e-16, max_chern_jump=0.0000, edge=1, corner=0
- (v=0.20, t=0.30): z2_flips=9, min_abs_gap=1.2212e-15, max_chern_jump=1.0000, edge=1, corner=1
- (v=0.50, t=0.10): z2_flips=8, min_abs_gap=2.2204e-16, max_chern_jump=0.0000, edge=1, corner=0
- (v=0.50, t=0.30): z2_flips=8, min_abs_gap=2.9143e-16, max_chern_jump=0.0000, edge=1, corner=0
- (v=0.90, t=0.50): z2_flips=8, min_abs_gap=5.2736e-16, max_chern_jump=0.0000, edge=1, corner=0
- (v=0.10, t=0.60): z2_flips=8, min_abs_gap=5.4123e-16, max_chern_jump=0.0000, edge=1, corner=0
- (v=1.00, t=0.10): z2_flips=8, min_abs_gap=5.5430e-16, max_chern_jump=0.0000, edge=1, corner=0
- (v=0.10, t=0.00): z2_flips=8, min_abs_gap=6.2450e-16, max_chern_jump=0.0000, edge=1, corner=0
- (v=0.90, t=0.30): z2_flips=8, min_abs_gap=8.7430e-16, max_chern_jump=0.0000, edge=1, corner=0
- (v=0.80, t=0.60): z2_flips=8, min_abs_gap=9.1593e-16, max_chern_jump=0.0000, edge=1, corner=0