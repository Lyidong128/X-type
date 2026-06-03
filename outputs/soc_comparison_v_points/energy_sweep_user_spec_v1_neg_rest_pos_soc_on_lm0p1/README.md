# OBC energy-sweep wavefunction package

- package_root: `/workspace/outputs/soc_comparison_v_points`
- soc_state: `soc_on`
- lm: `0.1`
- special_v: `1.0`
- special_energy_range: `[-0.5, 0.0] step 0.1`
- default_energy_range: `[0.0, 0.5] step 0.1`
- lattice: `20 x 20`

Per point outputs:
- `obc_wavefunction_target_E*.png`
- `obc_spectrum_e_vs_index_marked_E*.png`
- `energy_sweep_selection_summary.csv`

Global summary:
- `energy_sweep_selection_summary_all_points.csv`
