Physical meaningful special points package
=======================================
selection rules:
1) robust_topological: gap > 0.1 and Z2=1
2) near_transition: globally smallest |gap| points
total selected points: 34
robust_topological count: 14
near_transition count: 20
copied image files: 136
missing image files count: 0
each point folder includes:
- band.png
- ribbon.png
- obc_spectrum.png
- obc_wavefunction.png
- special_point_info.txt

newly added (auto special-state analysis):
- obc_wavefunction_subspace.png: weighted overlay of automatically selected special OBC modes
- special_modes.txt: selected mode indices/energies/localization metrics/weights

subspace energy-window setting (updated):
- window_rule=max(0.20, 8.0*min|E|)
- max picked modes per point: 8

observation guidance files:
- best_observation_positions.csv: global table of best viewing positions for all selected points
- best_observation_positions_README.txt: interpretation guide
- each point folder adds best_observation_position.txt
