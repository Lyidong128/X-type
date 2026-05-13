# First-stage per-point folders

Each point has its own folder:

- `v_*_t_*_lm_*`

Current baseline artifacts per point:

- `band.png`
- `ribbon.png`
- `obc_spectrum_e_vs_index.png`

Recommended way to extend later:

- Keep adding new figures into the same point folder.
- Use stable, descriptive names, for example:
  - `obc_wavefunction_filtered.png`
  - `obc_wavefunction_window_sum.png`
  - `chern_local_map.png`
  - `wilson_loop_local.png`
- Avoid renaming baseline files above, so batch scripts and summaries stay compatible.
