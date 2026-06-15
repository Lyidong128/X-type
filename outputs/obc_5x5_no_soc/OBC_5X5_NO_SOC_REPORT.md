# OBC_5X5_NO_SOC_REPORT

## 1) Hamiltonian dimension
- For Lx=5, Ly=5, 4 orbitals per cell (spinless no-SOC block), matrix size is N=100.

## 2) Half-filling gaps
- From `half_filling_gap_summary.csv`:
  - v=0.5: E_valence=-0.086460, E_conduction=-0.066861, gap=1.959879e-02
  - v=0.6: E_valence=-0.079269, E_conduction=-0.059912, gap=1.935685e-02
  - v=0.7: E_valence=-0.107604, E_conduction=-0.049352, gap=5.825156e-02
  - v=0.8: E_valence=-0.162731, E_conduction=-0.064172, gap=9.855914e-02

## 3) Gap trend near v=0.7
- Minimum half-filling gap among [0.5,0.6,0.7,0.8] occurs at v=0.6, gap=1.935685e-02.
- In this 5x5 finite-size OBC sample, the gap does not close exactly at v=0.7; instead the minimum appears near v=0.6.
- This is consistent with strong finite-size/boundary hybridization shifting the apparent minimum away from the bulk mass-zero point.

## 4) Localization character near half-filling
- Based on states 46..53 and (W_corner, W_edge, W_bulk):
  - v=0.5: <W_corner>=0.137, <W_edge>=0.711, <W_bulk>=0.153 -> edge-like tendency
  - v=0.6: <W_corner>=0.136, <W_edge>=0.648, <W_bulk>=0.216 -> edge-like tendency
  - v=0.7: <W_corner>=0.131, <W_edge>=0.586, <W_bulk>=0.282 -> edge-like tendency
  - v=0.8: <W_corner>=0.140, <W_edge>=0.540, <W_bulk>=0.320 -> edge-like tendency

## 5) Can 5x5 alone prove corner state?
- No. 5x5 is too small; corner/edge/bulk regions strongly mix in finite-size spectra.
- At most one can report corner-like tendency, not robust corner-state proof.

## 6) Recommended next step
- Use larger systems (e.g., 20x20, 30x30) and perform size scaling of W_corner/W_edge/IPR before final claims.

## Caution
- This is strictly lm=0 no-SOC analysis for finite-size OBC observation.
- Do not directly equate these 5x5 features with final topological evidence.
