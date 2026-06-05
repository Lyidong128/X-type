# KM SOC Executive Summary

## Parameter setup
- Fixed: `t=0.3`, `w=1.0`
- Scanned: `v in [0.00, 1.10]`, `lm = 0.0, 0.1`

## Model / basis identification
- Hamiltonian: `models/xtype_model.py::Hxtype(k)`
- Matrix size: 8 (interpreted as 4 orbital/site components x 2 spins)
- Spin up/down is explicitly identifiable; `Sz` can be constructed.
- SOC term is spin-diagonal and opposite in up/down channels, with dominant imaginary structure, consistent with Kane-Mele-like SOC symmetry form.

## Bulk topology summary
- In reliable gapped points, `C_total` stays near zero.
- Fu-Kane-backed reliable `Z2` remains `0` in this scan window.
- Main transitions detected are gap-closing / gapless corridors, not robust gapped `Z2` switch points.

## Spin-resolved topology summary
- Spin-Chern analysis succeeds.
- Around representative / transition-adjacent points:
  - `C_up ≈ -C_down`
  - `C_total ≈ 0`
  - `C_spin` becomes nonzero (including ±1 in several points)
- This supports spin-resolved Kane-Mele-type topology rather than a Chern-insulator regime.

## Gap-closing summary
- Main reliable gapless corridors:
  - `lm=0.0`: near `v ~ 0.698 -> 1.1`
  - `lm=0.1`: near `v ~ 0.598 -> 0.602`
- Representative gap-min momentum is around `(kx, ky) ≈ (pi, pi)` for key corridors.

## Wilson loop summary
- Full Wannier center flows were generated for points A/B/C/D/E and auto-selected F/G.
- Wilson parity at some points is direction-dependent (`kx-scan-ky` vs `ky-scan-kx`) and is marked with reliability flags.
- For near-gapless points, Wilson-based `Z2` is marked uncertain and should not be over-interpreted.

## Chiral-symmetry breaking summary
- Candidate chiral-breaking metric increases from `lm=0.0` to `lm=0.1` on average.
- This supports the interpretation that Kane-Mele SOC can weaken chiral pinning of zero-energy corner modes.
- This does not negate TRS-protected spin topology analysis.

## Recommended first-look files
- `KM_SOC_TOPOLOGY_REPORT.md`
- `gap_z2_scan/gap_z2_scan.csv`
- `gap_closing/transition_candidates.csv`
- `spin_chern/spin_chern_summary.csv`
- `wilson_loop/wilson_loop_summary.csv`
- `chiral_symmetry/chiral_breaking_summary.csv`
- `fu_kane/fu_kane_summary.csv`

