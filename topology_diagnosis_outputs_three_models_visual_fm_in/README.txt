three-model visual outputs
v_list=0.100,0.200,0.300,0.400,0.500,0.600,0.700,0.800,0.900,1.000,1.100,1.200,1.300,1.400,1.500,1.600
params: t=0.3, w=1.0, lm=0.1
ferromagnetism: fm_out=0.0, fm_in=0.1
Each model directory contains:
  01_bulk_band/ (red=spin-up dominant, blue=spin-down dominant)
  02_ribbon/ (red=spin-up lines, blue=spin-down lines)
  03_obc_marked/ (edge states spin-colored by dominant spin)
  04_wf_sum/
  summary_*.csv and obc_marked_states_*.csv
