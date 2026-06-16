# CHIRAL_20X20_REPORT

## Summary table
- v=0.5, alpha=0.0: chiral_error=0.000e+00, sym_err=2.013e-15, min|E|=4.899e-18, counts(<1e-2/1e-4/1e-6/1e-8)=72/60/52/22
- v=0.5, alpha=1.0: chiral_error=5.108e-01, sym_err=5.968e-01, min|E|=5.653e-07, counts(<1e-2/1e-4/1e-6/1e-8)=4/4/4/0
- v=0.6, alpha=0.0: chiral_error=0.000e+00, sym_err=5.107e-15, min|E|=6.249e-18, counts(<1e-2/1e-4/1e-6/1e-8)=68/60/18/14
- v=0.6, alpha=1.0: chiral_error=4.915e-01, sym_err=5.972e-01, min|E|=1.058e-03, counts(<1e-2/1e-4/1e-6/1e-8)=6/0/0/0

## Questions
1) alpha=0 是否严格满足手征对称性：是。
2) alpha=0 全谱是否关于 E=0 对称：是。
3) 20x20 是否比 5x5 更接近零能：是（以 min|E| 比较）。
4) 近零态是否主要局域在四角：v=0.5 corner-like比例=1.00, v=0.6 corner-like比例=1.00。
5) v=0.5 与 v=0.6 哪个角态更清晰：v=0.6（按 <W_corner-W_edge>）。
6) alpha=1 原始模型中零能钉扎是否消失：是。
7) alpha=1 近零态形态：v=0.5: edge-like/mixed (Wc=0.231, We=0.562, Wb=0.207); v=0.6: edge-like/mixed (Wc=0.210, We=0.586, Wb=0.204)。
8) 是否支持目标结论：支持“手征对称极限存在准零能角态，而原始模型中零能角态不受保护”的总体结论；但若 W_edge 同时较大，应表述为 corner-like / edge-mixed。
