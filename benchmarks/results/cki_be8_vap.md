# CKI Be8: collapsed HFB versus particle-number VAP

Date: 2026-09-16. Same unmodified CKI interaction as the previous benchmark;
2 valence neutrons, 2 valence protons, 12 single-particle modes. Energies are
valence-space MeV without a core offset or added mass scaling.

| State | Energy | Raw fidelity to exact ground state | N,Z sector probability |
|---|---:|---:|---:|
| Ordinary optimized HFB (collapsed) | -25.689032344 | 0.201813505 | 0.9999999995 |
| Intrinsic vacuum underlying PN-VAP | -15.462034073 | 0.046947247 | 0.120051626 |
| Normalized N,Z-projected VAP state | -27.398536786 | 0.391058817 | 1 |
| Exact shell-model ground state | -30.295394614 | 1 | 1 |

PN-VAP gains 1.709504441 MeV relative to the collapsed HF/HFB solution.
It remains 2.896857828 MeV above exact diagonalization. The unprojected VAP
vacuum is not optimized for its own energy or fidelity. Its raw fidelity is
smaller because most of its weight is outside the target number sector:
0.046947247 = 0.120051626 x 0.391058817.

For a defined comparison, the intrinsic VAP representative is chosen by positive
species scaling so that average N=Z=2 (verified to 5e-16). The normalized
projected state and its energy are invariant under this scaling. Intrinsic
||kappa||=1.164303394 and np block norm=0.459112028; canonical error=8.29e-15.
The projected state itself has <cc>=0 by exact number conservation.

## Grid and optimization checks

- 7 x 7 full-period gauge grid: -27.39853678570439 MeV.
- 9 x 9 grid, different offset: -27.39853678570438 MeV.
- Exact-sector expectation: -27.39853678570438 MeV.
- Three analytic-gradient reference optimizations converged to energies agreeing
  within 9e-12 MeV, with fixed-scale gradient norms around 1e-6.
- Gauge-only finite-difference refinement took one iteration from the reference
  result; reported success and gradient norm 1.31e-5. This warm start is explicit:
  we have not demonstrated independent gauge-only random-start convergence for
  Be8, or a global minimum.
- Independent full-Fock quasiparticle-vacuum reconstruction agrees with the
  normalized projected vector and sector weight to better than 1e-9.
- Fifteen regression tests pass. Benchmark run took about 21 seconds.

The gauge evaluator has polynomial cost per evaluation and does not enumerate
the 225-dimensional sector. The reference initialization and exact fidelities
do enumerate it; that part is only a small-system benchmark and is not claimed
to scale polynomially. M=0 conditioning is deliberately not applied to the
intrinsic fidelities in the table; all overlaps use normalized full states.

Raw data: `cki_be8_vap.json`; intrinsic U,V,Z and projected amplitudes:
`cki_be8_vap_state.npz`. Reproduce with `benchmarks/cki_be8_vap.py`.
