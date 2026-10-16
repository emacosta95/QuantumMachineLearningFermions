# HF/HFB energy and fidelity audit

> Historical results below use the original, unscaled repository interaction.
> On 2026-10-16, `data/usdb.nat` was aligned with the collaborator's supplied
> interaction, and optional mass scaling was added. See
> `docs/usdb_benchmark_conventions.md` and `results/taurus_import_audit/summary.json`
> for the subsequent direct TAURUS-state comparison. The old tables have not
> been recomputed and should not be treated as the matched TAURUS benchmark.

Interaction conventions: CKI with an inert He4 core for Be; USDB with an inert O16 core for Ne. Energies are in MeV. Fidelity means the raw squared overlap of two normalized states, without renormalizing the trial state in the target sector.

## Main Ne24 result

An additional direct energy check expands all 4096 canonical-pair configurations and evaluates the Hamiltonian matrix elements in that basis, independently of the library's density/Wick energy formula. The wavefunction norm is 0.9999999999999998. Its energy is -69.88668277120581 MeV, versus the variational output -69.88668277120640 MeV: a difference of 5.83e-13 MeV. Run `python verify_ne24_wavefunction_energy.py`; numerical output is in `results/ne24_wavefunction_energy_audit.json`.

The explicit Bloch-Messiah canonical-pair expansion gives fidelity 0.0803314263295867; the library gives 0.0803314263295908. The largest determinant-amplitude discrepancy after aligning the global phase is 2.749e-14. The canonical orbital rotation back to the shell-model basis is included.

| Ne24 state | Intrinsic energy | Fidelity | Infidelity |
|---|---:|---:|---:|
| HF, 8 starts | -69.6142044022 | 0.1099641005 | 0.8900358995 |
| HFB, 8 starts, tighter tolerance | -69.8866827712 | 0.0803314263 | 0.9196685737 |
| HFB, original 3-start settings | -69.8866839502 | 0.0803314389 | 0.9196685611 |
| HFB, deterministic first start | -69.7580661536 | 0.1179697187 | 0.8820302813 |
| Collaborator PDF (digitized vector curve) | Not supplied | ~0.1283030321 | ~0.8716969679 |

The original Ne24 point (-ln F = 2.52159422) is reproduced. It is genuinely paired: ||kappa|| = 0.97458. The tighter HFB run converged with gradient norm 6.78e-6. Pure HF is a different state and does not reproduce the collaborator point.

For the audited Ne24 HFB state, the exact (N,Z) sector weight is 0.5828450337. Normalizing after selecting that sector gives fidelity 0.1378263890. Normalizing after additionally selecting M=0 gives about 0.47764. Neither equals the collaborator point. His exact state, energy, and normalization/basis conventions are needed to locate the remaining discrepancy.

## Neon chain

All HF calculations converged. HFB uses 8 starts, maxiter=2000, real Bogoliubov coordinates, gradient_tolerance=1e-5, solver tolerance=1e-9. False convergence flags remain visible; no global optimum is certified. The PDF contains no energies or states.

| Isotope | HF infidelity | HFB infidelity | HFB converged | PDF infidelity |
|---|---:|---:|---|---:|
| Ne18 | 0.767434934 | 0.582392274 | True | ~0.582392469 |
| Ne20 | 0.897914608 | 0.897914608 | False | ~0.897387391 |
| Ne22 | 0.909451912 | 0.909451913 | False | ~0.908132298 |
| Ne24 | 0.890035899 | 0.919668574 | True | ~0.871696968 |
| Ne26 | 0.863983390 | 0.855617961 | False | ~0.808655656 |
| Ne28 | 0.847434870 | 0.848750176 | False | ~0.840131853 |
| Ne30 | 0.702523656 | 0.565796270 | True | ~0.564467811 |

## Be HF benchmark

| Isotope | HF | HF after J=0 projection | Exact |
|---|---:|---:|---:|
| Be6 | -0.635698912 | -2.956945148 | -3.048797984 |
| Be8 | -25.689032358 | -30.193003727 | -30.295394614 |
| Be10 | -34.163589933 | -38.656543073 | -39.453415798 |
| Be12 | -39.015345677 | -40.504859811 | -41.221454571 |

The supplied intrinsic values for Be6, Be8, and Be10 match the HF branch closely. Be10 projected HF is -38.656543073, differing from the quoted -38.65735 by +0.000806927 MeV (0.807 keV). The Be12 value -40.501673 instead matches a paired HFB state, not the HF minimum.

## Be HFB: distinguish angular from number projection

| Isotope | HFB | J=0 only | N,Z,J=0 | Strict convergence |
|---|---:|---:|---:|---|
| Be6 | -1.908835803 | -1.908835803 | -2.868290574 | True |
| Be8 | -25.689032420 | -30.193003725 | -30.193003725 | True |
| Be10 | -34.229245200 | -37.896444924 | -38.799453927 | False |
| Be12 | -40.501653444 | -40.501653444 | -40.945310089 | False |

Be10 and Be12 HFB failed the strict 1e-6 gradient criterion (residuals 1.62e-5 and 8.92e-5), although their state energies and amplitudes are independently validated. Their values are best found results, not certified converged minima. Be12 HFB is already J=0, so angular projection alone leaves its energy unchanged. Number projection does not.

## Independent correctness checks

- All Be Euler results agree with enlarged grids and an independent J2 eigenspace projection to numerical precision.
- An independent full-even-Fock-space reconstruction solves sum(beta_dagger beta)|Phi>=0 for every Be state. Energies agree with the density-functional evaluation to about 1.4e-12 MeV and fixed-sector amplitudes to about 3e-15.
- Direct Slater determinant overlaps reproduce all Ne HF fidelities.
- Three canonical-pair-product tests pass: a generic complex paired state, a partially paired state with singular U, and an exact HF state. They also check densities, quasiparticle annihilation, and energies.
- The actual Ne24 canonical-pair expansion agrees with the library at the amplitude level and reproduces its fidelity to 4e-15.

## Confirmed projection limitation

BogoliubovVacuumSeries.occupation_amplitudes currently fails for some genuinely paired states with singular or unstable particle-vacuum Thouless charts (observed for Be10 and Be12 HFB). The single-vacuum stable amplitude API is correct in these cases. The audit uses the same Euler quadrature acting in the many-body basis as a fallback, preserving phases. Converting these genuinely paired states to HF would be incorrect. No library physics implementation was changed in this audit.

## Reproduction

Run from the repository root with NumPy, SciPy and PFAPACK installed:

```sh
python verify_be_energy_table.py
python verify_be_full_fock.py
python verify_ne_hf_fidelity.py
python verify_ne_hfb_fidelity.py
python verify_ne24_bloch_messiah.py
python -m unittest discover -s tests -p test_bloch_messiah_amplitudes.py -v
```

To reproduce the original Ne24/Ne26 settings:

```sh
python verify_ne_hfb_fidelity.py --masses 24 26 --starts 3 --maxiter 500 --gradient-tolerance 0.001 --number-tolerance 1e-8 --output results/ne_original_settings_rerun.json
```

Raw numerical records and reusable states are in results/*audit.json and results/*audit.pkl. The collaborator curve was extracted from C:/Users/ecosta/Downloads/hfb_vs_faf_inf.pdf.
