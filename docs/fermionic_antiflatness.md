# Fermionic anti-flatness

`fermionic_antiflatness.py` implements the pure-state covariance measure

```text
F_k = L - Tr[(M^T M)^k] / 2,
M_mn = -i <[gamma_m, gamma_n]> / 2,
```

for `L` complex fermionic modes.  The implementation follows the convention in
P. Sierant, P. Stornati, and X. Turkeshi, *Fermionic Magic Resources of Quantum
Many-Body Systems*, arXiv:2506.00116.

For a pure Gaussian state, `M.T @ M` is the identity and `F_k=0`.  A coherent
sum of Gaussian vacua need not be Gaussian, so its anti-flatness is evaluated
only after all series amplitudes have been summed.

## Amplitude API

```python
from fermionic_antiflatness import fermionic_antiflatness

result = fermionic_antiflatness(
    exact_state,
    fermionic_hamiltonian.occupations,
    fermionic_hamiltonian.modes,
    order=2,
)
print(result.value, result.value_per_mode)
```

The occupation list may be sparse, but it must contain the complete support of
the state.  A fixed-N,Z exact eigenstate therefore uses its existing shell-model
basis directly.

## Bogoliubov-vacuum series API

```python
from fermionic_antiflatness import vacuum_series_antiflatness

result = vacuum_series_antiflatness(
    projected_series,
    fermionic_hamiltonian.occupations,
    order=2,
)
```

This expands and coherently sums the supplied series in `occupations`, normalizes
the resulting pure state, and then constructs its Majorana covariance matrix.
When the series is a number-projected state, `occupations` must span the complete
projected sector.  Supplying only part of an unprojected vacuum's support would
instead measure the normalized truncation and is not the anti-flatness of the
original vacuum.

The reproducible CKI example is:

```powershell
python benchmarks/cki_be10_antiflatness.py --order 2
```

It compares the Be10 exact ground state with the committed exact-grid VAP
Bogoliubov-vacuum series.  The full projector contains
`7*7*9*5*9 = 19,845` vacua.  Because the final determinant basis has fixed N,Z,
the benchmark analytically sums its 49 identical gauge-phase copies and performs
the amplitude calculation using 405 Euler-rotated vacua without changing the
projected state.

## Projection-after-variation and component convergence

`complete_fock_occupations` and `vacuum_series_prefix_antiflatness` support a
component-by-component analysis without silently discarding occupation sectors:

```python
from fermionic_antiflatness import (
    complete_fock_occupations,
    vacuum_series_prefix_antiflatness,
)

occupations = complete_fock_occupations(modes, parity="even")
trajectory = vacuum_series_prefix_antiflatness(
    series,
    occupations,
    component_counts=(1, 2, 4, 8, series.number_of_vacua),
)
```

A fixed-`(N,Z)` basis cannot diagnose convergence in the number-gauge component
count: after restriction to that sector, every gauge-rotated term is
proportional. The full even-parity basis is required for that question. Euler
rotations are different: after analytically collapsing the exact number-gauge
sum, their fixed-sector components remain distinct and their convergence can be
studied directly.

The isotope benchmark

```powershell
python benchmarks/cki_be_pav_faf_components.py
```

computes exact-state and `P_N P_Z P_J=0` PAV fidelity/FAF for Be6, Be8, Be10,
and Be12. It also records deterministic Euler-prefix trajectories. Those
prefixes are quadrature convergence diagnostics rather than optimized
multi-reference ansatzes, so neither fidelity nor FAF is required to change
monotonically with the number of retained components.

The committed production run gives the following full-grid results:

| Nucleus | PAV fidelity | PAV FAF | Exact-state FAF |
|---|---:|---:|---:|
| Be6  | 0.992014 | 5.821824 | 5.921629 |
| Be8  | 0.995483 | 11.706025 | 11.787413 |
| Be10 | 0.945998 | 9.043253 | 10.739276 |
| Be12 | 0.955402 | 4.331327 | 5.369539 |

For Be8, for example, the deterministic prefix fidelity rises from `0.201814`
for one rotated Slater determinant to `0.995483` for all 324 Euler components,
while FAF rises from numerical zero to `11.706025`. Small non-monotonic steps
in either quantity are expected from the non-variational prefix ordering.

## Extension to Ne isotopes

The same physics workflow is feasible with the USDB interaction in `data/usdb.nat`,
but the `sd` shell has 12 modes per species (24 total), rather than six per
species for CKI. With an O16 core and two valence protons, the fixed-`(N,Z)`
dimensions are 4,356 (Ne20), 32,670 (Ne22), 60,984 (Ne24), and 32,670 (Ne26).
Exact sparse diagonalization and fixed-sector FAF remain practical, but the
current dense/full-series route should not be used unchanged: the full even
Fock basis has `2^23 = 8,388,608` configurations and materializing the product
of exact number and Euler grids can require several gigabytes.

A scalable Ne implementation should collapse the number-gauge sum before
materializing transformed vacua, keep Hamiltonians and angular-momentum
operators sparse, and use either batched Euler amplitudes or the existing
Metropolis Euler sampler. This preserves exact `(N,Z)` while avoiding the full
24-mode Fock basis. Ne20 is the natural first validation point before extending
to Ne22--Ne26.
