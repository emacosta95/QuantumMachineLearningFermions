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
