# Closest pure fermionic Gaussian state

`gaussian_fidelity.py` optimizes

F_G = max_Omega |<Psi_target|Omega>|^2

for a normalized fixed-sector target. The objective is fidelity only: it does
not minimize energy, constrain average particle numbers, or project the trial
state. Complex nn, pp and np pairing and normal mixing are allowed.

For an even target, odd-parity Gaussian states have zero overlap and need not be
searched. The implementation handles two parts of the even Gaussian manifold:

- `maximize_gaussian_fidelity`: unrestricted Thouless vacua with nonzero vacuum
  overlap. It uses Pfaffian sector amplitudes, exact full-Gaussian normalization,
  finite-difference optimization and multiple ordinary/large-norm starts.
- `maximize_slater_fidelity`: the singular number-conserving Slater boundary.
  It optimizes four complex occupied orbitals on the Stiefel manifold, permits
  proton-neutron orbital mixing, and uses analytic determinant gradients.

The reported best-found Gaussian is the better of both searches. This addresses
a practical coordinate issue: a non-vacuum Slater determinant lies at infinite
Thouless norm and cannot be represented as a finite Z matrix.

## Interpretation

If non-Gaussianity is defined by geometric infidelity, use

N_G = 1 - F_G.

This is different from HFB energy optimization. The lowest-energy Gaussian need
not be the Gaussian with largest ground-state fidelity. Likewise, the intrinsic
vacuum produced by intrinsic HFB variation can have small raw overlap with the
exact state even when its normalized PAV state has much larger fidelity.

The problem is non-convex. Multiple starts and small gradients establish a
reproducible best-found stationary value, not a mathematical certificate of the
global maximum. For CKI Be8, all ten Slater starts converged to the same fidelity
within 4e-14, while finite Thouless searches approached a slightly smaller value
at the coordinate boundary. This is strong numerical evidence, not a proof,
that the closest Gaussian belongs to the Slater boundary.

## API

```python
from NSMFermions.gaussian_fidelity import (
    maximize_gaussian_fidelity, maximize_slater_fidelity)

interior = maximize_gaussian_fidelity(fermionic_hamiltonian, target, starts=16)
boundary = maximize_slater_fidelity(fermionic_hamiltonian, target, starts=10)
best_fidelity = max(interior.fidelity, boundary.fidelity)
non_gaussianity = 1 - best_fidelity
```

`fermionic_hamiltonian` is an existing `FermiHubbardHamiltonian` (or optimized
variant). It supplies the determinant ordering used by `target`; the fidelity
code does not build a second projected-space Hamiltonian. Consequently, this is
an exact benchmark tool whose target representation scales combinatorially. It
is not the polynomial gauge-kernel backend used for projected-energy evaluation.

The state-level calculation is also available directly:

```python
from NSMFermions.hfb import HFBState

state = HFBState.from_thouless(Z)
fidelity = state.fixed_sector_fidelity(
    target, fermionic_hamiltonian.occupations
)
```

This returns the raw intrinsic-Gaussian fidelity optimized by
`maximize_gaussian_fidelity`. Passing `projected=True` instead normalizes the
selected sector first and computes the fidelity after projection. The optimizer
remains a separate function because it manages analytic gradients, bounds, and
multiple starts rather than intrinsic state data.

## Verification

The analytic Slater gradient is checked against finite differences. Full
4096-dimensional quasiparticle-vacuum reconstruction can verify CKI overlaps
independently after the new PAV benchmark has been run. The unit tests also
compare the Gaussian objective directly with the `HFBState` amplitude API.
