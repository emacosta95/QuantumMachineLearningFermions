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
  analytic gradients and multiple ordinary/large-norm starts.
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
vacuum underlying a PN-VAP state is optimized only after projection and can have
small raw overlap with the exact state even when its normalized projected state
has much larger fidelity.

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

interior = maximize_gaussian_fidelity(space, exact_ground_state, starts=16)
boundary = maximize_slater_fidelity(space, exact_ground_state, starts=10)
best_fidelity = max(interior.fidelity, boundary.fidelity)
non_gaussianity = 1 - best_fidelity
```

`space` is a small-system `NumberProjectedSpace`, which supplies occupation
ordering and Pfaffian polynomials. Consequently, this implementation is an exact
benchmark tool whose target representation scales combinatorially. It is not
the polynomial gauge-kernel backend used for projected-energy evaluation.

## Verification

The analytic Gaussian and Slater gradients are checked against finite
differences. Full 4096-dimensional quasiparticle-vacuum reconstruction verifies
the CKI overlaps independently. Eighteen regression tests pass. Numerical CKI
results are in `benchmarks/results/cki_be8_best_gaussian.md`.
