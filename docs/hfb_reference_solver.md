# Unrestricted complex HFB reference solver

Implemented in `src/NSMFermions/hfb.py`. Requires NumPy and SciPy.

## Scope

All mode pairs are allowed in complex U,V: nn, pp, and np anomalous densities,
and neutron-proton normal mixing. Species indices define number constraints,
not variational blocks. This first implementation covers even total fermion
parity. It does not fix neutron or proton parity separately. Odd-total blocked
states are not implemented. Particle-number and J=0 projection after variation
are implemented in the projection modules; variation after projection is
deliberately outside the current scope.

The optimizer minimizes the physical energy subject to average N and Z using
SLSQP equality constraints. These are Lagrange-multiplier constraints, not a
quadratic number penalty. Chemical potentials are recovered from the final
stationarity equation; NaN indicates a rank-deficient constraint Jacobian where
they cannot be uniquely inferred. `converged` checks optimizer status, number
residuals, and a parameter-space stationarity residual. It is not a proof of
global optimality or stability against all perturbations.

## Conventions

`beta = U.conj().T c + V.conj().T c_dagger`.

`rho = V.conj() @ V.T`, `kappa = V.conj() @ U.T`.

Hamiltonian: H = sum h_ij c_i† c_j + (1/4) sum v_ijkl c_i† c_j† c_l c_k.

Interaction inputs must be antisymmetric in each index pair and Hermitian
under pair exchange. Dictionary entries are copied as supplied: missing
permutations are not silently reconstructed. This matches the operator order
in the existing `get_twobody_interaction`, whose call sets j1=i4, j2=i3.
The nuclear reader constructs antisymmetric permutations. Actual USDB input
and comparison with the existing builder remain to be validated.

Energy = Tr(h rho) + (1/2) sum v_ijkl rho_ki rho_lj
         + (1/4) sum v_ijkl kappa_ij* kappa_kl.

Exponentiating a complex antisymmetric generator preserves canonical constraints
and avoids inverse-U formulas. HF states with singular U are representable.
The finite-difference optimizer and dense rank-four interaction are intended
for small reference calculations; analytic gradients and a scalable optimizer
remain future work.

## Usage

In an environment with the repository's legacy package dependencies installed:

```python
import numpy as np
from NSMFermions.hfb import HFBHamiltonian, solve_hfb

h = np.diag([-2., 1., -3., 2.])
ham = HFBHamiltonian(h, np.zeros((4, 4, 4, 4)))
result = solve_hfb(ham, neutron_modes=[0, 1], targets=[1, 1], seed=3)
if not result.converged:
    raise RuntimeError(result.attempts)
print(result.energy, result.numbers, np.linalg.norm(result.state.kappa))
```

The legacy package initializer eagerly imports optional ML modules. For an
isolated NumPy/SciPy environment, add `src/NSMFermions` to `sys.path` and use
`from hfb import HFBHamiltonian, solve_hfb`; the tests use an explicit file
import to avoid changing the package's existing import behavior.

## Validation performed

`python -m unittest discover -s tests -v` (with NumPy/SciPy installed).

Five tests passed on 2026-09-16:
- Complex random HFB energy agrees with independent full Fock-space algebra.
- Canonical identities hold, including mixed normal and anomalous densities.
- A singular-U Slater limit is represented without inversion.
- Constrained noninteracting optimization reaches energy -5 and zero pairing.
- An attractive four-mode np pairing model reaches energy -1.5 with nonzero
  np pairing; malformed interactions are rejected (covered across five tests).

The first CKI Be8 benchmark is now recorded in
`benchmarks/results/cki_be8.md`: it reproduces a collapsed HF solution in a
bounded local optimization. No large computation has been run. Pairing
collapse is a permissible result: inspect paired starts, convergence and
stability rather than forcing a nonzero anomalous density. After number
projection, use number-conserving pairing diagnostics because the
projected state's anomalous expectation vanishes by number conservation.

Primary methodological references:
- TAURUS I: https://arxiv.org/abs/2010.14169
- Stoitsov et al.: https://arxiv.org/abs/nucl-th/0610061

This reference optimizer is not a reproduction of the TAURUS implementation.
