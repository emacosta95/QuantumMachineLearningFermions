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

The optimizer minimizes the physical energy subject to average N and Z.  Its
default path uses SLSQP finite differences.  With `analytic_jacobian=True`, a
quadratic-penalty continuation using exact gradients selects a suitable number
surface basin, then SLSQP performs the final equality-constrained refinement.
The final result is therefore a Lagrange-multiplier solution, not the minimum
of an energy with a retained finite penalty. Chemical potentials are recovered
from the final stationarity equation; NaN indicates a rank-deficient constraint
Jacobian where they cannot be uniquely inferred. `converged` checks optimizer status, number
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
The analytic Jacobian differentiates the exponential Bogoliubov chart and the
Wick-contracted energy in one pass.  The dense rank-four interaction and dense
matrix operations are still intended as a reference implementation rather
than a scalable production solver.

## Usage

In an environment with the repository's legacy package dependencies installed:

```python
import numpy as np
from NSMFermions.hfb import HFBHamiltonian, solve_hfb

h = np.diag([-2., 1., -3., 2.])
ham = HFBHamiltonian(h, np.zeros((4, 4, 4, 4)))
result = solve_hfb(
    ham,
    neutron_modes=[0, 1],
    targets=[1, 1],
    seed=3,
    analytic_jacobian=True,
)
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

The analytic energy and number derivatives are checked against centered
directional finite differences.  On the four-mode constrained test, the
analytic penalty continuation plus SLSQP reaches the same energy and particle
numbers as the finite-difference solver.  A bounded five-iteration Ne-20 pilot
gave 33.6 s for the analytic path and 35.3 s for the shared finite-difference
path.  Neither five-iteration pilot was converged, so these numbers demonstrate
only per-pilot cost, not a production Ne-20 solution.

The HFB module's eight focused tests passed on 2026-09-29, including:
- Complex random HFB energy agrees with independent full Fock-space algebra.
- Canonical identities hold, including mixed normal and anomalous densities.
- A singular-U Slater limit is represented without inversion.
- Constrained noninteracting optimization reaches energy -5 and zero pairing.
- An attractive four-mode np pairing model reaches energy -1.5 with nonzero
  np pairing; malformed interactions are rejected.
- Analytic energy and number Jacobians agree with independent centered finite
  differences, and the analytic-gradient optimizer reaches the constrained
  noninteracting minimum.

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
- P. Ring and P. Schuck, *The Nuclear Many-Body Problem*, Springer (1980),
  Chs. 7-8 (HFB densities and energy variations).
- N. J. Higham, *Functions of Matrices*, SIAM (2008), Secs. 3.1-3.2
  (Frechet derivative and divided differences).
- A. H. Al-Mohy and N. J. Higham, SIAM J. Matrix Anal. Appl. 30,
  1639-1657 (2009), https://doi.org/10.1137/080716426
  (matrix-exponential Frechet derivative).
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer
  (2006), Sec. 17.1 (quadratic-penalty continuation).

This reference optimizer is not a reproduction of the TAURUS implementation.
