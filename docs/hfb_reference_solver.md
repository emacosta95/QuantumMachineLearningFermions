# Constrained HF/HFB solver

The implementation is in `src/NSMFermions/hfb.py` and requires NumPy and
SciPy. `solve_hfb(..., method="hfb")` and `solve_hfb(..., method="hf")` share
one public interface, multistart policy, result type, and convergence
diagnostics. They differ only in the physical variational manifold: HFB allows
an anomalous density, while HF fixes `kappa=0` exactly.

## HFB algorithm

The production HFB path follows the gradient method used by TAURUS. At the
current quasiparticle vacuum it evaluates the local Thouless derivatives

```text
dU = V* dZ
dV = U* dZ
```

and obtains the energy gradient `H20` and number gradients `N20`, `Z20`
directly from the density variations. It then solves the two-constraint Gram
system for the neutron and proton chemical potentials,

```text
G = H20 - lambda_N N20 - lambda_Z Z20,
(Q Q^T) lambda = Q H20,
```

takes a heavy-ball step, and applies local minimum-norm corrections until the
requested average particle numbers are recovered. A backtracking safeguard
rejects energy-increasing steps. Each accepted step is a canonical
quasiparticle rotation, so the Bogoliubov identities are preserved without a
generic constrained optimizer.

This removes the former quadratic-penalty continuation and SLSQP refinement.
It also avoids constructing the Frechet derivative of a global matrix
exponential for every variational coordinate. The old
`analytic_jacobian`, `shared_finite_difference_jacobian`, and
`number_penalty_max` keywords remain accepted as compatibility no-ops, but new
code should not use them.

The implementation follows:

- B. Bally et al., *Eur. Phys. J. A* **57**, 69 (2021),
  https://doi.org/10.1140/epja/s10050-021-00369-z.
- P. Ring and P. Schuck, *The Nuclear Many-Body Problem*, Springer (1980),
  Chapters 7–8.

It is an independent implementation of the published method, not copied
TAURUS source code.

## HF algorithm

`method="hf"` optimizes separate neutron and proton occupied-orbital
Grassmann manifolds using the analytic Fock gradient, conjugate-gradient
momentum, a line search, and QR retraction. Integer neutron and proton numbers
are exact by construction. This is the `kappa=0` restriction of the physical
ansatz, not a separate command-line optimizer selection.

## Real and complex states

`real_bogoliubov=True` restricts HFB to real antisymmetric Thouless steps. It
uses `m*(m-1)/2` real coordinates: 66 for CKI and 276 for USDB. The complex
manifold uses twice as many. The repository study scripts default to the real
TAURUS-compatible manifold; pass `--complex-bogoliubov` when complex intrinsic
states are required.

All mode pairs remain available, including neutron-proton pairing. Neutron and
proton mode lists define expectation-value constraints, not forbidden matrix
blocks. The vacuum solver supports even total fermion parity; blocked odd-total
states are not implemented.

## Usage

```python
import numpy as np
from NSMFermions.hfb import HFBHamiltonian, solve_hfb

h = np.diag([-2.0, 1.0, -3.0, 2.0])
hamiltonian = HFBHamiltonian(h, np.zeros((4, 4, 4, 4)))
result = solve_hfb(
    hamiltonian,
    neutron_modes=[0, 1],
    targets=[1, 1],
    method="hfb",
    real_bogoliubov=True,
    starts=4,
    seed=3,
)
if not result.converged:
    raise RuntimeError(result.attempts)
print(result.energy, result.numbers, result.stationarity_error)
```

Every attempt reports the number residual, constrained `H20` norm, iteration
count, final step size, number of local constraint corrections, and solver
name. Multistart HFB uses a deterministic lowest-orbital Slater seed plus
perturbed paired seeds. This lets the calculation find either a collapsed HF
minimum or a lower paired solution without forcing either result.

The default HFB gradient tolerance is `1e-3`, consistent with the scale used
in TAURUS examples. Pass `gradient_tolerance=` to `solve_hfb` for a stricter
or looser calculation. `HFBResult.parameters` stores accumulated transported
step coordinates for diagnostics; the authoritative reusable solution is
`HFBResult.state` (`U` and `V`).

The dense four-index interaction remains the principal scaling limitation.
OpenMP-enabled BLAS can accelerate contractions, but increasing CPU count does
not remove that memory and arithmetic cost.
