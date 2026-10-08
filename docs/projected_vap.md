# Particle-number and J=0 variation after projection

`projected_vap.py` minimizes the energy of the same
`BogoliubovVacuumSeries` later used for projected observables.  Every objective
evaluation performs

1. finite-Thouless parameters to an `HFBState`;
2. the configured gauge/Euler transformations to a vacuum series;
3. projected energy evaluation from that series.

The final retained series is passed directly to `project_state_observables`,
so the reported energy and fidelity use identical transformations and weights.

## Energy backends

`energy_backend="kernel"` uses generalized-Wick transition kernels.  This is
the scalable formulation, but it requires a guaranteed-exact number grid by
default.  With a paired vacuum, `number_grid=(1,1)` does not select fixed N and
Z; using that aliased norm would not define fixed-particle-number VAP.

`energy_backend="fixed_sector_basis"` expands the series in an existing
fixed-(N,Z) `FermiHubbardHamiltonian` basis at every objective evaluation.  In
this small-space diagnostic backend, the basis selection itself removes all
other number sectors, so a 1x1 gauge grid is valid.  The Euler rule may also be
undersized when the projector was constructed with
`allow_inexact_euler_grid=True`.  This backend scales combinatorially and is
intended for CKI validation, not large calculations.

`energy_backend="analytic_fixed_sector"` differentiates the Pfaffian
coefficient of every determinant in the fixed N,Z basis.  The configured Euler
transformations remain explicit.  The U(1) sums are performed analytically:
inside the target sector each gauge phase cancels its Fourier character.  This
gives the exact real gradient with respect to every complex antisymmetric
Thouless entry and lets L-BFGS optimize 132 CKI coordinates without finite
differences.  The benchmark independently checks the final state with the full
transition-kernel series whenever the number and Euler grids are exact.

## Pairing

VAP varies all complex entries of the Thouless matrix, including neutron,
proton, and neutron-proton pairing.  Multiple weak and strong seed scales are
used instead of enforcing a nonzero pairing tensor.  The returned
`kappa_norm` shows whether projected-energy variation selected a paired
intrinsic state.  Forcing a minimum pairing norm would define a different,
constrained functional and is therefore not hidden in the solver.

The finite Thouless chart does not include a nonempty *intrinsic* Slater
determinant at a finite coordinate value.  For an even target, however,
`number_projected_slater_seed` embeds occupied orbitals in a finite BCS vacuum
whose target-number component is exactly that determinant.  Starting VAP from
this seed includes the HF-PAV state as a projected variational baseline.  The
solver retains the initial point unless it finds a lower projected energy.

## Python API

```python
from NSMFermions.angular_momentum import ParticleNumberJ0ProjectedEnergy
from NSMFermions.projected_vap import (
    number_projected_slater_seed,
    solve_projected_hfb_vap,
)

projector = ParticleNumberJ0ProjectedEnergy(
    ham,
    state_encoding,
    neutron_modes,
    targets,
    number_grid=(1, 1),
    euler_grid=(3, 3, 3),
    allow_inexact_number_grid=True,
    allow_inexact_euler_grid=True,
)

# Small-space CKI analysis: fixed-sector expansion makes the 1x1 number grid
# an exact sector selection even though the vacuum series itself is aliased.
result = solve_projected_hfb_vap(
    projector,
    energy_backend="fixed_sector_basis",
    fermionic_hamiltonian=fermionic,
    starts=4,
    seed=7,
    optimizer="SPSA",
)
```

For the polynomial transition-kernel formulation, use the exact number grid:

```python
projector = ParticleNumberJ0ProjectedEnergy(
    ham,
    state_encoding,
    neutron_modes,
    targets,
    number_grid=None,
    euler_grid=(3, 3, 3),
    allow_inexact_euler_grid=True,
)
result = solve_projected_hfb_vap(projector, energy_backend="kernel")
```

For an exact kernel calculation initialized from occupied HF orbitals, use
`initial_state=number_projected_slater_seed(hf_orbitals)`.  Columns must be
ordered in consecutive same-species pairs when neutron and proton numbers are
projected separately.

## Historical CKI result

An exact Be10 full-series calculation used a `7 x 7` number grid, a `9 x 5 x
9` Euler grid, the `analytic_fixed_sector` backend, and L-BFGS-B with a
`2e-6` gradient tolerance.

For Be10 this deterministic route lowers the HF-PAV energy from
`-38.6565430722` to `-39.4357813571` MeV and raises the exact-ground-state
fidelity from `0.9459990335` to `0.9994052173`.  The final projected-gradient
norm is `9.02e-7`; the full 19,845-vacuum kernel agrees with the analytic
objective within `7e-14` MeV.

`L-BFGS-B` uses the exact Pfaffian gradient with `analytic_fixed_sector`; other
backends would otherwise require an expensive finite-difference gradient.  The
`SPSA` option uses two simultaneous perturbations per iteration and is the
appropriate optimizer for Metropolis-projected objectives.  Its convergence is
stochastic, so repeat several seeds and compare exact-grid validations.

## Metropolis-projected variation

Euler projection can be importance sampled while the particle-number Fourier
sum remains exact. The historical calculation used the `fixed_sector_basis`
backend, 256 Metropolis samples after 500 burn-in steps, thinning by two, and
SPSA optimization.

The Markov chain samples Euler rotations with probability proportional to the
vacuum-overlap magnitude and stores the reciprocal-importance weights in the
projected vacuum series.  A fixed random seed supplies common random numbers
to nearby SPSA evaluations.  For CKI, use `fixed_sector_basis`: its finite-
sample Rayleigh quotient is real and variational.  The one-sided stochastic
transition-kernel ratio can have appreciable complex noise and should be
treated as a diagnostic rather than an optimization objective.

Every stochastic benchmark also reevaluates its final intrinsic state with the
deterministic exact grid.  For Be10, seed 41 gave:

| Euler samples | Sampled vacua after gauge collapse | Exact-grid energy | Exact-grid fidelity |
|---:|---:|---:|---:|
| 64 | 64 | -38.89248915 | 0.96210949 |
| 256 | 256 | -39.05819982 | 0.97446866 |
| exact `(9,5,9)` VAP | 405 | -39.43578136 | 0.99940522 |

The stochastic runs are exploratory SPSA solutions, not stationary points.
Increasing the sample count reduced projection bias but increased runtime
approximately linearly.

## Projection-grid references

The implementation follows the standard finite Fourier/Fomenko treatment of
particle-number and periodic Euler angles, with Gauss-Legendre quadrature in
`cos(beta)`.  Useful primary references are:

- [Bally and Bender, projection on particle number and angular momentum](https://arxiv.org/abs/2010.15224), including numerical implementation and mesh reduction for Bogoliubov vacua.
- [Johnson and Jiao, convergence and efficiency of angular-momentum projection](https://arxiv.org/abs/1808.05672), relating periodic trapezoidal sums, linear-algebra projection, and Fomenko projection.
- [Shimizu and Tsunoda, SO(3) quadratures in angular-momentum projection](https://arxiv.org/abs/2205.04119), deriving exactness conditions and comparing conventional and reduced SO(3) rules.
- [Anguiano, Egido, and Robledo, particle-number projection with effective forces](https://arxiv.org/abs/nucl-th/0105003), for particle-number projected HFB kernels and gradients.

The output records grid exactness, projected energy and fidelity, intrinsic
particle-number expectations, pairing norm, all optimizer attempts, and the
final state arrays.
