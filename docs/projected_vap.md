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

## CKI command line

```text
python benchmarks/cki_be_vap.py --mass 10 --number-grid 1 1 \
  --euler-grid 3 3 3 --allow-inexact-number-grid \
  --allow-inexact-euler-grid --backend fixed_sector_basis --optimizer SPSA
```

The exact Be10 full-series calculation can be reproduced from a saved HF
orbital file with:

```text
python benchmarks/cki_be_vap.py --mass 10 --number-grid 7 7 \
  --euler-grid 9 5 9 --backend analytic_fixed_sector \
  --optimizer L-BFGS-B --gradient-tolerance 2e-6 \
  --initial-state benchmarks/results/cki_be10_hf_pav_state.npz
```

For Be10 this deterministic route lowers the HF-PAV energy from
`-38.6565430722` to `-39.4357813571` MeV and raises the exact-ground-state
fidelity from `0.9459990335` to `0.9994052173`.  The final projected-gradient
norm is `9.02e-7`; the full 19,845-vacuum kernel agrees with the analytic
objective within `7e-14` MeV.

`L-BFGS-B` uses ordinary finite-difference gradients and therefore needs about
one objective evaluation per real Thouless coordinate for each gradient.  The
`SPSA` option uses two simultaneous perturbations per iteration and is the
practical default for the 132-coordinate CKI calculations.  Its convergence is
stochastic, so repeat several seeds and compare the best projected energy.

The output records grid exactness, projected energy and fidelity, intrinsic
particle-number expectations, pairing norm, all optimizer attempts, and the
final state arrays.
