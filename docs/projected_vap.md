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

## Pairing

VAP varies all complex entries of the Thouless matrix, including neutron,
proton, and neutron-proton pairing.  Multiple weak and strong seed scales are
used instead of enforcing a nonzero pairing tensor.  The returned
`kappa_norm` shows whether projected-energy variation selected a paired
intrinsic state.  Forcing a minimum pairing norm would define a different,
constrained functional and is therefore not hidden in the solver.

The finite Thouless chart does not include a nonempty Slater determinant at a
finite coordinate value.  Large-norm seeds and bounds approach that boundary,
but a separate projected-HF boundary calculation remains necessary when
deciding whether the global optimum is paired.

## Python API

```python
from NSMFermions.angular_momentum import ParticleNumberJ0ProjectedEnergy
from NSMFermions.projected_vap import solve_projected_hfb_vap

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

## CKI command line

```text
python benchmarks/cki_be_vap.py --mass 10 --number-grid 1 1 \
  --euler-grid 3 3 3 --allow-inexact-number-grid \
  --allow-inexact-euler-grid --backend fixed_sector_basis --optimizer SPSA
```

`L-BFGS-B` uses ordinary finite-difference gradients and therefore needs about
one objective evaluation per real Thouless coordinate for each gradient.  The
`SPSA` option uses two simultaneous perturbations per iteration and is the
practical default for the 132-coordinate CKI calculations.  Its convergence is
stochastic, so repeat several seeds and compare the best projected energy.

The output records grid exactness, projected energy and fidelity, intrinsic
particle-number expectations, pairing norm, all optimizer attempts, and the
final state arrays.
