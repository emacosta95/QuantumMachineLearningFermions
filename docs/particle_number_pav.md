# Particle-number projection after variation

`number_projection.py` implements particle-number projection after variation
(PAV). The HF/HFB optimizer finishes first. Projection then selects the desired
fixed-particle-number sector; the projected energy is never fed back into the
intrinsic optimization.

## Data flow

1. Build and solve an intrinsic `HFBHamiltonian` with `solve_hfb`.
2. Build the repository's existing `FermiHubbardHamiltonian` (or optimized
   variant) for the desired neutron and proton numbers.
3. Call `number_projected_series`. Two full-period trapezoidal grids construct
   `L_N*L_Z` gauge-rotated Bogoliubov vacua and their Fourier coefficients.
4. Keep this `BogoliubovVacuumSeries` representation for kernel energies and
   all subsequent symmetry operations.
5. Only when energy or fidelity in a determinant basis is required, call
   `projected_series_observables` with the existing `FermiHubbardHamiltonian`.
6. The series then evaluates Pfaffian or Slater-minor components in the exact
   `occupations` ordering owned by that Hamiltonian.
7. The squared norm of the coherently summed components gives the sector probability
   `<Phi|P_N P_Z|Phi>`.
8. Dividing by its square root constructs the normalized projected vector.
9. The energy is the Rayleigh quotient of the already assembled fermionic
   Hamiltonian matrix. An optional exact target supplies the fidelity.

There is no separate projected-space class and no duplicate many-body
Hamiltonian builder. PN-VAP gradients and optimizers are intentionally absent.

## Paired and Hartree-Fock states

For a paired vacuum with nonsingular `U`, `HFBState` obtains the Thouless matrix
`Z = (V U^{-1})*`. An even occupation coefficient is the Pfaffian of the
corresponding principal submatrix of `Z`, multiplied by the normalized vacuum
amplitude. PFAPACK performs the Pfaffian calculation.

A non-vacuum Slater determinant is an exact HF limit with singular `U`, so its
particle-vacuum Thouless matrix is not finite. `HFBState.from_slater` stores the
same state in canonical `U,V` form. When `U` is singular and the density is an
idempotent number-conserving projector, occupation coefficients are computed as
determinants of occupied-orbital minors. Thus collapsed HFB/HF solutions remain
projectable without regularizing or perturbing `U`.

## Minimal use

```python
from NSMFermions.number_projection import (
    exact_ground_state,
    number_projected_series,
    projected_series_observables,
)

exact_energy, target = exact_ground_state(fermionic_hamiltonian)
series = number_projected_series(
    hfb_result.state,
    fermionic_hamiltonian,
    grid=(7, 7),
)
result = projected_series_observables(
    series,
    fermionic_hamiltonian,
    target,
)

print(series.number_of_vacua)
print(result.sector_weight)
print(result.energy, result.fidelity)
```

`result.projected_vector`, `result.occupations`, and `result.masks` describe the
same fixed-number basis as `fermionic_hamiltonian.matrix`.

## Polynomial gauge-kernel check

`GaugeProjectedEnergy` evaluates the same PAV energy by discrete gauge-angle
quadrature and transition-density kernels without constructing a many-body
vector. It is useful for larger model spaces and for validation against the
explicit small-space route. It requires a finite Thouless matrix; the explicit
`project_particle_numbers` route is the supported path for singular-`U` HF
limits.

Particle-number grids remain deterministic rather than Metropolis sampled. The
U(1)xU(1) integral is only two-dimensional and has an exact finite Fourier
rule. More importantly, a finite random gauge sample is not an exact N,Z
projector: it leaves components in unwanted particle sectors that are absent
from a fixed-N,Z `FermiHubbardHamiltonian` basis. The optional Metropolis path
therefore samples only Euler rotations and includes the complete controlled
number grid for every retained rotation.

### Deliberately undersized grids

The default remains the state-independent exactness rule `m_species + 1`.
Arbitrary positive grid sizes can be selected explicitly for convergence tests
or for a known fixed-N,Z state:

```python
result = project_particle_numbers(
    state,
    fermionic_hamiltonian,
    grid=(1, 1),
    allow_inexact_grid=True,
)
```

The same opt-in is available on `number_projected_series` and
`GaugeProjectedEnergy`. An undersized grid emits `ProjectionGridWarning` and
the returned series records `number_grid_guaranteed_exact=False` together with
`minimum_number_grid`. A one-point rule is exact for a state already known to
be an eigenstate of both N and Z, but it is not a general particle-number
projector: other sectors congruent modulo the grid size can alias into the sum.
Without the explicit opt-in, undersized grids continue to raise `ValueError`.

The CKI Be8 workflow is in `benchmarks/cki_be8_pav.py`. It performs intrinsic
variation, exact fixed-sector PAV, optional gauge-kernel validation, and saves
the intrinsic `U,V` together with the projected vector and determinant masks.
