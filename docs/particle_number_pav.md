# Particle-number projection after variation

`number_projection.py` implements particle-number projection after variation
(PAV). The HF/HFB optimizer finishes first. Projection then selects the desired
fixed-particle-number sector; the projected energy is never fed back into the
intrinsic optimization.

## Data flow

1. Build and solve an intrinsic `HFBHamiltonian` with `solve_hfb`.
2. Build the repository's existing `FermiHubbardHamiltonian` (or optimized
   variant) for the desired neutron and proton numbers.
3. Pass the resulting `HFBState` and fermionic Hamiltonian to
   `project_particle_numbers`.
4. The function reads `occupations`, `masks`, and `matrix` directly from that
   Hamiltonian, so it cannot silently use a different determinant ordering.
5. `HFBState.occupation_amplitudes` evaluates the intrinsic coefficient of
   every determinant in the fixed sector.
6. Two full-period trapezoidal grids apply the discrete
   `U(1)_N x U(1)_Z` Fourier projector. A species with `c` modes uses at least
   `c+1` angles, which resolves every possible particle-number power exactly.
7. The squared norm of the Fourier-projected coefficients gives the sector probability
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
    project_particle_numbers,
)

exact_energy, target = exact_ground_state(fermionic_hamiltonian)
result = project_particle_numbers(
    hfb_result.state,
    fermionic_hamiltonian,
    target,
)

print(result.grid, result.grid_offset)
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

The CKI Be8 workflow is in `benchmarks/cki_be8_pav.py`. It performs intrinsic
variation, exact fixed-sector PAV, optional gauge-kernel validation, and saves
the intrinsic `U,V` together with the projected vector and determinant masks.
