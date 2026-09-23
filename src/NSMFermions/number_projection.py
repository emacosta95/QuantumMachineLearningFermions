"""Particle-number projection after variation using fermionic Hamiltonians.

The intrinsic HFB/HF state is optimized elsewhere. This module performs only
projection after variation (PAV): it reads the determinant basis and many-body
matrix already constructed by ``FermiHubbardHamiltonian`` (or its optimized
variant), applies exact finite U(1) x U(1) Fourier sums to the corresponding
``HFBState`` amplitudes, normalizes the projected vector, and evaluates
observables.

There is deliberately no PN-VAP optimizer, Pfaffian matching table, duplicated
Hamiltonian builder, or projected-energy gradient in this module.
"""

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
from scipy import sparse

if __package__:
    from .hfb import BogoliubovVacuumSeries, HFBState, ProjectionGridWarning
else:
    from hfb import BogoliubovVacuumSeries, HFBState, ProjectionGridWarning


# Restrict the public projection API to the repository's established exact
# Hamiltonian containers. Checking the class hierarchy also permits subclasses.
_FERMIONIC_HAMILTONIAN_TYPES = {
    "FermiHubbardHamiltonian",
    "FermiHubbardHamiltonianOptimized",
}


def _require_fermionic_hamiltonian(hamiltonian):
    """Validate and unpack the projection-facing Hamiltonian interface."""
    # Inspect every base-class name so a specialized nuclear Hamiltonian derived
    # from either supported FermiHubbardHamiltonian remains acceptable.
    hierarchy = {cls.__name__ for cls in type(hamiltonian).__mro__}
    if not hierarchy.intersection(_FERMIONIC_HAMILTONIAN_TYPES):
        raise TypeError(
            "Projection requires FermiHubbardHamiltonian or "
            "FermiHubbardHamiltonianOptimized"
        )

    # PAV needs the number of modes, determinant ordering, bit masks, and the
    # already assembled many-body matrix. The Hamiltonian classes expose these
    # through a common read-only interface.
    for attribute in (
        "modes",
        "occupations",
        "masks",
        "matrix",
        "species_modes",
        "particle_numbers",
    ):
        if not hasattr(hamiltonian, attribute):
            raise TypeError(
                f"Fermionic Hamiltonian is missing projection attribute {attribute}"
            )

    # Convert basis metadata to stable NumPy/Python representations. The order
    # is never changed because it is also the row/column order of the matrix.
    occupations = [tuple(int(i) for i in row) for row in hamiltonian.occupations]
    masks = np.asarray(hamiltonian.masks, dtype=np.int64)
    matrix = hamiltonian.matrix
    dimension = len(occupations)

    # All three representations must describe exactly the same determinant
    # basis before state coefficients can safely be contracted with the matrix.
    if masks.shape != (dimension,) or matrix.shape != (dimension, dimension):
        raise ValueError("Fermionic Hamiltonian basis and matrix sizes disagree")
    if any(tuple(sorted(row)) != row for row in occupations):
        raise ValueError("Fermionic occupations must use increasing mode order")
    return occupations, masks, matrix


def fermionic_basis_data(hamiltonian):
    """Return validated ``(occupations, masks, matrix)`` basis information."""
    # Keep validation centralized so fidelity and angular-momentum utilities
    # cannot silently adopt a different determinant ordering from PAV.
    return _require_fermionic_hamiltonian(hamiltonian)


def _number_projection_metadata(
    modes,
    species_modes,
    particle_numbers,
    grid=None,
    offset=0.137,
    allow_inexact_grid=False,
):
    """Validate species data and return the two finite Fourier grids."""
    # Convert the two species blocks to immutable integer sets for fast counting.
    species = tuple(frozenset(int(mode) for mode in block) for block in species_modes)
    # Exactly two independent gauge angles are used for neutron/proton projection.
    if len(species) != 2:
        raise ValueError("Number projection requires exactly two species blocks")

    # A mode must belong to one and only one species for separate N and Z phases.
    if species[0].intersection(species[1]):
        raise ValueError("Fermionic species blocks must be disjoint")
    # The largest mode index fixes the represented one-body space.
    represented_modes = species[0].union(species[1])
    if represented_modes != set(range(modes)):
        raise ValueError("Fermionic species blocks must cover every mode exactly once")

    # Convert requested particle counts to ordinary Python integers.
    targets = tuple(int(number) for number in particle_numbers)
    # Reject malformed or physically impossible target sectors before quadrature.
    if len(targets) != 2 or any(
        number < 0 or number > len(block)
        for number, block in zip(targets, species)
    ):
        raise ValueError("Invalid fixed particle numbers for species blocks")

    # A block with c modes has Fourier powers 0,...,c, so c+1 points integrate
    # every possible particle-number component exactly without aliasing.
    minimum_grid = tuple(len(block) + 1 for block in species)
    # Use the exact minimal grid unless the caller requests another positive grid.
    chosen_grid = minimum_grid if grid is None else tuple(grid)
    if len(chosen_grid) != 2 or any(
        not isinstance(points, (int, np.integer)) or points < 1
        for points in chosen_grid
    ):
        raise ValueError("Number grid must contain two positive integers")
    # Fewer than c+1 points identify particle numbers only modulo L and can
    # therefore leave aliased sectors in a general HFB vacuum.
    guaranteed_exact = all(
        points >= minimum
        for points, minimum in zip(chosen_grid, minimum_grid)
    )
    if not guaranteed_exact:
        message = (
            f"Number grid {chosen_grid} is below the finite-space exactness "
            f"bound {minimum_grid}; exact P_N P_Z symmetry restoration is not "
            "guaranteed. Use the bound or a larger grid for a guaranteed "
            "projector."
        )
        if not allow_inexact_grid:
            raise ValueError(message + " Set allow_inexact_grid=True to proceed.")
        warnings.warn(message, ProjectionGridWarning, stacklevel=3)
    # The common fractional shift avoids sampling frequent overlap-zero angles.
    if not np.isfinite(offset):
        raise ValueError("Gauge-grid offset must be finite")

    # Return validated immutable data shared by series construction and tests.
    return species, targets, chosen_grid, minimum_grid, guaranteed_exact


def number_projected_series(
    state,
    hamiltonian,
    grid=None,
    offset=0.137,
    *,
    allow_inexact_grid=False,
):
    """Return P_N P_Z|Phi> as a weighted series of Bogoliubov vacua.

    No determinant coefficients are computed here. Every term is a gauge-rotated
    vacuum, and ``grid=(L_A,L_B)`` directly controls the number ``L_A*L_B`` of
    vacua retained in the Fourier representation. Grids below the exactness
    bound require ``allow_inexact_grid=True`` and emit a warning.
    """
    # Projection requires a complete HFBState rather than densities alone.
    if not isinstance(state, HFBState):
        raise TypeError("state must be an HFBState")
    # Validate the established fermionic container and its mode dimension.
    _require_fermionic_hamiltonian(hamiltonian)
    if hamiltonian.modes != len(state.U):
        raise ValueError("State and fermionic Hamiltonian use different modes")
    # Validate species blocks, requested particle counts, and exact grid bounds.
    species, targets, chosen_grid, minimum_grid, guaranteed_exact = (
        _number_projection_metadata(
        hamiltonian.modes,
        hamiltonian.species_modes,
        hamiltonian.particle_numbers,
        grid=grid,
        offset=offset,
        allow_inexact_grid=allow_inexact_grid,
        )
    )

    # Allocate one transformation and coefficient per pair of gauge angles.
    transformations = []
    weights = []
    # Traverse subsystem A's full-period Fourier grid.
    for index_a in range(chosen_grid[0]):
        # The fractional offset changes nodes without changing an exact sum.
        angle_a = 2 * np.pi * (index_a + offset) / chosen_grid[0]
        # Traverse subsystem B's independent Fourier grid.
        for index_b in range(chosen_grid[1]):
            # Construct the second full-period angle.
            angle_b = 2 * np.pi * (index_b + offset) / chosen_grid[1]
            # Assign the proper gauge angle to every one-body mode.
            phases = np.ones(hamiltonian.modes, dtype=complex)
            phases[list(species[0])] = np.exp(1j * angle_a)
            phases[list(species[1])] = np.exp(1j * angle_b)
            # Gauge rotations are diagonal in the particle-mode basis.
            transformations.append(np.diag(phases))
            # The complex Fourier character selects the requested A,B numbers.
            character = np.exp(
                -1j * (targets[0] * angle_a + targets[1] * angle_b)
            )
            # Normalizing by both grid sizes implements the double U(1) integral.
            weights.append(character / np.prod(chosen_grid))

    # Keep the state as a group-orbit series until an observable requests a basis.
    return BogoliubovVacuumSeries(
        intrinsic_state=state,
        transformations=np.asarray(transformations),
        weights=np.asarray(weights),
        number_grid=chosen_grid,
        euler_grid=None,
        projection="P_N P_Z",
        number_offset=offset,
        number_grid_guaranteed_exact=guaranteed_exact,
        minimum_number_grid=minimum_grid,
    )


@dataclass
class NumberProjectionResult:
    """Normalized P_N P_Z state and PAV observables in the fermionic basis."""

    # The symmetry-breaking state produced by the preceding HF/HFB variation.
    intrinsic_state: HFBState
    # Weighted gauge/Euler vacuum series used before basis expansion.
    series: BogoliubovVacuumSeries
    # Coefficients after selecting and normalizing the Hamiltonian's N,Z basis.
    projected_vector: np.ndarray
    # Probability of that N,Z sector in the normalized intrinsic state.
    sector_weight: float
    # Rayleigh quotient of the normalized projected vector.
    energy: float
    # Optional squared overlap with an exact target in the identical basis.
    fidelity: Optional[float]
    # Basis metadata retained so saved vectors are self-describing.
    occupations: tuple
    masks: np.ndarray
    # Numbers of discrete gauge angles used for the two U(1) integrals.
    grid: tuple
    # Fractional shift of both periodic trapezoidal grids.
    grid_offset: float


def projected_series_observables(series, hamiltonian, target=None):
    """Expand a vacuum series in the fermionic basis only for observables."""
    # Reject arbitrary coefficient containers that do not preserve vacuum terms.
    if not isinstance(series, BogoliubovVacuumSeries):
        raise TypeError("series must be a BogoliubovVacuumSeries")
    # Read basis ordering and matrix from the existing fermionic Hamiltonian.
    occupations, masks, matrix = _require_fermionic_hamiltonian(hamiltonian)
    # Every one-body transformation must act on the Hamiltonian's mode space.
    if hamiltonian.modes != len(series.intrinsic_state.U):
        raise ValueError("Series and fermionic Hamiltonian use different modes")

    # This is the deliberately delayed Fock-space expansion requested for
    # fidelity: sum w_q <I|T_q|Phi> in Hamiltonian row order.
    amplitudes = series.occupation_amplitudes(occupations)
    # For exact N,Z grids, this fixed sector contains the complete projected ket.
    norm = float(np.vdot(amplitudes, amplitudes).real)
    if not np.isfinite(norm) or norm < 1e-14:
        raise ValueError("Projected vacuum series has vanishing norm")
    # Normalize only after all Bogoliubov-vacuum terms have been coherently summed.
    projected = amplitudes / np.sqrt(norm)
    # Apply the exact fixed-sector Hamiltonian to the normalized coefficient vector.
    energy_value = np.vdot(projected, matrix @ projected)
    if abs(energy_value.imag) > 1e-9:
        raise ValueError("Projected energy has a non-negligible imaginary part")

    # Fidelity is evaluated at the same late basis-expansion boundary.
    fidelity = None
    if target is not None:
        # Convert and validate the target in identical determinant ordering.
        target = np.asarray(target, dtype=complex)
        if target.shape != projected.shape:
            raise ValueError("Target must match the fermionic Hamiltonian basis")
        # Normalize defensively because eigenvector callers may rescale the target.
        target_norm = np.linalg.norm(target)
        if not np.isfinite(target_norm) or target_norm == 0:
            raise ValueError("Target must have finite nonzero norm")
        # The physical fidelity is the squared overlap of normalized vectors.
        fidelity = float(abs(np.vdot(target / target_norm, projected)) ** 2)

    # Preserve legacy field names while attaching the actual vacuum series.
    return NumberProjectionResult(
        intrinsic_state=series.intrinsic_state,
        series=series,
        projected_vector=projected,
        sector_weight=norm,
        energy=float(energy_value.real),
        fidelity=fidelity,
        occupations=tuple(occupations),
        masks=masks.copy(),
        grid=series.number_grid,
        grid_offset=series.number_offset,
    )


def project_particle_numbers(
    state,
    hamiltonian,
    target=None,
    grid=None,
    offset=0.137,
    *,
    allow_inexact_grid=False,
):
    """Project an already optimized HFB/HF state into the Hamiltonian sector.

    Parameters
    ----------
    state : HFBState
        Intrinsic state obtained before projection. Finite-Z paired vacua use
        Pfaffian occupation amplitudes; singular-U HF states automatically use
        Slater-determinant amplitudes.
    hamiltonian : FermiHubbardHamiltonian
        An assembled fixed-particle-number Hamiltonian. Its basis determines
        both the projected sector and coefficient ordering.
    target : array_like, optional
        Exact state in the same Hamiltonian basis. When supplied, the result
        includes the squared overlap with the normalized projected state.
    grid : tuple, optional
        Numbers of gauge angles for the two species. The default ``c+1`` rule
        exactly resolves every sector of a block containing ``c`` modes. Any
        positive grid is accepted when ``allow_inexact_grid=True``.
    offset : float, optional
        Common fractional shift of the full-period trapezoidal grids.
    allow_inexact_grid : bool, optional
        Permit a positive grid below the finite-space exactness bound. A
        :class:`ProjectionGridWarning` is emitted because unwanted particle
        sectors can then alias into the requested sector.
    """
    # Reject density-like stand-ins: projection requires the full quasiparticle
    # state because determinant amplitudes are not determined by rho alone.
    if not isinstance(state, HFBState):
        raise TypeError("state must be an HFBState")

    # Construct the finite Fourier expansion without introducing basis states.
    series = number_projected_series(
        state,
        hamiltonian,
        grid=grid,
        offset=offset,
        allow_inexact_grid=allow_inexact_grid,
    )
    # Expand only at the requested energy/fidelity evaluation boundary.
    return projected_series_observables(
        series, hamiltonian, target=target
    )


def exact_ground_state(hamiltonian):
    """Return the lowest eigenpair of an assembled small fermionic Hamiltonian."""
    # Reuse the same interface validation as projection so the returned vector
    # is guaranteed to follow the occupation ordering expected by HFBState.
    _, _, matrix = _require_fermionic_hamiltonian(hamiltonian)

    # This helper is deliberately a small-system reference path. Convert sparse
    # matrices to dense form and diagonalize the Hermitian matrix completely.
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    energies, vectors = np.linalg.eigh(dense)

    # NumPy orders eigenvalues increasingly, so column zero is the ground state.
    return float(energies[0]), vectors[:, 0]
