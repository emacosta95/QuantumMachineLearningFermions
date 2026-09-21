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

import numpy as np
from scipy import sparse

if __package__:
    from .hfb import HFBState
else:
    from hfb import HFBState


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


def _discrete_number_projector(
    amplitudes,
    occupations,
    species_modes,
    particle_numbers,
    grid=None,
    offset=0.137,
):
    """Apply the finite U(1) x U(1) Fourier projector to basis amplitudes."""
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
    if represented_modes != set(range(sum(len(block) for block in species))):
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
    # Use the exact minimal grid unless the caller requests a larger one.
    chosen_grid = minimum_grid if grid is None else tuple(grid)
    # Larger grids remain exact, while c or fewer points can alias two sectors.
    if len(chosen_grid) != 2 or any(
        not isinstance(points, (int, np.integer)) or points < minimum
        for points, minimum in zip(chosen_grid, minimum_grid)
    ):
        raise ValueError(f"Number grid must be at least {minimum_grid}")
    # The common fractional shift avoids sampling frequent overlap-zero angles.
    if not np.isfinite(offset):
        raise ValueError("Gauge-grid offset must be finite")

    # Store the normalized intrinsic amplitudes in the Hamiltonian's row order.
    intrinsic = np.asarray(amplitudes, dtype=complex)
    # Each basis row must have exactly one coefficient.
    if intrinsic.shape != (len(occupations),):
        raise ValueError("Amplitude and occupation-basis dimensions disagree")

    # Count subsystem-A and subsystem-B particles in every determinant row.
    counts = np.asarray(
        [
            [sum(mode in block for mode in occupation) for block in species]
            for occupation in occupations
        ],
        dtype=int,
    )
    # FermiHubbardHamiltonian is a fixed-sector container; checking this makes
    # its particle-number metadata and determinant basis mutually verifiable.
    if not np.all(counts == np.asarray(targets)[None, :]):
        raise ValueError("Fermionic Hamiltonian basis is not in its stated sector")

    # Begin the discrete representation of P_A P_B |Phi> with zero coefficients.
    projected = np.zeros_like(intrinsic)
    # Sum the first U(1) integral over equally spaced subsystem-A gauge angles.
    for index_a in range(chosen_grid[0]):
        # Shifted trapezoidal nodes still integrate a complete Fourier period.
        angle_a = 2 * np.pi * (index_a + offset) / chosen_grid[0]
        # Sum the second U(1) integral independently for subsystem B.
        for index_b in range(chosen_grid[1]):
            # Construct the second periodic gauge angle on its own exact grid.
            angle_b = 2 * np.pi * (index_b + offset) / chosen_grid[1]
            # The projector character exp[-i(N_a phi_a+N_b phi_b)] selects
            # the requested particle-number Fourier coefficient.
            character = np.exp(
                -1j * (targets[0] * angle_a + targets[1] * angle_b)
            )
            # A determinant with counts (n_a,n_b) acquires the gauge phase
            # exp[i(n_a phi_a+n_b phi_b)] under the rotated ket.
            rotated = intrinsic * np.exp(
                1j * (counts[:, 0] * angle_a + counts[:, 1] * angle_b)
            )
            # Add this quadrature node's character-weighted rotated ket.
            projected += character * rotated

    # Divide by both grid sizes to implement the normalized double integral.
    projected /= np.prod(chosen_grid)
    # Return the actual grid so benchmark output fully specifies the projector.
    return projected, chosen_grid


@dataclass
class NumberProjectionResult:
    """Normalized P_N P_Z state and PAV observables in the fermionic basis."""

    # The symmetry-breaking state produced by the preceding HF/HFB variation.
    intrinsic_state: HFBState
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


def project_particle_numbers(
    state,
    hamiltonian,
    target=None,
    grid=None,
    offset=0.137,
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
        exactly resolves every sector of a block containing ``c`` modes.
    offset : float, optional
        Common fractional shift of the full-period trapezoidal grids.
    """
    # Reject density-like stand-ins: projection requires the full quasiparticle
    # state because determinant amplitudes are not determined by rho alone.
    if not isinstance(state, HFBState):
        raise TypeError("state must be an HFBState")

    # Obtain the exact basis and Hamiltonian from the repository's canonical
    # fermionic Hamiltonian object rather than rebuilding either one here.
    occupations, masks, matrix = _require_fermionic_hamiltonian(hamiltonian)
    if hamiltonian.modes != len(state.U):
        raise ValueError("State and fermionic Hamiltonian use different modes")

    # Evaluate coefficients of the *normalized intrinsic state* on the
    # determinants retained by the fixed-N,Z Hamiltonian. HFBState selects
    # Pfaffians for paired vacua and determinants for collapsed HF states.
    intrinsic_amplitudes = state.occupation_amplitudes(
        occupations, normalized=True
    )

    # Apply P_N P_Z through the same finite, discretized group integral used by
    # the polynomial kernel implementation and by the J=0 combined projector.
    amplitudes, chosen_grid = _discrete_number_projector(
        intrinsic_amplitudes,
        occupations,
        hamiltonian.species_modes,
        hamiltonian.particle_numbers,
        grid=grid,
        offset=offset,
    )

    # The squared norm of this restricted vector is <Phi|P_N P_Z|Phi>, namely
    # the probability that the intrinsic state has the requested particle numbers.
    weight = float(np.vdot(amplitudes, amplitudes).real)
    if not np.isfinite(weight) or weight < 1e-14:
        raise ValueError("Intrinsic state has vanishing projected-sector weight")

    # Divide by sqrt(weight) to construct a unit vector representing
    # P_N P_Z|Phi>/sqrt(<Phi|P_N P_Z|Phi>).
    projected = amplitudes / np.sqrt(weight)

    # Apply the already assembled FermiHubbardHamiltonian matrix and form its
    # Rayleigh quotient. Hermiticity makes the result real up to roundoff.
    applied = matrix @ projected
    energy_value = np.vdot(projected, applied)
    if abs(energy_value.imag) > 1e-9:
        raise ValueError("Projected energy has a non-negligible imaginary part")

    # Fidelity is optional because PAV energy evaluation does not require an
    # exact eigenvector. If supplied, normalize the target defensively first.
    fidelity = None
    if target is not None:
        target = np.asarray(target, complex)
        if target.shape != projected.shape:
            raise ValueError("Target must match the fermionic Hamiltonian basis")
        target_norm = np.linalg.norm(target)
        if not np.isfinite(target_norm) or target_norm == 0:
            raise ValueError("Target must have finite nonzero norm")
        fidelity = float(
            abs(np.vdot(target / target_norm, projected)) ** 2
        )

    # Return state, observables, and basis identifiers together so a saved
    # projected vector cannot later be mistaken for a different basis ordering.
    return NumberProjectionResult(
        intrinsic_state=state,
        projected_vector=projected,
        sector_weight=weight,
        energy=float(energy_value.real),
        fidelity=fidelity,
        occupations=tuple(occupations),
        masks=masks.copy(),
        grid=tuple(int(points) for points in chosen_grid),
        grid_offset=float(offset),
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
