"""Fermionic anti-flatness for pure states in an occupation basis.

For ``L`` complex fermion modes and Majorana covariance matrix

    M_mn = -i <[gamma_m, gamma_n]> / 2,

the order-``k`` fermionic anti-flatness is

    F_k = L - Tr[(M.T @ M)**k] / 2.

It vanishes for every pure fermionic Gaussian state.  The routines below also
accept a coherent :class:`BogoliubovVacuumSeries`; its determinant amplitudes
are summed before the covariance matrix is evaluated, so interference between
different transformed vacua is retained.
"""

from dataclasses import dataclass

import numpy as np

if __package__:
    from .hfb import BogoliubovVacuumSeries
else:
    from hfb import BogoliubovVacuumSeries


@dataclass
class FermionicAntiflatnessResult:
    """Covariance data and fermionic anti-flatness of a normalized pure state."""

    value: float
    value_per_mode: float
    order: int
    modes: int
    covariance: np.ndarray
    covariance_squared_spectrum: np.ndarray
    state_norm: float


def _validated_state(coefficients, occupations, modes):
    """Normalize coefficients and return their distinct occupation masks."""
    coefficients = np.asarray(coefficients, dtype=complex)
    occupations = tuple(tuple(int(mode) for mode in row) for row in occupations)
    if coefficients.shape != (len(occupations),):
        raise ValueError("Coefficients and occupations have incompatible sizes")
    if not np.isfinite(coefficients).all():
        raise ValueError("State coefficients must be finite")
    if not isinstance(modes, (int, np.integer)) or modes < 1:
        raise ValueError("modes must be a positive integer")

    masks = []
    for occupied in occupations:
        if occupied != tuple(sorted(occupied)) or len(set(occupied)) != len(occupied):
            raise ValueError("Each occupation must contain distinct ordered modes")
        if any(mode < 0 or mode >= modes for mode in occupied):
            raise ValueError("Occupation contains a mode outside the one-body space")
        masks.append(sum(1 << mode for mode in occupied))
    if len(set(masks)) != len(masks):
        raise ValueError("Occupation configurations must be distinct")

    norm = float(np.vdot(coefficients, coefficients).real)
    if not np.isfinite(norm) or norm < 1e-24:
        raise ValueError("State has vanishing norm")
    return coefficients / np.sqrt(norm), tuple(masks), norm


def majorana_covariance(coefficients, occupations, modes):
    """Return the real Majorana covariance matrix of a pure fermionic state.

    ``occupations`` need only list configurations with nonzero amplitudes, but
    together they must contain the complete support of the state being studied.
    This is particularly useful for fixed-particle-number exact eigenvectors and
    exactly number-projected Bogoliubov-vacuum series.
    """
    coefficients, masks, _ = _validated_state(coefficients, occupations, modes)

    # gamma_(2j)|psi> and gamma_(2j+1)|psi> occupy the adjacent number sectors.
    # Store these sparse vectors first, then form their Gram matrix.  This avoids
    # allocating the full 2**modes Fock space when the supplied state is sparse.
    transformed = [dict() for _ in range(2 * modes)]
    for amplitude, mask in zip(coefficients, masks):
        for mode in range(modes):
            lower_mask = (1 << mode) - 1
            sign = -1.0 if (mask & lower_mask).bit_count() % 2 else 1.0
            occupied = bool(mask & (1 << mode))
            destination = (
                mask & ~(1 << mode) if occupied else mask | (1 << mode)
            )
            even_factor = sign
            # gamma_(2j+1) = -i(c_j-c_j^dagger).
            odd_factor = (-1j if occupied else 1j) * sign
            transformed[2 * mode][destination] = (
                transformed[2 * mode].get(destination, 0j)
                + even_factor * amplitude
            )
            transformed[2 * mode + 1][destination] = (
                transformed[2 * mode + 1].get(destination, 0j)
                + odd_factor * amplitude
            )

    destinations = sorted({mask for vector in transformed for mask in vector})
    destination_index = {mask: index for index, mask in enumerate(destinations)}
    gamma_states = np.zeros((2 * modes, len(destinations)), dtype=complex)
    for majorana, vector in enumerate(transformed):
        for mask, amplitude in vector.items():
            gamma_states[majorana, destination_index[mask]] = amplitude

    # G_mn=<gamma_m psi|gamma_n psi>=<psi|gamma_m gamma_n|psi>.
    gram = gamma_states.conj() @ gamma_states.T
    covariance = -0.5j * (gram - gram.T)
    if np.max(np.abs(covariance.imag)) > 1e-10:
        raise ValueError("Majorana covariance has a non-negligible imaginary part")
    covariance = covariance.real
    # Remove roundoff while enforcing the defining real antisymmetry exactly.
    return 0.5 * (covariance - covariance.T)


def fermionic_antiflatness(coefficients, occupations, modes, *, order=2):
    """Compute order-``k`` fermionic anti-flatness from state amplitudes."""
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    _, _, state_norm = _validated_state(coefficients, occupations, modes)
    covariance = majorana_covariance(coefficients, occupations, modes)
    covariance_squared = covariance.T @ covariance
    spectrum = np.linalg.eigvalsh(
        0.5 * (covariance_squared + covariance_squared.T)
    )
    if spectrum.min(initial=0.0) < -1e-9 or spectrum.max(initial=0.0) > 1 + 1e-8:
        raise ValueError("Covariance singular values lie outside the physical range")
    spectrum = np.clip(spectrum, 0.0, 1.0)
    value = float(modes - 0.5 * np.sum(spectrum ** int(order)))
    if value < 0 and value > -1e-10:
        value = 0.0
    return FermionicAntiflatnessResult(
        value=value,
        value_per_mode=value / modes,
        order=int(order),
        modes=int(modes),
        covariance=covariance,
        covariance_squared_spectrum=spectrum,
        state_norm=state_norm,
    )


def vacuum_series_antiflatness(series, occupations, *, order=2):
    """Compute anti-flatness after coherently summing a vacuum series."""
    if not isinstance(series, BogoliubovVacuumSeries):
        raise TypeError("series must be a BogoliubovVacuumSeries")
    amplitudes = series.occupation_amplitudes(occupations)
    return fermionic_antiflatness(
        amplitudes,
        occupations,
        len(series.intrinsic_state.U),
        order=order,
    )
