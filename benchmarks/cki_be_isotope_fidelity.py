"""Controlled CKI Be isotope test for HF and maximum-overlap Slater states.

The script keeps proton and neutron orbitals separate in the HF and overlap
optimizations, so every determinant has exact N and Z.  J=0 projection is
performed independently by diagonalizing J^2 in the fixed-(N,Z) basis.  This
is intended as a small-space diagnostic, not as the production projector.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from angular_momentum import polynomial_j0_grid, single_particle_angular_momentum
from gaussian_fidelity import (
    _slater_value_gradient,
    maximize_gaussian_fidelity,
    maximize_slater_fidelity,
)
from hfb import HFBHamiltonian
from number_projection import exact_ground_state


def _density(orbitals):
    rho = orbitals @ orbitals.conj().T
    return type("SlaterDensity", (), {
        "rho": rho,
        "kappa": np.zeros_like(rho),
    })()


def _combine(proton_orbitals, neutron_orbitals):
    """Embed species orbitals in the CKI ordering: protons, then neutrons."""
    zp = proton_orbitals.shape[1]
    nn = neutron_orbitals.shape[1]
    result = np.zeros((12, zp + nn), complex)
    result[:6, :zp] = proton_orbitals
    result[6:, zp:] = neutron_orbitals
    return result


def _random_frame(rng, rows, columns):
    if columns == rows:
        return np.eye(rows, dtype=complex)
    matrix = rng.normal(size=(rows, columns)) + 1j * rng.normal(
        size=(rows, columns)
    )
    return np.linalg.qr(matrix, mode="reduced")[0]


def _retract(frame, tangent, step):
    if frame.shape[0] == frame.shape[1]:
        return frame
    return np.linalg.qr(frame + step * tangent, mode="reduced")[0]


def _tangent(frame, gradient):
    if frame.shape[0] == frame.shape[1]:
        return np.zeros_like(frame)
    overlap = frame.conj().T @ gradient
    return gradient - frame @ ((overlap + overlap.conj().T) / 2)


def minimize_species_hf(hamiltonian, neutrons, protons=2, *, starts=12,
                        seed=0, maxiter=2500, tolerance=2e-8):
    """Minimize Slater energy on St(6,Z) x St(6,N)."""
    rng = np.random.default_rng(seed)
    candidates = []
    attempts = []
    for attempt in range(starts):
        cp = _random_frame(rng, 6, protons)
        cn = _random_frame(rng, 6, neutrons)
        c = _combine(cp, cn)
        value = hamiltonian.energy(_density(c))
        residual = np.inf
        for iteration in range(maxiter):
            rho = c @ c.conj().T
            fock = hamiltonian.h + np.einsum(
                "ajbl,lj->ab", hamiltonian.v, rho, optimize=True
            )
            gp = _tangent(cp, 2 * fock[:6, :6] @ cp)
            gn = _tangent(cn, 2 * fock[6:, 6:] @ cn)
            residual = float(np.sqrt(np.linalg.norm(gp) ** 2 +
                                     np.linalg.norm(gn) ** 2))
            if residual <= tolerance:
                break
            step = min(0.25, 0.5 / max(residual, 1e-12))
            accepted = False
            for _ in range(35):
                trial_p = _retract(cp, -gp, step)
                trial_n = _retract(cn, -gn, step)
                trial_c = _combine(trial_p, trial_n)
                trial_value = hamiltonian.energy(_density(trial_c))
                if trial_value <= value - 1e-4 * step * residual ** 2:
                    cp, cn, c, value = trial_p, trial_n, trial_c, trial_value
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
        attempts.append({
            "energy": float(value),
            "gradient_norm": residual,
            "iterations": iteration + 1,
            "converged": bool(residual <= tolerance),
        })
        candidates.append((value, residual, c.copy()))
    minimum_energy = min(item[0] for item in candidates)
    energy_degenerate = [item for item in candidates
                         if item[0] <= minimum_energy + 1e-10]
    value, residual, orbitals = min(energy_degenerate, key=lambda item: item[1])
    return orbitals, float(value), float(residual), attempts


def maximize_species_slater(hamiltonian, target, neutrons, protons=2, *,
                            initial_orbitals=None, starts=12, seed=0,
                            maxiter=3000, tolerance=2e-8):
    """Maximize fidelity while imposing exact species numbers."""
    occupations = np.asarray(hamiltonian.occupations, dtype=int)
    target = np.asarray(target, complex) / np.linalg.norm(target)
    rng = np.random.default_rng(seed)
    seeds = []
    if initial_orbitals is not None:
        seeds.append((initial_orbitals[:6, :protons],
                      initial_orbitals[6:, protons:]))
    dominant = occupations[int(np.argmax(abs(target)))]
    proton_rows = [row for row in dominant if row < 6]
    neutron_rows = [row - 6 for row in dominant if row >= 6]
    cp = np.zeros((6, protons), complex)
    cn = np.zeros((6, neutrons), complex)
    cp[proton_rows, range(protons)] = 1
    cn[neutron_rows, range(neutrons)] = 1
    seeds.append((cp, cn))
    while len(seeds) < starts:
        seeds.append((_random_frame(rng, 6, protons),
                      _random_frame(rng, 6, neutrons)))

    attempts = []
    candidates = []
    for cp, cn in seeds[:starts]:
        c = _combine(cp, cn)
        value = 0.0
        residual = np.inf
        for iteration in range(maxiter):
            value, gradient = _slater_value_gradient(c, occupations, target)
            gp = _tangent(cp, gradient[:6, :protons])
            gn = _tangent(cn, gradient[6:, protons:])
            residual = float(np.sqrt(np.linalg.norm(gp) ** 2 +
                                     np.linalg.norm(gn) ** 2))
            if residual <= tolerance:
                break
            step = min(1.0, 1.0 / max(residual, 1e-12))
            accepted = False
            for _ in range(35):
                trial_p = _retract(cp, gp, step)
                trial_n = _retract(cn, gn, step)
                trial_c = _combine(trial_p, trial_n)
                trial_value, _ = _slater_value_gradient(
                    trial_c, occupations, target
                )
                if trial_value >= value + 1e-4 * step * residual ** 2:
                    cp, cn, c, value = trial_p, trial_n, trial_c, trial_value
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
        attempts.append({
            "fidelity": float(value),
            "gradient_norm": residual,
            "iterations": iteration + 1,
            "converged": bool(residual <= tolerance),
        })
        candidates.append((value, residual, c.copy()))
    value, residual, orbitals = max(candidates, key=lambda item: item[0])
    return orbitals, float(value), float(residual), attempts


def determinant_vector(orbitals, occupations):
    vector = np.linalg.det(orbitals[np.asarray(occupations, dtype=int)])
    return vector / np.linalg.norm(vector)


def one_body_matrix(fermionic, operator):
    result = np.zeros(fermionic.matrix.shape, complex)
    for i, j in zip(*np.nonzero(abs(operator) > 1e-14)):
        result += operator[i, j] * fermionic.adag_a_matrix(i, j).toarray()
    return result


def j0_projector(fermionic, state_encoding):
    generators = single_particle_angular_momentum(state_encoding)
    many_body = [one_body_matrix(fermionic, generator)
                 for generator in generators]
    j2 = sum(generator @ generator for generator in many_body)
    values, vectors = np.linalg.eigh((j2 + j2.conj().T) / 2)
    zero = values < 1e-8
    if not np.any(zero):
        raise RuntimeError("No J=0 subspace found")
    basis = vectors[:, zero]
    return basis @ basis.conj().T, values, int(np.count_nonzero(zero))


def observables(vector, target, projector):
    raw = float(abs(np.vdot(target, vector)) ** 2)
    projected = projector @ vector
    weight = float(np.vdot(projected, projected).real)
    fidelity = (float(abs(np.vdot(target, projected)) ** 2 / weight)
                if weight > 1e-13 else None)
    return raw, weight, fidelity


def projected_mutual_fidelity(left, right, projector):
    left = projector @ left
    right = projector @ right
    return float(abs(np.vdot(left, right)) ** 2 /
                 (np.vdot(left, left).real * np.vdot(right, right).real))


def main(*, full_gaussian=False, gaussian_starts=8, gaussian_maxiter=1200):
    started = time.perf_counter()
    namespace = dict(globals(), trange=range)
    legacy_definitions(
        "cg_utils.py",
        ["CG", "ClebschGordan", "SelectCG", "CreateInitialCGList",
         "CalcInitialValues", "DivCalc", "CgJM"],
        namespace,
    )
    legacy_definitions(
        "nuclear_physics_utils.py",
        ["SingleParticleState", "krond", "scattering_matrix_reader",
         "compute_nuclear_twobody_matrix", "get_twobody_nuclearshell_model"],
        namespace,
    )
    interaction, eps = namespace["get_twobody_nuclearshell_model"](
        str(ROOT / "data/cki")
    )
    encoding = namespace["SingleParticleState"](
        str(ROOT / "data/cki")
    ).state_encoding
    intrinsic = HFBHamiltonian(np.diag(eps), interaction)
    rows = []
    for mass, neutrons in ((8, 2), (10, 4), (12, 6)):
        fermionic = build_fermionic_hamiltonian(
            interaction, eps, particles=(neutrons, 2)
        )
        exact_energy, target = exact_ground_state(fermionic)
        projector, j2_values, j0_dimension = j0_projector(fermionic, encoding)
        target_j2 = float(np.vdot(target, projector @ target).real)

        hf, hf_energy, hf_gradient, hf_attempts = minimize_species_hf(
            intrinsic, neutrons, starts=12, seed=100 + mass
        )
        hf_vector = determinant_vector(hf, fermionic.occupations)
        hf_raw, hf_j0_weight, hf_projected = observables(
            hf_vector, target, projector
        )

        closest, closest_raw, closest_gradient, closest_attempts = (
            maximize_species_slater(
                fermionic, target, neutrons, initial_orbitals=hf,
                starts=12, seed=200 + mass
            )
        )
        closest_vector = determinant_vector(closest, fermionic.occupations)
        closest_raw_check, closest_j0_weight, closest_projected = observables(
            closest_vector, target, projector
        )
        hf_closest_raw = float(abs(np.linalg.det(hf.conj().T @ closest)) ** 2)
        hf_closest_j0 = projected_mutual_fidelity(
            hf_vector, closest_vector, projector
        )
        closest_energy = intrinsic.energy(_density(closest))

        # The Slater solution is already a pure Gaussian boundary state.  An
        # optional finite-Thouless search completes the interior of the even
        # Gaussian manifold and selects whichever raw-overlap candidate wins.
        gaussian_kind = (
            "species_conserving_slater_boundary"
            if full_gaussian
            else "species_conserving_slater_boundary_only"
        )
        gaussian_raw_coefficients = closest_vector
        gaussian_raw_fidelity = closest_raw
        gaussian_energy = closest_energy
        gaussian_converged = True
        gaussian_gradient = closest_gradient
        gaussian_attempts = []
        if full_gaussian:
            interior = maximize_gaussian_fidelity(
                fermionic,
                target,
                starts=gaussian_starts,
                seed=400 + mass,
                maxiter=gaussian_maxiter,
                gradient_tolerance=2e-6,
            )
            gaussian_attempts = interior.attempts
            if interior.fidelity > gaussian_raw_fidelity:
                gaussian_kind = "interior_thouless"
                gaussian_raw_coefficients = (
                    interior.state.occupation_amplitudes(
                        fermionic.occupations, normalized=True
                    )
                )
                gaussian_raw_fidelity = interior.fidelity
                gaussian_energy = intrinsic.energy(interior.state)
                gaussian_converged = bool(interior.converged)
                gaussian_gradient = interior.gradient_norm

        gaussian_sector_weight = float(
            np.vdot(
                gaussian_raw_coefficients, gaussian_raw_coefficients
            ).real
        )
        gaussian_raw_mutual_hf = float(
            abs(np.vdot(hf_vector, gaussian_raw_coefficients)) ** 2
        )
        gaussian_raw_gs_check, gaussian_j0_weight, gaussian_projected_gs = (
            observables(gaussian_raw_coefficients, target, projector)
        )
        gaussian_projected_mutual_hf = projected_mutual_fidelity(
            hf_vector, gaussian_raw_coefficients, projector
        )

        unrestricted = maximize_slater_fidelity(
            fermionic, target, starts=12, seed=300 + mass,
            maxiter=3000, gradient_tolerance=2e-7,
            initial_orbitals=hf,
        )
        unrestricted_vector = determinant_vector(
            unrestricted.orbitals, fermionic.occupations
        )
        unrestricted_sector_weight = float(
            np.vdot(unrestricted_vector, unrestricted_vector).real
        )
        # determinant_vector normalizes inside the target sector; use the raw
        # objective reported by the optimizer for the actual full-state overlap.

        grid = polynomial_j0_grid(
            encoding, list(range(6, 12)), (neutrons, 2)
        ).minimum_grid
        rows.append({
            "nucleus": f"Be{mass}",
            "valence_neutrons": neutrons,
            "valence_protons": 2,
            "dimension_NZ": len(fermionic.occupations),
            "dimension_J0": j0_dimension,
            "minimum_euler_grid": list(grid),
            "exact_energy": exact_energy,
            "exact_target_J0_weight": target_j2,
            "hf_energy": hf_energy,
            "hf_energy_error": hf_energy - exact_energy,
            "hf_gradient_norm": hf_gradient,
            "hf_raw_fidelity": hf_raw,
            "hf_J0_weight": hf_j0_weight,
            "hf_J0_projected_fidelity": hf_projected,
            "closest_species_slater_raw_fidelity": closest_raw,
            "closest_species_slater_raw_crosscheck": closest_raw_check,
            "closest_species_slater_gradient_norm": closest_gradient,
            "closest_species_slater_J0_weight": closest_j0_weight,
            "closest_species_slater_J0_projected_fidelity": closest_projected,
            "closest_species_slater_energy": closest_energy,
            "hf_closest_species_slater_raw_mutual_fidelity": hf_closest_raw,
            "hf_closest_species_slater_J0_mutual_fidelity": hf_closest_j0,
            "unrestricted_slater_raw_fidelity": unrestricted.fidelity,
            "unrestricted_slater_converged": bool(unrestricted.converged),
            "unrestricted_slater_gradient_norm": unrestricted.gradient_norm,
            "unrestricted_target_sector_vector_norm_check":
                unrestricted_sector_weight,
            # Canonical systematic-study names requested for plotting.
            "gaussian_max_kind": gaussian_kind,
            "fidelity_HF_GS_raw": hf_raw,
            "fidelity_GaussianMax_GS_raw": gaussian_raw_fidelity,
            "fidelity_GaussianMax_GS_raw_crosscheck": gaussian_raw_gs_check,
            "fidelity_GaussianMax_HF_raw": gaussian_raw_mutual_hf,
            "fidelity_HF_GS_J0": hf_projected,
            "fidelity_GaussianMax_GS_J0": gaussian_projected_gs,
            "fidelity_GaussianMax_HF_J0": gaussian_projected_mutual_hf,
            "gaussian_max_target_NZ_sector_weight": gaussian_sector_weight,
            "gaussian_max_J0_weight": gaussian_j0_weight,
            "gaussian_max_intrinsic_energy": gaussian_energy,
            "gaussian_max_converged": gaussian_converged,
            "gaussian_max_gradient_norm": gaussian_gradient,
            "gaussian_interior_search_run": bool(full_gaussian),
            "gaussian_interior_attempts": gaussian_attempts,
            "hf_attempts": hf_attempts,
            "closest_species_slater_attempts": closest_attempts,
            "unrestricted_slater_attempts": unrestricted.attempts,
            "lowest_J2_eigenvalues": j2_values[:min(12, len(j2_values))].tolist(),
        })
        print(json.dumps(rows[-1], indent=2), flush=True)

    report = {
        "method": "species-conserving Slater HF and exact fixed-space J2 projection",
        "interaction": "data/cki",
        "results": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = ROOT / "benchmarks/results/cki_be_isotope_fidelity.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-gaussian",
        action="store_true",
        help="also search the finite-Thouless Gaussian interior",
    )
    parser.add_argument("--gaussian-starts", type=int, default=8)
    parser.add_argument("--gaussian-maxiter", type=int, default=1200)
    arguments = parser.parse_args()
    main(
        full_gaussian=arguments.full_gaussian,
        gaussian_starts=arguments.gaussian_starts,
        gaussian_maxiter=arguments.gaussian_maxiter,
    )
