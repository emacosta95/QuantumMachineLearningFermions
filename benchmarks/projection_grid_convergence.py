"""Number/Euler projection-grid convergence for CKI or USDB nuclei.

For every pair ``M=1..M_max`` and ``J=1..J_max``, this script constructs the
Euler grid ``(M,J,M)``: M points for each azimuthal angle and J Gauss-Legendre
points for beta.  The neutron/proton Fourier grid is controlled separately.
It records fidelity with the exact fixed-(N,Z) ground state, <J^2>, the
associated effective J, and fermionic anti-flatness (FAF).  Undersized grids
are deliberate convergence probes and are marked as not guaranteed exact.
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from angular_momentum import (
    ParticleNumberJ0ProjectedEnergy,
    single_particle_angular_momentum,
)
from fermionic_antiflatness import fermionic_antiflatness
from hfb import HFBHamiltonian, ProjectionGridWarning, solve_hfb
from number_projection import exact_ground_state, projected_series_observables


def _load_interaction(name):
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
    path = ROOT / ("data/cki" if name == "cki" else "data/usdb.nat")
    interaction, eps = namespace["get_twobody_nuclearshell_model"](str(path))
    encoding = namespace["SingleParticleState"](str(path)).state_encoding
    return interaction, np.asarray(eps), encoding, path


def _nucleus(interaction_name, mass, species_modes):
    if interaction_name == "cki":
        valence_neutrons = mass - 6
        label = f"Be{mass}"
    else:
        valence_neutrons = mass - 18
        label = f"Ne{mass}"
    if valence_neutrons < 0 or valence_neutrons > species_modes:
        raise ValueError("Nucleus lies outside the selected valence space")
    return label, (valence_neutrons, 2)


def _one_body_many_body_matrix(fermionic, operator):
    dimension = len(fermionic.occupations)
    result = sparse.csr_matrix((dimension, dimension), dtype=complex)
    for row, column in zip(*np.nonzero(np.abs(operator) > 1e-14)):
        # The benchmark's isolated legacy loader intentionally omits the
        # optional Numba helper used by ``adag_a_matrix_optimized``.
        term = fermionic.adag_a_matrix(int(row), int(column))
        result = result + operator[row, column] * sparse.csr_matrix(term)
    return result.tocsr()


def _j_operators(fermionic, encoding):
    return tuple(
        _one_body_many_body_matrix(fermionic, generator)
        for generator in single_particle_angular_momentum(encoding)
    )


def _angular_observables(vector, operators):
    j2 = float(sum(
        np.vdot(applied, applied).real
        for applied in (operator @ vector for operator in operators)
    ))
    effective_j = 0.5 * (np.sqrt(max(0.0, 1.0 + 4.0 * j2)) - 1.0)
    return j2, float(effective_j)


def main(
    *,
    interaction_name="cki",
    mass=8,
    m_max=3,
    j_max=3,
    intrinsic_method="hf",
    starts=8,
    maxiter=500,
    seed=8,
    faf_order=2,
    allow_unconverged_intrinsic=False,
    number_grid=None,
    output_dir=None,
):
    if m_max < 1 or j_max < 1:
        raise ValueError("m_max and j_max must be positive")
    started = time.perf_counter()
    interaction, eps, encoding, interaction_path = _load_interaction(
        interaction_name
    )
    modes = len(eps)
    species_modes = modes // 2
    label, targets = _nucleus(interaction_name, int(mass), species_modes)
    neutron_modes = list(range(species_modes, modes))
    intrinsic_hamiltonian = HFBHamiltonian(np.diag(eps), interaction)
    # Rotated states and J operators need the complete fixed-(N,Z) basis, but
    # the expensive exact diagonalization only needs M=0.  Embed that exact
    # eigenvector back into the full basis for fidelities and J observables.
    fermionic = build_fermionic_hamiltonian(
        interaction, eps, particles=targets
    )
    print(
        f"[{label}] full projection basis: {len(fermionic.occupations)} states",
        flush=True,
    )
    magnetic_projections = np.asarray(
        [float(state[3]) for state in encoding]
    )

    def m_zero(occupied):
        return abs(float(np.sum(
            magnetic_projections[list(occupied)]
        ))) < 1e-10

    exact_fermionic = build_fermionic_hamiltonian(
        interaction, eps, particles=targets, symmetries=[m_zero]
    )
    print(
        f"[{label}] exact M=0 diagonalization: "
        f"{len(exact_fermionic.occupations)} states",
        flush=True,
    )
    exact_energy, exact_m0_state = exact_ground_state(exact_fermionic)
    exact_state = np.zeros(len(fermionic.occupations), complex)
    full_indices = {
        tuple(occupied): index
        for index, occupied in enumerate(fermionic.occupations)
    }
    for coefficient, occupied in zip(
        exact_m0_state, exact_fermionic.occupations
    ):
        exact_state[full_indices[tuple(occupied)]] = coefficient
    exact_faf = fermionic_antiflatness(
        exact_state,
        fermionic.occupations,
        fermionic.modes,
        order=faf_order,
    )
    print(
        f"[{label}] optimizing {intrinsic_method.upper()} with {starts} starts",
        flush=True,
    )
    intrinsic = solve_hfb(
        intrinsic_hamiltonian,
        neutron_modes,
        targets,
        starts=starts,
        seed=seed,
        maxiter=maxiter,
        tolerance=1e-8,
        analytic_jacobian=(intrinsic_method == "hfb"),
        method=intrinsic_method,
    )
    if not intrinsic.converged and not allow_unconverged_intrinsic:
        raise RuntimeError(
            "Intrinsic optimization did not converge; use "
            "--allow-unconverged-intrinsic only for diagnostics"
        )
    angular_operators = _j_operators(fermionic, encoding)
    if number_grid is None and intrinsic_method == "hf":
        # A species-conserving determinant already has exact N,Z; one gauge
        # representative is therefore sufficient even though the generic HFB
        # finite-space guarantee still reports the formal c+1 bound.
        number_grid = (1, 1)
    result_directory = (
        ROOT / "benchmarks/results"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    result_directory.mkdir(parents=True, exist_ok=True)
    output = result_directory / (
        f"{interaction_name}_{label.lower()}_projection_grid_convergence.json"
    )
    report = {
        "status": "running",
        "interaction": interaction_name,
        "interaction_path": str(interaction_path.relative_to(ROOT)),
        "nucleus": label,
        "targets": list(targets),
        "intrinsic_method": intrinsic_method,
        "intrinsic_converged": bool(intrinsic.converged),
        "intrinsic_energy": intrinsic.energy,
        "intrinsic_numbers": intrinsic.numbers.tolist(),
        "intrinsic_pairing_norm": float(np.linalg.norm(intrinsic.state.kappa)),
        "intrinsic_attempts": intrinsic.attempts,
        "exact_energy": exact_energy,
        "exact_diagonalization_sector": "fixed (N,Z,M=0)",
        "exact_dimension_NZ_M0": len(exact_fermionic.occupations),
        "projection_dimension_NZ": len(fermionic.occupations),
        "exact_ground_state_faf": exact_faf.value,
        "exact_ground_state_faf_per_mode": exact_faf.value_per_mode,
        "faf_order": int(faf_order),
        "grid_rows": [],
        "elapsed_seconds": 0.0,
    }
    rows = report["grid_rows"]
    for magnetic_points in range(1, m_max + 1):
        for angular_points in range(1, j_max + 1):
            row = {
                "M_grid_points": magnetic_points,
                "J_grid_points": angular_points,
                "number_grid": (
                    None if number_grid is None else list(number_grid)
                ),
                "euler_grid": [magnetic_points, angular_points, magnetic_points],
            }
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ProjectionGridWarning)
                    projector = ParticleNumberJ0ProjectedEnergy(
                        intrinsic_hamiltonian,
                        encoding,
                        neutron_modes,
                        targets,
                        number_grid=number_grid,
                        euler_grid=(
                            magnetic_points,
                            angular_points,
                            magnetic_points,
                        ),
                        allow_inexact_number_grid=True,
                        allow_inexact_euler_grid=True,
                    )
                    series = projector.projected_series(intrinsic.state)
                projected = projected_series_observables(
                    series, fermionic, exact_state
                )
                faf = fermionic_antiflatness(
                    projected.projected_vector,
                    fermionic.occupations,
                    fermionic.modes,
                    order=faf_order,
                )
                j2, effective_j = _angular_observables(
                    projected.projected_vector, angular_operators
                )
                row.update({
                    "components": series.number_of_vacua,
                    "number_projection_redundant_for_hf":
                        intrinsic_method == "hf",
                    "number_grid_guaranteed_exact":
                        series.number_grid_guaranteed_exact,
                    "euler_grid_guaranteed_exact":
                        series.euler_grid_guaranteed_exact,
                    "minimum_number_grid": list(series.minimum_number_grid),
                    "minimum_euler_grid": list(series.minimum_euler_grid),
                    "projected_energy": projected.energy,
                    "projected_energy_relative_error": float(
                        abs((projected.energy - exact_energy) / exact_energy)
                    ),
                    "fidelity": projected.fidelity,
                    "J2_expectation": j2,
                    "effective_J": effective_j,
                    "faf": faf.value,
                    "faf_per_mode": faf.value_per_mode,
                    "faf_difference_from_exact_ground_state": float(
                        faf.value - exact_faf.value
                    ),
                    "faf_ratio_to_exact_ground_state": (
                        float(faf.value / exact_faf.value)
                        if abs(exact_faf.value) > 1e-14 else None
                    ),
                })
            except ValueError as error:
                row["error"] = str(error)
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
            report["elapsed_seconds"] = time.perf_counter() - started
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction", choices=("cki", "usdb"), default="cki")
    parser.add_argument("--mass", type=int, default=8)
    parser.add_argument("--m-max", type=int, default=3)
    parser.add_argument("--j-max", type=int, default=3)
    parser.add_argument("--intrinsic-method", choices=("hf", "hfb"), default="hf")
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--faf-order", type=int, default=2)
    parser.add_argument(
        "--number-grid", nargs=2, type=int, metavar=("LN", "LZ")
    )
    parser.add_argument("--allow-unconverged-intrinsic", action="store_true")
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    main(
        interaction_name=arguments.interaction,
        mass=arguments.mass,
        m_max=arguments.m_max,
        j_max=arguments.j_max,
        intrinsic_method=arguments.intrinsic_method,
        starts=arguments.starts,
        maxiter=arguments.maxiter,
        seed=arguments.seed,
        faf_order=arguments.faf_order,
        allow_unconverged_intrinsic=arguments.allow_unconverged_intrinsic,
        number_grid=(
            tuple(arguments.number_grid) if arguments.number_grid else None
        ),
        output_dir=arguments.output_dir,
    )
