"""USDB Ne-isotope PAV fidelity and fermionic anti-flatness study.

The O16 core leaves two valence protons and A-18 valence neutrons for Ne-A.
Number projection is collapsed analytically in the fixed-(N,Z) determinant
basis, so only Euler-rotated vacua are materialized. This is essential in the
24-mode sd shell, where an explicit 13x13 gauge product is unnecessarily large.
"""

import argparse
import json
import time
from dataclasses import replace
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from angular_momentum import (
    euler_rotation,
    polynomial_j0_grid,
    single_particle_angular_momentum,
)
from fermionic_antiflatness import fermionic_antiflatness
from hfb import BogoliubovVacuumSeries, HFBHamiltonian, HFBState, solve_hfb
from number_projection import exact_ground_state, projected_series_observables


def _working_state(state, particles):
    rho = (state.rho + state.rho.conj().T) / 2
    idempotency = float(np.linalg.norm(rho @ rho - rho))
    pairing = float(np.linalg.norm(state.kappa))
    if pairing < 1e-3 and idempotency < 1e-6:
        _, orbitals = np.linalg.eigh(rho)
        return (
            HFBState.from_slater(orbitals[:, -particles:]),
            "Slater (pairing collapsed)",
            pairing,
            idempotency,
        )
    # A fully occupied species can make the global U singular even when another
    # species remains paired. The present amplitude API cannot represent that
    # partial particle-hole chart, so fail with a targeted explanation.
    try:
        state.thouless_matrix
    except ValueError as error:
        raise ValueError(
            "HFB state has a singular mixed Slater/paired chart; use another "
            "seed or implement a blocked Pfaffian amplitude chart"
        ) from error
    return state, "paired Bogoliubov vacuum", pairing, idempotency


def _one_term_series(state, modes, species_modes):
    """Represent exact fixed-sector P_N P_Z projection without gauge copies."""
    return BogoliubovVacuumSeries(
        intrinsic_state=state,
        transformations=np.eye(modes, dtype=complex)[None, :, :],
        weights=np.ones(1, dtype=complex),
        number_grid=(species_modes + 1, species_modes + 1),
        projection="P_N P_Z (fixed-sector gauge sum collapsed)",
        number_grid_guaranteed_exact=True,
        minimum_number_grid=(species_modes + 1, species_modes + 1),
    )


def _j0_series(state, encoding, neutron_modes, targets, grid=None):
    """Build explicit full-space Euler-rotated Gaussian components."""
    rule = polynomial_j0_grid(
        encoding,
        neutron_modes,
        targets,
        grid=grid,
        allow_inexact_grid=(grid is not None),
    )
    generators = single_particle_angular_momentum(encoding)
    transformations = []
    weights = []
    for alpha in rule.alpha:
        for cos_beta, beta_weight in zip(rule.cos_beta, rule.beta_weights):
            beta = np.arccos(cos_beta)
            for gamma in rule.gamma:
                transformations.append(
                    euler_rotation(alpha, beta, gamma, generators)
                )
                weights.append(
                    beta_weight
                    / (2 * len(rule.alpha) * len(rule.gamma))
                )
    species_modes = len(encoding) // 2
    return BogoliubovVacuumSeries(
        intrinsic_state=state,
        transformations=np.asarray(transformations),
        weights=np.asarray(weights, complex),
        number_grid=(species_modes + 1, species_modes + 1),
        euler_grid=(len(rule.alpha), len(rule.cos_beta), len(rule.gamma)),
        projection="P_N P_Z P_J=0 (fixed-sector gauge sum collapsed)",
        euler_offset=0.173,
        number_grid_guaranteed_exact=True,
        euler_grid_guaranteed_exact=rule.guaranteed_exact,
        minimum_number_grid=(species_modes + 1, species_modes + 1),
        minimum_euler_grid=rule.minimum_grid,
    )


def _counts(total, maximum_points=10):
    candidates = {1, total}
    value = 2
    while value < total:
        candidates.add(value)
        value *= 2
    ordered = sorted(candidates)
    if len(ordered) <= maximum_points:
        return tuple(ordered)
    indices = np.linspace(0, len(ordered) - 1, maximum_points, dtype=int)
    return tuple(ordered[index] for index in sorted(set(indices)))


def _prefix_rows(series, fermionic, target, order):
    target = np.asarray(target, complex) / np.linalg.norm(target)
    rows = []
    for count in _counts(series.number_of_vacua):
        prefix = replace(
            series,
            transformations=series.transformations[:count],
            weights=series.weights[:count],
            euler_grid_guaranteed_exact=(
                series.euler_grid_guaranteed_exact
                and count == series.number_of_vacua
            ),
        )
        amplitudes = prefix.occupation_amplitudes(fermionic.occupations)
        norm = float(np.vdot(amplitudes, amplitudes).real)
        if norm < 1e-14:
            rows.append({"components": count, "state_norm": norm})
            continue
        state = amplitudes / np.sqrt(norm)
        faf = fermionic_antiflatness(
            state, fermionic.occupations, fermionic.modes, order=order
        )
        rows.append({
            "components": count,
            "state_norm": norm,
            "fidelity": float(abs(np.vdot(target, state)) ** 2),
            "faf": faf.value,
            "faf_per_mode": faf.value_per_mode,
        })
    return rows


def main(
    *,
    isotopes=(20,),
    starts=2,
    maxiter=120,
    order=2,
    seed=8,
    euler_grid=None,
    exact_only=False,
    analytic_jacobian=True,
):
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
    interaction_path = ROOT / "data/usdb.nat"
    interaction, eps = namespace["get_twobody_nuclearshell_model"](
        str(interaction_path)
    )
    encoding = namespace["SingleParticleState"](
        str(interaction_path)
    ).state_encoding
    modes = len(eps)
    species_modes = modes // 2
    neutron_modes = list(range(species_modes, modes))
    intrinsic_hamiltonian = HFBHamiltonian(np.diag(eps), interaction)
    results = []

    for mass in isotopes:
        valence_neutrons = int(mass) - 18
        targets = (valence_neutrons, 2)
        if valence_neutrons < 0 or valence_neutrons > species_modes:
            raise ValueError("Ne isotope lies outside the O16-core sd shell")
        isotope_started = time.perf_counter()
        magnetic_projections = np.asarray(
            [float(state[3]) for state in encoding]
        )

        def m_zero(occupied):
            return abs(float(np.sum(magnetic_projections[list(occupied)]))) < 1e-10

        fermionic = build_fermionic_hamiltonian(
            interaction, eps, particles=targets, symmetries=[m_zero]
        )
        exact_energy, target = exact_ground_state(fermionic)
        exact_faf = fermionic_antiflatness(
            target, fermionic.occupations, fermionic.modes, order=order
        )
        exact_row = {
            "nucleus": f"Ne{mass}",
            "valence_neutrons": valence_neutrons,
            "valence_protons": 2,
            "modes": modes,
            "dimension_NZ_M0": len(fermionic.occupations),
            "basis_sector": "fixed (N,Z,M=0)",
            "exact_energy": exact_energy,
            "exact_ground_state_faf": exact_faf.value,
            "exact_ground_state_faf_per_mode": exact_faf.value_per_mode,
        }
        if exact_only:
            exact_row["elapsed_seconds"] = time.perf_counter() - isotope_started
            results.append(exact_row)
            print(json.dumps(exact_row, indent=2), flush=True)
            continue
        hfb = solve_hfb(
            intrinsic_hamiltonian,
            neutron_modes,
            targets,
            starts=starts,
            seed=seed,
            maxiter=maxiter,
            tolerance=1e-8,
            stationarity_diagnostics=True,
            shared_finite_difference_jacobian=not analytic_jacobian,
            analytic_jacobian=analytic_jacobian,
        )
        state, chart, pairing, idempotency = _working_state(
            hfb.state, sum(targets)
        )
        number_series = _one_term_series(state, modes, species_modes)
        number_pav = projected_series_observables(
            number_series, fermionic, target
        )
        number_faf = fermionic_antiflatness(
            number_pav.projected_vector,
            fermionic.occupations,
            fermionic.modes,
            order=order,
        )
        angular_series = _j0_series(
            state, encoding, neutron_modes, targets, grid=euler_grid
        )
        angular_pav = projected_series_observables(
            angular_series, fermionic, target
        )
        trajectory = _prefix_rows(angular_series, fermionic, target, order)
        row = {
            **exact_row,
            "hfb_energy": hfb.energy,
            "hfb_converged": bool(hfb.converged),
            "hfb_pairing_norm": pairing,
            "hfb_rho_idempotency": idempotency,
            "projection_state_chart": chart,
            # These are explicitly conditioned on the exact diagonalization's
            # M=0 basis; they are not observables of the full number-only PAV.
            "number_and_M0_pav_energy": number_pav.energy,
            "number_and_M0_pav_fidelity": number_pav.fidelity,
            "number_and_M0_pav_faf": number_faf.value,
            "euler_grid": list(angular_series.euler_grid),
            "minimum_euler_grid": list(angular_series.minimum_euler_grid),
            "euler_grid_guaranteed_exact":
                angular_series.euler_grid_guaranteed_exact,
            "j0_pav_energy": angular_pav.energy,
            "j0_pav_fidelity": angular_pav.fidelity,
            "j0_pav_faf": trajectory[-1].get("faf"),
            "component_trajectory": trajectory,
            "elapsed_seconds": time.perf_counter() - isotope_started,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    report = {
        "method": (
            "USDB exact diagonalization and exact-ground-state FAF in "
            "fixed (N,Z,M=0)"
            if exact_only
            else "USDB HFB followed by fixed-sector P_N P_Z P_J=0 PAV"
        ),
        "interaction": "data/usdb.nat",
        "faf_order": int(order),
        "hfb_analytic_jacobian": None if exact_only else analytic_jacobian,
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_name = (
        "usdb_ne_exact_faf.json"
        if exact_only
        else "usdb_ne_pav_faf_components.json"
    )
    output = ROOT / "benchmarks/results" / output_name
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--isotopes", nargs="+", type=int, default=[20])
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--euler-grid", nargs=3, type=int)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument(
        "--finite-difference-jacobian",
        action="store_true",
        help="use the shared numerical HFB Jacobian instead of the analytic one",
    )
    arguments = parser.parse_args()
    main(
        isotopes=tuple(arguments.isotopes),
        starts=arguments.starts,
        maxiter=arguments.maxiter,
        order=arguments.order,
        seed=arguments.seed,
        euler_grid=(
            tuple(arguments.euler_grid) if arguments.euler_grid else None
        ),
        exact_only=arguments.exact_only,
        analytic_jacobian=not arguments.finite_difference_jacobian,
    )
