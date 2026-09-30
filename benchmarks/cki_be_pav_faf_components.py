"""J=0 projection-after-variation fidelity and FAF for CKI Be isotopes.

The trajectory adds Euler-rotated components in the deterministic quadrature
order after analytically collapsing the exact N,Z gauge sum on the fixed-sector
basis.  Prefix values are convergence diagnostics; they need not be monotonic,
because a quadrature prefix is not a variationally optimized multi-reference
ansatz.
"""

import argparse
import json
import time
from dataclasses import replace
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from angular_momentum import ParticleNumberJ0ProjectedEnergy
from fermionic_antiflatness import fermionic_antiflatness
from hfb import BogoliubovVacuumSeries, HFBHamiltonian, HFBState, solve_hfb
from number_projection import exact_ground_state, projected_series_observables


def _collapse_fixed_sector_gauge_sum(series):
    """Retain one weighted representative per Euler point at fixed N,Z."""
    gauge_points = int(np.prod(series.number_grid))
    euler_points = int(np.prod(series.euler_grid))
    if series.number_of_vacua != gauge_points * euler_points:
        raise ValueError("Series size does not match its number and Euler grids")
    # projected_series stores Euler points in the inner loop, so the first
    # contiguous block contains every Euler rotation at one gauge point.
    selected = np.arange(euler_points)
    return BogoliubovVacuumSeries(
        intrinsic_state=series.intrinsic_state,
        transformations=series.transformations[selected],
        weights=series.weights[selected] * gauge_points,
        number_grid=series.number_grid,
        euler_grid=series.euler_grid,
        projection=series.projection + " (fixed-sector gauge sum collapsed)",
        number_offset=series.number_offset,
        euler_offset=series.euler_offset,
        number_grid_guaranteed_exact=series.number_grid_guaranteed_exact,
        euler_grid_guaranteed_exact=series.euler_grid_guaranteed_exact,
        minimum_number_grid=series.minimum_number_grid,
        minimum_euler_grid=series.minimum_euler_grid,
    )


def _default_counts(total):
    counts = {1, total}
    value = 2
    while value < total:
        counts.add(value)
        value *= 2
    return tuple(sorted(counts))


def _prefix_observables(series, fermionic, target, counts, order):
    rows = []
    target = np.asarray(target, complex) / np.linalg.norm(target)
    for count in counts:
        prefix = replace(
            series,
            transformations=series.transformations[:count],
            weights=series.weights[:count],
            projection=series.projection + f"; first {count} Euler components",
            euler_grid_guaranteed_exact=(count == series.number_of_vacua),
        )
        amplitudes = prefix.occupation_amplitudes(fermionic.occupations)
        norm = float(np.vdot(amplitudes, amplitudes).real)
        if norm < 1e-14:
            rows.append({
                "components": count,
                "state_norm": norm,
                "fidelity": None,
                "faf": None,
                "faf_per_mode": None,
            })
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
    isotopes=(6, 8, 10, 12),
    starts=2,
    maxiter=120,
    order=2,
    seed=8,
    intrinsic_method="hfb",
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
    interaction, eps = namespace["get_twobody_nuclearshell_model"](
        str(ROOT / "data/cki")
    )
    encoding = namespace["SingleParticleState"](
        str(ROOT / "data/cki")
    ).state_encoding
    intrinsic_hamiltonian = HFBHamiltonian(np.diag(eps), interaction)
    neutron_modes = list(range(6, 12))
    results = []

    for mass in isotopes:
        neutrons = int(mass) - 6
        targets = (neutrons, 2)
        if neutrons not in (0, 2, 4, 6):
            raise ValueError("CKI even Be isotopes are Be6, Be8, Be10, Be12")
        fermionic = build_fermionic_hamiltonian(
            interaction, eps, particles=targets
        )
        exact_energy, target = exact_ground_state(fermionic)
        hfb = solve_hfb(
            intrinsic_hamiltonian,
            neutron_modes,
            targets,
            starts=starts,
            seed=seed,
            maxiter=maxiter,
            tolerance=1e-8,
            method=intrinsic_method,
        )
        # Move numerically collapsed HFB solutions onto the exact Slater chart;
        # their nearly singular U matrices do not define a stable Thouless Z.
        rho = (hfb.state.rho + hfb.state.rho.conj().T) / 2
        rho_idempotency = float(np.linalg.norm(rho @ rho - rho))
        pairing_norm = float(np.linalg.norm(hfb.state.kappa))
        if pairing_norm < 1e-3 and rho_idempotency < 1e-6:
            _, natural_orbitals = np.linalg.eigh(rho)
            working_state = HFBState.from_slater(
                natural_orbitals[:, -sum(targets):]
            )
            state_chart = "Slater (pairing collapsed)"
        else:
            working_state = hfb.state
            state_chart = "paired Bogoliubov vacuum"
        projector = ParticleNumberJ0ProjectedEnergy(
            intrinsic_hamiltonian,
            encoding,
            neutron_modes,
            targets,
        )
        full_series = projector.projected_series(working_state)
        series = _collapse_fixed_sector_gauge_sum(full_series)
        final = projected_series_observables(series, fermionic, target)
        exact_faf = fermionic_antiflatness(
            target, fermionic.occupations, fermionic.modes, order=order
        )
        counts = _default_counts(series.number_of_vacua)
        trajectory = _prefix_observables(
            series, fermionic, target, counts, order
        )
        row = {
            "nucleus": f"Be{mass}",
            "valence_neutrons": neutrons,
            "valence_protons": 2,
            "dimension_NZ": len(fermionic.occupations),
            "exact_energy": exact_energy,
            "exact_ground_state_faf": exact_faf.value,
            "exact_ground_state_faf_per_mode": exact_faf.value_per_mode,
            "hfb_energy": hfb.energy,
            "hfb_converged": bool(hfb.converged),
            "intrinsic_method": intrinsic_method,
            "intrinsic_attempts": hfb.attempts,
            "hfb_pairing_norm": pairing_norm,
            "hfb_rho_idempotency": rho_idempotency,
            "projection_state_chart": state_chart,
            "number_grid": list(full_series.number_grid),
            "euler_grid": list(full_series.euler_grid),
            "full_gaussian_components": full_series.number_of_vacua,
            "euler_components_after_fixed_sector_gauge_collapse":
                series.number_of_vacua,
            "pav_energy": final.energy,
            "pav_fidelity": final.fidelity,
            "pav_faf": trajectory[-1]["faf"],
            "pav_faf_per_mode": trajectory[-1]["faf_per_mode"],
            "component_trajectory": trajectory,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

    report = {
        "method": (
            f"{intrinsic_method.upper()} followed by exact "
            "P_N P_Z P_J=0 projection (PAV)"
        ),
        "faf_definition": "F_k = L - Tr[(M^T M)^k]/2",
        "faf_order": int(order),
        "component_trajectory_note": (
            "Euler-quadrature prefixes are deterministic convergence checks, "
            "not variationally optimized component expansions; monotonicity is "
            "therefore neither assumed nor required."
        ),
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = ROOT / "benchmarks/results/cki_be_pav_faf_components.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--isotopes", nargs="+", type=int, default=[6, 8, 10, 12])
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--intrinsic-method", choices=("hf", "hfb"), default="hfb"
    )
    arguments = parser.parse_args()
    main(
        isotopes=tuple(arguments.isotopes),
        starts=arguments.starts,
        maxiter=arguments.maxiter,
        order=arguments.order,
        seed=arguments.seed,
        intrinsic_method=arguments.intrinsic_method,
    )
