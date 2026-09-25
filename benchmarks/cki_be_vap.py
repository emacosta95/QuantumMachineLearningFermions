"""Projected-HFB VAP for CKI beryllium isotopes.

Example exploratory fixed-sector run with the user's coarse grid:

  python benchmarks/cki_be_vap.py --mass 10 --number-grid 1 1 \
      --euler-grid 3 3 3 --allow-inexact-number-grid \
      --allow-inexact-euler-grid --backend fixed_sector_basis

Use ``--backend kernel`` with the default/exact number grid for the scalable
transition-kernel formulation.  The fixed-sector backend is intended for the
small CKI analysis because it explicitly expands determinant amplitudes during
every VAP energy evaluation.
"""

import argparse
import json
import time
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from angular_momentum import (
    ParticleNumberJ0ProjectedEnergy,
    project_state_observables,
)
from hfb import HFBHamiltonian, HFBState
from number_projection import exact_ground_state
from projected_vap import solve_projected_hfb_vap


def main(arguments):
    started = time.perf_counter()
    valence_neutrons = arguments.mass - 6
    if arguments.mass not in (8, 10, 12):
        raise ValueError("This CKI benchmark supports Be8, Be10, and Be12")

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
    state_encoding = namespace["SingleParticleState"](
        str(ROOT / "data/cki")
    ).state_encoding
    intrinsic_hamiltonian = HFBHamiltonian(np.diag(eps), interaction)
    neutron_modes = list(range(6, 12))
    targets = (valence_neutrons, 2)
    fermionic = build_fermionic_hamiltonian(
        interaction, eps, particles=targets
    )
    exact_energy, exact_target = exact_ground_state(fermionic)

    projector = ParticleNumberJ0ProjectedEnergy(
        intrinsic_hamiltonian,
        state_encoding,
        neutron_modes,
        targets,
        number_grid=(tuple(arguments.number_grid)
                     if arguments.number_grid else None),
        euler_grid=(tuple(arguments.euler_grid)
                    if arguments.euler_grid else None),
        allow_inexact_number_grid=arguments.allow_inexact_number_grid,
        allow_inexact_euler_grid=arguments.allow_inexact_euler_grid,
    )

    initial_state = None
    if arguments.initial_state is not None:
        saved = np.load(arguments.initial_state)
        stored_z = saved["Z"] if "Z" in saved and saved["Z"].size else None
        initial_state = HFBState(saved["U"], saved["V"], Z=stored_z)

    result = solve_projected_hfb_vap(
        projector,
        energy_backend=arguments.backend,
        fermionic_hamiltonian=(
            fermionic if arguments.backend == "fixed_sector_basis" else None
        ),
        allow_aliased_number_kernel=arguments.allow_aliased_number_kernel,
        starts=arguments.starts,
        seed=arguments.seed,
        maxiter=arguments.maxiter,
        tolerance=arguments.tolerance,
        gradient_tolerance=arguments.gradient_tolerance,
        parameter_bound=arguments.parameter_bound,
        initial_state=initial_state,
        optimizer=arguments.optimizer,
        spsa_learning_rate=arguments.spsa_learning_rate,
        spsa_perturbation=arguments.spsa_perturbation,
    )
    initial_vacuum = HFBState.from_thouless(
        projector.unpack(result.initial_parameters)
    )
    initial_series = projector.projected_series(initial_vacuum)
    initial_projected = project_state_observables(
        initial_series, fermionic, exact_target
    )
    projected = project_state_observables(
        result.projected_series, fermionic, exact_target
    )
    intrinsic_numbers = [
        float(np.diag(result.state.rho)[neutron_modes].real.sum()),
        float(np.diag(result.state.rho)[:6].real.sum()),
    ]
    report = {
        "nucleus": f"Be{arguments.mass}",
        "method": "P_N P_Z P_J=0 variation after projection",
        "energy_backend": result.energy_backend,
        "number_grid": list(result.projected_series.number_grid),
        "euler_grid": list(result.projected_series.euler_grid),
        "minimum_number_grid": list(result.projected_series.minimum_number_grid),
        "minimum_euler_grid": list(result.projected_series.minimum_euler_grid),
        "number_grid_guaranteed_exact": result.number_grid_guaranteed_exact,
        "euler_grid_guaranteed_exact": result.euler_grid_guaranteed_exact,
        "projected_vap_energy": result.projected_energy,
        "initial_projected_basis_energy": initial_projected.energy,
        "initial_projected_fidelity": initial_projected.fidelity,
        "projected_basis_energy": projected.energy,
        "projected_fidelity": projected.fidelity,
        "fidelity_improvement": (
            projected.fidelity - initial_projected.fidelity
        ),
        "projected_sector_weight": projected.sector_weight,
        "exact_ground_energy": exact_energy,
        "intrinsic_numbers": intrinsic_numbers,
        "intrinsic_kappa_norm": result.kappa_norm,
        "converged": result.converged,
        "optimizer": arguments.optimizer,
        "gradient_norm": result.gradient_norm,
        "attempts": result.attempts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = ROOT / "benchmarks/results"
    output.mkdir(exist_ok=True)
    stem = f"cki_be{arguments.mass}_vap"
    (output / f"{stem}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez(
        output / f"{stem}_state.npz",
        U=result.state.U,
        V=result.state.V,
        Z=result.state.thouless_matrix,
        parameters=result.parameters,
        initial_parameters=result.initial_parameters,
        projected_vector=projected.projected_vector,
        masks=projected.masks,
    )
    print(json.dumps(report, indent=2))


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--mass", type=int, choices=(8, 10, 12), default=10)
    result.add_argument("--number-grid", nargs=2, type=int)
    result.add_argument("--euler-grid", nargs=3, type=int)
    result.add_argument("--allow-inexact-number-grid", action="store_true")
    result.add_argument("--allow-inexact-euler-grid", action="store_true")
    result.add_argument(
        "--backend",
        choices=("kernel", "fixed_sector_basis"),
        default="kernel",
    )
    result.add_argument("--allow-aliased-number-kernel", action="store_true")
    result.add_argument("--starts", type=int, default=4)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--maxiter", type=int, default=300)
    result.add_argument("--tolerance", type=float, default=1e-10)
    result.add_argument("--gradient-tolerance", type=float, default=2e-5)
    result.add_argument("--parameter-bound", type=float, default=8.0)
    result.add_argument(
        "--optimizer", choices=("L-BFGS-B", "SPSA"), default="SPSA"
    )
    result.add_argument("--spsa-learning-rate", type=float, default=0.08)
    result.add_argument("--spsa-perturbation", type=float, default=0.12)
    result.add_argument("--initial-state")
    return result


if __name__ == "__main__":
    main(parser().parse_args())
