"""Gauge/Euler-series P_N P_Z P_J=0 projection of the CKI Be8 HFB vacuum."""

import argparse
import json
import time
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

# Importing cki_be8 first establishes the isolated src/NSMFermions path used by
# all benchmark scripts without importing optional package-level dependencies.
from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from angular_momentum import (
    ParticleNumberJ0ProjectedEnergy,
    project_state_observables,
)
from hfb import HFBHamiltonian, HFBState
from number_projection import exact_ground_state


def main(number_grid=None, euler_grid=None):
    """Project with caller-controlled number and Euler quadrature dimensions."""
    # Include data loading, series construction, and fidelity in elapsed time.
    start = time.perf_counter()

    # Load only the legacy CKI definitions required by this benchmark.
    namespace = dict(globals(), trange=range)
    legacy_definitions(
        "cg_utils.py",
        [
            "CG", "ClebschGordan", "SelectCG", "CreateInitialCGList",
            "CalcInitialValues", "DivCalc", "CgJM",
        ],
        namespace,
    )
    legacy_definitions(
        "nuclear_physics_utils.py",
        [
            "SingleParticleState", "krond", "scattering_matrix_reader",
            "compute_nuclear_twobody_matrix", "get_twobody_nuclearshell_model",
        ],
        namespace,
    )

    # Read the antisymmetrized CKI interaction and one-body energies.
    interaction, eps = namespace["get_twobody_nuclearshell_model"](
        str(ROOT / "data/cki")
    )
    # Read spherical quantum numbers needed to construct spatial rotations.
    single_particle = namespace["SingleParticleState"](str(ROOT / "data/cki"))
    # The tensor Hamiltonian supplies transition-density energy kernels.
    intrinsic_hamiltonian = HFBHamiltonian(np.diag(eps), interaction)
    # CKI places its six neutron modes after its six proton modes.
    neutron_modes = list(range(6, 12))

    # Introduce a determinant basis only for the final components and target.
    fermionic = build_fermionic_hamiltonian(interaction, eps, particles=(2, 2))
    # Obtain the exact target in precisely fermionic.occupations ordering.
    exact_energy, target = exact_ground_state(fermionic)

    # Load the intrinsic HFB state produced before symmetry projection.
    saved = np.load(ROOT / "benchmarks/results/cki_be8_pav_state.npz")
    # Preserve Z only if the intrinsic state has a finite Thouless chart.
    stored_z = saved["Z"] if bool(saved["has_thouless"]) else None
    # Reconstruct the common U,V vacuum used by every group-orbit term.
    state = HFBState(saved["U"], saved["V"], Z=stored_z)

    # Configure simultaneous N,Z,J=0 quadrature. The Euler tuple directly
    # controls the alpha, beta, and gamma discretization dimensions.
    evaluator = ParticleNumberJ0ProjectedEnergy(
        intrinsic_hamiltonian,
        single_particle.state_encoding,
        neutron_modes,
        [2, 2],
        number_grid=number_grid,
        euler_grid=euler_grid,
    )
    # Construct M transformed Bogoliubov vacua without J^2 diagonalization or
    # determinant coefficients.
    series = evaluator.projected_series(state)
    # For finite Z, evaluate transition-density kernels over the identical
    # stored (T_q,w_q) terms before introducing determinant configurations.
    if stored_z is None:
        kernel_energy = None
    else:
        kernel_energy = evaluator.series_energy(series)

    # Expand the coherent series only now, when components and target fidelity
    # in the FermiHubbardHamiltonian determinant basis are actually required.
    result = project_state_observables(series, fermionic, target)
    # Compare the polynomial transition kernel with the late basis evaluation.
    kernel_basis_difference = (
        None if kernel_energy is None else kernel_energy - result.energy
    )
    # Record every discretization parameter for reproducible M convergence scans.
    report = {
        "nucleus": "Be8",
        "method": "Bogoliubov-vacuum gauge/Euler series",
        "target_J": 0,
        "number_grid": list(series.number_grid),
        "euler_grid": list(series.euler_grid),
        "number_grid_points": int(np.prod(series.number_grid)),
        "euler_grid_points": int(np.prod(series.euler_grid)),
        "M_vacua": series.number_of_vacua,
        "M_bound": evaluator.euler_grid.m_bound,
        "J_bound": evaluator.euler_grid.j_bound,
        "projected_series_norm": result.sector_weight,
        "pnj0_energy_from_series_components": result.energy,
        "pnj0_fidelity_from_series_components": result.fidelity,
        "pnj0_energy_transition_kernels": kernel_energy,
        "kernel_basis_energy_difference": kernel_basis_difference,
        "exact_ground_energy": exact_energy,
        "elapsed_seconds": time.perf_counter() - start,
    }

    # Save normalized components and masks only after the series fidelity step.
    output = ROOT / "benchmarks/results"
    output.mkdir(exist_ok=True)
    (output / "cki_be8_j0_projection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez(
        output / "cki_be8_j0_projected_state.npz",
        projected_vector=result.projected_vector,
        masks=result.masks,
        number_grid=np.asarray(series.number_grid),
        euler_grid=np.asarray(series.euler_grid),
        M_vacua=series.number_of_vacua,
    )
    # Print the same structured record for interactive convergence studies.
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # Expose both discretizations directly at the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-grid", nargs=2, type=int, metavar=("LN", "LZ"))
    parser.add_argument(
        "--euler-grid", nargs=3, type=int,
        metavar=("LALPHA", "LBETA", "LGAMMA"),
    )
    arguments = parser.parse_args()
    # Convert argparse lists to immutable tuples expected by the projector.
    main(
        number_grid=(tuple(arguments.number_grid) if arguments.number_grid else None),
        euler_grid=(tuple(arguments.euler_grid) if arguments.euler_grid else None),
    )
