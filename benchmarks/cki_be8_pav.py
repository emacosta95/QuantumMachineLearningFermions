"""CKI Be8 gauge-series particle-number projection after HFB variation."""

import argparse
import json
import time

import numpy as np

from cki_be8 import (
    ROOT,
    build_fermionic_hamiltonian,
    legacy_definitions,
)
from typing import List, Dict, Tuple, Optional, Callable, ClassVar
from gauge_projection import GaugeProjectedEnergy
from hfb import HFBHamiltonian, solve_hfb
from number_projection import exact_ground_state, projected_series_observables


def main(number_grid=None):
    """Optimize the intrinsic state first, then apply exact N,Z projection."""
    # Include loading, variation, projection, and validation in the elapsed time.
    start = time.perf_counter()

    # Load only the legacy definitions needed to read CKI h and v data without
    # importing unrelated optional machine-learning dependencies.
    namespace = dict(globals(), trange=range)
    legacy_definitions(
        "cg_utils.py",
        [
            "CG",
            "ClebschGordan",
            "SelectCG",
            "CreateInitialCGList",
            "CalcInitialValues",
            "DivCalc",
            "CgJM",
        ],
        namespace,
    )
    legacy_definitions(
        "nuclear_physics_utils.py",
        [
            "SingleParticleState",
            "krond",
            "scattering_matrix_reader",
            "compute_nuclear_twobody_matrix",
            "get_twobody_nuclearshell_model",
        ],
        namespace,
    )

    # Read the antisymmetrized CKI interaction and diagonal one-body energies.
    interaction, eps = namespace["get_twobody_nuclearshell_model"](
        str(ROOT / "data/cki")
    )

    # HFBHamiltonian retains raw h,v tensors required by intrinsic variation and
    # optional gauge-kernel validation.
    intrinsic_hamiltonian = HFBHamiltonian(np.diag(eps), interaction)

    # FermiHubbardHamiltonian owns the exact N=2,Z=2 determinant basis and the
    # many-body matrix used after variation. No second matrix is built by the
    # projection module.
    fermionic_hamiltonian = build_fermionic_hamiltonian(
        interaction, eps, particles=(2, 2)
    )

    # CKI stores proton modes 0:6 and neutron modes 6:12.
    neutron_modes = list(range(6, 12))

    # Variation occurs only at the intrinsic HFB level with average N=Z=2.
    # The projected energy does not feed back into this optimization.
    hfb_result = solve_hfb(
        intrinsic_hamiltonian,
        neutron_modes,
        [2, 2],
        starts=3,
        seed=15,
        maxiter=1000,
        tolerance=1e-10,
    )

    # Diagonalize the same FermiHubbardHamiltonian matrix to obtain the exact
    # reference energy and target vector in its native determinant ordering.
    exact_energy, target = exact_ground_state(fermionic_hamiltonian)

    # Configure the user-controlled double Fourier sum before any determinant
    # amplitudes are evaluated.
    gauge = GaugeProjectedEnergy(
        intrinsic_hamiltonian,
        neutron_modes,
        [2, 2],
        grid=number_grid,
    )
    # Keep P_N P_Z|Phi> as L_N*L_Z gauge-rotated Bogoliubov vacua.
    series = gauge.projected_series(hfb_result.state)
    # Gauge-kernel projection is an independent polynomial-memory energy check.
    # It requires a finite particle-vacuum Thouless chart; a collapsed occupied
    # HF determinant has singular U and is instead handled exactly above through
    # Slater-determinant amplitudes.
    try:
        z = hfb_result.state.thouless_matrix
    except ValueError:
        z = np.empty((0, 0), complex)
        gauge_energy = None
    else:
        # Transition kernels consume the identical stored series terms.
        gauge_energy = gauge.series_energy(series)

    # Expand that same series only now, at the final target-fidelity boundary.
    projected = projected_series_observables(
        series, fermionic_hamiltonian, target
    )

    # Record intrinsic and projected quantities separately so PAV cannot be
    # mistaken for a variation-after-projection calculation.
    report = {
        "method": "particle-number projection after variation",
        "intrinsic_variation": "average-number constrained HFB",
        "hfb_converged": hfb_result.converged,
        "hfb_energy": hfb_result.energy,
        "intrinsic_numbers": hfb_result.numbers.tolist(),
        "intrinsic_kappa_norm": float(np.linalg.norm(hfb_result.state.kappa)),
        "intrinsic_canonical_error": float(
            hfb_result.state.canonical_error()
        ),
        "projected_energy": projected.energy,
        "projected_fidelity": projected.fidelity,
        "projected_sector_weight": projected.sector_weight,
        "number_projection_grid": list(projected.grid),
        "number_projection_grid_offset": projected.grid_offset,
        "M_vacua": series.number_of_vacua,
        "exact_energy": exact_energy,
        "gauge_kernel_energy": gauge_energy,
        "elapsed_seconds": time.perf_counter() - start,
    }

    # Save both the intrinsic Bogoliubov amplitudes and normalized PAV vector.
    # ``has_thouless`` tells downstream code whether the stored Z chart is valid.
    output = ROOT / "benchmarks/results"
    output.mkdir(exist_ok=True)
    (output / "cki_be8_pav.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez(
        output / "cki_be8_pav_state.npz",
        U=hfb_result.state.U,
        V=hfb_result.state.V,
        Z=z,
        has_thouless=bool(z.size),
        projected_vector=projected.projected_vector,
        masks=projected.masks,
    )

    # Print the same structured report written to disk for interactive runs.
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # Permit direct N,Z discretization control in convergence studies.
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-grid", nargs=2, type=int, metavar=("LN", "LZ"))
    arguments = parser.parse_args()
    main(
        number_grid=(tuple(arguments.number_grid) if arguments.number_grid else None)
    )
