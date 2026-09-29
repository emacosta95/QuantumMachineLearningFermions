"""Fermionic anti-flatness of the CKI Be10 exact state and VAP vacuum series.

The benchmark reuses the committed exact-grid Be10 VAP state.  It constructs
the same J=0 Euler series and evaluates both states in the identical fixed
(N,Z) determinant basis.  Because that basis already fixes N=4,Z=2, the gauge
sum is collapsed to one point; the Euler grid remains the exact (9,5,9) rule.
"""

import argparse
import json
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from angular_momentum import ParticleNumberJ0ProjectedEnergy
from fermionic_antiflatness import (
    fermionic_antiflatness,
    vacuum_series_antiflatness,
)
from hfb import BogoliubovVacuumSeries, HFBHamiltonian, HFBState
from number_projection import exact_ground_state


def main(order=2):
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
    targets = (4, 2)
    fermionic = build_fermionic_hamiltonian(interaction, eps, particles=targets)
    exact_energy, exact_state = exact_ground_state(fermionic)

    saved_path = (
        ROOT / "benchmarks/results/cki_be10_vap_n7x7_j9x5x9_state.npz"
    )
    saved = np.load(saved_path)
    intrinsic = HFBState(saved["U"], saved["V"], Z=saved["Z"])
    projector = ParticleNumberJ0ProjectedEnergy(
        HFBHamiltonian(np.diag(eps), interaction),
        state_encoding,
        list(range(6, 12)),
        targets,
        number_grid=(7, 7),
        euler_grid=(9, 5, 9),
    )
    exact_series = projector.projected_series(intrinsic)

    # Every supplied determinant already has target N,Z.  On this basis all 49
    # gauge copies have the same net Fourier phase, so sum them analytically and
    # retain one representative per Euler rotation.  This evaluates the exact
    # (7,7)x(9,5,9) projected series with 405, rather than 19,845, Pfaffian rows.
    gauge_points = int(np.prod(exact_series.number_grid))
    euler_points = int(np.prod(exact_series.euler_grid))
    if exact_series.number_of_vacua != gauge_points * euler_points:
        raise ValueError("Series size does not match its number and Euler grids")
    # ParticleNumberJ0ProjectedEnergy stores the Euler loop innermost.  The
    # first contiguous block is therefore one representative of every Euler
    # rotation at a single gauge point.
    selected = np.arange(euler_points)
    series = BogoliubovVacuumSeries(
        intrinsic_state=intrinsic,
        transformations=exact_series.transformations[selected],
        weights=exact_series.weights[selected] * gauge_points,
        number_grid=exact_series.number_grid,
        euler_grid=exact_series.euler_grid,
        projection=exact_series.projection + " (fixed-sector gauge sum collapsed)",
        number_offset=exact_series.number_offset,
        euler_offset=exact_series.euler_offset,
        number_grid_guaranteed_exact=True,
        euler_grid_guaranteed_exact=True,
        minimum_number_grid=exact_series.minimum_number_grid,
        minimum_euler_grid=exact_series.minimum_euler_grid,
    )

    exact = fermionic_antiflatness(
        exact_state, fermionic.occupations, fermionic.modes, order=order
    )
    projected_vap = vacuum_series_antiflatness(
        series, fermionic.occupations, order=order
    )
    report = {
        "nucleus": "Be10",
        "definition": "F_k = L - Tr[(M^T M)^k]/2",
        "order": int(order),
        "modes": int(fermionic.modes),
        "exact_ground_energy": exact_energy,
        "exact_ground_state_antiflatness": exact.value,
        "exact_ground_state_antiflatness_per_mode": exact.value_per_mode,
        "projected_vap_series_antiflatness": projected_vap.value,
        "projected_vap_series_antiflatness_per_mode": projected_vap.value_per_mode,
        "series_number_grid": list(series.number_grid),
        "series_euler_grid": list(series.euler_grid),
        "exact_series_vacua": exact_series.number_of_vacua,
        "evaluated_vacua_after_fixed_sector_gauge_collapse": series.number_of_vacua,
        "source_state": str(saved_path.relative_to(ROOT)),
    }
    output = ROOT / "benchmarks/results/cki_be10_antiflatness.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=2)
    arguments = parser.parse_args()
    main(order=arguments.order)
