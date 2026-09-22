"""Test whether the CKI Be8 HFB and closest Slater states differ by symmetry."""

import json
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from characterize_gaussian_overlap import generators, rotation
from gaussian_fidelity import maximize_slater_fidelity
from number_projection import exact_ground_state


def main():
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
    interaction, eps = namespace["get_twobody_nuclearshell_model"](
        str(ROOT / "data/cki")
    )
    fermionic = build_fermionic_hamiltonian(interaction, eps, particles=(2, 2))
    _, target = exact_ground_state(fermionic)

    saved = np.load(ROOT / "benchmarks/results/cki_be8_state.npz")
    rho = saved["V"].conj() @ saved["V"].T
    _, eigenvectors = np.linalg.eigh((rho + rho.conj().T) / 2)
    hfb_orbitals = eigenvectors[:, -4:]

    closest = maximize_slater_fidelity(
        fermionic,
        target,
        starts=10,
        seed=42,
        maxiter=1500,
        gradient_tolerance=2e-7,
        initial_orbitals=hfb_orbitals,
    )
    gaussian_orbitals = closest.orbitals

    def determinant_fidelity(left, right):
        return float(abs(np.linalg.det(left.conj().T @ right)) ** 2)

    # The exact target is not itself a Slater determinant, so expand the HFB
    # determinant through its occupied-orbital minors in the target ordering.
    occupations = np.asarray(fermionic.occupations, dtype=int)
    hfb_coefficients = np.linalg.det(hfb_orbitals[occupations])
    hfb_ground = float(abs(np.vdot(target, hfb_coefficients)) ** 2)
    gaussian_ground = float(closest.fidelity)
    raw_mutual = determinant_fidelity(hfb_orbitals, gaussian_orbitals)

    states = namespace["SingleParticleState"](
        str(ROOT / "data/cki")
    ).state_encoding
    j_generators, t_generators = generators(states)

    rng = np.random.default_rng(18)
    aligned = {}
    angles = {}
    for name, use_j, use_t, dimensions in (
        ("J", True, False, 3),
        ("T", False, True, 3),
        ("JT", True, True, 6),
    ):
        def transform(parameters):
            result = np.eye(12, dtype=complex)
            if use_j:
                result = rotation(parameters[:3], j_generators) @ result
            if use_t:
                result = rotation(parameters[-3:], t_generators) @ result
            return result

        fits = []
        for _ in range(30):
            initial = rng.uniform(-np.pi, np.pi, dimensions)
            fits.append(minimize(
                lambda x: -determinant_fidelity(
                    hfb_orbitals, transform(x) @ gaussian_orbitals
                ),
                initial,
                method="BFGS",
                options={"maxiter": 300, "gtol": 1e-10},
            ))
        winner = min(fits, key=lambda fit: fit.fun)
        aligned[name] = float(-winner.fun)
        angles[name] = winner.x.tolist()

    report = {
        "hfb_ground_fidelity": hfb_ground,
        "closest_gaussian_ground_fidelity": gaussian_ground,
        "raw_mutual_fidelity": raw_mutual,
        "best_aligned_fidelity": aligned,
        "best_angles_radians": angles,
        "closest_gaussian_converged": bool(closest.converged),
        "closest_gaussian_gradient_norm": float(closest.gradient_norm),
        "closest_gaussian_attempts": closest.attempts,
    }
    output = ROOT / "benchmarks/results/cki_be8_rotation_equivalence.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
