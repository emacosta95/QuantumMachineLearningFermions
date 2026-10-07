"""Compare energy-optimized and closest-Gaussian states with exact nuclei.

Choosing ``cki`` selects the p-shell Be chain, while ``usdb`` selects the
sd-shell Ne chain.  Exact USDB diagonalization uses only the M=0 sector; this
restriction is never imposed on the intrinsic HF/HFB optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
for directory in (ROOT / "src" / "NSMFermions", ROOT / "benchmarks"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cki_be8 import build_fermionic_hamiltonian  # noqa: E402
from gaussian_fidelity import maximize_best_gaussian_fidelity  # noqa: E402
from hfb import HFBHamiltonian, HFBState, solve_hfb  # noqa: E402
from number_projection import exact_ground_state  # noqa: E402
from projection_grid_convergence import _load_interaction, _nucleus  # noqa: E402


DEFAULT_ISOTOPES = {"cki": (8, 10, 12), "usdb": (20, 22, 24)}


def _relative_error(value: float, exact: float) -> float:
    return float(abs((value - exact) / exact))


def _state_overlap(state, occupations, target):
    """Return the intrinsic fidelity ``|<Psi_0|Phi>|^2``.

    Although ``occupations`` enumerate the symmetry sector supporting the exact
    state, the Gaussian is neither projected nor renormalized in that sector.
    """
    amplitudes = state.stable_normalized_occupation_amplitudes(occupations)
    return float(abs(np.vdot(target, amplitudes)) ** 2)


def _stable_variational_fidelity_state(state, particles):
    """Replace a pairing-collapsed vacuum by its stable Slater boundary.

    The particle-vacuum Thouless chart requires an invertible ``U``.  Near an
    HF solution ``U`` becomes singular, so recovering ``V* (U*)^-1`` can lose
    antisymmetry even though the physical Bogoliubov state remains canonical.
    In that limit the occupied natural orbitals provide the exact boundary
    representation and determinant amplitudes remain numerically stable.
    """
    rho = (state.rho + state.rho.conj().T) / 2
    pairing_norm = float(np.linalg.norm(state.kappa))
    idempotency = float(np.linalg.norm(rho @ rho - rho))
    if pairing_norm < 1e-3 and idempotency < 1e-6:
        _, natural_orbitals = np.linalg.eigh(rho)
        stable = HFBState.from_slater(natural_orbitals[:, -int(particles):])
        return stable, "Slater (pairing-collapsed HFB)"
    return state, "finite particle-vacuum Thouless chart"


def run_study(
    interaction_name="cki",
    isotopes=None,
    variational_method="hfb",
    starts=4,
    maxiter=500,
    gaussian_starts=4,
    gaussian_maxiter=1000,
    gaussian_hf_starts=None,
    gaussian_hf_maxiter=None,
    real_bogoliubov=True,
    variational_only=False,
    seed=8,
    output=None,
):
    interaction_name = interaction_name.lower()
    interaction, eps, encoding, interaction_path = _load_interaction(
        interaction_name
    )
    modes = len(eps)
    species_modes = modes // 2
    neutron_modes = list(range(species_modes, modes))
    intrinsic_hamiltonian = HFBHamiltonian(np.diag(eps), interaction)
    masses = tuple(isotopes or DEFAULT_ISOTOPES[interaction_name])
    magnetic_projections = np.asarray([float(state[3]) for state in encoding])
    results = []
    started = time.perf_counter()
    output_path = (
        Path(output).expanduser().resolve()
        if output
        else ROOT / "results" / (
            f"{interaction_name}_{variational_method}_optimization.json"
            if variational_only
            else f"{interaction_name}_gaussian_fidelity.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for mass in masses:
        isotope_started = time.perf_counter()
        label, targets = _nucleus(interaction_name, int(mass), species_modes)
        print(
            f"[{label}] optimizing {variational_method.upper()} with "
            f"the constrained-gradient solver and {starts} starts",
            flush=True,
        )
        variational = solve_hfb(
            intrinsic_hamiltonian,
            neutron_modes,
            targets,
            starts=starts,
            seed=seed + int(mass),
            maxiter=maxiter,
            tolerance=1e-8,
            real_bogoliubov=real_bogoliubov,
            method=variational_method,
        )
        if variational_only:
            row = {
                "nucleus": label,
                "interaction": interaction_name,
                "valence_neutrons": targets[0],
                "valence_protons": targets[1],
                "variational_method": variational_method,
                "variational_solver": "constrained manifold gradient",
                "variational_real_bogoliubov": bool(
                    real_bogoliubov and variational_method == "hfb"
                ),
                "variational_parameter_count": int(
                    np.asarray(variational.parameters).size
                ),
                "variational_converged": bool(variational.converged),
                "variational_energy": variational.energy,
                "variational_numbers": variational.numbers.tolist(),
                "variational_pairing_norm": float(
                    np.linalg.norm(variational.state.kappa)
                ),
                "variational_stationarity_error": (
                    variational.stationarity_error
                ),
                "variational_chemical_potentials": (
                    variational.chemical_potentials.tolist()
                ),
                "variational_attempts": variational.attempts,
                "stationarity_error": variational.stationarity_error,
                "attempts": variational.attempts,
                "elapsed_seconds": time.perf_counter() - isotope_started,
            }
            results.append(row)
            print(json.dumps(row, indent=2), flush=True)
            checkpoint = {
                "status": "running",
                "study": "intrinsic variational optimization only",
                "interaction": interaction_name,
                "interaction_path": str(interaction_path.relative_to(ROOT)),
                "isotopes": list(masses),
                "results": results,
                "elapsed_seconds": time.perf_counter() - started,
            }
            output_path.write_text(
                json.dumps(checkpoint, indent=2), encoding="utf-8"
            )
            continue

        print(f"[{label}] building the optimized exact sector", flush=True)
        symmetries = None
        exact_sector = "fixed (N,Z)"
        if interaction_name == "usdb":
            def m_zero(occupied):
                return abs(float(np.sum(
                    magnetic_projections[list(occupied)]
                ))) < 1e-10

            symmetries = [m_zero]
            exact_sector = "fixed (N,Z,M=0)"

        exact_build_started = time.perf_counter()
        fermionic = build_fermionic_hamiltonian(
            interaction, eps, particles=targets, symmetries=symmetries
        )
        exact_build_seconds = time.perf_counter() - exact_build_started
        print(
            f"[{label}] exact matrix built: dimension="
            f"{len(fermionic.occupations)}, nnz={fermionic.matrix.nnz}, "
            f"elapsed={exact_build_seconds:.3f} s",
            flush=True,
        )
        print(f"[{label}] diagonalizing the exact sector", flush=True)
        exact_diagonalization_started = time.perf_counter()
        exact_energy, target = exact_ground_state(fermionic)
        exact_diagonalization_seconds = (
            time.perf_counter() - exact_diagonalization_started
        )
        print(
            f"[{label}] exact ground state found: E={exact_energy:.10f}, "
            f"elapsed={exact_diagonalization_seconds:.3f} s",
            flush=True,
        )
        target = np.asarray(target, complex) / np.linalg.norm(target)

        variational_fidelity_state, variational_fidelity_chart = (
            _stable_variational_fidelity_state(
                variational.state, sum(targets)
            )
        )
        var_fidelity = _state_overlap(
            variational_fidelity_state, fermionic.occupations, target
        )

        print(
            f"[{label}] optimizing closest Gaussian in both Bogoliubov and "
            f"HF families ({gaussian_starts} and "
            f"{gaussian_hf_starts or gaussian_starts} starts)",
            flush=True,
        )
        initial_bogoliubov_parameters = None
        try:
            initial_z = variational.state.thouless_matrix
            initial_upper = initial_z[np.triu_indices(modes, 1)]
            if not real_bogoliubov:
                initial_bogoliubov_parameters = np.concatenate((
                    initial_upper.real, initial_upper.imag
                ))
            elif np.max(np.abs(initial_upper.imag), initial=0.0) < 1e-10:
                initial_bogoliubov_parameters = initial_upper.real
        except ValueError:
            # A collapsed Slater determinant is a singular boundary state and
            # has no finite particle-vacuum Thouless coordinates.
            pass
        natural_occupations, natural_orbitals = np.linalg.eigh(
            (variational.state.rho + variational.state.rho.conj().T) / 2
        )
        initial_hartree_fock_orbitals = natural_orbitals[
            :, np.argsort(natural_occupations)[-sum(targets):]
        ]
        if real_bogoliubov:
            initial_hartree_fock_orbitals = (
                initial_hartree_fock_orbitals.real
            )
        closest = maximize_best_gaussian_fidelity(
            fermionic,
            target,
            bogoliubov_starts=gaussian_starts,
            hartree_fock_starts=gaussian_hf_starts,
            seed=seed + 1000 + int(mass),
            bogoliubov_maxiter=gaussian_maxiter,
            hartree_fock_maxiter=gaussian_hf_maxiter,
            gradient_tolerance=2e-6,
            real_parameters=real_bogoliubov,
            initial_bogoliubov_parameters=initial_bogoliubov_parameters,
            initial_hartree_fock_orbitals=initial_hartree_fock_orbitals,
        )
        gauss_fidelity = _state_overlap(
            closest.state, fermionic.occupations, target
        )
        closest_energy = float(intrinsic_hamiltonian.energy(closest.state))
        closest_bogoliubov_energy = float(
            intrinsic_hamiltonian.energy(closest.bogoliubov.state)
        )
        closest_hartree_fock_state = HFBState.from_slater(
            closest.hartree_fock.orbitals
        )
        closest_hartree_fock_energy = float(
            intrinsic_hamiltonian.energy(closest_hartree_fock_state)
        )

        row = {
            "nucleus": label,
            "interaction": interaction_name,
            "valence_neutrons": targets[0],
            "valence_protons": targets[1],
            "exact_diagonalization_sector": exact_sector,
            "exact_dimension": len(fermionic.occupations),
            "exact_energy": exact_energy,
            "exact_build_seconds": exact_build_seconds,
            "exact_diagonalization_seconds": exact_diagonalization_seconds,
            "variational_method": variational_method,
            "variational_solver": "constrained manifold gradient",
            "variational_real_bogoliubov": bool(
                real_bogoliubov and variational_method == "hfb"
            ),
            "variational_parameter_count": int(
                np.asarray(variational.parameters).size
            ),
            "variational_converged": bool(variational.converged),
            "variational_energy": variational.energy,
            "variational_energy_relative_error": _relative_error(
                variational.energy, exact_energy
            ),
            "variational_ground_state_fidelity": var_fidelity,
            "variational_numbers": variational.numbers.tolist(),
            "variational_pairing_norm": float(
                np.linalg.norm(variational.state.kappa)
            ),
            "variational_fidelity_state_chart": variational_fidelity_chart,
            "variational_stationarity_error": (
                variational.stationarity_error
            ),
            "variational_chemical_potentials": (
                variational.chemical_potentials.tolist()
            ),
            "variational_attempts": variational.attempts,
            "closest_gaussian_converged": bool(closest.converged),
            "closest_gaussian_family": closest.family,
            "closest_gaussian_solver": (
                "compact analytic F20 heavy ball"
                if closest.family == "bogoliubov"
                else "analytic Slater Stiefel ascent"
            ),
            "closest_gaussian_real_parameters": bool(real_bogoliubov),
            "closest_gaussian_parameter_count": int(
                np.asarray(closest.parameters).size
            ),
            "closest_gaussian_ground_state_fidelity": gauss_fidelity,
            "closest_gaussian_energy": closest_energy,
            "closest_gaussian_energy_relative_error": _relative_error(
                closest_energy, exact_energy
            ),
            "closest_gaussian_gradient_norm": closest.gradient_norm,
            "closest_gaussian_attempts": closest.attempts,
            "closest_bogoliubov_fidelity": closest.bogoliubov.fidelity,
            "closest_bogoliubov_converged": bool(
                closest.bogoliubov.converged
            ),
            "closest_bogoliubov_gradient_norm": (
                closest.bogoliubov.gradient_norm
            ),
            "closest_bogoliubov_energy": closest_bogoliubov_energy,
            "closest_bogoliubov_energy_relative_error": _relative_error(
                closest_bogoliubov_energy, exact_energy
            ),
            "closest_bogoliubov_attempts": closest.bogoliubov.attempts,
            "closest_hartree_fock_fidelity": closest.hartree_fock.fidelity,
            "closest_hartree_fock_converged": bool(
                closest.hartree_fock.converged
            ),
            "closest_hartree_fock_gradient_norm": (
                closest.hartree_fock.gradient_norm
            ),
            "closest_hartree_fock_energy": closest_hartree_fock_energy,
            "closest_hartree_fock_energy_relative_error": _relative_error(
                closest_hartree_fock_energy, exact_energy
            ),
            "closest_hartree_fock_attempts": closest.hartree_fock.attempts,
            "elapsed_seconds": time.perf_counter() - isotope_started,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

        # Checkpoint after every isotope so a scheduler time limit does not
        # discard nuclei that already completed.
        checkpoint = {
            "status": "running",
            "study": "energy-optimized versus closest pure Gaussian state",
            "interaction": interaction_name,
            "interaction_path": str(interaction_path.relative_to(ROOT)),
            "isotopes": list(masses),
            "results": results,
            "elapsed_seconds": time.perf_counter() - started,
        }
        output_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    report = {
        "study": (
            "intrinsic variational optimization only"
            if variational_only
            else "energy-optimized versus closest pure Gaussian state"
        ),
        "interaction": interaction_name,
        "interaction_path": str(interaction_path.relative_to(ROOT)),
        "isotopes": list(masses),
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["status"] = "complete"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare exact nuclei with HFB/HF and closest Gaussians."
    )
    parser.add_argument(
        "--interaction", type=str.lower, choices=("cki", "usdb"), required=True
    )
    parser.add_argument(
        "--isotopes", nargs="+", type=int,
        help="Mass numbers; defaults to Be 8,10,12 or Ne 20,22,24.",
    )
    parser.add_argument(
        "--variational-method", choices=("hf", "hfb"), default="hfb"
    )
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--gaussian-starts", type=int, default=4)
    parser.add_argument("--gaussian-maxiter", type=int, default=1000)
    parser.add_argument(
        "--gaussian-hf-starts", type=int,
        help="HF maximum-overlap starts; defaults to --gaussian-starts.",
    )
    parser.add_argument(
        "--gaussian-hf-maxiter", type=int,
        help="HF maximum-overlap iterations; defaults to --gaussian-maxiter.",
    )
    parser.add_argument(
        "--real-bogoliubov",
        dest="real_bogoliubov",
        action="store_true",
        help="use real HFB and closest-Gaussian manifolds (default)",
    )
    parser.add_argument(
        "--complex-bogoliubov",
        dest="real_bogoliubov",
        action="store_false",
        help="allow complex HFB and closest-Gaussian manifolds",
    )
    parser.set_defaults(real_bogoliubov=True)
    parser.add_argument(
        "--variational-only",
        action="store_true",
        help="skip exact diagonalization and closest-Gaussian optimization",
    )
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    run_study(
        interaction_name=args.interaction,
        isotopes=args.isotopes,
        variational_method=args.variational_method,
        starts=args.starts,
        maxiter=args.maxiter,
        gaussian_starts=args.gaussian_starts,
        gaussian_maxiter=args.gaussian_maxiter,
        gaussian_hf_starts=args.gaussian_hf_starts,
        gaussian_hf_maxiter=args.gaussian_hf_maxiter,
        real_bogoliubov=args.real_bogoliubov,
        variational_only=args.variational_only,
        seed=args.seed,
        output=args.output,
    )
