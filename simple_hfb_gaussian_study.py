"""Minimal HFB-energy and maximum-Gaussian-overlap nuclear study.

Edit the configuration block below, then run

    python simple_hfb_gaussian_study.py

The output is a pickled nested dictionary indexed as
``data["results"][valence_protons][valence_neutrons]``.  Pickle files must only
be loaded from trusted sources.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration: these are the only values normally changed by the user.
# ---------------------------------------------------------------------------

INTERACTION = "usdb"              # "usdb" or "cki"
VALENCE_PROTONS = [2]
VALENCE_NEUTRONS = [2]             # for example [2, 4, 6] for Ne20,22,24
# Each species has capacity 12 for USDB and 6 for CKI. For an even-even grid,
# use range(0, 13, 2) for USDB or range(0, 7, 2) for CKI.

HFB_STARTS = 3                     # use at least 8 for production
HFB_MAXITER = 500
GAUSSIAN_STARTS = 3                # use at least 8 for production
GAUSSIAN_MAXITER = 1000
REAL_BOGOLIUBOV = True
SEED = 8

OUTPUT = Path("results") / f"{INTERACTION}_simple_hfb_gaussian.pkl"


# ---------------------------------------------------------------------------
# Library imports.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "src" / "NSMFermions"
if str(LIBRARY) not in sys.path:
    sys.path.insert(0, str(LIBRARY))

from gaussian_fidelity import maximize_best_gaussian_fidelity  # noqa: E402
from hfb import HFBHamiltonian, HFBState, solve_hfb  # noqa: E402
from nuclear_workflow import (  # noqa: E402
    build_fermionic_hamiltonian,
    load_nuclear_interaction,
)
from number_projection import exact_ground_state  # noqa: E402


def stable_energy_state(state, particle_number):
    """Return the energy solution in a stable HFBState representation.

    A pairing-collapsed HFB solution is physically a Hartree-Fock determinant.
    Its particle-vacuum Thouless matrix is singular, so we reconstruct the
    equivalent Slater state from the occupied natural orbitals.  A genuinely
    paired solution is already stored by its canonical ``U,V`` matrices.
    """
    rho = (state.rho + state.rho.conj().T) / 2
    pairing_norm = float(np.linalg.norm(state.kappa))
    idempotency_error = float(np.linalg.norm(rho @ rho - rho))

    if pairing_norm < 1e-3 and idempotency_error < 1e-6:
        occupations, orbitals = np.linalg.eigh(rho)
        order = np.argsort(occupations)
        selected = order[-particle_number:] if particle_number else order[:0]
        occupied = orbitals[:, selected]
        return (
            HFBState.from_slater(occupied),
            "hartree_fock",
            pairing_norm,
            idempotency_error,
        )

    return state, "hfb", pairing_norm, idempotency_error


def overlap_and_fidelity(state, occupations, exact_state):
    """Return ``<exact|state>`` and its squared magnitude."""
    amplitudes = state.stable_normalized_occupation_amplitudes(occupations)
    overlap = complex(np.vdot(exact_state, amplitudes))
    return overlap, float(abs(overlap) ** 2)


def particle_number_expectations(state, neutron_modes):
    """Return ``[<N>, <Z>]`` from the normal density."""
    neutron_mask = np.zeros(len(state.U), dtype=bool)
    neutron_mask[np.asarray(neutron_modes, dtype=int)] = True
    diagonal = np.diag(state.rho).real
    return np.array(
        [diagonal[neutron_mask].sum(), diagonal[~neutron_mask].sum()]
    )


def validate_particle_lists(protons, neutrons, capacity):
    """Validate the requested valence-particle grid for unblocked HFB."""
    proton_values = [int(value) for value in protons]
    neutron_values = [int(value) for value in neutrons]
    if not proton_values or not neutron_values:
        raise ValueError("VALENCE_PROTONS and VALENCE_NEUTRONS cannot be empty")
    if any(value < 0 or value > capacity for value in proton_values):
        raise ValueError(f"Valence protons must lie between 0 and {capacity}")
    if any(value < 0 or value > capacity for value in neutron_values):
        raise ValueError(f"Valence neutrons must lie between 0 and {capacity}")
    odd_total = [
        (protons_value, neutrons_value)
        for protons_value in proton_values
        for neutrons_value in neutron_values
        if (protons_value + neutrons_value) % 2
    ]
    if odd_total:
        raise ValueError(
            "Unblocked HFBState has even number parity. Remove odd-total "
            f"(Zval, Nval) pairs or implement blocked HFB: {odd_total}"
        )
    return proton_values, neutron_values


def save_checkpoint(data, output):
    """Save the complete nested dictionary after every finished nucleus."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)


def run_study():
    interaction, eps, single_particle, interaction_path = (
        load_nuclear_interaction(INTERACTION, repository_root=ROOT)
    )
    modes = len(eps)
    species_capacity = modes // 2
    proton_values, neutron_values = validate_particle_lists(
        VALENCE_PROTONS, VALENCE_NEUTRONS, species_capacity
    )
    neutron_modes = list(range(species_capacity, modes))
    intrinsic_hamiltonian = HFBHamiltonian(np.diag(eps), interaction)

    data = {
        "interaction": INTERACTION,
        "interaction_path": str(interaction_path.relative_to(ROOT)),
        "valence_protons": proton_values,
        "valence_neutrons": neutron_values,
        "single_particle_encoding": tuple(single_particle.state_encoding),
        "settings": {
            "hfb_starts": HFB_STARTS,
            "hfb_maxiter": HFB_MAXITER,
            "gaussian_starts": GAUSSIAN_STARTS,
            "gaussian_maxiter": GAUSSIAN_MAXITER,
            "real_bogoliubov": REAL_BOGOLIUBOV,
            "seed": SEED,
        },
        "results": {},
    }

    for valence_protons in proton_values:
        data["results"][valence_protons] = {}

        for valence_neutrons in neutron_values:
            targets = (valence_neutrons, valence_protons)  # solver order: N,Z
            particles = valence_neutrons + valence_protons
            nucleus_seed = (
                SEED
                + valence_protons * (species_capacity + 1)
                + valence_neutrons
            )
            print(
                f"\n[Zval={valence_protons}, Nval={valence_neutrons}]",
                flush=True,
            )

            # 1. HFB energy minimization only. It may collapse to the HF boundary.
            variational = solve_hfb(
                intrinsic_hamiltonian,
                neutron_modes,
                targets,
                method="hfb",
                starts=HFB_STARTS,
                maxiter=HFB_MAXITER,
                seed=nucleus_seed,
                tolerance=1e-8,
                real_bogoliubov=REAL_BOGOLIUBOV,
            )
            energy_state, energy_family, pairing_norm, idempotency_error = (
                stable_energy_state(variational.state, particles)
            )

            # 2. Exact ground state in the fixed-(N,Z,M=0) determinant basis.
            exact_hamiltonian = build_fermionic_hamiltonian(
                interaction,
                eps,
                particles=targets,
                symmetries=[single_particle.total_M_zero],
            )
            exact_energy, exact_state = exact_ground_state(exact_hamiltonian)
            exact_state = np.asarray(exact_state, complex)
            exact_state /= np.linalg.norm(exact_state)
            occupations = tuple(exact_hamiltonian.occupations)

            energy_overlap, energy_fidelity = overlap_and_fidelity(
                energy_state, occupations, exact_state
            )
            energy_state_energy = intrinsic_hamiltonian.energy(energy_state)

            # 3. Direct maximum-overlap search over paired HFB and HF boundary.
            closest = maximize_best_gaussian_fidelity(
                exact_hamiltonian,
                exact_state,
                bogoliubov_starts=GAUSSIAN_STARTS,
                hartree_fock_starts=GAUSSIAN_STARTS,
                seed=nucleus_seed + 1000,
                bogoliubov_maxiter=GAUSSIAN_MAXITER,
                hartree_fock_maxiter=GAUSSIAN_MAXITER,
                gradient_tolerance=2e-6,
                real_parameters=REAL_BOGOLIUBOV,
            )
            overlap_state = closest.state  # always an HFBState, also for HF
            overlap_family = (
                "hartree_fock"
                if closest.family == "hartree_fock"
                else "hfb"
            )
            gaussian_overlap, gaussian_fidelity = overlap_and_fidelity(
                overlap_state, occupations, exact_state
            )
            gaussian_energy = intrinsic_hamiltonian.energy(overlap_state)
            gaussian_numbers = particle_number_expectations(
                overlap_state, neutron_modes
            )

            data["results"][valence_protons][valence_neutrons] = {
                "targets_N_Z": targets,
                "exact": {
                    "state": exact_state,
                    "energy": float(exact_energy),
                    "occupations": occupations,
                    "sector": "fixed (N,Z,M=0)",
                },
                "energy_solution": {
                    "state": energy_state,
                    "family": energy_family,
                    "energy": float(energy_state_energy),
                    "raw_optimizer_energy": float(variational.energy),
                    "overlap_with_ground_state": energy_overlap,
                    "ground_state_fidelity": energy_fidelity,
                    "converged": bool(variational.converged),
                    "numbers_N_Z": np.asarray(variational.numbers),
                    "pairing_norm": pairing_norm,
                    "density_idempotency_error": idempotency_error,
                    "stationarity_error": float(
                        variational.stationarity_error
                    ),
                },
                "maximum_overlap_solution": {
                    "state": overlap_state,
                    "family": overlap_family,
                    "energy": float(gaussian_energy),
                    "numbers_N_Z": gaussian_numbers,
                    "overlap_with_ground_state": gaussian_overlap,
                    "ground_state_fidelity": gaussian_fidelity,
                    "converged": bool(closest.converged),
                    "gradient_norm": float(closest.gradient_norm),
                    "bogoliubov_fidelity": float(
                        closest.bogoliubov.fidelity
                    ),
                    "hartree_fock_fidelity": float(
                        closest.hartree_fock.fidelity
                    ),
                },
            }
            save_checkpoint(data, OUTPUT)

            print(
                f"  exact: E={exact_energy:.10f}\n"
                f"  energy {energy_family}: E={energy_state_energy:.10f}, "
                f"F={energy_fidelity:.10f}\n"
                f"  maximum-overlap {overlap_family}: "
                f"E={gaussian_energy:.10f}, F={gaussian_fidelity:.10f}",
                flush=True,
            )

    print(f"\nSaved {OUTPUT.resolve()}")
    return data


if __name__ == "__main__":
    run_study()
