from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Tuple
import itertools
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = Path.cwd()
if not (ROOT / "src" / "NSMFermions").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src" / "NSMFermions"))

from nuclear_workflow import (
    build_fermionic_hamiltonian,
    load_nuclear_interaction,
)
from hfb import HFBHamiltonian, HFBState, BogoliubovVacuumSeries, solve_hfb
from number_projection import (
    exact_ground_state,
    number_projected_series,
    projected_series_observables,
)
from angular_momentum import (
    ParticleNumberJ0ProjectedEnergy,
    single_particle_angular_momentum,
    euler_rotation,
    project_state_observables,
)
from gaussian_fidelity import maximize_gaussian_fidelity, maximize_slater_fidelity
from fermionic_antiflatness import (
    fermionic_antiflatness,
    vacuum_series_antiflatness,
)

interaction, eps, state_encoding, _ = load_nuclear_interaction(
    "cki", repository_root=ROOT
)
ham = HFBHamiltonian(np.diag(eps), interaction)
neutron_modes = list(range(6, 12))
proton_modes = list(range(6))

proton_numbers = [2]
neutron_numbers = np.arange(0, 8, 2)

relative_errors_in_energy_bhf = []
fidelities_bhf = []
is_hartree_fock_energy = []
fidelities_bhf_gaussian = []
is_hartree_fock_gaussian = []
measure_from_relative_errors = []
measure_from_fidelities = []
pav_fidelities = []
pav_sector_weights = []
exact_ground_state_faf = []
pav_faf = []
labels = []
for Z in proton_numbers:
    for N in neutron_numbers:

        targets = [N, Z]
        if any(number < 0 or number > 6 for number in targets):
            raise ValueError("CKI has only six valence modes for each species")
        if sum(targets) == 0:
            raise ValueError("This tutorial expects at least one valence particle")
        if sum(targets) % 2:
            raise ValueError(
                "The present HFB vacuum and J=0 projector require even total valence number; "
                "odd systems need blocked HFB and a J>0 projection workflow."
            )
        fermionic = build_fermionic_hamiltonian(
            interaction, eps, particles=tuple(targets)
        )
        exact_energy, exact_target = exact_ground_state(fermionic)

        print("valence (N,Z):", tuple(targets))
        print("nuclear (N,Z) with the 4He core:", (targets[0] + 2, targets[1] + 2))
        print("modes:", len(eps))
        print("fixed-(N,Z) dimension:", len(fermionic.occupations))
        print("exact ground-state energy:", exact_energy, "MeV")
        labels.append(rf"$^{{{targets[0]+2+targets[1]+2}}}$Be$")
        ##################### compute the HFB variational problem ####################
        hfb_result = solve_hfb(
            ham,
            neutron_modes,
            targets,
            starts=2,
            seed=8,
            maxiter=120,
            tolerance=1e-8,
        )
        hfb_raw = hfb_result.state
        print("optimizer converged:", hfb_result.converged)
        print("optimizer attempts:", hfb_result.attempts)

        rho_hfb = (hfb_raw.rho + hfb_raw.rho.conj().T) / 2
        rho_idempotency = np.linalg.norm(rho_hfb @ rho_hfb - rho_hfb)
        pairing_norm = np.linalg.norm(hfb_raw.kappa)
        occupations_rho, natural_orbitals = np.linalg.eigh(rho_hfb)
        hfb_orbitals = None
        # if pairing collapse the best mean field solution is HF.
        if pairing_norm < 1e-4 and rho_idempotency < 1e-6:
            # Use the exact Slater chart only when the optimized state has collapsed.
            hfb_orbitals = natural_orbitals[:, -sum(targets) :]
            hfb_state = HFBState.from_slater(hfb_orbitals)
            hfb_chart = "Slater (pairing collapsed)"
        else:
            # Retain genuine pairing for nuclei whose HFB minimum is not a Slater state.
            hfb_state = hfb_raw
            hfb_chart = "paired Bogoliubov vacuum"

        print("raw HFB energy:", ham.energy(hfb_raw), "MeV")
        print("state used below:", hfb_chart)
        print("working-state energy:", ham.energy(hfb_state), "MeV")
        print("||kappa||:", pairing_norm)
        print("||rho^2-rho||:", rho_idempotency)
        print("working-state canonical error:", hfb_state.canonical_error())

        hfb_fidelity = hfb_state.fixed_sector_fidelity(
            exact_target, fermionic.occupations
        )
        # Particle-number projection after variation (PAV).  The fidelity and
        # FAF are evaluated from the same coherent gauge-vacuum series.
        pav_series = number_projected_series(hfb_state, fermionic)
        pav = projected_series_observables(
            pav_series, fermionic, exact_target
        )
        exact_faf = fermionic_antiflatness(
            exact_target,
            fermionic.occupations,
            fermionic.modes,
            order=2,
        )
        projected_faf = vacuum_series_antiflatness(
            pav_series, fermionic.occupations, order=2
        )
        pav_fidelities.append(pav.fidelity)
        pav_sector_weights.append(pav.sector_weight)
        exact_ground_state_faf.append(exact_faf.value)
        pav_faf.append(projected_faf.value)
        print("particle-number PAV fidelity:", pav.fidelity)
        print("particle-number PAV sector weight:", pav.sector_weight)
        print("exact-ground-state FAF (k=2):", exact_faf.value)
        print("particle-number PAV FAF (k=2):", projected_faf.value)
        relative_error = abs(ham.energy(hfb_raw) - exact_energy) / abs(exact_energy)
        relative_errors_in_energy_bhf.append(relative_error)
        fidelities_bhf.append(hfb_fidelity)

        # hartree fock ansatz
        slater_best = maximize_slater_fidelity(
            fermionic,
            exact_target,
            starts=2,
            seed=42,
            maxiter=1500,
            gradient_tolerance=2e-7,
            initial_orbitals=hfb_orbitals,
        )
        # gaussian_candidates = [("Slater boundary", slater_best.fidelity)]
        # interior_best = None
        # # bhf ansatz
        # interior_best = maximize_gaussian_fidelity(
        #     fermionic,
        #     exact_target,
        #     starts=2,
        #     seed=41,
        #     maxiter=1500,
        #     gradient_tolerance=2e-6,
        # )
        # gaussian_candidates.append(("finite-Z interior", interior_best.fidelity))

        # best_kind, exact_best_gaussian_fidelity = max(
        #     gaussian_candidates, key=lambda item: item[1]
        # )
        # if best_kind == "Slater boundary":
        #     best_gaussian_state = HFBState.from_slater(slater_best.orbitals)
        # else:
        #     best_gaussian_state = interior_best.state

        # print("candidates:", gaussian_candidates)
        # print("selected:", best_kind)
        # print("best-found Gaussian fidelity:", exact_best_gaussian_fidelity)

        fidelities_bhf_gaussian.append(slater_best.fidelity)
        is_hartree_fock_energy.append(pairing_norm < 1e-4)
        is_hartree_fock_gaussian.append(
            True
        )  # the best Gaussian is always a Slater state in this tutorial

        measure_from_relative_errors.append(-np.log10(1 - relative_error))
        measure_from_fidelities.append(-np.log10(hfb_fidelity))


import pickle as pkl

with open("data/results_gaussianity/results.pkl", "wb") as f:
    pkl.dump(
        {
            "relative_errors_in_energy_bhf": relative_errors_in_energy_bhf,
            "fidelities_bhf": fidelities_bhf,
            "is_hartree_fock_energy": is_hartree_fock_energy,
            "fidelities_bhf_gaussian": fidelities_bhf_gaussian,
            "is_hartree_fock_gaussian": is_hartree_fock_gaussian,
            "measure_from_relative_errors": measure_from_relative_errors,
            "measure_from_fidelities": measure_from_fidelities,
            "pav_fidelities": pav_fidelities,
            "pav_sector_weights": pav_sector_weights,
            "exact_ground_state_faf_k2": exact_ground_state_faf,
            "pav_faf_k2": pav_faf,
            "labels": labels,
        },
        f,
    )
