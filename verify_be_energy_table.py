"""Reproduce the CKI Be energy table, with independent J=0 checks.

Run: python verify_be_energy_table.py
Energies are valence-space energies relative to the inert He4 core.
"""
import argparse
import json
import pickle
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "NSMFermions"))
if sys.platform == "win32" and (ROOT / ".hfb-test-deps").exists():
    sys.path.insert(0, str(ROOT / ".hfb-test-deps"))

import numpy as np
from scipy.sparse import csr_matrix, diags

from angular_momentum import (
    euler_rotation, polynomial_j0_grid, single_particle_angular_momentum,
)
from hfb import BogoliubovVacuumSeries, HFBHamiltonian, HFBState, solve_hfb
from nuclear_workflow import build_fermionic_hamiltonian, load_nuclear_interaction
from number_projection import exact_ground_state

QUOTED = {
    6: (-0.635699, -2.95695, -3.04880),
    8: (-25.689036, -30.19300, -30.29539),
    10: (-34.163596, -38.65735, -39.45342),
    12: (-40.501673, -40.50167, -41.22145),
}


def j_squared(fermionic, encoding, generators):
    jp = generators[0] + 1j * generators[1]
    dim = len(fermionic.occupations)
    raising = csr_matrix((dim, dim), dtype=complex)
    for i, j in zip(*np.nonzero(jp)):
        raising += jp[i, j] * fermionic.adag_a_matrix(int(i), int(j))
    magnetic = np.array([
        sum(encoding[i][3] for i in occ) for occ in fermionic.occupations
    ])
    return (
        (raising.conj().T @ raising + diags(magnetic * (magnetic + 1))).tocsr(),
        ((raising - raising.conj().T) / (2j)).toarray(), magnetic,
    )


def euler_series(state, encoding, neutron_modes, targets, generators, shape):
    grid = polynomial_j0_grid(encoding, neutron_modes, targets, grid=shape)
    rotations, weights = [], []
    for a, (x, weight), g in product(
        grid.alpha, zip(grid.cos_beta, grid.beta_weights), grid.gamma
    ):
        rotations.append(euler_rotation(a, np.arccos(x), g, generators))
        weights.append(weight / (2 * len(grid.alpha) * len(grid.gamma)))
    return BogoliubovVacuumSeries(
        state, np.asarray(rotations), np.asarray(weights),
        number_grid=(1, 1), number_grid_guaranteed_exact=False,
        euler_grid=shape, euler_grid_guaranteed_exact=grid.guaranteed_exact,
        projection="Euler J=0; exact N,Z restriction in occupation basis",
    )


def normalized(vector):
    norm = float(np.vdot(vector, vector).real)
    if norm < 1e-20:
        raise ValueError("Unresolved projected state norm")
    return vector / np.sqrt(norm), norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    parser.add_argument("--masses", nargs="+", type=int, default=[6, 8, 10, 12])
    parser.add_argument("--methods", nargs="+", default=["hf", "hfb"], choices=["hf", "hfb"])
    parser.add_argument("--output", default="results/be_energy_audit.json")
    parser.add_argument("--reuse-states", action="store_true", help="Recheck saved states without repeating optimization")
    args = parser.parse_args()
    started = time.perf_counter()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if args.reuse_states:
        with output.with_suffix(".pkl").open("rb") as stream:
            cached = pickle.load(stream)
    interaction, eps, sp, interaction_path = load_nuclear_interaction("cki", repository_root=ROOT)
    encoding = sp.state_encoding
    neutron_modes = list(range(len(eps) // 2, len(eps)))
    ham = HFBHamiltonian(np.diag(eps), interaction)
    generators = single_particle_angular_momentum(encoding)
    report = {
        "interaction": str(interaction_path), "settings": vars(args),
        "energy_convention": "MeV relative to inert He4 core",
        "projection": "P_N P_Z P_J=0 after intrinsic variation",
        "hfb_manifold": "real Bogoliubov", "results": [],
    }
    states = {}

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        with output.with_suffix(".pkl").open("wb") as stream:
            pickle.dump(states, stream, protocol=pickle.HIGHEST_PROTOCOL)

    for mass in args.masses:
        targets = (mass - 6, 2)
        fermionic = build_fermionic_hamiltonian(interaction, eps, particles=targets)
        exact_energy, ground = exact_ground_state(fermionic)
        ground, _ = normalized(np.asarray(ground, complex))
        J2, Jy, magnetic = j_squared(fermionic, encoding, generators)
        jy_values, jy_vectors = np.linalg.eigh(Jy)
        j2_values, j2_vectors = np.linalg.eigh(J2.toarray())
        singlets = j2_vectors[:, np.abs(j2_values) < 1e-8]
        grid = polynomial_j0_grid(encoding, neutron_modes, targets).minimum_grid
        enlarged = tuple(n + 2 for n in grid)
        print(f"Be{mass}: exact={exact_energy:.10f}; Euler grid={grid}", flush=True)
        states[mass] = {"ground": ground, "occupations": fermionic.occupations}
        for method in args.methods:
            print(f"Be{mass}: starting {method}, {args.starts} starts", flush=True)
            solution = cached[mass][method]["solution"] if args.reuse_states else solve_hfb(
                ham, neutron_modes, targets, method=method,
                starts=args.starts, seed=args.seed + mass, maxiter=args.maxiter,
                tolerance=1e-9, gradient_tolerance=args.gradient_tolerance,
                real_bogoliubov=(method == "hfb"),
            )
            state = solution.state
            rho = (state.rho + state.rho.conj().T) / 2
            pairing = float(np.linalg.norm(state.kappa))
            idempotency = float(np.linalg.norm(rho @ rho - rho))
            converted = pairing < 1e-3 and idempotency < 1e-6
            if converted:
                _, orbitals = np.linalg.eigh(rho)
                state = HFBState.from_slater(orbitals[:, -sum(targets):])
            row = {
                "nucleus": f"Be{mass}", "method": method,
                "targets_N_Z": targets, "converged": bool(solution.converged),
                "stationarity_error": float(solution.stationarity_error),
                "numbers_N_Z": solution.numbers.tolist(),
                "pairing_norm": pairing, "idempotency_error": idempotency,
                "slater_boundary_conversion": bool(converted),
                "intrinsic_energy_raw": float(solution.energy),
                "intrinsic_energy_working": float(ham.energy(state)),
                "exact_energy": float(exact_energy),
                "ground_J2_residual": float(np.linalg.norm(J2 @ ground)),
                "euler_grid": grid, "enlarged_euler_grid": enlarged,
                "attempts": solution.attempts,
            }
            print(f"Be{mass} {method}: intrinsic={solution.energy:.10f}, converged={solution.converged}, pairing={pairing:.3e}", flush=True)
            states[mass][method] = {"solution": solution, "working_state": state}
            intrinsic = state.stable_normalized_occupation_amplitudes(fermionic.occupations)
            spectral, spectral_weight = normalized(singlets @ (singlets.conj().T @ intrinsic))
            spectral_energy = float(np.vdot(spectral, fermionic.matrix @ spectral).real)
            row["spectral_check_energy"] = spectral_energy
            row["spectral_check_weight"] = spectral_weight
            for key, shape in (("projected", grid), ("enlarged", enlarged)):
                series = euler_series(state, encoding, neutron_modes, targets, generators, shape)
                try:
                    psi, weight = normalized(series.occupation_amplitudes(fermionic.occupations))
                except ValueError as error:
                    row["euler_projection_error"] = str(error)
                    # Same Euler quadrature acting directly in the fixed-N,Z
                    # space. This preserves phases without a singular U chart.
                    rule = polynomial_j0_grid(encoding, neutron_modes, targets, grid=shape)
                    alpha = np.exp(-1j * rule.alpha[:, None] * magnetic).mean(axis=0)
                    gamma = np.exp(-1j * rule.gamma[:, None] * magnetic).mean(axis=0)
                    beta = (rule.beta_weights[:, None] * np.exp(
                        -1j * np.arccos(rule.cos_beta)[:, None] * jy_values
                    )).sum(axis=0) / 2
                    psi, weight = normalized(alpha * (jy_vectors @ (
                        beta * (jy_vectors.conj().T @ (gamma * intrinsic))
                    )))
                    row["euler_fallback"] = "Exact many-body representation of the same Euler grid"
                row[f"{key}_energy"] = float(np.vdot(psi, fermionic.matrix @ psi).real)
                row[f"{key}_weight"] = weight
                row[f"{key}_fidelity"] = float(abs(np.vdot(ground, psi)) ** 2)
                row[f"{key}_J2_residual"] = float(np.linalg.norm(J2 @ psi))
                row[f"{key}_spectral_infidelity"] = float(1 - abs(np.vdot(spectral, psi)) ** 2)
                if key == "projected":
                    states[mass][method] = {"solution": solution, "working_state": state, "projected": psi}
            row["delta_intrinsic_vs_quoted"] = row["intrinsic_energy_raw"] - QUOTED[mass][0]
            if "projected_energy" in row:
                row["delta_projected_vs_quoted"] = row["projected_energy"] - QUOTED[mass][1]
            row["delta_exact_vs_quoted"] = row["exact_energy"] - QUOTED[mass][2]
            report["results"].append(row)
            save()
            print(f"Be{mass} {method}: E={row['intrinsic_energy_raw']:.10f}, E_J0={row.get('projected_energy', 'chart failure')}, exact={exact_energy:.10f}, J2 residual={row.get('projected_J2_residual')}", flush=True)
    report["status"] = "complete"
    save()
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
