"""Check intrinsic HF fidelities using independent Slater determinants."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "NSMFermions"))
if sys.platform == "win32" and (ROOT / ".hfb-test-deps").exists():
    sys.path.insert(0, str(ROOT / ".hfb-test-deps"))
import numpy as np
from hfb import HFBHamiltonian, solve_hfb
from nuclear_workflow import load_nuclear_interaction, build_fermionic_hamiltonian
from number_projection import exact_ground_state

interaction, eps, sp, _ = load_nuclear_interaction("usdb", repository_root=ROOT)
ham = HFBHamiltonian(np.diag(eps), interaction)
report = {"interaction": "usdb", "starts": 8, "maxiter": 2000, "results": []}
states = {}
for N in range(0, 13, 2):
    mass = 18 + N
    print(f"Ne{mass}: optimizing HF", flush=True)
    hf = solve_hfb(ham, range(12, 24), (N, 2), method="hf", starts=8,
                   maxiter=2000, seed=8 + mass, tolerance=1e-9)
    print(f"Ne{mass}: E_HF={hf.energy:.10f}; building M=0 matrix", flush=True)
    exact = build_fermionic_hamiltonian(interaction, eps, particles=(N, 2),
                                      symmetries=[sp.total_M_zero])
    energy, ground = exact_ground_state(exact)
    ground = np.asarray(ground, complex) / np.linalg.norm(ground)
    occ = np.asarray(exact.occupations, dtype=int)
    amplitudes = hf.state.stable_normalized_occupation_amplitudes(exact.occupations)
    orbitals = hf.state.slater_orbitals
    determinants = np.linalg.det(orbitals[occ, :])
    fidelity = float(abs(np.vdot(ground, amplitudes)) ** 2)
    direct_fidelity = float(abs(np.vdot(ground, determinants)) ** 2)
    weight = float(np.vdot(amplitudes, amplitudes).real)
    row = dict(mass=mass, valence_neutrons=N, intrinsic_energy=hf.energy,
               exact_energy=float(energy), converged=bool(hf.converged),
               stationarity_error=float(hf.stationarity_error),
               fidelity=fidelity, infidelity=1-fidelity,
               determinant_fidelity=direct_fidelity, M0_weight=weight,
               fidelity_after_M0_normalization=fidelity/weight,
               negative_log_fidelity=float(-np.log(fidelity)),
               attempts=hf.attempts)
    report["results"].append(row)
    states[mass] = dict(hf=hf, ground=ground, occupations=exact.occupations)
    (ROOT / "results/ne_hf_fidelity_audit.json").write_text(json.dumps(report, indent=2))
    with (ROOT / "results/ne_hf_fidelity_audit.pkl").open("wb") as stream:
        pickle.dump(states, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Ne{mass}: fidelity={fidelity:.9f}, infidelity={1-fidelity:.9f}, "
          f"independent difference={fidelity-direct_fidelity:.2e}", flush=True)
