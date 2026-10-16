"""Independent HF/HFB optimization against the matched Ne24 TAURUS benchmark."""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src/NSMFermions'))
if sys.platform == 'win32':
    sys.path.insert(0, str(ROOT/'.hfb-test-deps'))
import numpy as np
from hfb import HFBHamiltonian, solve_hfb
from nuclear_workflow import load_nuclear_interaction, build_fermionic_hamiltonian

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--methods', nargs='+', default=['hf', 'hfb'], choices=['hf', 'hfb'])
parser.add_argument('--starts', type=int, default=8)
parser.add_argument('--maxiter', type=int, default=2000)
args = parser.parse_args()
interaction, eps, sp, _ = load_nuclear_interaction('usdb', mass_number=24)
ham = HFBHamiltonian(np.diag(eps), interaction)
reference = np.load(ROOT/'results/taurus_import_audit/Ne24.npz')
ground, occupations = reference['ground'], reference['occupations']
exact = build_fermionic_hamiltonian(interaction, eps, (6, 2), symmetries=[sp.total_M_zero])
assert np.array_equal(occupations, np.asarray(exact.occupations))
assert np.linalg.norm(exact.hamiltonian@ground-reference['ground_energy']*ground) < 1e-7
rows = []
states = {}
output = ROOT/'results/ne24_scaled_variational_audit.json'
for method in args.methods:
    print(f'Optimizing Ne24 {method}: {args.starts} independent starts', flush=True)
    result = solve_hfb(ham, range(12,24), (6,2), method=method,
                       starts=args.starts, maxiter=args.maxiter,
                       seed=32 if method == 'hf' else 40,
                       tolerance=1e-9, gradient_tolerance=1e-5,
                       real_bogoliubov=True)
    state = result.state
    amplitudes = state.stable_normalized_occupation_amplitudes(occupations)
    fidelity = float(abs(np.vdot(ground, amplitudes))**2)
    row = dict(method=method, energy=float(result.energy), fidelity=fidelity,
               infidelity=1-fidelity, converged=bool(result.converged),
               gradient_norm=float(result.stationarity_error),
               numbers=result.numbers.tolist(),
               proton_pairing_norm=float(np.linalg.norm(state.kappa[:12,:12])),
               neutron_pairing_norm=float(np.linalg.norm(state.kappa[12:,12:])),
               attempts=result.attempts)
    if method == 'hf':
        determinants = np.linalg.det(state.slater_orbitals[occupations, :])
        row['determinant_fidelity'] = float(abs(np.vdot(ground, determinants))**2)
        assert abs(row['determinant_fidelity']-fidelity) < 1e-10
    rows.append(row)
    states[method] = result
    output.write_text(json.dumps(dict(mass=24, mass_factor=(18/24)**0.3,
                                     exact_energy=float(reference['ground_energy']),
                                     results=rows), indent=2))
    with output.with_suffix('.pkl').open('wb') as f:
        pickle.dump(states, f)
    print({k:v for k,v in row.items() if k!='attempts'}, flush=True)
