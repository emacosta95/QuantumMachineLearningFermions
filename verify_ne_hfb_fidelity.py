"""Check paired HFB against the same exact states as the HF audit."""
import argparse
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
from nuclear_workflow import load_nuclear_interaction

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--starts',type=int,default=8)
parser.add_argument('--maxiter',type=int,default=2000)
parser.add_argument('--masses',type=int,nargs='+',default=list(range(18,31,2)))
parser.add_argument('--gradient-tolerance',type=float,default=1e-5)
parser.add_argument('--number-tolerance',type=float,default=1e-9,
                    help='The solve_hfb tolerance argument; internal number bound is max(1e-8, 10*tolerance).')
parser.add_argument('--output',default='results/ne_hfb_fidelity_audit.json')
args=parser.parse_args()
output=ROOT/args.output

with (ROOT/'results/ne_hf_fidelity_audit.pkl').open('rb') as stream:
    exact = pickle.load(stream)
interaction, eps, sp, _ = load_nuclear_interaction('usdb',repository_root=ROOT)
ham = HFBHamiltonian(np.diag(eps),interaction)
rows=[]
states={}
for mass, data in exact.items():
    if mass not in args.masses:
        continue
    N=mass-18
    print(f'Ne{mass}: HFB optimization',flush=True)
    result = solve_hfb(ham,range(12,24),(N,2),method='hfb',real_bogoliubov=True,
                       starts=args.starts,maxiter=args.maxiter,seed=34+N,
                       tolerance=args.number_tolerance,gradient_tolerance=args.gradient_tolerance)
    state=result.state
    amplitudes=state.stable_normalized_occupation_amplitudes(data['occupations'])
    fidelity=float(abs(np.vdot(data['ground'],amplitudes))**2)
    rho=state.rho
    row=dict(mass=mass,energy=float(result.energy),fidelity=fidelity,infidelity=1-fidelity,
             pairing_norm=float(np.linalg.norm(state.kappa)),
             idempotency_error=float(np.linalg.norm(rho@rho-rho)),
             converged=bool(result.converged),gradient_norm=float(result.stationarity_error),
             numbers=result.numbers.tolist(),sector_weight=float(np.vdot(amplitudes,amplitudes).real),
             attempts=result.attempts)
    rows.append(row)
    states[mass]=result
    output.write_text(json.dumps(rows,indent=2))
    with output.with_suffix('.pkl').open('wb') as stream:
        pickle.dump(states,stream,protocol=pickle.HIGHEST_PROTOCOL)
    print({k:v for k,v in row.items() if k!='attempts'},flush=True)
