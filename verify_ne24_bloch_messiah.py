"""Independent explicit canonical-pair expansion of the audited real Ne24 state."""
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src/NSMFermions'))
if sys.platform=='win32' and (ROOT/'.hfb-test-deps').exists():
    sys.path.insert(0,str(ROOT/'.hfb-test-deps'))
import numpy as np

with (ROOT/'results/ne_hfb_fidelity_audit.pkl').open('rb') as f:
    state=pickle.load(f)[24].state
with (ROOT/'results/ne_hf_fidelity_audit.pkl').open('rb') as f:
    exact=pickle.load(f)[24]
assert np.linalg.norm(state.rho.imag)+np.linalg.norm(state.kappa.imag)<1e-12
occupations=np.asarray(exact['occupations'],int)
eigenvalues,D=np.linalg.eigh((state.rho.real+state.rho.real.T)/2)
# The audited real state has adjacent doubly degenerate natural occupations.
assert np.max(abs(eigenvalues[::2]-eigenvalues[1::2]))<1e-10
n=np.clip((eigenvalues[::2]+eigenvalues[1::2])/2,0,1)
canonical_kappa=D.T@state.kappa.real@D
signs=np.where(canonical_kappa[np.arange(0,24,2),np.arange(1,24,2)]<0,-1.,1.)
u=np.sqrt(1-n)
v=signs*np.sqrt(n)
K=np.zeros((24,24))
for k in range(12):
    K[2*k,2*k+1]=u[k]*v[k]
    K[2*k+1,2*k]=-u[k]*v[k]
rho_error=float(np.linalg.norm(D@np.diag(np.repeat(n,2))@D.T-state.rho))
kappa_error=float(np.linalg.norm(D@K@D.T-state.kappa))
assert rho_error<1e-9 and kappa_error<1e-7, (rho_error,kappa_error)

# Directly expand product_k (u_k+v_k d_(2k)^dag d_(2k+1)^dag)|0>.
# Only four occupied pairs contribute to this eight-particle sector.
amplitudes=np.zeros(len(occupations),complex)
orbital_rows=D[occupations,:]
terms=0
for selected in combinations(range(12),4):
    factors=u.copy()
    factors[list(selected)]=v[list(selected)]
    coefficient=np.prod(factors)
    if abs(coefficient)<1e-15:
        continue
    columns=[i for k in selected for i in (2*k,2*k+1)]
    amplitudes+=coefficient*np.linalg.det(orbital_rows[:,:,columns])
    terms+=1
library=state.stable_normalized_occupation_amplitudes(exact['occupations'])
phase=np.vdot(amplitudes,library)
phase/=abs(phase)
error=float(np.max(abs(library-phase*amplitudes)))
result=dict(canonical_pair_terms=terms,rho_reconstruction_error=rho_error,
            kappa_reconstruction_error=kappa_error,amplitude_max_error=error,
            bloch_messiah_fidelity=float(abs(np.vdot(exact['ground'],amplitudes))**2),
            library_fidelity=float(abs(np.vdot(exact['ground'],library))**2))
assert error<1e-7, result
(ROOT/'results/ne24_bloch_messiah_audit.json').write_text(json.dumps(result,indent=2))
print(result,flush=True)
