"""Bounded CKI Be8 benchmark. Run from repository root with NumPy/SciPy.

Reads selected legacy definitions by AST to avoid optional ML/Numba imports;
the interaction conversion code itself is executed unchanged.
"""
import ast
import itertools
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, ClassVar
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/NSMFermions'))
from hfb import HFBHamiltonian, HFBState, solve_hfb


def legacy_definitions(filename, names, namespace):
    tree = ast.parse((ROOT/'src/NSMFermions'/filename).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Legacy definition missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)


def annihilators(m):
    result = []
    for i in range(m):
        cols = [n for n in range(2**m) if n & (1 << i)]
        rows = [n ^ (1 << i) for n in cols]
        values = [(-1)**bin(n & ((1 << i)-1)).count('1') for n in cols]
        result.append(sparse.csr_matrix((values,(rows,cols)),shape=(2**m,2**m)))
    return result


def main():
    start = time.perf_counter()
    ns = dict(globals(), trange=range)
    legacy_definitions('cg_utils.py',['CG','ClebschGordan','SelectCG',
        'CreateInitialCGList','CalcInitialValues','DivCalc','CgJM'],ns)
    legacy_definitions('nuclear_physics_utils.py',
        ['SingleParticleState','krond','scattering_matrix_reader',
         'compute_nuclear_twobody_matrix','get_twobody_nuclearshell_model'],ns)
    interaction, eps = ns['get_twobody_nuclearshell_model'](str(ROOT/'data/cki'))
    sp = ns['SingleParticleState'](str(ROOT/'data/cki'))
    ham = HFBHamiltonian(np.diag(eps),interaction)
    m = len(eps)
    # Existing study labels t_z=-1/2 as neutrons; reader emits protons first.
    neutrons = [i for i,s in enumerate(sp.state_encoding) if s[-1] == -.5]
    protons = [i for i in range(m) if i not in neutrons]
    occupations = [tuple(sorted(n+p)) for n in itertools.combinations(neutrons,2)
                   for p in itertools.combinations(protons,2)]
    masks = [sum(1<<i for i in occ) for occ in occupations]
    lookup = {mask:i for i,mask in enumerate(masks)}
    exact = np.diag([sum(eps[i] for i in occ) for occ in occupations]).astype(complex)
    # Independent bit-string implementation of c_i† c_j† c_l c_k.
    for (i,j,k,l),v in interaction.items():
        for col,mask in enumerate(masks):
            sign = 1
            for mode,create in [(k,False),(l,False),(j,True),(i,True)]:
                if bool(mask & (1<<mode)) == create:
                    break
                sign *= (-1)**bin(mask & ((1<<mode)-1)).count('1')
                mask ^= 1<<mode
            else:
                if mask in lookup:
                    exact[lookup[mask],col] += .25*v*sign
    assert np.allclose(exact,exact.conj().T)
    mzero = np.array([sum(sp.state_encoding[i][3] for i in occ)==0 for occ in occupations])
    e,vec = np.linalg.eigh(exact[np.ix_(mzero,mzero)])
    # Compare unchanged legacy two-body operator assembly for every tensor entry.
    ns.update(lil_matrix=sparse.lil_matrix,coo_matrix=sparse.coo_matrix)
    legacy_definitions('fermi_hubbard_library.py',['FemionicBasis'],ns)
    basis = ns['FemionicBasis'].__new__(ns['FemionicBasis'])
    basis.size_a = basis.size_b = 6
    basis.basis = np.array([[int(mask>>i & 1) for i in range(m)] for mask in masks])
    basis.encode = basis._get_the_encode()
    legacy = np.diag([sum(eps[i] for i in occ) for occ in occupations]).astype(complex)
    for (i,j,k,l),v in interaction.items():
        legacy += .25*v*basis.adag_adag_a_a_matrix(i,j,l,k).toarray()
    discrepancy = float(np.max(np.abs(legacy-exact)))
    assert discrepancy < 1e-10
    print('Exact energy',e[0], 'dimension',len(masks),'M0',int(mzero.sum()),flush=True)
    # General complex Slater baseline, allowing species mixing with mean N=2.
    rng = np.random.default_rng(4)
    def slater(x):
        a = (x[:48]+1j*x[48:]).reshape(12,4)
        c = np.linalg.qr(a)[0]
        r = c@c.conj().T
        class Density:
            rho = r
            kappa = np.zeros_like(r)
        return Density()
    hf_runs=[]
    for _ in range(2):
        fit=minimize(lambda x:ham.energy(slater(x)),rng.normal(size=96),method='SLSQP',
            constraints={'type':'eq','fun':lambda x:slater(x).rho.diagonal()[neutrons].real.sum()-2},
            options={'maxiter':120,'ftol':1e-8})
        hf_runs.append({'energy':float(fit.fun),'success':bool(fit.success),
                        'number_error':float(abs(slater(fit.x).rho.diagonal()[neutrons].real.sum()-2))})
    result=solve_hfb(ham,neutrons,[2,2],starts=2,seed=8,maxiter=120,tolerance=1e-8)
    print('HFB',result.energy,result.converged,result.attempts,flush=True)
    # Independent normalized Gaussian vacuum in full 4096-dimensional Fock space.
    a=annihilators(m)
    beta=[sum(result.state.U[i,j].conjugate()*a[i]+
              result.state.V[i,j].conjugate()*a[i].T for i in range(m)) for j in range(m)]
    parent=sum(b.conj().T@b for b in beta)
    ev,pv=eigsh(parent,k=1,which='SA',tol=1e-10,v0=np.random.default_rng(9).normal(size=2**m))
    psi=pv[:,0]
    sector=psi[masks]
    weight=float(np.vdot(sector,sector).real)
    sector0=sector[mzero]
    weight0=float(np.vdot(sector0,sector0).real)
    overlap=float(abs(np.vdot(vec[:,0],sector0))**2)
    report={'interaction':'data/cki, unchanged; no added mass scaling or core offset',
        'valence_neutrons':2,'valence_protons':2,'modes':m,'dimension_NZ':len(masks),
        'dimension_NZM0':int(mzero.sum()),'exact_M0_energy':float(e[0]),
        'exact_NZ_energy':float(np.linalg.eigvalsh(exact)[0]),
        'legacy_matrix_max_error':discrepancy,'hf_attempts':hf_runs,
        'hfb_energy':result.energy,'hfb_converged':result.converged,
        'hfb_attempts':result.attempts,'numbers':result.numbers.tolist(),
        'canonical_error':float(result.state.canonical_error()),
        'stationarity_error':result.stationarity_error,
        'kappa_norm':float(np.linalg.norm(result.state.kappa)),
        'kappa_np_norm':float(np.linalg.norm(result.state.kappa[np.ix_(neutrons,protons)])),
        'rho_np_norm':float(np.linalg.norm(result.state.rho[np.ix_(neutrons,protons)])),
        'parent_vacuum_energy':float(ev[0]),'weight_NZ':weight,'weight_NZM0':weight0,
        'raw_fidelity':overlap,'conditional_fidelity_NZ':overlap/weight,
        'conditional_fidelity_NZM0':overlap/weight0,
        'projected_NZ_energy':float(np.vdot(sector,exact@sector).real/weight),
        'elapsed_seconds':time.perf_counter()-start}
    out=ROOT/'benchmarks/results'
    out.mkdir(exist_ok=True)
    (out/'cki_be8.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez(out/'cki_be8_state.npz',U=result.state.U,V=result.state.V,
             hfb_parameters=result.parameters,exact_M0_vector=vec[:,0],
             basis_masks=np.array(masks),mzero=mzero)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
