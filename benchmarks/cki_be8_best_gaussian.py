"""Best-found unrestricted Gaussian approximation to the CKI Be8 ground state."""
import json
import time
import numpy as np
from scipy.sparse.linalg import eigsh
from cki_be8 import ROOT, legacy_definitions, annihilators
from typing import List, Dict, Tuple, Optional, Callable, ClassVar
from hfb import HFBHamiltonian
from number_projection import NumberProjectedSpace
from gaussian_fidelity import maximize_gaussian_fidelity, maximize_slater_fidelity


def full_gaussian_vector(U,V,annihilation,seed):
    beta=[sum(U[i,j].conjugate()*annihilation[i]+
              V[i,j].conjugate()*annihilation[i].T for i in range(len(U)))
          for j in range(len(U))]
    value,vector=eigsh(sum(b.conj().T@b for b in beta),k=1,which='SA',tol=1e-11,
                       v0=np.random.default_rng(seed).normal(size=2**len(U)))
    return float(value[0]),vector[:,0]/np.linalg.norm(vector[:,0])


def full_slater_vector(orbitals):
    from itertools import combinations
    vector=np.zeros(2**len(orbitals),complex)
    for occupation in combinations(range(len(orbitals)),orbitals.shape[1]):
        vector[sum(1<<i for i in occupation)]=np.linalg.det(orbitals[list(occupation)])
    return vector/np.linalg.norm(vector)


def main():
    start=time.perf_counter()
    ns=dict(globals(),trange=range)
    legacy_definitions('cg_utils.py',['CG','ClebschGordan','SelectCG',
        'CreateInitialCGList','CalcInitialValues','DivCalc','CgJM'],ns)
    legacy_definitions('nuclear_physics_utils.py',['SingleParticleState','krond',
        'scattering_matrix_reader','compute_nuclear_twobody_matrix',
        'get_twobody_nuclearshell_model'],ns)
    interaction,eps=ns['get_twobody_nuclearshell_model'](str(ROOT/'data/cki'))
    ham=HFBHamiltonian(np.diag(eps),interaction)
    space=NumberProjectedSpace(ham,list(range(6,12)),[2,2])
    energies,vectors=np.linalg.eigh(space.matrix.toarray())
    target=vectors[:,0]
    vap=np.load(ROOT/'benchmarks/results/cki_be8_vap_state.npz')
    ij=np.triu_indices(12,1); q=vap['Z'][ij]
    initial=np.r_[q.real,q.imag]
    result=maximize_gaussian_fidelity(space,target,starts=16,seed=41,maxiter=2000,
        tolerance=1e-15,gradient_tolerance=2e-6,initial_parameters=initial)
    collapsed=np.load(ROOT/'benchmarks/results/cki_be8_state.npz')
    eigenvalues,eigenvectors=np.linalg.eigh(collapsed['V'].conj()@collapsed['V'].T)
    slater=maximize_slater_fidelity(space,target,starts=10,seed=42,maxiter=1500,
        gradient_tolerance=2e-7,initial_orbitals=eigenvectors[:,-4:])
    print('Interior Gaussian',result.fidelity,result.converged,
          'Slater boundary',slater.fidelity,slater.converged,flush=True)
    annihilation=annihilators(12)
    vacuum_values=[]; states=[]
    for seed,U,V in [(1,result.state.U,result.state.V),
                     (2,collapsed['U'],collapsed['V']),
                     (3,vap['U'],vap['V'])]:
        value,state=full_gaussian_vector(U,V,annihilation,seed)
        vacuum_values.append(value); states.append(state)
    interior,ordinary,vap_intrinsic=states
    boundary=full_slater_vector(slater.orbitals)
    best=boundary if slater.fidelity>=result.fidelity else interior
    best_kind='slater_boundary' if slater.fidelity>=result.fidelity else 'interior_thouless'
    exact_full=np.zeros(2**12,complex)
    exact_full[np.array(space.masks)]=target
    projected_vap=np.zeros(2**12,complex)
    projected_vap[vap['masks']]=vap['projected_vector']
    fidelities={
        'best_gaussian_exact':float(abs(np.vdot(exact_full,best))**2),
        'interior_gaussian_exact':float(abs(np.vdot(exact_full,interior))**2),
        'slater_boundary_exact':float(abs(np.vdot(exact_full,boundary))**2),
        'collapsed_hfb_exact':float(abs(np.vdot(exact_full,ordinary))**2),
        'vap_intrinsic_exact':float(abs(np.vdot(exact_full,vap_intrinsic))**2),
        'best_gaussian_collapsed_hfb':float(abs(np.vdot(best,ordinary))**2),
        'best_gaussian_vap_intrinsic':float(abs(np.vdot(best,vap_intrinsic))**2),
        'best_gaussian_vap_projected':float(abs(np.vdot(best,projected_vap))**2),
        'collapsed_hfb_vap_intrinsic':float(abs(np.vdot(ordinary,vap_intrinsic))**2),
    }
    rho=result.state.rho
    report={'definition':'max over unrestricted pure even fermionic Gaussian vacua',
        'global_optimum_certified':False,'selected_best_kind':best_kind,
        'thouless_chart_boundary_certified':bool(slater.converged),
        'best_result_converged':bool(slater.converged if best_kind=='slater_boundary' else result.converged),
        'best_gradient_norm':float(slater.gradient_norm if best_kind=='slater_boundary' else result.gradient_norm),
        'interior_attempts':result.attempts,'slater_attempts':slater.attempts,
        'fidelities':fidelities,
        'interior_objective_crosscheck_error':abs(fidelities['interior_gaussian_exact']-result.fidelity),
        'slater_objective_crosscheck_error':abs(fidelities['slater_boundary_exact']-slater.fidelity),
        'best_intrinsic_numbers':[float(np.diag(slater.orbitals@slater.orbitals.conj().T)[6:].real.sum()),
                                  float(np.diag(slater.orbitals@slater.orbitals.conj().T)[:6].real.sum())],
        'best_NZ_sector_weight':float(np.vdot(best[np.array(space.masks)],best[np.array(space.masks)]).real),
        'best_kappa_norm':0.0,'best_kappa_np_norm':0.0,
        'best_orbital_orthogonality_error':float(np.linalg.norm(slater.orbitals.conj().T@slater.orbitals-np.eye(4))),
        'best_energy':ham.energy(type('SlaterDensity',(),{
            'rho':slater.orbitals@slater.orbitals.conj().T,
            'kappa':np.zeros((12,12),complex)})()),
        'interior_energy':ham.energy(result.state),'exact_energy':float(energies[0]),
        'independent_parent_vacuum_eigenvalues':vacuum_values,
        'elapsed_seconds':time.perf_counter()-start}
    out=ROOT/'benchmarks/results'
    (out/'cki_be8_best_gaussian.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez(out/'cki_be8_best_gaussian_state.npz',Z=result.thouless_matrix,
             U=result.state.U,V=result.state.V,parameters=result.parameters,
             slater_orbitals=slater.orbitals,full_vector=best,
             interior_full_vector=interior)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
