"""Small exact-sector VAP benchmark; not a polynomial-scaling backend."""
import json
import time
import numpy as np
from cki_be8 import ROOT, legacy_definitions
from typing import List, Dict, Tuple, Optional, Callable, ClassVar
from hfb import HFBHamiltonian
from number_projection import NumberProjectedSpace, solve_number_vap, state_from_thouless
from gauge_projection import GaugeProjectedEnergy
from scipy.optimize import least_squares


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
    exact,vec=np.linalg.eigh(space.matrix.toarray())
    result=solve_number_vap(space,starts=3,seed=15,maxiter=3000,
                            tolerance=1e-15,gradient_tolerance=2e-5)
    print('Reference VAP',result.energy,result.converged,flush=True)
    evaluator=GaugeProjectedEnergy(ham,list(range(6,12)),[2,2])
    ij=np.triu_indices(12,1)
    q=result.thouless_matrix[ij]
    grid_fit=evaluator.solve(initial_parameters=np.r_[q.real,q.imag],maxiter=40,tolerance=1e-13)
    z=evaluator.unpack(grid_fit.x)
    grid_energy=evaluator.energy(z)
    check_energy=GaugeProjectedEnergy(ham,list(range(6,12)),[2,2],grid=(9,9),offset=.319).energy(z)
    # Positive species rescaling leaves the normalized projected state invariant.
    # Select the intrinsic representative with average N=Z=2 for fair fidelity.
    def scaled(logs):
        d=np.exp(np.r_[np.repeat(logs[1],6),np.repeat(logs[0],6)])
        return d[:,None]*z*d[None,:]
    def numbers(logs):
        rho=state_from_thouless(scaled(logs)).rho.diagonal().real
        return np.array([rho[6:].sum(),rho[:6].sum()])
    scaling=least_squares(lambda logs:numbers(logs)-2,np.zeros(2),
                          bounds=(-8,8),gtol=1e-12,ftol=1e-12,xtol=1e-12)
    if np.max(np.abs(numbers(scaling.x)-2))>1e-8:
        raise RuntimeError('Could not choose intrinsic mean-number representative')
    z=scaled(scaling.x)
    intrinsic=state_from_thouless(z)
    q=z[ij]; parameters=np.r_[q.real,q.imag]
    amps,_=space.amplitudes_and_jacobian(parameters)
    gaussian_norm=np.exp(.5*np.linalg.slogdet(np.eye(12)+z.conj().T@z)[1])
    sector=amps/np.sqrt(gaussian_norm)
    weight=float(np.vdot(sector,sector).real)
    projected=sector/np.sqrt(weight)
    projected_energy,projected_gradient=space.energy_and_gradient(parameters)
    if abs(grid_energy-projected_energy)>1e-8 or abs(grid_energy-check_energy)>1e-8:
        raise RuntimeError('Gauge-grid validation failed')
    report={'backend':'explicit_sector_initialization_then_gauge_grid_refinement', 'dimension':len(space.masks),
        'projected_energy':projected_energy,'converged':result.converged,
        'gradient_norm':result.gradient_norm,'attempts':result.attempts,
        'exact_energy':float(exact[0]),
        'ground_state_fidelity_projected':float(abs(np.vdot(vec[:,0],projected))**2),
        'ground_state_fidelity_intrinsic':float(abs(np.vdot(vec[:,0],sector))**2),
        'weight_NZ':weight,
        'intrinsic_kappa_norm_mean_number_gauge':float(np.linalg.norm(intrinsic.kappa)),
        'intrinsic_kappa_np_norm_mean_number_gauge':float(np.linalg.norm(intrinsic.kappa[:6,6:])),
        'intrinsic_canonical_error':float(intrinsic.canonical_error()),
        'intrinsic_energy':ham.energy(intrinsic),
        'intrinsic_numbers':[float(intrinsic.rho.diagonal()[6:].real.sum()),
                             float(intrinsic.rho.diagonal()[:6].real.sum())],
        'grid':[7,7],'grid_refinement':[9,9],
        'grid_energy':grid_energy,'refined_grid_energy':check_energy,
        'grid_optimizer_success':bool(grid_fit.success),
        'grid_optimizer_message':str(grid_fit.message),
        'grid_optimizer_iterations':int(grid_fit.nit),
        'grid_optimizer_gradient_norm':float(np.linalg.norm(grid_fit.jac)),
        'post_scaling_analytic_gradient_norm':float(np.linalg.norm(projected_gradient)),
        'exact_projected_numbers':[2,2], 'elapsed_seconds':time.perf_counter()-start}
    out=ROOT/'benchmarks/results'
    (out/'cki_be8_vap.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez(out/'cki_be8_vap_state.npz',Z=z,U=intrinsic.U,V=intrinsic.V,
             projected_vector=projected,masks=np.array(space.masks))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
