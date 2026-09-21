"""PN and J=0 projection of the saved CKI Be8 PN-VAP vacuum."""
import json
import time
import numpy as np
from cki_be8 import ROOT, legacy_definitions
from typing import List, Dict, Tuple, Optional, Callable, ClassVar
from hfb import HFBHamiltonian
from number_projection import NumberProjectedSpace
from angular_momentum import (ParticleNumberJ0ProjectedEnergy,
    exact_j0_projector, projected_observables)


def main():
    start=time.perf_counter(); ns=dict(globals(),trange=range)
    legacy_definitions('cg_utils.py',['CG','ClebschGordan','SelectCG',
        'CreateInitialCGList','CalcInitialValues','DivCalc','CgJM'],ns)
    legacy_definitions('nuclear_physics_utils.py',['SingleParticleState','krond',
        'scattering_matrix_reader','compute_nuclear_twobody_matrix',
        'get_twobody_nuclearshell_model'],ns)
    interaction,eps=ns['get_twobody_nuclearshell_model'](str(ROOT/'data/cki'))
    sp=ns['SingleParticleState'](str(ROOT/'data/cki'))
    ham=HFBHamiltonian(np.diag(eps),interaction)
    neutrons=list(range(6,12)); space=NumberProjectedSpace(ham,neutrons,[2,2])
    energies,vectors=np.linalg.eigh(space.matrix.toarray()); target=vectors[:,0]
    reference=exact_j0_projector(space,sp.state_encoding)
    target_j2=float(np.vdot(target,reference.j2@target).real)
    saved=np.load(ROOT/'benchmarks/results/cki_be8_vap_state.npz')
    pn=saved['projected_vector']
    exact_result=projected_observables(pn,space,reference,target)
    evaluator=ParticleNumberJ0ProjectedEnergy(ham,sp.state_encoding,neutrons,[2,2])
    grid_energy=evaluator.energy(saved['Z'])
    report={'nucleus':'Be8','target_J':0,
        'number_grid':list(evaluator.grid),
        'euler_grid':[len(evaluator.euler_grid.alpha),len(evaluator.euler_grid.cos_beta),
                      len(evaluator.euler_grid.gamma)],
        'number_grid_points':int(np.prod(evaluator.grid)),
        'euler_grid_points':evaluator.euler_grid.size,
        'combined_kernel_points':int(np.prod(evaluator.grid))*evaluator.euler_grid.size,
        'M_bound':evaluator.euler_grid.m_bound,'J_bound':evaluator.euler_grid.j_bound,
        'exact_J0_subspace_dimension':reference.rank,'exact_ground_J2':target_j2,
        'pn_energy':float(np.vdot(pn,space.matrix@pn).real),
        'pn_fidelity':float(abs(np.vdot(target,pn))**2),
        'J0_weight_within_NZ':exact_result['j0_weight'],
        'pnj0_energy_exact_reference':exact_result['energy'],
        'pnj0_fidelity_exact_reference':exact_result['fidelity'],
        'pnj0_energy_polynomial_grid':grid_energy,
        'grid_reference_energy_difference':grid_energy-exact_result['energy'],
        'exact_ground_energy':float(energies[0]),
        'elapsed_seconds':time.perf_counter()-start}
    out=ROOT/'benchmarks/results'
    (out/'cki_be8_j0_projection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez(out/'cki_be8_j0_projected_state.npz',projected_vector=exact_result['vector'],
             masks=np.array(space.masks))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
