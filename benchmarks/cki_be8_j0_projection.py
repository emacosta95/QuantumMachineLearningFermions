"""Apply particle-number and J=0 PAV to the saved CKI Be8 HFB vacuum."""
import json
import time
import numpy as np
from cki_be8 import ROOT, build_fermionic_hamiltonian, legacy_definitions
from typing import List, Dict, Tuple, Optional, Callable, ClassVar
from hfb import HFBHamiltonian, HFBState
from number_projection import exact_ground_state, project_particle_numbers
from angular_momentum import (ParticleNumberJ0ProjectedEnergy,
    exact_j0_projector, project_state_observables)


def main():
    # Rebuild the same CKI Hamiltonian and single-particle angular-momentum
    # encoding used to generate the saved intrinsic HFB vacuum. Raw h,v tensors are
    # required by the gauge/Euler kernels; the base CKI benchmark validates
    # their matrix against the legacy fermionic-basis construction.
    start=time.perf_counter(); ns=dict(globals(),trange=range)
    legacy_definitions('cg_utils.py',['CG','ClebschGordan','SelectCG',
        'CreateInitialCGList','CalcInitialValues','DivCalc','CgJM'],ns)
    legacy_definitions('nuclear_physics_utils.py',['SingleParticleState','krond',
        'scattering_matrix_reader','compute_nuclear_twobody_matrix',
        'get_twobody_nuclearshell_model'],ns)
    interaction,eps=ns['get_twobody_nuclearshell_model'](str(ROOT/'data/cki'))
    sp=ns['SingleParticleState'](str(ROOT/'data/cki'))
    ham=HFBHamiltonian(np.diag(eps),interaction)
    neutrons=list(range(6,12))

    # Reuse the established fermionic Hamiltonian for the N=2,Z=2 basis and
    # exact many-body matrix; the projection code does not rebuild either one.
    fermionic=build_fermionic_hamiltonian(interaction,eps,particles=(2,2))

    # Exact diagonalization supplies the fixed-N,Z ground-state target for the
    # fidelity.  This step is possible only because the benchmark space is small.
    exact_energy,target=exact_ground_state(fermionic)

    # Build P_J=0 explicitly by diagonalizing J^2 in the same determinant basis.
    reference=exact_j0_projector(fermionic,sp.state_encoding)
    target_j2=float(np.vdot(target,reference.j2@target).real)

    # Load the intrinsic state produced before particle-number PAV.
    saved=np.load(ROOT/'benchmarks/results/cki_be8_pav_state.npz')

    # Reconstruct HFBState from its canonical U,V representation. Preserve Z
    # only when the intrinsic state had a finite particle-vacuum chart.
    stored_z=saved['Z'] if bool(saved['has_thouless']) else None
    state=HFBState(saved['U'],saved['V'],Z=stored_z)

    # Construct the state in two explicit stages:
    #   Z -> normalized P_N P_Z|Phi(Z)> -> normalized P_J=0 P_N P_Z|Phi(Z)>.
    # The helper also computes the J=0 weight, energy, and target fidelity.
    pn_result=project_particle_numbers(state,fermionic,target)
    pn=pn_result.projected_vector
    exact_result=project_state_observables(
        state,fermionic,reference,target
    )

    # Independently evaluate the same projected energy with gauge and Euler
    # kernels, never materializing the many-body vector.  Agreement validates
    # the polynomial-memory projection against the explicit reference route.
    evaluator=ParticleNumberJ0ProjectedEnergy(ham,sp.state_encoding,neutrons,[2,2])
    grid_energy=(evaluator.energy(stored_z) if stored_z is not None else None)
    report={'nucleus':'Be8','target_J':0,
        'number_grid':list(evaluator.grid),
        'explicit_number_grid':list(pn_result.grid),
        'explicit_number_grid_offset':pn_result.grid_offset,
        'euler_grid':[len(evaluator.euler_grid.alpha),len(evaluator.euler_grid.cos_beta),
                      len(evaluator.euler_grid.gamma)],
        'number_grid_points':int(np.prod(evaluator.grid)),
        'euler_grid_points':evaluator.euler_grid.size,
        'combined_kernel_points':int(np.prod(evaluator.grid))*evaluator.euler_grid.size,
        'M_bound':evaluator.euler_grid.m_bound,'J_bound':evaluator.euler_grid.j_bound,
        'exact_J0_subspace_dimension':reference.rank,'exact_ground_J2':target_j2,
        'pn_energy':pn_result.energy,
        'pn_fidelity':pn_result.fidelity,
        'J0_weight_within_NZ':exact_result['j0_weight'],
        'pnj0_energy_exact_reference':exact_result['energy'],
        'pnj0_fidelity_exact_reference':exact_result['fidelity'],
        'pnj0_energy_polynomial_grid':grid_energy,
        'grid_reference_energy_difference':(
            grid_energy-exact_result['energy'] if grid_energy is not None else None),
        'exact_ground_energy':exact_energy,
        'elapsed_seconds':time.perf_counter()-start}
    out=ROOT/'benchmarks/results'
    # Save the normalized N,Z,J=0 vector together with its determinant masks so
    # downstream observables use the correct basis ordering.
    (out/'cki_be8_j0_projection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez(out/'cki_be8_j0_projected_state.npz',projected_vector=exact_result['vector'],
             masks=np.array(fermionic.masks))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
