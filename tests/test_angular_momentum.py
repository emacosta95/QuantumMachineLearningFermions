import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src/NSMFermions'))
from hfb import HFBHamiltonian, HFBState
from number_projection import project_particle_numbers
from test_number_projection import FermiHubbardHamiltonian
from angular_momentum import (single_particle_angular_momentum,
    polynomial_j0_grid, ParticleNumberJ0ProjectedEnergy, exact_j0_projector,
    projected_observables, project_state_observables)


def spin_half_model():
    # proton m=-1/2,+1/2 followed by neutron m=-1/2,+1/2
    states=[(0,0,.5,-.5,.5,.5),(0,0,.5,.5,.5,.5),
            (0,0,.5,-.5,.5,-.5),(0,0,.5,.5,.5,-.5)]
    # J=0 pair: (p+ n- - p- n+)/sqrt(2).
    w=np.zeros((4,4),complex)
    w[1,2]=1/np.sqrt(2); w[0,3]=-1/np.sqrt(2); w-=w.T
    v=-np.einsum('ij,kl->ijkl',w,w.conj())
    return states,HFBHamiltonian(np.zeros((4,4)),v)


class TestAngularMomentum(unittest.TestCase):
    def test_generators_and_polynomial_grid(self):
        states,_=spin_half_model()
        jx,jy,jz=single_particle_angular_momentum(states)
        np.testing.assert_allclose(jx@jy-jy@jx,1j*jz,atol=1e-12)
        grid=polynomial_j0_grid(states,[2,3],[1,1])
        self.assertEqual((len(grid.alpha),len(grid.cos_beta),len(grid.gamma)),(3,1,3))
        self.assertEqual(grid.size,9)
        with self.assertRaises(ValueError):
            polynomial_j0_grid(states,[2,3],[1,1],grid=(2,1,3))
        with self.assertRaises(ValueError):
            polynomial_j0_grid(states,[2,3],[1,0])

    def test_gauge_euler_kernel_against_exact_projector(self):
        states,ham=spin_half_model()
        exact_hamiltonian=FermiHubbardHamiltonian(ham,[2,3],[1,1])
        reference=exact_j0_projector(exact_hamiltonian,states)
        self.assertEqual(reference.rank,1)
        rng=np.random.default_rng(31)
        x=rng.normal(size=12)*.4
        evaluator=ParticleNumberJ0ProjectedEnergy(ham,states,[2,3],[1,1])
        z=evaluator.unpack(x)
        state=HFBState.from_thouless(z)
        number_result=project_particle_numbers(state,exact_hamiltonian)
        exact=projected_observables(
            number_result.projected_vector,exact_hamiltonian,reference)
        direct=project_state_observables(
            state,exact_hamiltonian,reference,exact['vector'])
        grid_energy=evaluator.energy(z)
        self.assertAlmostEqual(exact['energy'],-1.,places=11)
        self.assertAlmostEqual(direct['fidelity'],1.,places=12)
        np.testing.assert_allclose(direct['vector'],exact['vector']/np.linalg.norm(exact['vector']))
        self.assertAlmostEqual(grid_energy,exact['energy'],places=10)


if __name__=='__main__':unittest.main()
