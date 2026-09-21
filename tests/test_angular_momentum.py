import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src/NSMFermions'))
from hfb import HFBHamiltonian, HFBState
from number_projection import exact_ground_state
from test_number_projection import FermiHubbardHamiltonian
from angular_momentum import (single_particle_angular_momentum,
    polynomial_j0_grid, ParticleNumberJ0ProjectedEnergy,
    project_state_observables)


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

    def test_gauge_euler_series_against_exact_ground_state(self):
        states,ham=spin_half_model()
        exact_hamiltonian=FermiHubbardHamiltonian(ham,[2,3],[1,1])
        exact_energy,target=exact_ground_state(exact_hamiltonian)
        rng=np.random.default_rng(31)
        x=rng.normal(size=12)*.4
        evaluator=ParticleNumberJ0ProjectedEnergy(ham,states,[2,3],[1,1])
        z=evaluator.unpack(x)
        state=HFBState.from_thouless(z)
        # The production path constructs only gauge/Euler-rotated vacua and is
        # checked against the independently diagonalized Hamiltonian ground state.
        # Energy consumes the exact same finite vacuum series as the fidelity.
        series=evaluator.projected_series(state)
        direct=project_state_observables(series,exact_hamiltonian,target)
        grid_energy=evaluator.series_energy(series)
        self.assertAlmostEqual(exact_energy,-1.,places=11)
        self.assertAlmostEqual(direct.fidelity,1.,places=12)
        self.assertEqual(series.number_of_vacua,81)
        self.assertEqual(series.euler_grid,(3,1,3))
        self.assertAlmostEqual(direct.energy,exact_energy,places=10)
        self.assertAlmostEqual(grid_energy,direct.energy,places=10)

        # Explicit user grids directly control M without changing the API.
        larger=ParticleNumberJ0ProjectedEnergy(
            ham,states,[2,3],[1,1],number_grid=(4,5),euler_grid=(4,2,5))
        larger_series=larger.projected_series(state)
        self.assertEqual(larger_series.number_of_vacua,4*5*4*2*5)
        larger_result=project_state_observables(
            larger_series,exact_hamiltonian,target)
        self.assertAlmostEqual(larger_result.fidelity,1.,places=11)


if __name__=='__main__':unittest.main()
