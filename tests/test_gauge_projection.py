import unittest
import numpy as np
from test_number_projection import pairing_model
from number_projection import NumberProjectedSpace
from gauge_projection import GaugeProjectedEnergy, pfaffian
from hfb import HFBHamiltonian


class TestGaugeProjection(unittest.TestCase):
    def test_pfaffian_sign_and_determinant(self):
        a=np.array([[0,2,3,4],[-2,0,5,6],[-3,-5,0,7],[-4,-6,-7,0]],complex)
        self.assertAlmostEqual(pfaffian(a),2*7-3*6+4*5)
        rng=np.random.default_rng(4)
        a=rng.normal(size=(12,12))+1j*rng.normal(size=(12,12)); a-=a.T
        np.testing.assert_allclose(pfaffian(a)**2,np.linalg.det(a),rtol=1e-10)

    def test_complex_grid_against_exact_projection(self):
        ham=pairing_model()
        space=NumberProjectedSpace(ham,[0,1],[1,1])
        for seed in range(3):
            x=np.random.default_rng(seed).normal(size=12)*.3
            z=space.unpack(x)
            expected=space.energy_and_gradient(x)[0]
            for grid,offset in [((3,3),.137),((4,5),.319)]:
                actual=GaugeProjectedEnergy(ham,[0,1],[1,1],grid,offset).energy(z)
                self.assertAlmostEqual(expected,actual,places=10)

    def test_polynomial_grid_optimizer(self):
        evaluator=GaugeProjectedEnergy(pairing_model(),[0,1],[1,1])
        fit=evaluator.solve(seed=7,maxiter=100,tolerance=1e-13)
        self.assertTrue(fit.success,fit.message)
        self.assertAlmostEqual(fit.fun,-.2-np.sqrt(4+.2**2),places=8)

    def test_undersized_grid_rejected(self):
        with self.assertRaises(ValueError):
            GaugeProjectedEnergy(pairing_model(),[0,1],[1,1],grid=(2,2))

    def test_four_particle_complex_hamiltonian(self):
        rng=np.random.default_rng(28)
        m=6; species=np.array([0,0,0,1,1,1])
        h=rng.normal(size=(m,m))+1j*rng.normal(size=(m,m))
        h=(h+h.conj().T)*(species[:,None]==species[None,:])
        v=rng.normal(size=(m,)*4)+1j*rng.normal(size=(m,)*4)
        v-=v.swapaxes(0,1); v-=v.swapaxes(2,3)
        v=(v+v.transpose(2,3,0,1).conj())/2
        charges=species[:,None]+species[None,:]
        v*=charges[:,:,None,None]==charges[None,None,:,:]
        ham=HFBHamiltonian(h,v)
        space=NumberProjectedSpace(ham,[0,1,2],[2,2])
        x=rng.normal(size=30)*.3
        z=space.unpack(x)
        grid=GaugeProjectedEnergy(ham,[0,1,2],[2,2])
        self.assertAlmostEqual(grid.energy(z),space.energy_and_gradient(x)[0],places=10)
        d=np.exp([.2,.2,.2,-.4,-.4,-.4])
        self.assertAlmostEqual(grid.energy(d[:,None]*z*d[None,:]),grid.energy(z),places=10)
