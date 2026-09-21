import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from test_number_projection import pairing_model
from number_projection import NumberProjectedSpace
from number_projection import state_from_thouless
from gaussian_fidelity import (GaussianFidelityObjective, maximize_gaussian_fidelity,
                               maximize_slater_fidelity, _slater_value_gradient)


class TestGaussianFidelity(unittest.TestCase):
    def test_gradient(self):
        space=NumberProjectedSpace(pairing_model(),[0,1],[1,1])
        target=np.array([1,2j,-.3,.7+1j])
        objective=GaussianFidelityObjective(space,target)
        x=np.random.default_rng(8).normal(size=12)*.4
        value,gradient=objective.fidelity_and_gradient(x)
        state=state_from_thouless(space.unpack(x))
        self.assertAlmostEqual(
            value,
            state.fixed_sector_fidelity(target,space.occupations),
            places=12)
        numerical=approx_derivative(lambda y:objective.fidelity_and_gradient(y)[0],x).ravel()
        self.assertGreater(value,0)
        np.testing.assert_allclose(gradient,numerical,atol=2e-8,rtol=2e-7)

    def test_unrestricted_optimization_improves_random_start(self):
        space=NumberProjectedSpace(pairing_model(),[0,1],[1,1])
        target=np.linalg.eigh(space.matrix.toarray())[1][:,0]
        rng=np.random.default_rng(9)
        x=rng.normal(size=12)*.3
        initial=GaussianFidelityObjective(space,target).fidelity_and_gradient(x)[0]
        result=maximize_gaussian_fidelity(space,target,starts=2,seed=9,
                                          initial_parameters=x)
        self.assertGreater(result.fidelity,initial+.1)
        self.assertGreaterEqual(result.fidelity,0)
        self.assertLessEqual(result.fidelity,1+1e-10)

    def test_slater_gradient_and_boundary_optimizer(self):
        space=NumberProjectedSpace(pairing_model(),[0,1],[1,1])
        target=np.array([1,0,0,0],complex)
        rng=np.random.default_rng(12)
        c=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))[0]
        value,gradient=_slater_value_gradient(c,space.occupations,target)
        direction=rng.normal(size=c.shape)+1j*rng.normal(size=c.shape)
        direction-=c@((c.conj().T@direction+direction.conj().T@c)/2)
        eps=1e-6
        plus=np.linalg.qr(c+eps*direction)[0]
        minus=np.linalg.qr(c-eps*direction)[0]
        numerical=(_slater_value_gradient(plus,space.occupations,target)[0]-
                   _slater_value_gradient(minus,space.occupations,target)[0])/(2*eps)
        analytic=np.real(np.vdot(gradient,direction))
        self.assertAlmostEqual(numerical,analytic,places=6)
        result=maximize_slater_fidelity(space,target,starts=3,seed=2)
        self.assertTrue(result.converged,result.attempts)
        self.assertAlmostEqual(result.fidelity,1.,places=10)


if __name__=='__main__':
    unittest.main()
