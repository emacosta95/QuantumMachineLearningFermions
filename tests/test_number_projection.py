import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0,str(Path(__file__).parents[1]/'src/NSMFermions'))
from hfb import HFBHamiltonian, HFBState, solve_hfb
from number_projection import NumberProjectedSpace, solve_number_vap, state_from_thouless
from test_hfb import annihilators


def pairing_model(g=.2):
    v=np.zeros((4,)*4)
    for i,j in [(0,2),(1,3)]:
        for k,l in [(0,2),(1,3)]:
            v[i,j,k,l]=v[j,i,l,k]=-g
            v[j,i,k,l]=v[i,j,l,k]=g
    return HFBHamiltonian(np.diag([-1.,1.,-1.,1.]),v)


class TestNumberProjection(unittest.TestCase):
    def test_analytic_gradient(self):
        space=NumberProjectedSpace(pairing_model(),[0,1],[1,1])
        x=np.random.default_rng(3).normal(size=12)
        _,gradient=space.energy_and_gradient(x)
        numerical=approx_derivative(lambda y:space.energy_and_gradient(y)[0],x).ravel()
        np.testing.assert_allclose(gradient,numerical,atol=2e-9)
        a,_=space.amplitudes_and_jacobian(x)
        z=space.unpack(x)
        np.testing.assert_allclose(a,[z[occ] for occ in space.occupations])
        projected=space.projected_state(z)
        np.testing.assert_allclose(projected,a/np.linalg.norm(a))
        self.assertAlmostEqual(space.projected_fidelity(z,projected),1.)
        state=state_from_thouless(z)
        np.testing.assert_allclose(
            state.occupation_amplitudes(space.occupations),a)
        # U,V determine the same chart when U is nonsingular, even if the
        # original Z is not retained explicitly on the state object.
        recovered=HFBState(state.U,state.V).thouless_matrix
        np.testing.assert_allclose(recovered,z,atol=1e-12)
        self.assertLess(state.canonical_error(),1e-12)

    def test_four_particle_pfaffian_gradient(self):
        space=NumberProjectedSpace(HFBHamiltonian(np.diag(np.arange(6.)),np.zeros((6,)*4)),
                                   [0,1,2],[2,2])
        x=np.random.default_rng(10).normal(size=30)
        a,_=space.amplitudes_and_jacobian(x)
        z=space.unpack(x)
        expected=[z[i,j]*z[k,l]-z[i,k]*z[j,l]+z[i,l]*z[j,k]
                  for i,j,k,l in space.occupations]
        np.testing.assert_allclose(a,expected)
        _,g=space.energy_and_gradient(x)
        num=approx_derivative(lambda y:space.energy_and_gradient(y)[0],x).ravel()
        np.testing.assert_allclose(g,num,atol=2e-9)

    def test_weak_pairing_collapse_and_vap_recovery(self):
        ham=pairing_model()
        hfb=solve_hfb(ham,[0,1],[1,1],starts=2,seed=3)
        self.assertTrue(hfb.converged)
        self.assertLess(np.linalg.norm(hfb.state.kappa),1e-3)
        space=NumberProjectedSpace(ham,[0,1],[1,1])
        vap=solve_number_vap(space,starts=2,seed=7,tolerance=1e-13)
        self.assertTrue(vap.converged,vap.attempts)
        exact=np.linalg.eigvalsh(space.matrix.toarray())[0]
        self.assertAlmostEqual(exact,-.2-np.sqrt(4+.2**2),places=12)
        self.assertAlmostEqual(vap.energy,exact,places=9)
        self.assertLess(vap.energy,hfb.energy-.005)

    def test_guards(self):
        ham=pairing_model()
        with self.assertRaises(ValueError):
            NumberProjectedSpace(ham,[0,1],[1,0])
        with self.assertRaises(ValueError):
            NumberProjectedSpace(ham,[0,1],[1,1],max_dimension=1)
        space=NumberProjectedSpace(ham,[0,1],[1,1])
        with self.assertRaises(ValueError):
            space.energy_and_gradient(np.zeros(12))

    def test_legacy_hamiltonian_matrix_comparison_reorders_basis(self):
        space=NumberProjectedSpace(pairing_model(),[0,1],[1,1])
        permutation=np.array([2,0,3,1])
        class LegacyHamiltonian:
            pass
        legacy=LegacyHamiltonian()
        legacy.basis=np.array([
            [int(space.masks[i]>>mode & 1) for mode in range(space.modes)]
            for i in permutation
        ])
        legacy.hamiltonian=space.matrix[permutation][:,permutation]
        self.assertEqual(space.many_body_matrix_error(legacy),0.)

    def test_thouless_amplitudes_are_bogoliubov_vacuum(self):
        rng=np.random.default_rng(2)
        z=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); z=(z-z.T)*.2
        state=state_from_thouless(z)
        psi=np.zeros(16,complex); psi[0]=1
        for i in range(4):
            for j in range(i+1,4):
                psi[(1<<i)+(1<<j)]=z[i,j]
        psi[15]=z[0,1]*z[2,3]-z[0,2]*z[1,3]+z[0,3]*z[1,2]
        norm=np.exp(.5*np.linalg.slogdet(np.eye(4)+z.conj().T@z)[1])
        self.assertAlmostEqual(np.vdot(psi,psi).real,norm,places=12)
        a=annihilators(4)
        for j in range(4):
            beta=sum(state.U[i,j].conjugate()*a[i]+state.V[i,j].conjugate()*a[i].T for i in range(4))
            self.assertLess(np.linalg.norm(beta@psi),1e-12)


if __name__=='__main__':
    unittest.main()
