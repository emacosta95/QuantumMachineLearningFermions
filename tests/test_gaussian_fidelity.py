import unittest
import numpy as np
from test_number_projection import fermionic_pairing_model
from hfb import HFBState, apply_thouless_step
from gaussian_fidelity import (GaussianFidelityObjective, maximize_gaussian_fidelity,
                               maximize_slater_fidelity,
                               maximize_best_gaussian_fidelity,
                               _slater_value, _slater_value_gradient,
                               _batch_pfaffian,
                               _batch_pfaffian_cofactors)
from pfapack import pfaffian as pf


class TestGaussianFidelity(unittest.TestCase):
    def test_batched_four_particle_pfaffian(self):
        rng=np.random.default_rng(31)
        matrices=rng.normal(size=(9,4,4))+1j*rng.normal(size=(9,4,4))
        matrices=matrices-matrices.transpose(0,2,1)
        expected=(matrices[:,0,1]*matrices[:,2,3]
                  -matrices[:,0,2]*matrices[:,1,3]
                  +matrices[:,0,3]*matrices[:,1,2])
        np.testing.assert_allclose(_batch_pfaffian(matrices),expected,atol=1e-12)

    def test_hybrid_pfaffian_matches_pfapack(self):
        rng=np.random.default_rng(52)
        for size in (0,2,4,6,8,10,12):
            matrices=(rng.normal(size=(5,size,size))+
                      1j*rng.normal(size=(5,size,size)))
            matrices-=matrices.transpose(0,2,1)
            expected=np.asarray([
                pf.pfaffian(matrix,overwrite_a=False,method='P')
                if size else 1.
                for matrix in matrices
            ])
            np.testing.assert_allclose(
                _batch_pfaffian(matrices),expected,rtol=2e-11,atol=2e-11
            )

    def test_pfaffian_cofactors_match_direct_minors(self):
        rng=np.random.default_rng(53)
        for size in (4,8,10):
            matrices=(rng.normal(size=(4,size,size))+
                      1j*rng.normal(size=(4,size,size)))
            matrices-=matrices.transpose(0,2,1)
            values,cofactors=_batch_pfaffian_cofactors(matrices)
            np.testing.assert_allclose(
                values,_batch_pfaffian(matrices),rtol=2e-10,atol=2e-10
            )
            rows,columns=np.triu_indices(size,1)
            expected=np.empty_like(cofactors)
            for pair_index,(row,column) in enumerate(zip(rows,columns)):
                retained=[index for index in range(size)
                          if index not in (row,column)]
                expected[:,pair_index]=(
                    (-1)**(row+column+1)
                    *_batch_pfaffian(
                        matrices[:,retained,:][:,:,retained]
                    )
                )
            np.testing.assert_allclose(
                cofactors,expected,rtol=2e-9,atol=2e-9
            )

    def test_pfaffian_cofactors_remain_defined_at_singular_matrix(self):
        matrix=np.zeros((1,8,8),complex)
        matrix[0,0,1]=2.; matrix[0,1,0]=-2.
        matrix[0,2,3]=3.; matrix[0,3,2]=-3.
        matrix[0,4,5]=5.; matrix[0,5,4]=-5.
        values,cofactors=_batch_pfaffian_cofactors(matrix)
        rows,columns=np.triu_indices(8,1)
        final_pair=np.flatnonzero((rows==6)&(columns==7))[0]
        self.assertEqual(values[0],0.)
        self.assertAlmostEqual(cofactors[0,final_pair],30.)

        odd_values,odd_cofactors=_batch_pfaffian_cofactors(
            np.zeros((2,3,3),complex)
        )
        np.testing.assert_array_equal(odd_values,np.zeros(2))
        np.testing.assert_array_equal(odd_cofactors,np.zeros((2,3)))

    def test_objective_matches_state_api(self):
        _,hamiltonian=fermionic_pairing_model()
        target=np.array([1,2j,-.3,.7+1j])
        objective=GaussianFidelityObjective(hamiltonian,target)
        x=np.random.default_rng(8).normal(size=12)*.4
        value=objective.fidelity(x)
        state=HFBState.from_thouless(objective.unpack(x))
        self.assertAlmostEqual(
            value,
            state.fixed_sector_fidelity(target,hamiltonian.occupations),
            places=12)
        self.assertGreater(value,0)

    def test_unrestricted_optimization_improves_random_start(self):
        _,hamiltonian=fermionic_pairing_model()
        target=np.linalg.eigh(hamiltonian.matrix.toarray())[1][:,0]
        rng=np.random.default_rng(9)
        x=rng.normal(size=12)*.3
        initial=GaussianFidelityObjective(hamiltonian,target).fidelity(x)
        result=maximize_gaussian_fidelity(hamiltonian,target,starts=2,seed=9,
                                          initial_parameters=x,maxiter=100)
        self.assertGreater(result.fidelity,initial+.1)
        self.assertGreaterEqual(result.fidelity,0)
        self.assertLessEqual(result.fidelity,1+1e-10)

    def test_zero_overlap_stationary_point_is_not_convergence(self):
        _,hamiltonian=fermionic_pairing_model()
        target=np.array([1.,0.,0.,0.])
        result=maximize_gaussian_fidelity(
            hamiltonian,target,starts=1,maxiter=3,
            initial_parameters=np.zeros(12),
        )
        self.assertFalse(result.converged)
        self.assertEqual(
            result.attempts[0]['message'],'Zero-overlap stationary point'
        )

    def test_real_gaussian_chart_uses_half_the_coordinates(self):
        _,hamiltonian=fermionic_pairing_model()
        target=np.array([1., .2, -.3, .1])
        objective=GaussianFidelityObjective(
            hamiltonian,target,real_parameters=True
        )
        x=np.random.default_rng(18).normal(size=6)*.2
        z=objective.unpack(x)
        self.assertEqual(x.size, 6)
        self.assertLess(np.linalg.norm(z.imag), 1e-14)
        result=maximize_gaussian_fidelity(
            hamiltonian,target,starts=1,seed=18,maxiter=5,
            real_parameters=True,
        )
        self.assertEqual(result.parameters.size, 6)
        self.assertLess(np.linalg.norm(result.thouless_matrix.imag), 1e-14)

    def test_local_overlap_gradient_matches_centered_difference(self):
        _,hamiltonian=fermionic_pairing_model()
        target=np.array([1.,.2j,-.3,.7+1j])
        rng=np.random.default_rng(42)
        for real_parameters in (True,False):
            objective=GaussianFidelityObjective(
                hamiltonian,target,real_parameters=real_parameters
            )
            coordinate_count=(len(objective.pairs) if real_parameters
                              else 2*len(objective.pairs))
            parameters=rng.normal(scale=.3,size=coordinate_count)
            state=HFBState.from_thouless(objective.unpack(parameters))
            value,gradient=objective.local_value_gradient(state)
            field_value,field=objective.local_value_field(state)
            self.assertAlmostEqual(value,field_value,places=13)
            np.testing.assert_allclose(field,-field.T,atol=1e-13)
            upper=field[objective.ij]
            field_gradient=(2*upper.real if real_parameters else
                            np.r_[2*upper.real,2*upper.imag])
            np.testing.assert_allclose(gradient,field_gradient,atol=1e-13)
            direction=rng.normal(size=coordinate_count)
            direction/=np.linalg.norm(direction)
            step=2e-6
            plus=apply_thouless_step(
                state,step*direction,real_parameters=real_parameters
            )
            minus=apply_thouless_step(
                state,-step*direction,real_parameters=real_parameters
            )
            numerical=(objective.fidelity_state(plus)-
                       objective.fidelity_state(minus))/(2*step)
            self.assertAlmostEqual(float(gradient@direction),numerical,places=8)
            self.assertAlmostEqual(value,objective.fidelity_state(state),places=12)

    def test_slater_gradient_and_boundary_optimizer(self):
        _,hamiltonian=fermionic_pairing_model()
        target=np.array([1,0,0,0],complex)
        rng=np.random.default_rng(12)
        c=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))[0]
        value,gradient=_slater_value_gradient(c,hamiltonian.occupations,target)
        self.assertAlmostEqual(
            value,_slater_value(c,hamiltonian.occupations,target),places=14
        )
        direction=rng.normal(size=c.shape)+1j*rng.normal(size=c.shape)
        direction-=c@((c.conj().T@direction+direction.conj().T@c)/2)
        eps=1e-6
        plus=np.linalg.qr(c+eps*direction)[0]
        minus=np.linalg.qr(c-eps*direction)[0]
        numerical=(_slater_value_gradient(plus,hamiltonian.occupations,target)[0]-
                   _slater_value_gradient(minus,hamiltonian.occupations,target)[0])/(2*eps)
        analytic=np.real(np.vdot(gradient,direction))
        self.assertAlmostEqual(numerical,analytic,places=6)
        result=maximize_slater_fidelity(hamiltonian,target,starts=3,seed=2)
        self.assertTrue(result.converged,result.attempts)
        self.assertAlmostEqual(result.fidelity,1.,places=10)

    def test_real_slater_and_combined_search(self):
        _,hamiltonian=fermionic_pairing_model()
        target=np.array([1.,.2,-.3,.1])
        real_slater=maximize_slater_fidelity(
            hamiltonian,target,starts=2,seed=4,maxiter=100,
            real_parameters=True,
        )
        self.assertLess(np.linalg.norm(real_slater.orbitals.imag),1e-14)

        best=maximize_best_gaussian_fidelity(
            hamiltonian,target,bogoliubov_starts=1,
            hartree_fock_starts=2,seed=4,bogoliubov_maxiter=30,
            hartree_fock_maxiter=100,real_parameters=True,
        )
        expected=max(best.bogoliubov.fidelity,best.hartree_fock.fidelity)
        self.assertAlmostEqual(best.fidelity,expected,places=12)
        self.assertIn(best.family,('bogoliubov','hartree_fock'))
        self.assertAlmostEqual(
            best.state.fixed_sector_fidelity(
                target,hamiltonian.occupations
            ),
            best.fidelity,
            places=10,
        )


if __name__=='__main__':
    unittest.main()
