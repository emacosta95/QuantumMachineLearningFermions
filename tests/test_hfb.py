"""Small independent Fock-space checks; avoids legacy package eager imports."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.linalg import expm

spec = importlib.util.spec_from_file_location('hfb', Path(__file__).parents[1] /
                                             'src/NSMFermions/hfb.py')
hfb = importlib.util.module_from_spec(spec)
sys.modules['hfb'] = hfb
spec.loader.exec_module(hfb)


def annihilators(m):
    result = []
    for i in range(m):
        a = np.zeros((2**m, 2**m), complex)
        for n in range(2**m):
            if n & (1 << i):
                a[n ^ (1 << i), n] = (-1)**bin(n & ((1 << i)-1)).count('1')
        result.append(a)
    return result


class TestHFB(unittest.TestCase):
    def test_complex_energy_against_full_fock_space(self):
        m = 4
        rng = np.random.default_rng(19)
        x = rng.normal(size=12) * .3
        state = hfb.state_from_parameters(x, m)
        self.assertLess(state.canonical_error(), 1e-12)
        np.testing.assert_allclose(state.kappa, -state.kappa.T, atol=1e-12)
        np.testing.assert_allclose(state.rho-state.rho@state.rho,
                                   state.kappa@state.kappa.conj().T, atol=1e-12)
        # Obtain the vacuum independently as the zero eigenstate of sum beta†beta.
        a = annihilators(m)
        beta = [sum(state.U[i,j].conjugate()*a[i] +
                    state.V[i,j].conjugate()*a[i].conj().T for i in range(m))
                for j in range(m)]
        _, eigenvectors = np.linalg.eigh(sum(b.conj().T@b for b in beta))
        psi = eigenvectors[:, 0]
        h = rng.normal(size=(m,m)) + 1j*rng.normal(size=(m,m))
        h = h+h.conj().T
        v = rng.normal(size=(m,)*4)+1j*rng.normal(size=(m,)*4)
        v = v-v.swapaxes(0,1)
        v = v-v.swapaxes(2,3)
        v = (v+v.transpose(2,3,0,1).conj())/2
        matrix = sum(h[i,j]*a[i].conj().T@a[j] for i in range(m) for j in range(m))
        for i,j,k,l in np.ndindex((m,)*4):
            matrix += .25*v[i,j,k,l]*a[i].conj().T@a[j].conj().T@a[l]@a[k]
        self.assertAlmostEqual(hfb.HFBHamiltonian(h,v).energy(state),
                               np.vdot(psi,matrix@psi).real, places=11)
        self.assertGreater(np.linalg.norm(state.kappa[:2,2:]), .01)
        self.assertGreater(np.linalg.norm(state.rho[:2,2:]), .01)

    def test_singular_u_hf_limit(self):
        x = np.zeros(12)
        x[0] = np.pi/2
        state = hfb.state_from_parameters(x,4)
        np.testing.assert_allclose(state.rho, np.diag([1,1,0,0]), atol=1e-12)
        self.assertLess(np.linalg.norm(state.kappa), 1e-12)
        with self.assertRaises(ValueError):
            _ = state.thouless_matrix

    def test_analytic_energy_and_number_jacobian(self):
        modes = 4
        rng = np.random.default_rng(97)
        h_one = rng.normal(size=(modes, modes))
        h_one = h_one + h_one.T
        interaction = rng.normal(size=(modes,) * 4)
        interaction = interaction - interaction.swapaxes(0, 1)
        interaction = interaction - interaction.swapaxes(2, 3)
        interaction = (
            interaction + interaction.transpose(2, 3, 0, 1)
        ) / 2
        hamiltonian = hfb.HFBHamiltonian(h_one, interaction)
        parameters = rng.normal(scale=0.15, size=modes * (modes - 1))
        direction = rng.normal(size=len(parameters))
        direction /= np.linalg.norm(direction)
        energy, numbers, gradient, jacobian, _ = (
            hfb.hfb_energy_number_jacobian(
                parameters, hamiltonian, [0, 1]
            )
        )
        step = 2e-6

        def values(point):
            state = hfb.state_from_parameters(point, modes)
            occupation = state.rho.diagonal().real
            return (
                hamiltonian.energy(state),
                np.array([occupation[:2].sum(), occupation[2:].sum()]),
            )

        plus_energy, plus_numbers = values(parameters + step * direction)
        minus_energy, minus_numbers = values(parameters - step * direction)
        self.assertAlmostEqual(
            float(gradient @ direction),
            (plus_energy - minus_energy) / (2 * step),
            places=7,
        )
        np.testing.assert_allclose(
            jacobian @ direction,
            (plus_numbers - minus_numbers) / (2 * step),
            atol=2e-8,
        )
        self.assertAlmostEqual(energy, values(parameters)[0], places=12)
        np.testing.assert_allclose(numbers, values(parameters)[1], atol=1e-12)

    def test_constrained_noninteracting_minimum(self):
        ham = hfb.HFBHamiltonian(np.diag([-2.,1.,-3.,2.]), np.zeros((4,)*4))
        result = hfb.solve_hfb(ham,[0,1],[1,1],starts=2,seed=3)
        self.assertTrue(result.converged, result.attempts)
        np.testing.assert_allclose(result.numbers,[1,1],atol=1e-7)
        self.assertAlmostEqual(result.energy,-5.,places=6)
        self.assertLess(np.linalg.norm(result.state.kappa),1e-3)

    def test_real_bogoliubov_chart_and_analytic_jacobian(self):
        modes = 4
        rng = np.random.default_rng(109)
        one_body = rng.normal(size=(modes, modes))
        one_body = one_body + one_body.T
        interaction = rng.normal(size=(modes,) * 4)
        interaction = interaction - interaction.swapaxes(0, 1)
        interaction = interaction - interaction.swapaxes(2, 3)
        interaction = (
            interaction + interaction.transpose(2, 3, 0, 1)
        ) / 2
        hamiltonian = hfb.HFBHamiltonian(one_body, interaction)
        parameter_count = modes * (modes - 1) // 2
        parameters = rng.normal(scale=0.15, size=parameter_count)
        direction = rng.normal(size=parameter_count)
        direction /= np.linalg.norm(direction)
        energy, numbers, gradient, jacobian, state = (
            hfb.hfb_energy_number_jacobian(
                parameters,
                hamiltonian,
                [0, 1],
                real_parameters=True,
            )
        )
        self.assertLess(np.linalg.norm(state.U.imag), 1e-13)
        self.assertLess(np.linalg.norm(state.V.imag), 1e-13)
        step = 2e-6

        def values(point):
            candidate = hfb.state_from_parameters(
                point, modes, real_parameters=True
            )
            occupation = candidate.rho.diagonal().real
            return (
                hamiltonian.energy(candidate),
                np.array([
                    occupation[:2].sum(), occupation[2:].sum()
                ]),
            )

        plus_energy, plus_numbers = values(parameters + step * direction)
        minus_energy, minus_numbers = values(parameters - step * direction)
        self.assertAlmostEqual(
            float(gradient @ direction),
            (plus_energy - minus_energy) / (2 * step),
            places=7,
        )
        np.testing.assert_allclose(
            jacobian @ direction,
            (plus_numbers - minus_numbers) / (2 * step),
            atol=2e-8,
        )
        self.assertAlmostEqual(energy, values(parameters)[0], places=12)
        np.testing.assert_allclose(numbers, values(parameters)[1], atol=1e-12)

        result = hfb.solve_hfb(
            hfb.HFBHamiltonian(
                np.diag([-2., 1., -3., 2.]), np.zeros((4,) * 4)
            ),
            [0, 1],
            [1, 1],
            starts=1,
            seed=3,
            analytic_jacobian=True,
            real_bogoliubov=True,
        )
        self.assertTrue(result.converged, result.attempts)
        self.assertEqual(result.parameters.size, parameter_count)
        self.assertAlmostEqual(result.energy, -5., places=6)

    def test_hartree_fock_gradient_and_multistart_solver(self):
        modes = 5
        rng = np.random.default_rng(41)
        one_body = rng.normal(size=(modes, modes))
        one_body = one_body + one_body.T
        interaction = rng.normal(size=(modes,) * 4)
        interaction = interaction - interaction.swapaxes(0, 1)
        interaction = interaction - interaction.swapaxes(2, 3)
        interaction = (
            interaction + interaction.transpose(2, 3, 0, 1)
        ) / 2
        ham = hfb.HFBHamiltonian(one_body, interaction)
        orbitals = np.linalg.qr(
            rng.normal(size=(modes, 2))
            + 1j * rng.normal(size=(modes, 2))
        )[0]
        _, gradient, _, _ = hfb.hartree_fock_energy_gradient(
            orbitals, ham
        )
        direction = (
            rng.normal(size=orbitals.shape)
            + 1j * rng.normal(size=orbitals.shape)
        )
        direction -= orbitals @ (orbitals.conj().T @ direction)
        step = 1e-6
        plus = np.linalg.qr(orbitals + step * direction)[0]
        minus = np.linalg.qr(orbitals - step * direction)[0]
        numerical = (
            ham.energy(hfb.HFBState.from_slater(plus))
            - ham.energy(hfb.HFBState.from_slater(minus))
        ) / (2 * step)
        self.assertAlmostEqual(
            numerical, float(np.vdot(gradient, direction).real), places=7
        )

        noninteracting = hfb.HFBHamiltonian(
            np.diag([-2., 1., -3., 2.]), np.zeros((4,) * 4)
        )
        result = hfb.solve_hfb(
            noninteracting,
            [0, 1],
            [1, 1],
            method="hf",
            starts=3,
            seed=3,
            maxiter=100,
        )
        self.assertTrue(result.converged, result.attempts)
        self.assertEqual(len(result.attempts), 3)
        self.assertAlmostEqual(result.energy, -5., places=10)
        np.testing.assert_array_equal(result.numbers, [1., 1.])
        self.assertLess(np.linalg.norm(result.state.kappa), 1e-12)

    def test_stationarity_audit_can_be_skipped(self):
        ham = hfb.HFBHamiltonian(
            np.diag([-2., 1., -3., 2.]), np.zeros((4,) * 4)
        )
        result = hfb.solve_hfb(
            ham,
            [0, 1],
            [1, 1],
            starts=2,
            seed=3,
            stationarity_diagnostics=False,
            shared_finite_difference_jacobian=True,
        )
        np.testing.assert_allclose(result.numbers, [1, 1], atol=1e-7)
        self.assertTrue(np.isnan(result.stationarity_error))
        self.assertTrue(np.isnan(result.chemical_potentials).all())

    def test_analytic_jacobian_optimizer(self):
        ham = hfb.HFBHamiltonian(
            np.diag([-2., 1., -3., 2.]), np.zeros((4,) * 4)
        )
        result = hfb.solve_hfb(
            ham,
            [0, 1],
            [1, 1],
            starts=2,
            seed=3,
            analytic_jacobian=True,
        )
        self.assertTrue(result.converged, result.attempts)
        self.assertAlmostEqual(result.energy, -5., places=6)
        self.assertLess(result.stationarity_error, 1e-5)

    def test_invalid_interaction_rejected(self):
        with self.assertRaises(ValueError):
            hfb.HFBHamiltonian(np.eye(2), np.ones((2,)*4))

    def test_attractive_pn_pairing(self):
        v = np.zeros((4,)*4)
        for i,j in [(0,2),(1,3)]:
            for k,l in [(0,2),(1,3)]:
                v[i,j,k,l] = v[j,i,l,k] = -1
                v[j,i,k,l] = v[i,j,l,k] = 1
        result = hfb.solve_hfb(hfb.HFBHamiltonian(np.zeros((4,4)),v),
                               [0,1],[1,1],starts=2,seed=8)
        self.assertTrue(result.converged, result.attempts)
        self.assertAlmostEqual(result.energy,-1.5,places=6)
        self.assertGreater(np.linalg.norm(result.state.kappa[:2,2:]), .5)


if __name__ == '__main__':
    unittest.main()
