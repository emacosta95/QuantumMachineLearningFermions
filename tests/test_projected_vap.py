import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src/NSMFermions"))
sys.path.insert(0, str(Path(__file__).parent))

from angular_momentum import (
    ParticleNumberJ0ProjectedEnergy,
    project_state_observables,
)
from projected_vap import (
    ProjectedVAPObjective,
    number_projected_slater_seed,
    solve_projected_hfb_vap,
)
from test_angular_momentum import spin_half_model
from test_number_projection import FermiHubbardHamiltonian


class TestProjectedVAP(unittest.TestCase):
    def test_six_particle_pfaffian_gradient(self):
        rng = np.random.default_rng(91)
        raw = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
        matrix = raw - raw.T
        rows, columns = np.triu_indices(6, 1)
        pair_index = {
            (int(row), int(column)): index
            for index, (row, column) in enumerate(zip(rows, columns))
        }
        terms = tuple(
            (sign, tuple(pair_index[pair] for pair in matching))
            for sign, matching in ProjectedVAPObjective._perfect_matchings(
                tuple(range(6))
            )
        )
        values, cofactors = ProjectedVAPObjective._batch_pfaffian_cofactors(
            matrix[rows, columns][None, :], terms
        )
        expected, _, _, expected_cofactors = (
            ProjectedVAPObjective._pfaffian_with_cofactor(matrix)
        )
        np.testing.assert_allclose(values[0], expected, atol=1e-12)
        np.testing.assert_allclose(cofactors[0], expected_cofactors, atol=1e-12)

    def test_number_projected_slater_seed_recovers_determinant(self):
        orbitals = np.zeros((4, 2), complex)
        orbitals[[0, 3], [0, 1]] = 1
        state = number_projected_slater_seed(orbitals, pairing_scale=0.7)
        occupations = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        amplitudes = state.occupation_amplitudes(occupations, normalized=False)
        amplitudes /= np.linalg.norm(amplitudes)
        expected = np.zeros(6, complex)
        expected[2] = 1
        self.assertAlmostEqual(abs(np.vdot(expected, amplitudes)) ** 2, 1.0)

    def test_fixed_sector_backend_uses_custom_inexact_series(self):
        states, raw = spin_half_model()
        exact = FermiHubbardHamiltonian(raw, [2, 3], [1, 1])
        with self.assertWarns(UserWarning):
            projector = ParticleNumberJ0ProjectedEnergy(
                raw,
                states,
                [2, 3],
                [1, 1],
                number_grid=(1, 1),
                euler_grid=(1, 1, 1),
                allow_inexact_number_grid=True,
                allow_inexact_euler_grid=True,
            )
        result = solve_projected_hfb_vap(
            projector,
            energy_backend="fixed_sector_basis",
            fermionic_hamiltonian=exact,
            starts=1,
            seed=4,
            maxiter=25,
            gradient_tolerance=1e-3,
        )
        self.assertTrue(np.isfinite(result.projected_energy))
        self.assertEqual(result.projected_series.number_of_vacua, 1)
        self.assertFalse(result.number_grid_guaranteed_exact)
        self.assertFalse(result.euler_grid_guaranteed_exact)
        direct = project_state_observables(
            result.projected_series, exact
        )
        self.assertAlmostEqual(result.projected_energy, direct.energy, places=11)

    def test_kernel_backend_rejects_aliased_number_vap_by_default(self):
        states, raw = spin_half_model()
        with self.assertWarns(UserWarning):
            projector = ParticleNumberJ0ProjectedEnergy(
                raw,
                states,
                [2, 3],
                [1, 1],
                number_grid=(1, 1),
                euler_grid=(3, 1, 3),
                allow_inexact_number_grid=True,
            )
        with self.assertRaisesRegex(ValueError, "exact number grid"):
            ProjectedVAPObjective(projector, energy_backend="kernel")

    def test_analytic_gradient_matches_central_difference(self):
        states, raw = spin_half_model()
        exact = FermiHubbardHamiltonian(raw, [2, 3], [1, 1])
        projector = ParticleNumberJ0ProjectedEnergy(
            raw, states, [2, 3], [1, 1]
        )
        objective = ProjectedVAPObjective(
            projector,
            energy_backend="analytic_fixed_sector",
            fermionic_hamiltonian=exact,
        )
        rng = np.random.default_rng(73)
        parameters = rng.normal(scale=0.4, size=12)
        direction = rng.normal(size=12)
        direction /= np.linalg.norm(direction)
        energy, gradient = objective.energy_and_gradient(parameters)
        step = 2e-6
        finite_difference = (
            objective.energy(parameters + step * direction)
            - objective.energy(parameters - step * direction)
        ) / (2 * step)
        self.assertTrue(np.isfinite(energy))
        self.assertAlmostEqual(
            float(gradient @ direction), finite_difference, places=7
        )

        series = projector.projected_series(objective.state(parameters))
        self.assertAlmostEqual(
            energy, projector.series_energy(series), places=10
        )


if __name__ == "__main__":
    unittest.main()
