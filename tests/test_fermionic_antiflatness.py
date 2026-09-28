import itertools
import unittest

import numpy as np

from fermionic_antiflatness import (
    fermionic_antiflatness,
    vacuum_series_antiflatness,
)
from hfb import BogoliubovVacuumSeries, HFBState


def all_occupations(modes):
    return [
        occupied
        for particles in range(modes + 1)
        for occupied in itertools.combinations(range(modes), particles)
    ]


class TestFermionicAntiflatness(unittest.TestCase):
    def test_finite_thouless_gaussian_vanishes(self):
        rng = np.random.default_rng(81)
        raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        state = HFBState.from_thouless(0.3 * (raw - raw.T))
        occupations = all_occupations(4)
        amplitudes = state.occupation_amplitudes(occupations, normalized=True)

        result = fermionic_antiflatness(amplitudes, occupations, 4)

        self.assertLess(abs(result.value), 1e-10)
        np.testing.assert_allclose(
            result.covariance.T @ result.covariance,
            np.eye(8),
            atol=1e-10,
        )

    def test_non_gaussian_two_determinant_state_is_positive(self):
        occupations = [(0, 1), (2, 3)]
        coefficients = np.array([1.0, 1.0]) / np.sqrt(2)

        result = fermionic_antiflatness(coefficients, occupations, 4)

        self.assertAlmostEqual(result.value, 4.0, places=12)
        self.assertAlmostEqual(result.value_per_mode, 1.0, places=12)

    def test_coherent_vacuum_series_is_evaluated_after_summation(self):
        orbitals = np.eye(4, dtype=complex)[:, :2]
        state = HFBState.from_slater(orbitals)
        permutation = np.eye(4, dtype=complex)[:, [2, 3, 0, 1]]
        series = BogoliubovVacuumSeries(
            intrinsic_state=state,
            transformations=np.array([np.eye(4), permutation]),
            weights=np.array([1.0, 1.0]),
            number_grid=(1, 1),
        )

        result = vacuum_series_antiflatness(
            series, [(0, 1), (2, 3)]
        )

        self.assertAlmostEqual(result.value, 4.0, places=12)

    def test_incomplete_or_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            fermionic_antiflatness([1], [(0,), (1,)], 2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fermionic_antiflatness([1], [()], 2, order=0)


if __name__ == "__main__":
    unittest.main()
