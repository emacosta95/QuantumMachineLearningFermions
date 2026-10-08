import unittest
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src" / "NSMFermions"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from hfb import HFBState
from simple_hfb_gaussian_study import (
    particle_number_expectations,
    stable_energy_state,
    validate_particle_lists,
)


class SimpleHFBGaussianStudyTests(unittest.TestCase):
    def test_collapsed_state_is_saved_as_hartree_fock(self):
        state = HFBState.from_slater(np.eye(4, dtype=complex)[:, :2])

        saved, family, pairing_norm, idempotency_error = (
            stable_energy_state(state, 2)
        )

        self.assertEqual(family, "hartree_fock")
        self.assertEqual(saved.slater_orbitals.shape, (4, 2))
        self.assertLess(pairing_norm, 1e-12)
        self.assertLess(idempotency_error, 1e-12)

    def test_empty_slater_state_keeps_zero_occupied_orbitals(self):
        state = HFBState.from_slater(np.empty((4, 0), complex))

        saved, family, _, _ = stable_energy_state(state, 0)

        self.assertEqual(family, "hartree_fock")
        self.assertEqual(saved.slater_orbitals.shape, (4, 0))

    def test_paired_state_remains_hfb(self):
        z = np.zeros((4, 4), complex)
        z[0, 1] = 0.4
        z[1, 0] = -0.4
        state = HFBState.from_thouless(z)

        saved, family, pairing_norm, _ = stable_energy_state(state, 2)

        self.assertEqual(family, "hfb")
        self.assertIs(saved, state)
        self.assertGreater(pairing_norm, 1e-3)

    def test_particle_numbers_are_returned_in_neutron_proton_order(self):
        orbitals = np.eye(4, dtype=complex)[:, [0, 2, 3]]
        state = HFBState.from_slater(orbitals)

        numbers = particle_number_expectations(state, neutron_modes=[2, 3])

        np.testing.assert_allclose(numbers, [2, 1], atol=1e-12)

    def test_particle_grid_requires_even_total_parity(self):
        with self.assertRaisesRegex(ValueError, "even number parity"):
            validate_particle_lists([2], [1, 2], capacity=6)

        protons, neutrons = validate_particle_lists(
            [1, 3], [1, 3], capacity=6
        )
        self.assertEqual(protons, [1, 3])
        self.assertEqual(neutrons, [1, 3])


if __name__ == "__main__":
    unittest.main()
