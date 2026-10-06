import unittest

import numpy as np

from hfb import HFBState, apply_thouless_step
from study_gaussian_fidelity import _stable_variational_fidelity_state


class StableVariationalFidelityStateTests(unittest.TestCase):
    def test_pairing_collapsed_state_uses_slater_boundary(self):
        orbitals = np.eye(6, dtype=complex)[:, :2]
        slater = HFBState.from_slater(orbitals)
        coordinates = np.zeros(15)
        coordinates[-1] = 1e-9
        near_slater = apply_thouless_step(
            slater, coordinates, real_parameters=True
        )

        stable, chart = _stable_variational_fidelity_state(near_slater, 2)

        self.assertEqual(chart, "Slater (pairing-collapsed HFB)")
        self.assertLess(np.linalg.norm(stable.kappa), 1e-12)
        self.assertLess(np.linalg.norm(stable.rho @ stable.rho - stable.rho), 1e-12)
        self.assertEqual(stable.slater_orbitals.shape, (6, 2))


if __name__ == "__main__":
    unittest.main()
