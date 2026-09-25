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
from projected_vap import ProjectedVAPObjective, solve_projected_hfb_vap
from test_angular_momentum import spin_half_model
from test_number_projection import FermiHubbardHamiltonian


class TestProjectedVAP(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
