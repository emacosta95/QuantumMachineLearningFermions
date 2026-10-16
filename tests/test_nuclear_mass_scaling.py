"""Check optional mass scaling through the actual interaction loader."""
import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "NSMFermions"))
from nuclear_workflow import load_nuclear_interaction


class NuclearMassScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.base, cls.eps, cls.sp, cls.path = load_nuclear_interaction("usdb")

    def load(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_nuclear_interaction("usdb", **kwargs)

    def test_ne24_scales_every_two_body_element_only_once(self):
        scaled, eps, sp, path = self.load(mass_number=24)
        self.assertTrue(self.base)
        self.assertEqual(scaled.keys(), self.base.keys())
        np.testing.assert_allclose(
            list(scaled.values()),
            np.array(list(self.base.values())) * (18 / 24) ** 0.3,
            rtol=1e-14, atol=0,
        )
        np.testing.assert_array_equal(eps, self.eps)
        self.assertEqual(sp.state_encoding, self.sp.state_encoding)
        self.assertEqual(path, self.path)

    def test_reference_mass_preserves_original_values(self):
        scaled, eps, _, _ = self.load(mass_number=18)
        self.assertEqual(scaled, self.base)
        np.testing.assert_array_equal(eps, self.eps)

    def test_custom_reference_and_exponent(self):
        scaled, _, _, _ = self.load(
            mass_number=24, reference_mass=12, mass_exponent=1,
        )
        np.testing.assert_allclose(
            list(scaled.values()), np.array(list(self.base.values())) / 2,
            rtol=1e-14, atol=0,
        )

    def test_invalid_parameters(self):
        for kwargs in (
            {"mass_number": 0}, {"mass_number": -24},
            {"mass_number": np.nan}, {"mass_number": np.inf},
            {"mass_number": 24, "reference_mass": 0},
            {"mass_number": 24, "reference_mass": np.inf},
            {"mass_number": 24, "mass_exponent": np.nan},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.load(**kwargs)


if __name__ == "__main__":
    unittest.main()
