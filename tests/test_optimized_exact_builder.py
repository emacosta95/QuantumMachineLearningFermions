import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src" / "NSMFermions", ROOT / "benchmarks"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cki_be8 import build_fermionic_hamiltonian  # noqa: E402


class OptimizedExactBuilderTests(unittest.TestCase):
    def test_matches_legacy_two_body_operator_in_reduced_basis(self):
        # Two modes per species with one particle in each block.  Retaining two
        # determinants exercises the missing-row filtering needed by M sectors.
        def paired_indices(occupied):
            return tuple(occupied) in ((0, 2), (1, 3))

        interaction = {
            # Diagonal and sector-preserving scattering terms.
            (0, 2, 2, 0): 1.2,
            (1, 3, 3, 1): -0.7,
            (1, 3, 2, 0): 0.4,
            (0, 2, 3, 1): 0.4,
        }
        eps = np.array([0.1, 0.3, -0.2, 0.5])
        optimized = build_fermionic_hamiltonian(
            interaction, eps, particles=(1, 1), symmetries=[paired_indices]
        )

        # Reconstruct the previous implementation term by term using the same
        # basis class, so this is an independent regression of operator order,
        # fermionic phases, and the symmetry-reduced lookup.
        legacy = optimized.external_potential.copy().tocsr()
        two_body = legacy * 0
        for (i1, i2, i3, i4), value in interaction.items():
            two_body += (value / 4) * optimized.adag_adag_a_a_matrix(
                i1=i1, i2=i2, j1=i4, j2=i3
            )
        legacy += two_body

        np.testing.assert_allclose(
            optimized.hamiltonian.toarray(), legacy.toarray(), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
