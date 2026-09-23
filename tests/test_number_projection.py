"""Projection-after-variation tests using the fermionic-Hamiltonian interface."""

from itertools import combinations
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parents[1] / "src/NSMFermions"))
from hfb import HFBHamiltonian, HFBState
from number_projection import exact_ground_state, project_particle_numbers


def pairing_model(g=0.2):
    """Return raw h,v tensors for the four-mode regression pairing model."""
    # Couple the two proton-neutron pairs (0,2) and (1,3) with strength -g.
    interaction = np.zeros((4,) * 4)
    for i, j in [(0, 2), (1, 3)]:
        for k, l in [(0, 2), (1, 3)]:
            # Fill all antisymmetric permutations expected by HFBHamiltonian.
            interaction[i, j, k, l] = interaction[j, i, l, k] = -g
            interaction[j, i, k, l] = interaction[i, j, l, k] = g
    return HFBHamiltonian(np.diag([-1.0, 1.0, -1.0, 1.0]), interaction)


class FermiHubbardHamiltonian:
    """Minimal test double for the production projection-facing interface."""

    def __init__(self, raw_hamiltonian, neutron_modes, targets):
        # Generate the exact fixed-N,Z determinant basis in the same mode order
        # used by the production FermiHubbardHamiltonian class.
        self.modes = len(raw_hamiltonian.h)
        neutrons = list(neutron_modes)
        protons = [i for i in range(self.modes) if i not in neutrons]
        # Expose the same two-block metadata used by the production Hamiltonian.
        self.species_modes = (tuple(neutrons), tuple(protons))
        # Store the fixed counts in the identical species order.
        self.particle_numbers = tuple(int(number) for number in targets)
        self.occupations = [
            tuple(sorted(neutron_occupation + proton_occupation))
            for neutron_occupation in combinations(neutrons, targets[0])
            for proton_occupation in combinations(protons, targets[1])
        ]
        self.masks = np.array([
            sum(1 << mode for mode in occupation)
            for occupation in self.occupations
        ])

        # Build the reference many-body matrix independently by applying every
        # second-quantized one- and two-body operator to every determinant.
        lookup = {mask: index for index, mask in enumerate(self.masks)}
        terms = [
            ([(j, False), (i, True)], raw_hamiltonian.h[i, j])
            for i, j in zip(*np.nonzero(raw_hamiltonian.h))
        ]
        terms += [
            (
                [(k, False), (l, False), (j, True), (i, True)],
                raw_hamiltonian.v[i, j, k, l] / 4,
            )
            for i, j, k, l in zip(*np.nonzero(raw_hamiltonian.v))
        ]
        rows, columns, values = [], [], []
        for column, initial in enumerate(self.masks):
            for operations, value in terms:
                state, phase = int(initial), 1
                for mode, create in operations:
                    # Invalid creation/annihilation makes this term vanish.
                    if bool(state & (1 << mode)) == create:
                        break
                    # Count occupied lower modes to obtain the fermionic sign.
                    phase *= (-1) ** bin(state & ((1 << mode) - 1)).count("1")
                    state ^= 1 << mode
                else:
                    if state in lookup:
                        rows.append(lookup[state])
                        columns.append(column)
                        values.append(value * phase)
        self.hamiltonian = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(len(self.masks),) * 2,
        ).tocsr()
        self.hamiltonian.sum_duplicates()

    @property
    def matrix(self):
        """Expose the assembled fixed-sector matrix expected by PAV."""
        return self.hamiltonian


def fermionic_pairing_model():
    """Return matching raw and fixed-sector Hamiltonian representations."""
    raw = pairing_model()
    exact = FermiHubbardHamiltonian(raw, [0, 1], [1, 1])
    return raw, exact


class TestNumberProjection(unittest.TestCase):
    def test_paired_state_projection_energy_and_fidelity(self):
        raw, exact = fermionic_pairing_model()

        # Construct a general finite-Z paired vacuum before applying projection.
        rng = np.random.default_rng(3)
        z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        z = 0.3 * (z - z.T)
        state = HFBState.from_thouless(z)

        # Diagonalize the exact fermionic Hamiltonian only to provide a target.
        ground_energy, target = exact_ground_state(exact)
        result = project_particle_numbers(state, exact, target)

        # Projection returns a normalized vector in exact.occupations order.
        self.assertAlmostEqual(np.linalg.norm(result.projected_vector), 1.0)
        self.assertGreater(result.sector_weight, 0.0)
        self.assertGreaterEqual(result.fidelity, 0.0)
        self.assertLessEqual(result.fidelity, 1.0 + 1e-12)
        self.assertGreaterEqual(result.energy, ground_energy - 1e-12)
        # Four modes split into two blocks require the exact 3 x 3 Fourier grid.
        self.assertEqual(result.grid, (3, 3))

        # Independently form the expected Pfaffian coefficient of each two-body
        # determinant; for two particles it is simply Z_ij.
        expected = np.array([z[occupation] for occupation in exact.occupations])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result.projected_vector, expected)

        # A larger shifted grid represents the same continuous U(1) x U(1)
        # projector and must therefore return the same normalized vector.
        larger = project_particle_numbers(
            state, exact, grid=(5, 4), offset=0.319
        )
        np.testing.assert_allclose(larger.projected_vector, result.projected_vector)
        self.assertAlmostEqual(larger.energy, result.energy)

        # The intrinsic U,V remain canonical before and after projection.
        self.assertLess(state.canonical_error(), 1e-12)
        self.assertTrue(np.allclose(raw.h, raw.h.conj().T))

    def test_collapsed_hf_state_uses_slater_amplitudes(self):
        _, exact = fermionic_pairing_model()

        # Occupy modes 0 and 2. This is a nonempty HF determinant with kappa=0,
        # singular U, and therefore no finite particle-vacuum Thouless matrix.
        orbitals = np.eye(4, dtype=complex)[:, [0, 2]]
        state = HFBState.from_slater(orbitals)
        with self.assertRaises(ValueError):
            _ = state.thouless_matrix

        # PAV nevertheless works because HFBState switches from Pfaffians to
        # determinants of occupied-orbital submatrices at the Slater boundary.
        result = project_particle_numbers(state, exact)
        expected = np.zeros(len(exact.occupations), complex)
        expected[exact.occupations.index((0, 2))] = 1.0
        np.testing.assert_allclose(abs(result.projected_vector), expected)
        self.assertAlmostEqual(result.sector_weight, 1.0)
        self.assertLess(np.linalg.norm(state.kappa), 1e-12)

        # A state with exact neutron and proton numbers needs no nontrivial
        # gauge sum. The explicit opt-in permits the useful 1x1 discretization.
        with self.assertWarnsRegex(
            UserWarning, "exact P_N P_Z symmetry restoration is not guaranteed"
        ):
            one_point = project_particle_numbers(
                state,
                exact,
                grid=(1, 1),
                allow_inexact_grid=True,
            )
        np.testing.assert_allclose(
            abs(one_point.projected_vector), expected, atol=1e-12
        )
        self.assertFalse(one_point.series.number_grid_guaranteed_exact)
        self.assertEqual(one_point.series.minimum_number_grid, (3, 3))

    def test_wrong_container_is_rejected(self):
        state = HFBState.from_thouless(np.zeros((2, 2)))
        with self.assertRaises(TypeError):
            project_particle_numbers(state, object())


if __name__ == "__main__":
    unittest.main()
