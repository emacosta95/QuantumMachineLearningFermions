"""Independent canonical-pair products versus library HFB amplitudes."""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "NSMFermions"))
from hfb import HFBState, HFBHamiltonian


def pair_product(D, u, v):
    """Expand product_k (u_k+v_k d_2k^dag d_(2k+1)^dag)|0>."""
    modes = len(D)
    psi = np.zeros(1 << modes, complex)
    for selected in itertools.product((False, True), repeat=len(u)):
        columns = [i for k, used in enumerate(selected) if used for i in (2*k, 2*k+1)]
        coefficient = np.prod([v[k] if used else u[k] for k, used in enumerate(selected)])
        for occupied in itertools.combinations(range(modes), len(columns)):
            minor = D[np.ix_(occupied, columns)]
            psi[sum(1 << i for i in occupied)] += coefficient * np.linalg.det(minor)
    return psi


def annihilators(modes):
    result = []
    for i in range(modes):
        a = np.zeros((1 << modes, 1 << modes), complex)
        for mask in range(1 << modes):
            if mask & (1 << i):
                a[mask ^ (1 << i), mask] = (-1) ** ((mask & ((1 << i)-1)).bit_count())
        result.append(a)
    return result


class TestBlochMessiahAmplitudes(unittest.TestCase):
    def check_product(self, angles):
        rng = np.random.default_rng(719)
        modes = 2 * len(angles)
        D, _ = np.linalg.qr(rng.normal(size=(modes, modes)) + 1j*rng.normal(size=(modes, modes)))
        u, v = np.cos(angles), np.sin(angles)
        Ubar = np.diag(np.repeat(u, 2)).astype(complex)
        Vbar = np.zeros((modes, modes), complex)
        for k, value in enumerate(v):
            Vbar[2*k, 2*k+1] = value
            Vbar[2*k+1, 2*k] = -value
        state = HFBState(D @ Ubar, D.conj() @ Vbar)
        psi = pair_product(D, u, v)
        self.assertAlmostEqual(np.linalg.norm(psi), 1., places=12)
        operators = annihilators(modes)
        for j in range(modes):
            beta_psi = sum(state.U[i,j].conjugate()*(operators[i]@psi)
                           + state.V[i,j].conjugate()*(operators[i].conj().T@psi)
                           for i in range(modes))
            self.assertLess(np.linalg.norm(beta_psi), 2e-13)
        for number in range(0, modes+1, 2):
            occ = list(itertools.combinations(range(modes), number))
            direct = psi[[sum(1 << i for i in row) for row in occ]]
            if np.linalg.norm(direct) < 1e-12:
                continue
            library = state.stable_normalized_occupation_amplitudes(occ)
            phase = np.vdot(direct, library)
            phase /= abs(phase)
            np.testing.assert_allclose(library, phase*direct, atol=3e-12)
        apsi = np.array([a@psi for a in operators])
        rho = (apsi.conj() @ apsi.T).T
        kappa = np.array([[np.vdot(psi, operators[j]@(operators[i]@psi))
                           for j in range(modes)] for i in range(modes)])
        np.testing.assert_allclose(state.rho, rho, atol=2e-13)
        np.testing.assert_allclose(state.kappa, kappa, atol=2e-13)
        h = rng.normal(size=(modes,modes)) + 1j*rng.normal(size=(modes,modes))
        h = h + h.conj().T
        interaction = rng.normal(size=(modes,)*4) + 1j*rng.normal(size=(modes,)*4)
        interaction -= interaction.swapaxes(0,1)
        interaction -= interaction.swapaxes(2,3)
        interaction = (interaction+interaction.transpose(2,3,0,1).conj())/2
        pair_vectors = np.array([operators[j]@(operators[i]@psi)
                                for i in range(modes) for j in range(modes)])
        two_body = (pair_vectors.conj() @ pair_vectors.T).reshape((modes,)*4)
        direct_energy = np.einsum('ij,ji->',h,rho) + .25*np.einsum('ijkl,ijkl->',interaction,two_body)
        self.assertAlmostEqual(HFBHamiltonian(h,interaction).energy(state), direct_energy.real, places=11)

    def test_paired_complex_vacuum(self):
        self.check_product(np.array([.23,.61,1.1]))

    def test_partially_paired_singular_vacuum(self):
        self.check_product(np.array([np.pi/2,.61,0.]))

    def test_hartree_fock_boundary(self):
        self.check_product(np.array([np.pi/2,np.pi/2,0.]))


if __name__ == '__main__':
    unittest.main()
