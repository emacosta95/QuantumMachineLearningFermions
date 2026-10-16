"""Compute <Phi|H|Phi> directly from all 4096 canonical-pair configurations.

This does not use the library's Wick/density energy routine for the comparison.
The Hamiltonian is rotated to the independently reconstructed canonical basis.
"""
import json
import pickle

from verify_ne24_bloch_messiah import ROOT, D, u, v
import numpy as np
from hfb import HFBHamiltonian
from nuclear_workflow import load_nuclear_interaction

interaction, eps, _, _ = load_nuclear_interaction('usdb', repository_root=ROOT)
ham = HFBHamiltonian(np.diag(eps), interaction)
h = D.conj().T @ ham.h @ D
interaction_canonical = np.einsum(
    'ia,jb,ijkl,kc,ld->abcd', D.conj(), D.conj(), ham.v, D, D,
    optimize=True,
)

pairs = len(u)
pair_masks = range(1 << pairs)
coefficients = np.array([
    np.prod([v[k] if mask & (1 << k) else u[k] for k in range(pairs)])
    for mask in pair_masks
], dtype=complex)
Hpsi = np.zeros_like(coefficients)
for mask in pair_masks:
    occupied_pairs = [k for k in range(pairs) if mask & (1 << k)]
    empty_pairs = [k for k in range(pairs) if not mask & (1 << k)]
    occupied_modes = [i for k in occupied_pairs for i in (2*k, 2*k+1)]
    diagonal = sum(h[i,i] for i in occupied_modes)
    diagonal += sum(
        interaction_canonical[i,j,i,j]
        for pos,i in enumerate(occupied_modes) for j in occupied_modes[pos+1:]
    )
    Hpsi[mask] += diagonal * coefficients[mask]
    # Between paired determinants the remaining nonzero off-diagonal terms
    # transfer a whole canonical pair. Adjacent pair operators have + sign.
    for source in occupied_pairs:
        for destination in empty_pairs:
            new_mask = mask ^ (1 << source) ^ (1 << destination)
            Hpsi[new_mask] += (
                interaction_canonical[2*destination,2*destination+1,2*source,2*source+1]
                * coefficients[mask]
            )

norm = float(np.vdot(coefficients, coefficients).real)
energy = np.vdot(coefficients, Hpsi) / norm
with (ROOT/'results/ne_hfb_fidelity_audit.pkl').open('rb') as stream:
    variational = pickle.load(stream)[24]
result = dict(
    canonical_configurations=len(coefficients), wavefunction_norm=norm,
    variational_energy=float(variational.energy),
    direct_wavefunction_energy=float(energy.real),
    imaginary_energy=float(energy.imag),
    energy_difference=float(energy.real-variational.energy),
    gradient_norm=float(variational.stationarity_error),
    variational_converged=bool(variational.converged),
)
assert abs(norm-1)<1e-12
assert abs(energy.imag)<1e-10
assert abs(result['energy_difference'])<1e-8, result
(ROOT/'results/ne24_wavefunction_energy_audit.json').write_text(json.dumps(result,indent=2))
print(result, flush=True)
