# CKI Be8: closest Gaussian benchmark

Date: 2026-09-16. The target is the normalized exact CKI ground state in the
full fixed N=2,Z=2 space. The optimization permits complex amplitudes and full
neutron-proton mixing.

The best Gaussian state found is a number-conserving Slater determinant:

F_G(best found) = 0.20214721969217

N_G(best-found geometric infidelity) = 1 - F_G = 0.79785278030783.

All ten Slater starts converged to the same fidelity within 4e-14, with tangent
gradient norms below 2e-7. The unrestricted finite-Thouless search approached
0.20197595087785 while hitting the chart boundary. These results strongly
support a Slater optimum, but do not constitute a global-optimality proof.

| Pair of normalized states | Fidelity |
|---|---:|
| Closest Gaussian found vs exact ground state | 0.202147219692 |
| Energy-minimizing collapsed HFB vs exact | 0.201813504504 |
| Intrinsic PN-VAP vacuum vs exact | 0.046947246857 |
| Projected PN-VAP state vs exact | 0.391058816533 |
| Closest Gaussian found vs collapsed HFB | 0.000760488849 |
| Closest Gaussian found vs intrinsic PN-VAP vacuum | 0.013353787977 |
| Closest Gaussian found vs projected PN-VAP state | 0.111233712111 |

The closest Gaussian has exact average N=Z=2, N,Z-sector weight equal to one to
8e-14, and kappa=0. Its energy is -25.654949261 MeV, whereas the energy-minimized
collapsed HFB energy is lower (-25.689032344 MeV). This explicitly demonstrates
that maximum fidelity and minimum energy select different Gaussian states.

The very small overlap between the two Slater determinants should be interpreted
with care. The exact state has good symmetries, while individual intrinsic
determinants can break them; symmetry-related or otherwise distinct intrinsic
representatives may have similar overlap with the exact state but little overlap
with one another. No angular-momentum projection has been applied here.

A direct symmetry diagnostic confirms this explanation. The four principal
angles between the occupied subspaces are all about 65.95 degrees, giving the
unaligned fidelity 0.000760489. Optimizing only a common spatial SO(3) rotation
of the closest-fidelity determinant raises its fidelity with the collapsed HFB
determinant to 0.997433315. An isospin rotation alone leaves the small overlap
unchanged. Both determinants have <J^2> near 7.9 and <T^2> below 2e-9, so they
are strongly rotational-symmetry-breaking but essentially T=0 intrinsic states.
The remaining 0.26% infidelity after alignment measures their small difference
in intrinsic shape/orbitals. Diagnostic data are saved in
`cki_be8_gaussian_overlap_diagnostic.json`.

Raw data and state: `cki_be8_best_gaussian.json` and
`cki_be8_best_gaussian_state.npz`. Reproduce with
`benchmarks/cki_be8_best_gaussian.py`.
