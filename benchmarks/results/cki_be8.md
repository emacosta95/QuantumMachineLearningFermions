# CKI Be8 bounded benchmark — 2026-09-16

The repository's `data/cki` is used unchanged, including its single-particle
energies. Energies are valence-space energies in MeV, with no added core energy
or mass scaling. Two valence neutrons and two valence protons occupy 12 modes.
The reader emits t_z=+1/2 first; this benchmark follows the existing CKI study's
labeling of t_z=-1/2 as neutron. Equal N,Z makes this labeling immaterial here.

| Calculation | Energy (MeV) |
|---|---:|
| Exact fixed N,Z (225 states) | -30.295394614 |
| Exact fixed N,Z,M=0 (51 states) | -30.295394614 |
| Unrestricted complex Slater HF baseline | -25.689032351 |
| Unrestricted complex HFB, accepted start | -25.689032344 |
| Explicit N,Z projection of accepted HFB state | -25.689032351 |

The two independently assembled fixed-N,Z matrices agree elementwise (maximum
error zero). One uses the existing `FemionicBasis.adag_adag_a_a_matrix`; the
other applies fermionic bit-string operations directly. Both use the same
legacy coupled-to-uncoupled interaction reader, so this is not independent
validation of that reader's angular-momentum coefficients or of CKI input data.

HFB used two seeded paired starts, at most 120 SLSQP iterations each. One failed
to converge (number error 2.84e-4) and was rejected, despite its slightly lower
reported energy. The accepted start has maximum average-number error 1.31e-10,
canonical error 3.62e-15, and parameter-gradient stationarity residual 1.26e-4.
The two separate HF runs converged to the same energy within 8e-9 MeV.

The accepted HFB state has ||kappa|| = 1.12e-5, with np block norm 5.53e-6.
It therefore reproduces pairing collapse to numerical precision in this
bounded run. It lies 4.60636 MeV above the exact result. These local optimizations
do not prove global optimality; a pairing-stability calculation and additional
starts would be needed to exclude a lower paired solution.

The Gaussian state was reconstructed independently as the ground state of
sum beta† beta in the 4096-dimensional full Fock space. Its vacuum eigenvalue
is -4.65e-16. Diagnostics from explicit sector selection:

- N,Z sector probability: 0.99999999949.
- N,Z,M=0 sector probability: 0.35182614555.
- Raw fidelity to the exact M=0 ground-state vector: 0.20181350450.
- Fidelity conditional on N,Z: 0.20181350461.
- Fidelity conditional on N,Z,M=0: 0.57361713181.

M=0 restriction is not total-J projection. The M=0 weight/conditional fidelity
can depend on the intrinsic orientation of a symmetry-breaking state. No claim
of an orientation-independent or J-projected fidelity is made. Explicit Fock
reconstruction is a small-system benchmark, not a scalable projection method.

Runtime of the successful complete run: 23.74 seconds. Five existing HFB tests
also passed after the convergence-flag serialization fix. Raw results and
state arrays are saved alongside this report in `cki_be8.json` and
`cki_be8_state.npz`; script: `benchmarks/cki_be8.py`.

Next scientific step: pairing stability around the collapsed state, followed
by particle-number PAV. Because this solution is already almost fixed in N,Z,
projection is expected to leave its energy nearly unchanged.
