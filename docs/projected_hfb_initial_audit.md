# Projected HFB: initial audit and development plan

Date: 2026-09-16. Scope: preliminary static inspection; no numerical validation or solver changes performed.

Update: the initial audit below is retained as a record. A new complex,
species-unrestricted HFB reference solver and a PAV-only projection workflow now
exist; see `hfb_reference_solver.md` and `particle_number_pav.md` for verified
behavior and limitations. Variation after projection is not in the current
scope.
The agreed target allows full neutron-proton normal and pairing mixing with
separate average-N,Z constraints. Existing HF is a validation limit, not the
development objective.

## Context and working rules

The objective is to extend NSMFermions with HFB, projection after variation,
and comparisons with exact nuclear-shell-model states. The prior ChatGPT
discussion supplies the following working preferences: consult project/Notion
context at session start; communicate concisely and rigorously; explain code
structure and purpose before implementation; ask before expensive computations.
These are a summary from that conversation, not a separately verified
project-instructions file. No tracked AGENTS.md was found.

Sources consulted:
- ChatGPT conversation: Notion access confirmation.
- Research Hub: https://app.notion.com/p/3748ff1ca19781bc86bbfcb329898c79
- Vita Pratica: https://app.notion.com/p/3798ff1ca19781fa9d1deecfb0d18078

The Research Hub records pairing collapse in Neon/USDB calculations and an unresolved HFB versus HF energy discrepancy. These are reported research observations, not reproduced results. Its analytical formulas and complexity claims still require independent verification before implementation.

## Repository observations

- `src/NSMFermions/hamiltonian_utils.py`: fixed-species-number Hamiltonian builders, symmetry filtering, sparse diagonalization, and two-body assembly with an explicit factor of 1/4. Trace the annihilation-index order through the basis operator before defining a new interaction adapter.
- `src/NSMFermions/hartree_fock_library.py`: several distinct HF implementations. The legacy `HartreeFock` contracts all columns of a square orbital matrix without an occupation-number argument in its self-consistency loop. It mixes orbital matrices without subsequent orthonormalization. This needs validation before reuse as a reference solver.
- The same legacy class computes a species-aware column list in `create_hf_psi`, but then selects the first total-particle-count columns instead. Check the intended species convention against actual callers.
- `HFEnergyFunctional` and `HFEnergyFunctionalNuclear` use QR orbitals and transpose-based densities, suitable for their current real parameterization but requiring conjugate-transpose handling for a complex extension. Their interaction contractions must be compared against the exact builder using identical tensors.
- The nuclear functional accepts `neutron_indices` but uses `proton_indices` as a scalar block boundary. Clarify the public interface and mode ordering.
- `utils_quasiparticle_approximation.py` contains basis conversion utilities; its name alone does not establish a general HFB implementation.
- Packaging metadata disagree: `pyproject.toml` declares no dependencies, while `setup.py` lists NumPy, SciPy, and Matplotlib. Inspected modules additionally import packages including torch, tqdm, joblib, and tqdm_joblib.
- The tracked test-name search found a two-body notebook, but no dedicated automated test suite. Notebook validation has not yet been reviewed.

These observations do not establish the cause of the collaborators' HFB energy discrepancy: their implementation has not been inspected.

## First milestone: establish a trustworthy reference

1. Trace species/mode ordering, antisymmetrization, operator signs, interaction normalization, and the M=0 filter from input interaction through matrix assembly.
2. Compare HF functional energies against explicit Slater-state expectations in a tiny complete fixed-N,Z basis. A generic Slater determinant need not have M=0; restricting and renormalizing its amplitudes would instead test a projected state.
3. Check particle counts, orbital orthogonality, Hamiltonian Hermiticity, and agreement between ordinary and optimized matrix builders.
4. Establish a small reproducible baseline before a physical USDB benchmark. Select the first nucleus and computational budget after assessing basis dimensions.

## Subsequent milestones

1. Specify and test complex U,V conventions and canonical constraints, with HF as a limiting case.
2. Implement phase-consistent overlap and Hamiltonian kernels; explicitly address singular U and zero overlaps.
3. Implement N,Z projection, with gauge-grid exactness/convergence checks.
4. Implement particle-number PAV, checking the explicit fixed-sector vector
   against polynomial gauge-angle kernels without reoptimizing after projection.
5. Add exact-basis fidelity benchmarks with explicit sector weights and normalization. Distinguish N,Z projection from M=0 restriction and total-J projection.
6. Choose the non-Gaussianity definition explicitly; fidelity to the energy-minimizing HFB state is not automatically fidelity to the closest Gaussian state.
7. Add angular-momentum projection after validating the preceding layers.

Before claiming scaling guarantees or coding TAURUS formulas, consult the primary papers and derive bounds under explicit model-space and quadrature assumptions. Kernel cost, grid size, optimizer iteration count, and exact-basis benchmark cost must be reported separately.
