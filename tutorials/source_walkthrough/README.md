# NSMFermions source-code walkthrough

This series explains the source used by the CKI/HFB/Gaussian-fidelity workflow
for a reader who is new to Python classes and symmetry projection.  The order is
dependency-first: each chapter introduces only concepts needed by later ones.

1. `01_hfb_state_line_by_line.ipynb` — Python/NumPy notation, dataclasses,
   Bogoliubov amplitudes, Thouless and Slater charts, occupation amplitudes,
   densities, and canonical checks.
2. `02_hfb_energy_and_optimization.ipynb` — `HFBHamiltonian`, Wick contractions,
   parameterization, number constraints, `solve_hfb`, and `HFBResult`.
3. `03_number_projection_line_by_line.ipynb` — determinant-basis interface,
   discrete Fourier projection, `BogoliubovVacuumSeries`, normalization, energy,
   and fidelity.
4. `04_pfaffian_and_angular_momentum.ipynb` — overlap Pfaffians, transition
   densities, single-particle angular momentum, Euler rotations, exact grids,
   and the simultaneous `N,Z,J=0` projector.
5. `05_gaussian_fidelity_and_alignment.ipynb` — finite-Thouless optimization,
   the Slater boundary, Stiefel gradients, rotation alignment, and geometric
   non-Gaussianity.
6. `06_end_to_end_trace.ipynb` — trace one CKI calculation through the objects
   above, with shapes, invariants, debugging checks, and computational cost.

The chapters intentionally distinguish:

- a Python operation from the mathematical operation it represents;
- an exact identity from a numerical approximation;
- an intrinsic symmetry-breaking state from a projected physical state;
- a certified statement from a best-found optimization result.

