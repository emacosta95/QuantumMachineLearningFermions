# Nuclear-shell-model fermions

The two main study programs are located at the repository root.  They accept
`--interaction cki` for the p-shell Be isotopes and `--interaction usdb` for
the sd-shell Ne isotopes.

## Gaussian fidelity and variational energy

Compare the energy-optimized HF/HFB state and the maximum-overlap pure
Gaussian state with the exact ground state:

```bash
python study_gaussian_fidelity.py --interaction cki
python study_gaussian_fidelity.py --interaction usdb --isotopes 20 22 24
```

To test only the intrinsic Bogoliubov optimization, without exact
diagonalization or the separate closest-Gaussian search, use:

```bash
python study_gaussian_fidelity.py --interaction usdb --isotopes 20 \
  --variational-method hfb --variational-only
```

The JSON report includes the intrinsic fidelity
`|<exact ground state|Gaussian>|^2`, exact and variational energies, relative
energy errors, convergence diagnostics, particle numbers, and pairing norms.
It does not report a sector-conditioned fidelity or a target-sector weight:
those are properties of a projected analysis, not of the intrinsic Gaussian
comparison performed here. USDB exact diagonalization is restricted to the
M=0 sector; M=0 is not imposed on the intrinsic HF/HFB calculation.

New reports use the unambiguous fields `variational_ground_state_fidelity` and
`closest_gaussian_ground_state_fidelity`. The plotter still accepts older JSON
files whose corresponding field names ended in `_raw`.

Use `--variational-method hf` for the number-conserving `kappa=0` manifold or
`--variational-method hfb` for the paired Bogoliubov manifold. Both use the
library's constrained analytic-gradient interface and multistart policy. HFB
uses local `H20` gradients, per-iteration neutron/proton multipliers,
heavy-ball updates, and local number corrections rather than SLSQP or a
quadratic number penalty. Both variational searches are non-convex, so increase
`--starts` and `--gaussian-starts` for production results.

The study scripts use real HFB and closest-Gaussian states by default, matching
the real TAURUS formulation. Pass `--complex-bogoliubov` to allow complex
states. The real restriction makes intrinsic `U`, `V`, `rho`, and `kappa`
real and halves
the Gaussian coordinates: CKI (12 modes) uses 66 instead of 132 parameters,
and USDB (24 modes) uses 276 instead of 552.  Do not use this restriction when
complex or time-reversal-breaking intrinsic states are physically required.

The implementation and references are described in
[`docs/hfb_reference_solver.md`](docs/hfb_reference_solver.md).

The closest-Gaussian search uses no numerical differentiation. It accumulates
the Pfaffian cofactors once, reverse-contracts them into a compact local
`F20` field, applies canonical heavy-ball updates, and evaluates only scalar
overlaps during backtracking. This is the overlap analogue of the energy
`H20` solver and replaces the previous finite-difference L-BFGS-B search.

Every closest-Gaussian calculation now runs both the finite-Thouless
Bogoliubov search and the number-conserving HF/Slater-boundary search, then
returns the larger intrinsic fidelity. The JSON records the selected family
and the convergence data for both searches. `--gaussian-starts` and
`--gaussian-maxiter` control the Bogoliubov search; use
`--gaussian-hf-starts` and `--gaussian-hf-maxiter` to override those values for
the HF overlap search. This HF boundary fixes the target's total particle
number but permits neutron-proton orbital mixing; it does not impose separate
intrinsic neutron and proton numbers.

A step-by-step Neon tutorial, including the distinction between intrinsic and
rotationally projected fidelities, is available
in [`NeonGaussianFidelityTutorial.ipynb`](NeonGaussianFidelityTutorial.ipynb).

Visualize any JSON report from either study script with:

```bash
python plot_study_results.py results/usdb_gaussian_fidelity.json --per-isotope
python plot_study_results.py \
  results/usdb_ne20_projection_grid_convergence.json \
  --output results/ne20_projection.png
```

The first format produces intrinsic-fidelity, energy, relative-error, and optimizer
residual panels. The projection format produces heat maps against the two Euler
grid sizes. The image defaults to the JSON filename with a `.png` extension;
`--output figure.pdf` or `--output figure.svg` creates a vector figure instead.
With `--per-isotope`, the Gaussian study also writes one four-panel image for
every completed nucleus, for example `..._ne20.png`, `..._ne22.png`, and
`..._ne24.png`.

## Projected non-Gaussianity

Increase the Euler quadrature from one point through `(M_MAX,J_MAX,M_MAX)`:

```bash
python study_projected_nongaussianity.py \
  --interaction cki --mass 8 --m-max 9 --j-max 4

python study_projected_nongaussianity.py \
  --interaction usdb --mass 20 --m-max 17 --j-max 6
```

For every `(M,J)` pair the report stores projected fidelity, relative energy
error, FAF non-Gaussianity, `<J^2>`, effective `J`, and the FAF difference and
ratio relative to the exact ground state.  The exact-ground-state FAF is also
stored once at report level.  Results default to the root `results/` directory.

## Slurm

Create `logs/` before submitting because Slurm opens its output files before
the job script runs:

```bash
mkdir -p logs
sbatch --export=ALL,RUN_MODE=projection,INTERACTION=cki,MASS=8,M_MAX=9,J_MAX=4 \
  slurm/run_nuclear_projection.sbatch

ISOTOPES="20 22 24" sbatch \
  --export=ALL,RUN_MODE=fidelity,INTERACTION=usdb \
  slurm/run_nuclear_projection.sbatch
```
