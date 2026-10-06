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

The JSON report includes raw and target-sector-conditioned fidelities, exact
and variational energies, relative energy errors, convergence diagnostics,
particle numbers, and pairing norms.  USDB exact diagonalization is restricted
to the M=0 sector; M=0 is not imposed on the intrinsic HF/HFB calculation.

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

The closest-Gaussian search also uses an analytic local Thouless gradient. It
differentiates the normalized Pfaffian overlap, applies canonical heavy-ball
updates, and evaluates only scalar overlaps during backtracking. This replaces
the previous finite-difference L-BFGS-B search over every Gaussian coordinate.

A step-by-step Neon tutorial, including the distinction between raw,
target-sector-conditioned, and rotationally projected fidelities, is available
in [`NeonGaussianFidelityTutorial.ipynb`](NeonGaussianFidelityTutorial.ipynb).

Visualize any JSON report from either study script with:

```bash
python plot_study_results.py results/usdb_gaussian_fidelity.json
python plot_study_results.py \
  results/usdb_ne20_projection_grid_convergence.json \
  --output results/ne20_projection.png
```

The first format produces fidelity, sector-weight, energy-error, and optimizer
residual panels. The projection format produces heat maps against the two Euler
grid sizes. The image defaults to the JSON filename with a `.png` extension;
`--output figure.pdf` or `--output figure.svg` creates a vector figure instead.

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
