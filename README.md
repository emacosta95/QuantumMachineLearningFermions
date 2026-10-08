# Nuclear-shell-model fermions

The main entry point is
[`NeonHFHFBStepByStep.ipynb`](NeonHFHFBStepByStep.ipynb). It develops the
USDB Neon-isotope calculation one operation at a time:

1. load the interaction and define the valence space;
2. minimize the HF and HFB energies;
3. diagonalize the exact fixed-$(N,Z,M=0)$ Hamiltonian;
4. compute intrinsic HF and HFB fidelities;
5. maximize the exact-state overlap over both Bogoliubov vacua and the
   Hartree-Fock boundary;
6. optionally separate HFB particle-number sector weight from projected
   fidelity.

The notebook begins with $^{20}$Ne and conservative learning settings. Increase
the multistart and iteration counts before treating a result as a production
calculation, then repeat the same cells for $^{22}$Ne and $^{24}$Ne.

The reusable numerical implementation is in `src/NSMFermions`. The batch
command-line equivalent remains available as `study_gaussian_fidelity.py`.

## Installation

Python 3.8 or newer is required.

```bash
python -m pip install -e .
```

For notebook use, install Jupyter in the same environment and open
`NeonHFHFBStepByStep.ipynb` from the repository root.

## Tests

```bash
python -m unittest discover -s tests
```
