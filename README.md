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

## Simple saved-data study

Edit the configuration block at the top of `simple_hfb_gaussian_study.py` to
choose the interaction and lists of valence proton and neutron numbers, then
run:

```bash
python simple_hfb_gaussian_study.py
```

The script performs HFB energy minimization, exact diagonalization, and the
best-of-HFB/HF Gaussian-overlap search for every selected particle-number
pair. It saves a checkpointed pickle dictionary indexed by
`data["results"][valence_protons][valence_neutrons]`. Each entry contains the
exact state and basis, both variational `HFBState` objects, their family labels,
energies, overlaps, fidelities, and convergence diagnostics. Only load pickle
files that you trust.

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
