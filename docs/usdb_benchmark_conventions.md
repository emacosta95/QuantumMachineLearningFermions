# USDB benchmark conventions

The repository input `data/usdb.nat` was aligned with the collaborator's
`results.zip` / `24Ne/usdb.sho` on 2026-10-16. Its single-particle energies and
all coupled two-body elements now match that supplied file. This alignment is
not an independent determination of which published USDB parameter set is
authoritative.

Changes to the input:

- In the `205 203 203 1001` block, the T=1, J=1 element changed from
  -0.0736 to -0.0456 MeV.
- Core metadata now lists 8 protons and 8 neutrons, matching the O16 core in
  the collaborator's TAURUS input. Our Python reader does not apply this
  metadata automatically.

Enable mass scaling explicitly for each isotope:

```python
interaction, eps, sp, path = load_nuclear_interaction("usdb", mass_number=24)
```

Only two-body elements are multiplied by `(18 / A)**0.3`. `A` is the total
nuclear mass, including the core. Single-particle energies are unchanged.
The low-level `compute_nuclear_twobody_matrix` and
`get_twobody_nuclearshell_model` accept the same `mass_number`,
`reference_mass`, and `mass_exponent` keywords. Omitting `mass_number` keeps
scaling disabled for backward compatibility.

Use the same scaled interaction for the variational solver and exact
diagonalization. Saved results from the previous unscaled input are historical
and must be recomputed before comparisons to this benchmark.

`verify_taurus_import.py` imports the supplied TAURUS U,V matrices, converts
the ordering of magnetic substates, checks energies against matching state
labels, and computes raw and number-projected fidelities against a ground
state diagonalized with matching interaction parameters. Ne24's binary label
matches `out2.txt`, not `out.txt`.

## Independent Ne24 optimization

`verify_ne24_scaled_variational.py` reruns both methods with eight starting
states, without importing TAURUS U,V as an optimization seed. The exact-state
cache is checked against the newly assembled Hamiltonian before overlaps are
evaluated. Results are in `results/ne24_scaled_variational_audit.json` and
the corresponding `.pkl`.

| State | Energy (MeV) | Raw fidelity | Infidelity |
|---|---:|---:|---:|
| Independent HF | -66.2245803934 | 0.1197595583 | 0.8802404417 |
| Independent HFB, best found | -66.4023493320 | 0.0901476510 | 0.9098523490 |
| Supplied TAURUS HFB | -66.3278135878 | 0.1284417660 | 0.8715582340 |

The selected HF and HFB results converged (gradient norms 2.83e-7 and
9.94e-6, respectively). Six of the eight HFB attempts stalled before the
requested 1e-5 gradient tolerance; the selected best result did converge.
The independent HFB state has neutron pairing (kappa-block norm 0.904515)
and almost no proton pairing (0.000155). The supplied TAURUS state has proton
pairing (0.829309) and almost no neutron pairing (4.63e-7). Their energy
difference is about 74.54 keV. Several independent starts also approached
the TAURUS energy branch but did not meet the stricter gradient tolerance.

Thus the imported-state agreement validates the reconstruction, while the
independent optimizer selects a distinct, lower-energy paired solution.
Neither calculation establishes a certified global minimum.
