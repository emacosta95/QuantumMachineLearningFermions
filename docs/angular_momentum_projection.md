# Particle-number and angular-momentum projection

## Scope

`angular_momentum.py` adds simultaneous neutron number, proton number, and
total-angular-momentum projection of a fully general complex Bogoliubov vacuum.
The first implemented target is (J=0), appropriate for the CKI
(^{8}\mathrm{Be}) ground state. The intrinsic vacuum may mix neutron and
proton orbitals and all magnetic substates; no axial, time-reversal, or
separate-pairing restriction is imposed.

For (J=0),

\[
 P_0={1\over 8\pi^2}\int_0^{2\pi}d\alpha
 \int_{-1}^{1}d(\cos\beta)\int_0^{2\pi}d\gamma\,
 R(\alpha,\beta,\gamma).
\]

The combined state and energy are

\[
 |\Psi_{NZ0}\rangle =
 {P_NP_ZP_0|\Omega\rangle\over
  \sqrt{\langle\Omega|P_NP_ZP_0|\Omega\rangle}},\qquad
 E_{NZ0}={\langle\Omega|H P_NP_ZP_0|\Omega\rangle\over
 \langle\Omega|P_NP_ZP_0|\Omega\rangle}.
\]

`ParticleNumberJ0ProjectedEnergy` evaluates this ratio directly from Pfaffian
overlaps and transition densities. It does not construct a many-body vector.
Neutron-proton mixing in the intrinsic state is retained. Physical rotations
preserve the species label, so (P_N), (P_Z), and (P_0) commute.

## Polynomial quadrature

The number grids remain (L_N=m_N+1) and (L_Z=m_Z+1). For the Euler grid,
the code derives conservative finite-space bounds (M_{\max}) and
(J_{\max}) for the requested particle numbers and uses

\[
 L_\alpha=L_\gamma=2M_{\max}+1,\qquad
 L_\beta=\left\lceil {J_{\max}+1\over2}\right\rceil.
\]

The alpha and gamma rules are periodic trapezoidal grids. The beta rule is
Gauss-Legendre in (\cos\beta). After the periodic sums select (M=K=0),
the remaining finite (J=0,1,\ldots,J_{\max}) content is a finite Legendre
expansion, which this beta rule integrates exactly. The constructor rejects a
user-supplied grid below these bounds.

For CKI (^{8}\mathrm{Be}), the automatically selected grids are

- number grid: (7\times7=49) points;
- Euler grid: (9\times4\times9=324) points;
- combined grid: 15,876 kernels per energy evaluation.

Thus one energy evaluation uses a number of kernels polynomial in the number
of single-particle modes. With a dense two-body interaction, its cost is

\[
 O(L_NL_ZL_\alpha L_\beta L_\gamma(m^4+m^3)).
\]

This is a per-evaluation complexity statement. It does not make the global
nonlinear optimization of the Bogoliubov vacuum polynomial or guarantee that
a local optimizer finds the global minimum.

## Validation backends

`exact_j0_projector` constructs (J^2=J_x^2+J_y^2+J_z^2) in an enumerated
fixed-(N,Z) space and diagonalizes it. This is deliberately a small-system
reference, since its dimension is combinatorial. It is used to validate the
polynomial kernel and to compute benchmark fidelities, not as the production
projection algorithm.

The regression test compares both implementations in a spin-half proton-neutron
model. In CKI (^{8}\mathrm{Be}), the polynomial-grid projected energy differs
from the exact-(J^2) reference by (4.97\times10^{-14}) MeV.

General (J>0) projection is not represented by the scalar (J=0) integral.
A triaxial intrinsic vacuum then needs the full (P^J_{MK}) norm and Hamiltonian
matrices followed by (K) mixing. Odd systems additionally need blocked HFB
vacua. Both are left as explicit extensions rather than hidden approximations.

## Minimal use

```python
from NSMFermions.angular_momentum import ParticleNumberJ0ProjectedEnergy

objective = ParticleNumberJ0ProjectedEnergy(
    ham, state_encoding, neutron_modes, targets=(2, 2)
)
energy = objective.energy(Z)
print(objective.grid, objective.euler_grid.size, energy)
```

Here `Z` is the antisymmetric Thouless matrix of the intrinsic Bogoliubov
vacuum. The saved (^{8}\mathrm{Be}) benchmark can be reproduced with
`python benchmarks/cki_be8_j0_projection.py`.

## References

- Bally and Bender, particle-number and angular-momentum projection of
  triaxial Bogoliubov states: https://arxiv.org/abs/2010.15224
- Johnson and Jiao, SO(3) quadratures in angular-momentum projection:
  https://arxiv.org/abs/2205.04119
- Robledo, Pfaffian evaluation of HFB overlaps:
  https://arxiv.org/abs/0901.3213
