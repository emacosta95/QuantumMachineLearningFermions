# CKI 8Be: particle-number plus J=0 projection

The input intrinsic state is the saved unrestricted complex PN-VAP
Bogoliubov vacuum. The calculation applies rotational projection after the
PN-VAP optimization; it does not reoptimize the intrinsic vacuum with the
rotationally projected energy.

| State | Energy (MeV) | Exact-ground-state fidelity |
|---|---:|---:|
| Exact CKI ground state | -30.295394614 | 1 |
| PN-VAP, PnPz projected | -27.398536786 | 0.391058817 |
| Same vacuum, PnPzJ=0 projected | -30.079706080 | 0.990856904 |

The J=0 probability within the normalized N=Z=2 projected state is
0.394667298. Removing its J>0 components gains 2.681169294 MeV and leaves a
0.215688534 MeV difference from the exact ground state.

The automatically derived grid has 7 x 7 number angles and 9 x 4 x 9 Euler
points, or 15,876 combined kernels. Its energy agrees with direct projection
onto the J=0 eigenspace of J-squared to 4.97e-14 MeV. The direct projector and
the stored projected vector are validation artifacts for this small system;
the Pfaffian/transition-density grid is the polynomial-size implementation.

The result is projection after variation with respect to angular momentum.
Simultaneous PnPzJ=0 variation may improve the remaining energy and fidelity,
but it is a separate nonlinear optimization and has no polynomial global
convergence guarantee.
