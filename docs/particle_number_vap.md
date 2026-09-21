# Particle-number VAP and polynomial gauge grids

## Objective and interpretation

Ordinary HFB minimizes the unprojected energy with average-number constraints.
Particle-number variation after projection (PN-VAP) instead minimizes

E_NZ = <Phi|H P_N P_Z|Phi> / <Phi|P_N P_Z|Phi>.

Pairing is allowed to improve this projected objective. No lower bound on the
pairing tensor and no artificial pairing penalty is imposed. This approach
can recover correlations where HFB collapses, but is not a promise of nonzero
pairing for every Hamiltonian or convergence to a global minimum.

All complex nn, pp, np pair amplitudes and normal species mixing are allowed.
The present vacuum Thouless chart covers even total parity with nonzero vacuum
overlap; exact singular-U HF states require a different chart or a limiting
sequence. Odd-total blocked vacua are not supported yet.

## Two implementations

- `gauge_projection.py`: `GaugeProjectedEnergy` evaluates polynomial-cost
  Pfaffian overlaps and transition-density kernels on a full-period gauge
  grid. `.solve(...)` runs a finite-difference L-BFGS-B local optimization and
  returns a SciPy OptimizeResult. Inspect both `success` and the gradient.
- `number_projection.py`: `NumberProjectedSpace` and `solve_number_vap`
  enumerate the exact N,Z sector, evaluate Pfaffian amplitudes and analytic
  derivatives, and minimize a Rayleigh quotient. This is a combinatorial
  reference backend with resource guards, not the scalable production route.

General complex Pfaffians are evaluated by the external `pfapack` package.
The exact-sector optimizer separately retains a signed matching expansion
because it needs analytic derivatives of every amplitude with respect to Z.

The sector backend also provides exact small-system fidelity benchmarks.
Neither solver imposes M=0 or total J projection.

## Grid size and complexity

Let m_n and m_p be the numbers of single-particle modes. Use full-period angles
phi_s = 2 pi (k+offset)/L_s with L_n=m_n+1 and L_p=m_p+1. The finite Fock space
has particle numbers 0,...,m_s. Discrete Fourier orthogonality then removes
every non-target sector exactly in exact arithmetic: no nonzero difference
of represented particle numbers is a multiple of L_s. Both species parities
must be retained when np mixing is allowed. No half-period assumption is used.

The number of grid pairs is (m_n+1)(m_p+1), quadratic in total modes. For the
dense two-body tensor, a projected-energy evaluation costs
O(L_n L_p (m^4+m^3)); stored interaction size is O(m^4). Finite-difference
gradients add O(m^2) evaluations. Neither statement bounds the number of
iterations required to reach a global optimum.

For CKI Be8: m_n=m_p=6, so 7 x 7 = 49 angle pairs. A 9 x 9 grid and a different
offset provide a numerical cross-check. Undersized grids are rejected. The
kernel currently raises on ill-conditioned transition matrices or unresolved
projected norms. Shifted grids avoid common overlap zeros but do not solve
every possible singularity. A reference-chart/zero-overlap treatment remains
necessary for a robust large-scale implementation.

## Fidelity conventions

For a normalized exact fixed-N,Z ground state |Psi0>:

- Intrinsic fidelity F_intrinsic = |<Psi0|Phi>|^2.
- Sector probability p_NZ = <Phi|P_N P_Z|Phi>.
- Projected fidelity F_projected = F_intrinsic / p_NZ.

In a vacuum Thouless chart, positive species scaling Z -> D Z D multiplies
every fixed-N,Z amplitude by the same factor, leaving the normalized projected
state unchanged. Thus a VAP minimum alone does not uniquely specify its
intrinsic fidelity. For the Be8 comparison we select the representative with
average N=Z=2 using two positive scaling factors. This is possible for the
reported state; it is not asserted for all ansatz boundaries.

The projected state's anomalous expectation <cc> is zero by number conservation.
Reported nonzero kappa refers to the intrinsic vacuum. Energy improvement and
number-conserving correlations, not projected <cc>, diagnose retained pairing.

## Minimal API

```python
from NSMFermions.gauge_projection import GaugeProjectedEnergy

# ham is an HFBHamiltonian; indices are explicit, never assumed by species.
objective = GaugeProjectedEnergy(ham, neutron_modes, targets=(2, 2))
fit = objective.solve(seed=15, maxiter=200)
Z = objective.unpack(fit.x)
energy = objective.energy(Z)
print(fit.success, fit.message, energy)
```

For an explicit small-space projected vector and fidelity:

```python
from NSMFermions.hfb import HFBState
from NSMFermions.number_projection import NumberProjectedSpace

state = HFBState.from_thouless(Z)
projected_vector = state.fixed_sector_state(space.occupations)
raw_fidelity = state.fixed_sector_fidelity(
    exact_ground_state, space.occupations)
projected_fidelity = state.fixed_sector_fidelity(
    exact_ground_state, space.occupations, projected=True)

# Equivalent convenience API owned by the selected N,Z space:
projected_vector = space.projected_state(Z)
projected_fidelity = space.projected_fidelity(Z, exact_ground_state)
```

The state owns generic Pfaffian occupation amplitudes. `NumberProjectedSpace`
owns the list of determinants defining the desired N,Z sector and the analytic
derivatives required by PN-VAP.

For lightweight environments, as with `hfb.py`, add `src/NSMFermions` to
`sys.path` and import the module directly to avoid the legacy eager ML imports.

## Verification

21 regression tests pass: canonical constraints, complex Fock-space energies,
HF limits, weak-pairing HFB collapse and VAP recovery, analytic reference
gradients, Pfaffian signs, complex gauge kernels, species-scale invariance,
grid refinement and a gauge-only optimizer on a small pairing model.

The CKI benchmark first optimizes the exact-sector reference with three starts,
then performs a gauge-grid refinement from that solution. It therefore
validates the gauge objective at a stationary solution but is not an independent
random-start convergence study of the gauge-only Be8 solver. See
`benchmarks/results/cki_be8_vap.md` for numerical results.

## References

- TAURUS I, particle-number VAP: https://arxiv.org/abs/2010.14169
- Stoitsov et al., number-projected HFB: https://arxiv.org/abs/nucl-th/0610061
- Worst-case global HF complexity (not a hardness proof for this CKI problem):
  https://arxiv.org/abs/2103.08215

These references motivate the approach; this code is an independent reference
implementation, not a port of TAURUS.
