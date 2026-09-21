"""Polynomial-cost number-projected kernels in a vacuum Thouless chart.

No occupation-basis construction. Singular transition matrices are reported,
not regularized silently. Shifted full-period Fourier grids avoid common
overlap zeros but cannot guarantee conditioning for every state.
"""
import numpy as np
from pfapack import pfaffian as pf

if __package__:
    from .hfb import BogoliubovVacuumSeries, HFBState
else:
    from hfb import BogoliubovVacuumSeries, HFBState


def pfaffian(matrix):
    """Return the complex Pfaffian using PFAPACK's Parlett-Reid routine.

    This wrapper preserves the projection module's explicit validation and
    ``ValueError`` interface while delegating the numerical algorithm to the
    maintained :mod:`pfapack` package.
    """
    a = np.asarray(matrix, dtype=complex)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) % 2:
        raise ValueError("Pfaffian requires even square dimension")
    if not np.isfinite(a).all() or not np.allclose(a, -a.T, atol=1e-10):
        raise ValueError("Pfaffian requires finite antisymmetric matrix")

    # method="P" selects PFAPACK's O(n^3) Parlett-Reid implementation.
    # Do not overwrite the overlap matrix: it may be useful for diagnostics.
    return pf.pfaffian(a, overwrite_a=False, method="P")


class GaugeProjectedEnergy:
    """Exact full-period N,Z Fourier projection for finite mode spaces.

    Default L_n=m_n+1, L_p=m_p+1 resolves all particle-number sectors, even
    with np mixing and odd species parities. Cost per energy evaluation is
    O(L_n L_p (m^4+m^3)); dense interaction storage is O(m^4). Fixed undersized
    grids are rejected. This says nothing about global optimization complexity.

    This class evaluates projected *kernels* without constructing a many-body
    state vector. That is what keeps its memory polynomial in the number of
    modes. Use ``project_particle_numbers`` with an assembled
    ``FermiHubbardHamiltonian`` when an explicit small-space vector is required
    for fidelity or other observables.
    """
    def __init__(self,hamiltonian,neutron_modes,targets,grid=None,offset=.137):
        # Retain the raw one- and two-body tensors used by every transition kernel.
        self.ham=hamiltonian
        # The one-body matrix dimension is the number of fermionic modes.
        m=len(hamiltonian.h)

        # Mark which one-body indices are neutrons.  Every unmarked mode is
        # treated as a proton; the modes need not be stored in species blocks.
        ns=list(neutron_modes)
        invalid_type=any(not isinstance(i,(int,np.integer)) for i in ns)
        repeated_mode=len(set(ns))!=len(ns)
        out_of_range=any(i<0 or i>=m for i in ns)
        if invalid_type or repeated_mode or out_of_range:
            raise ValueError('Invalid neutron indices')
        # Start with every mode labeled as proton.
        self.mask=np.zeros(m,bool)
        # Mark precisely the caller-supplied neutron modes.
        self.mask[ns]=True
        # The two block sizes bound the allowed neutron and proton numbers.
        caps=[len(ns),m-len(ns)]

        # The two integer targets define P_N P_Z.  An unblocked Thouless vacuum
        # contains only even total number parity, hence the parity guard.
        malformed_targets=len(targets)!=2
        invalid_targets=(not malformed_targets and any(
            not isinstance(t,(int,np.integer)) or t<0 or t>c
            for t,c in zip(targets,caps)
        ))
        if malformed_targets or invalid_targets:
            raise ValueError('Invalid integer particle numbers')
        # An unblocked even HFB vacuum has no odd-total-particle component.
        if sum(targets)%2:
            raise ValueError('Even total parity required')
        # Store an immutable target used in every Fourier character.
        self.targets=tuple(targets)

        # A species with c modes has sectors 0,...,c, so c+1 equally spaced
        # angles resolve its finite Fourier polynomial exactly.
        default_grid=tuple(c+1 for c in caps)
        # A user grid may be larger than the exact minimal rule, never smaller.
        self.grid=default_grid if grid is None else tuple(grid)
        invalid_grid=len(self.grid)!=2 or any(
            not isinstance(points,(int,np.integer)) or points<=capacity
            for points,capacity in zip(self.grid,caps)
        )
        if invalid_grid:
            raise ValueError('Exact grid requires more points than modes of each species')
        # Nonfinite offsets would contaminate every complex phase.
        if not np.isfinite(offset):
            raise ValueError('Offset must be finite')
        # A common fractional offset moves the quadrature away from common
        # overlap zeros without changing a complete periodic Fourier sum.
        self.offset=float(offset)

        # Separate N and Z projectors require H to conserve both species.  The
        # labels on the incoming and outgoing legs of every nonzero term must
        # therefore match.
        labels=self.mask.astype(int)
        # Locate every numerically nonzero one-body matrix element.
        i,j=np.nonzero(np.abs(hamiltonian.h)>1e-12)
        # Locate every numerically nonzero two-body tensor element.
        a,b,c,d=np.nonzero(np.abs(hamiltonian.v)>1e-12)
        # One-body terms must preserve the species of their particle line.
        one_body_breaks_species=np.any(labels[i]!=labels[j])
        # Two-body terms must conserve total neutron label across both lines.
        two_body_breaks_species=np.any(
            labels[a]+labels[b]!=labels[c]+labels[d]
        )
        if one_body_breaks_species or two_body_breaks_species:
            raise ValueError('Hamiltonian must conserve each species number')

    def projected_series(self,state):
        """Return P_N P_Z|Phi> as the configured finite gauge-vacuum series."""
        # Projection acts on the full HFB/HF state rather than density matrices.
        if not isinstance(state,HFBState):
            raise TypeError('state must be an HFBState')
        # The raw Hamiltonian fixes the expected one-body dimension.
        m=len(self.ham.h)
        if len(state.U)!=m:
            raise ValueError('State and projection Hamiltonian sizes differ')

        # Materialize one transformed vacuum for each N,Z Fourier point.
        transformations=[]
        weights=[]
        # Traverse the neutron gauge grid.
        for i in range(self.grid[0]):
            # Convert its integer index into a shifted full-period angle.
            pn=2*np.pi*(i+self.offset)/self.grid[0]
            # Traverse the independent proton gauge grid.
            for j in range(self.grid[1]):
                # Convert the proton index to its gauge angle.
                pp=2*np.pi*(j+self.offset)/self.grid[1]
                # Assign neutron or proton phase to every particle mode.
                phase=np.exp(1j*np.where(self.mask,pn,pp))
                # The one-body gauge transformation is diagonal.
                transformations.append(np.diag(phase))
                # Its Fourier character selects precisely the target N,Z sector.
                character=np.exp(
                    -1j*(pn*self.targets[0]+pp*self.targets[1])
                )
                # Divide by both grid sizes to represent the normalized integrals.
                weights.append(character/np.prod(self.grid))

        # Return the series without evaluating a single determinant amplitude.
        return BogoliubovVacuumSeries(
            intrinsic_state=state,
            transformations=np.asarray(transformations),
            weights=np.asarray(weights),
            number_grid=self.grid,
            euler_grid=None,
            projection='P_N P_Z',
            number_offset=self.offset,
        )

    def series_energy(self,series):
        """Evaluate number-projected energy from a stored gauge-vacuum series."""
        # Require the same structured series later consumed by fidelity expansion.
        if not isinstance(series,BogoliubovVacuumSeries):
            raise TypeError('series must be a BogoliubovVacuumSeries')
        # A series made on a different grid would not represent this evaluator.
        if series.number_grid!=tuple(self.grid) or series.euler_grid is not None:
            raise ValueError('Series grid differs from number projector grid')
        # Equal dimensions are insufficient if the stored Fourier nodes differ.
        if not np.isclose(series.number_offset,self.offset):
            raise ValueError('Series number-grid offset differs from evaluator')
        # Transition kernels require a finite particle-vacuum Thouless chart.
        z=series.intrinsic_state.thouless_matrix
        # Infer and reuse the one-body identity matrix.
        m=len(self.ham.h)
        identity=np.eye(m)
        # Accumulate Hamiltonian and norm kernels independently.
        numerator=0j
        denominator=0j
        # Normalize the common intrinsic ket once for every transformed term.
        norm_matrix=identity+z.conj().T@z
        log_norm_determinant=np.linalg.slogdet(norm_matrix)[1]
        scale=np.exp(.5*log_norm_determinant)

        # Use exactly the transformations and coefficients stored in the series.
        for transform,series_weight in zip(series.transformations,series.weights):
                # A one-body group action rotates both legs of each created pair.
                zg=transform@z@transform.T

                # B=I+Z^dagger Z(phi) enters the generalized Wick theorem.  A
                # poorly conditioned B makes all transition contractions unsafe.
                b=identity+z.conj().T@zg
                if np.linalg.cond(b)>1e12:
                    raise ValueError('Singular gauge kernel; change grid offset or chart')
                inverse=np.linalg.solve(b,identity)
                # Normal, pair-annihilation, and pair-creation transition
                # densities, each divided by the vacuum overlap.
                rho=zg@inverse@z.conj().T
                kappa=zg@inverse
                creation=-inverse@z.conj().T

                # The Pfaffian gives the complex overlap with an unambiguous
                # sign.  A square root of a determinant would have a sign branch.
                overlap_matrix=np.block(
                    [[zg,-identity],[identity,-z.conj()]]
                )
                pfaffian_sign=(-1)**(m*(m+1)//2)
                overlap=(
                    pfaffian_sign*pfaffian(overlap_matrix)/scale
                )

                # Multiply the signed overlap by this series coefficient w_q.
                weight=overlap*series_weight

                # Generalized-Wick energy kernel: one-body, normal two-body,
                # and anomalous pairing contributions.
                one_body=np.einsum('ij,ji->',self.ham.h,rho)
                normal_two_body=.5*np.einsum(
                    'ijkl,ki,lj->',self.ham.v,rho,rho
                )
                pairing_two_body=.25*np.einsum(
                    'ijkl,ij,kl->',self.ham.v,creation,kappa
                )
                kernel=one_body+normal_two_body+pairing_two_body
                # Add the overlap kernel to the projected norm sum.
                denominator+=weight
                # Add H times the overlap kernel to the energy numerator.
                numerator+=weight*kernel

        # Series weights already include both quadrature normalization factors.
        norm=denominator
        if abs(norm.imag)>1e-8 or norm.real<1e-14:
            raise ValueError('Vanishing or numerically unresolved projected norm')
        # The common quadrature normalization cancels in this ratio.
        energy=numerator/denominator
        if abs(energy.imag)>1e-7:
            raise ValueError('Projection cancellation error exceeds tolerance')
        # Return the physically real value after verifying cancellation error.
        return float(energy.real)

    def energy(self,z):
        """Build the gauge-vacuum series from Z and evaluate its energy."""
        # Convert the finite Thouless matrix into the common intrinsic-state API.
        state=HFBState.from_thouless(np.asarray(z,complex))
        # Energy and later fidelity now use the identical finite series.
        return self.series_energy(self.projected_series(state))

    def unpack(self,x):
        """Map real optimizer coordinates to an antisymmetric complex Z."""
        # Determine the number of independent entries above the diagonal.
        m=len(self.ham.h)
        ij=np.triu_indices(m,1)
        p=len(ij[0])
        # Optimizers supply separate real and imaginary coordinate blocks.
        x=np.asarray(x,float)
        if x.shape!=(2*p,) or not np.isfinite(x).all():
            raise ValueError('Invalid complex Thouless parameters')
        # The first p entries are Re(Z_ij), the remaining p are Im(Z_ij).
        z=np.zeros((m,m),complex)
        z[ij]=x[:p]+1j*x[p:]
        # Subtracting the transpose fills the lower triangle with -Z_ij.
        return z-z.T
