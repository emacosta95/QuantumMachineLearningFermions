"""Polynomial-cost number-projected kernels in a vacuum Thouless chart.

No occupation-basis construction. Singular transition matrices are reported,
not regularized silently. Shifted full-period Fourier grids avoid common
overlap zeros but cannot guarantee conditioning for every state.
"""
import warnings

import numpy as np
from pfapack import pfaffian as pf

if __package__:
    from .hfb import BogoliubovVacuumSeries, HFBState, ProjectionGridWarning
else:
    from hfb import BogoliubovVacuumSeries, HFBState, ProjectionGridWarning


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


def transformed_vacuum_overlap(state, transform):
    """Return the phase-consistent overlap ``<Phi|T|Phi>``.

    The overlap magnitude defines the optional Metropolis importance density.
    Finite-Z vacua use the same Pfaffian convention as the energy kernels;
    singular-U Slater determinants use the occupied-orbital determinant.
    """
    # Convert and validate the one-body group transformation.
    transform=np.asarray(transform,dtype=complex)
    modes=len(state.U)
    if transform.shape!=(modes,modes) or not np.isfinite(transform).all():
        raise ValueError('Transform must be a finite one-body square matrix')
    try:
        # Recover the particle-vacuum chart when it exists.
        z=state.thouless_matrix
    except ValueError as chart_error:
        try:
            # A singular-U number-conserving state is handled as a Slater state.
            orbitals=state.slater_orbitals
        except ValueError:
            raise chart_error
        # The overlap of two Slater determinants is det(C^dagger T C).
        return complex(np.linalg.det(orbitals.conj().T@transform@orbitals))

    # Rotate both creation legs of the Thouless pair matrix.
    rotated_z=transform@z@transform.T
    # Construct the antisymmetric Robledo overlap matrix.
    identity=np.eye(modes)
    overlap_matrix=np.block(
        [[rotated_z,-identity],[identity,-z.conj()]]
    )
    # Normalize both vacua; unitary T preserves their common norm.
    scale=np.exp(
        .5*np.linalg.slogdet(identity+z.conj().T@z)[1]
    )
    # PFAPACK supplies the signed overlap rather than an ambiguous square root.
    sign=(-1)**(modes*(modes+1)//2)
    return complex(sign*pfaffian(overlap_matrix)/scale)


def _reflect_unit_interval(values):
    """Reflect real coordinates into [0,1] with a symmetric proposal map."""
    # Folding a period-two coordinate implements reflection at both boundaries.
    folded=np.mod(values,2.0)
    # Values in the second half are mirrored back into the unit interval.
    return np.where(folded<=1.0,folded,2.0-folded)


def metropolis_vacuum_terms(
    state,
    coordinate_to_term,
    dimensions,
    periodic_dimensions,
    *,
    samples,
    burn_in=500,
    thinning=5,
    proposal_scale=0.15,
    seed=0,
    overlap_floor=1e-10,
):
    """Importance-sample transformed vacua with Metropolis-Hastings.

    ``coordinate_to_term(u)`` maps unit-cube coordinates to ``(T, character)``.
    The chain targets ``q(u) proportional to max(|<Phi|T|Phi>|, floor)``.
    Returned coefficients are proportional to ``character/q``; the unknown
    normalization of q is common to the whole projected ket and cancels after
    state normalization and in projected-energy ratios.
    """
    # Validate integer chain controls before allocating the retained series.
    if (not isinstance(samples,(int,np.integer)) or samples<1
            or not isinstance(burn_in,(int,np.integer)) or burn_in<0
            or not isinstance(thinning,(int,np.integer)) or thinning<1):
        raise ValueError('Metropolis samples, burn-in, and thinning are invalid')
    # Use one positive random-walk scale for all unit-cube coordinates.
    if not np.isfinite(proposal_scale) or proposal_scale<=0:
        raise ValueError('Metropolis proposal_scale must be positive and finite')
    # A positive floor guarantees support even where the overlap is exactly zero.
    if not np.isfinite(overlap_floor) or overlap_floor<=0:
        raise ValueError('Metropolis overlap_floor must be positive and finite')

    # Mark coordinates such as gauge, alpha, and gamma angles as periodic.
    periodic=np.asarray(periodic_dimensions,dtype=bool)
    if periodic.shape!=(dimensions,):
        raise ValueError('Periodic-coordinate mask has the wrong dimension')
    # Seed a reproducible chain uniformly with respect to the base Haar measure.
    rng=np.random.default_rng(seed)
    current=rng.random(dimensions)
    # Map the initial coordinate to its one-body transform and projector character.
    current_transform,current_character=coordinate_to_term(current)
    # The target density is the positive overlap magnitude, never its complex phase.
    current_overlap=transformed_vacuum_overlap(state,current_transform)
    current_score=max(abs(current_overlap),overlap_floor)

    # Store only post-burn-in, thinned terms of the Markov chain.
    transformations=[]
    raw_weights=[]
    overlap_reweights=[]
    accepted=0
    total_steps=burn_in+samples*thinning
    for step in range(total_steps):
        # Symmetric Gaussian random walk in normalized group coordinates.
        proposal=current+rng.normal(scale=proposal_scale,size=dimensions)
        # Wrap periodic angles around the unit circle.
        proposal[periodic]=np.mod(proposal[periodic],1.0)
        # Reflect nonperiodic x=cos(beta) coordinates at their boundaries.
        proposal[~periodic]=_reflect_unit_interval(proposal[~periodic])
        # Build the proposed transformed vacuum and its complex character.
        proposed_transform,proposed_character=coordinate_to_term(proposal)
        # Evaluate the positive importance density at the proposal.
        proposed_overlap=transformed_vacuum_overlap(state,proposed_transform)
        proposed_score=max(abs(proposed_overlap),overlap_floor)
        # Symmetric proposals leave only the target-density ratio in MH.
        acceptance=min(1.0,proposed_score/current_score)
        if rng.random()<acceptance:
            # Accept all state associated with the proposed coordinate.
            current=proposal
            current_transform=proposed_transform
            current_character=proposed_character
            current_overlap=proposed_overlap
            current_score=proposed_score
            accepted+=1

        # Discard burn-in and retain one state every ``thinning`` transitions.
        if step>=burn_in and (step-burn_in)%thinning==0:
            transformations.append(current_transform.copy())
            # Importance reweighting restores the original Haar integral.
            raw_weights.append(current_character/current_score)
            # Track the complex overlap cancellation omitted by the positive target.
            overlap_reweights.append(current_overlap/current_score)

    # Divide by M; the unknown normalization of q is an irrelevant global factor.
    weights=np.asarray(raw_weights,dtype=complex)/samples
    # A magnitude-only importance ESS diagnoses uneven reweighting, while the
    # complex average phase separately exposes cancellations/sign problems.
    importance=np.abs(weights)
    importance_ess=float(
        importance.sum()**2/np.square(importance).sum()
    )
    overlap_reweights=np.asarray(overlap_reweights,dtype=complex)
    average_overlap_phase=float(
        abs(overlap_reweights.sum())/np.abs(overlap_reweights).sum()
    )
    diagnostics={
        'samples':int(samples),
        'burn_in':int(burn_in),
        'thinning':int(thinning),
        'proposal_scale':float(proposal_scale),
        'acceptance_rate':float(accepted/total_steps),
        'importance_ess':importance_ess,
        'average_overlap_phase':average_overlap_phase,
        'overlap_floor':float(overlap_floor),
        'seed':int(seed),
    }
    # Return the sampled group actions, reweighted coefficients, and diagnostics.
    return np.asarray(transformations),weights,diagnostics


class GaugeProjectedEnergy:
    """Exact full-period N,Z Fourier projection for finite mode spaces.

    Default L_n=m_n+1, L_p=m_p+1 resolves all particle-number sectors, even
    with np mixing and odd species parities. Cost per energy evaluation is
    O(L_n L_p (m^4+m^3)); dense interaction storage is O(m^4). Fixed undersized
    grids require ``allow_inexact_grid=True`` and then emit a warning. This says
    nothing about global optimization complexity.

    This class evaluates projected *kernels* without constructing a many-body
    state vector. That is what keeps its memory polynomial in the number of
    modes. Use ``project_particle_numbers`` with an assembled
    ``FermiHubbardHamiltonian`` when an explicit small-space vector is required
    for fidelity or other observables.
    """
    def __init__(
        self,
        hamiltonian,
        neutron_modes,
        targets,
        grid=None,
        offset=.137,
        *,
        allow_inexact_grid=False,
    ):
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
        self.minimum_grid=default_grid
        # Positive undersized grids are permitted only through an explicit opt-in.
        self.grid=default_grid if grid is None else tuple(grid)
        invalid_grid=len(self.grid)!=2 or any(
            not isinstance(points,(int,np.integer)) or points<1
            for points in self.grid
        )
        if invalid_grid:
            raise ValueError('Number grid must contain two positive integers')
        self.number_grid_guaranteed_exact=all(
            points>=minimum
            for points,minimum in zip(self.grid,self.minimum_grid)
        )
        if not self.number_grid_guaranteed_exact:
            message=(
                f'Number grid {self.grid} is below the finite-space exactness '
                f'bound {self.minimum_grid}; exact P_N P_Z symmetry restoration '
                'is not guaranteed. Use the bound or a larger grid for a '
                'guaranteed projector.'
            )
            if not allow_inexact_grid:
                raise ValueError(
                    message+' Set allow_inexact_grid=True to proceed.'
                )
            warnings.warn(message,ProjectionGridWarning,stacklevel=2)
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
            number_grid_guaranteed_exact=self.number_grid_guaranteed_exact,
            minimum_number_grid=self.minimum_grid,
        )

    def series_energy(self,series):
        """Evaluate number-projected energy from a stored gauge-vacuum series."""
        # Require the same structured series later consumed by fidelity expansion.
        if not isinstance(series,BogoliubovVacuumSeries):
            raise TypeError('series must be a BogoliubovVacuumSeries')
        # Deterministic series must reproduce this evaluator's exact grid.
        if series.sampling_method=='quadrature':
            if series.number_grid!=tuple(self.grid) or series.euler_grid is not None:
                raise ValueError('Series grid differs from number projector grid')
            # Equal dimensions are insufficient if Fourier nodes differ.
            if not np.isclose(series.number_offset,self.offset):
                raise ValueError('Series number-grid offset differs from evaluator')
        else:
            raise ValueError('Number-only Metropolis projection is unsupported')
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
        # A finite stochastic series has ordinary Monte Carlo imaginary noise;
        # record it instead of applying deterministic cancellation tolerances.
        if series.sampling_method=='metropolis':
            if abs(norm)<1e-14:
                raise ValueError('Metropolis projected norm is unresolved')
            energy=numerator/denominator
            series.sampling_diagnostics['norm_kernel_real']=float(norm.real)
            series.sampling_diagnostics['norm_kernel_imag']=float(norm.imag)
            series.sampling_diagnostics['energy_imaginary']=float(energy.imag)
            return float(energy.real)
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
