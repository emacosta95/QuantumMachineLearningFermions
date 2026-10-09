"""Best-overlap pure fermionic Gaussian state for a fixed-sector target.

The optimizer differentiates normalized Pfaffian amplitudes analytically and
uses local canonical Thouless heavy-ball updates. It returns the best stationary
point found, not a certificate of the global maximum. Singular boundary states
such as exact non-vacuum Slater determinants are approached as limits.
"""
from dataclasses import dataclass
import numpy as np
from pfapack import pfaffian as pf

if __package__:
    from .hfb import HFBState, apply_thouless_step
    from .number_projection import fermionic_basis_data
else:
    from hfb import HFBState, apply_thouless_step
    from number_projection import fermionic_basis_data


@dataclass
class GaussianFidelityResult:
    """Best interior Gaussian state and multi-start diagnostics."""

    fidelity: float
    parameters: np.ndarray
    thouless_matrix: np.ndarray
    state: object
    converged: bool
    gradient_norm: float
    attempts: list


@dataclass
class SlaterFidelityResult:
    """Best number-conserving Slater boundary state and diagnostics."""

    fidelity: float
    orbitals: np.ndarray
    converged: bool
    gradient_norm: float
    attempts: list


@dataclass
class BestGaussianFidelityResult:
    """Best result from the Bogoliubov interior and Slater boundary."""

    fidelity: float
    state: object
    family: str
    parameters: np.ndarray
    converged: bool
    gradient_norm: float
    attempts: list
    bogoliubov: object
    hartree_fock: object


def _batch_pfaffian(matrices):
    """Evaluate a batch of antisymmetric Pfaffians with a hybrid kernel.

    Vectorized expansion is faster than calling PFAPACK separately for the
    small occupation minors common near the bottom of a shell.  Its
    double-factorial operation count becomes prohibitive for larger particle
    numbers, so matrices of order ten and above use PFAPACK's cubic
    Parlett--Reid implementation instead.
    """
    matrices=np.asarray(matrices,complex)
    if matrices.ndim!=3 or matrices.shape[1]!=matrices.shape[2]:
        raise ValueError('Expected a batch of square matrices')
    size=matrices.shape[1]
    if size%2:
        return np.zeros(len(matrices),complex)
    if size==0:
        return np.ones(len(matrices),complex)
    if size==2:
        return matrices[:,0,1]
    if size>=10:
        return np.asarray([
            pf.pfaffian(matrix,overwrite_a=False,method='P')
            for matrix in matrices
        ],complex)
    result=np.zeros(len(matrices),complex)
    for column in range(1,size):
        retained=[index for index in range(1,size) if index!=column]
        minor=matrices[:,retained,:][:,:,retained]
        result+=((-1)**(column+1))*matrices[:,0,column]*_batch_pfaffian(minor)
    return result


def _batch_pfaffian_cofactors(matrices):
    """Return Pfaffians and every independent-entry derivative in one pass.

    Cofactors are ordered like ``np.triu_indices(size, 1)``.  For a regular
    antisymmetric matrix ``A`` they obey

    ``d Pf(A) / d A[i,j] = Pf(A) * inv(A)[j,i]``.

    This obtains all derivatives from one factorization instead of evaluating
    one Pfaffian minor per occupied pair.  Near a singular matrix the inverse
    identity is ill-conditioned, so those batch elements fall back to direct
    minor Pfaffians and retain the exact derivative at Pfaffian zeros.
    """
    matrices=np.asarray(matrices,complex)
    if matrices.ndim!=3 or matrices.shape[1]!=matrices.shape[2]:
        raise ValueError('Expected a batch of square matrices')
    size=matrices.shape[1]
    if size%2:
        return (np.zeros(len(matrices),complex),
                np.zeros((len(matrices),size*(size-1)//2),complex))

    values=_batch_pfaffian(matrices)
    rows,columns=np.triu_indices(size,1)
    cofactors=np.empty((len(matrices),len(rows)),complex)
    if not len(rows) or not len(matrices):
        return values,cofactors

    # For very small matrices, one vectorized minor expansion remains faster
    # than a Python loop through PFAPACK followed by a batched inverse.
    if size<=6:
        for pair_index,(row,column) in enumerate(zip(rows,columns)):
            retained=[index for index in range(size)
                      if index not in (row,column)]
            cofactors[:,pair_index]=(
                (-1)**(row+column+1)
                *_batch_pfaffian(
                    matrices[:,retained,:][:,:,retained]
                )
            )
        return values,cofactors

    fallback=np.ones(len(matrices),bool)
    maximum=np.max(np.abs(matrices),axis=(1,2))
    with np.errstate(divide='ignore',invalid='ignore'):
        relative_log=(
            np.log(np.abs(values))-(size/2)*np.log(maximum)
        )
    regular=np.isfinite(relative_log) & (relative_log>np.log(1e-9))

    if np.any(regular):
        regular_indices=np.flatnonzero(regular)
        try:
            inverses=np.linalg.inv(matrices[regular_indices])
            identity=np.eye(size)
            residuals=np.max(
                np.abs(matrices[regular_indices]@inverses-identity),
                axis=(1,2),
            )
            valid=(
                np.isfinite(inverses).all(axis=(1,2))
                & np.isfinite(residuals)
                & (residuals<1e-7)
            )
            valid_indices=regular_indices[valid]
            cofactors[valid_indices]=(
                values[valid_indices,None]
                *inverses[valid][:,columns,rows]
            )
            fallback[valid_indices]=False
        except np.linalg.LinAlgError:
            # Direct minors below remain valid for exactly singular matrices.
            pass

    if np.any(fallback):
        singular_matrices=matrices[fallback]
        for pair_index,(row,column) in enumerate(zip(rows,columns)):
            retained=[index for index in range(size)
                      if index not in (row,column)]
            cofactors[fallback,pair_index]=(
                (-1)**(row+column+1)
                *_batch_pfaffian(
                    singular_matrices[:,retained,:][:,:,retained]
                )
            )
    return values,cofactors


class GaussianFidelityObjective:
    """Maximize raw |<target|Omega(Z)>|^2 over unrestricted complex Z.

    ``hamiltonian`` supplies the target's fixed-N,Z determinant ordering. No
    energy minimization or projection is included in the objective: the raw
    overlap includes the probability that the intrinsic Gaussian occupies the
    target sector.
    """
    def __init__(self, hamiltonian, target, *, real_parameters=False):
        # Obtain the canonical determinant ordering directly from the assembled
        # FermiHubbardHamiltonian; no projection-specific space is rebuilt.
        occupations, _, _ = fermionic_basis_data(hamiltonian)

        # The target coefficients must use exactly that determinant ordering.
        # Normalize here so every subsequent overlap is a fidelity.
        target=np.asarray(target,complex)
        if target.shape!=(len(occupations),) or not np.isfinite(target).all():
            raise ValueError('Target must match the fermionic Hamiltonian basis')
        norm=np.linalg.norm(target)
        if norm<1e-14:
            raise ValueError('Target cannot vanish')
        self.hamiltonian=hamiltonian
        self.occupations=occupations
        self.occupation_array=np.asarray(occupations,int)
        self.modes=hamiltonian.modes
        self.target=target/norm
        self.real_parameters=bool(real_parameters)

        # Every unordered mode pair contributes one complex Z entry, stored as
        # consecutive real and imaginary blocks in the optimizer vector.
        self.ij=np.triu_indices(self.modes,1)
        self.pairs=list(zip(*self.ij))
        particles=self.occupation_array.shape[1]
        self.occupation_pair_rows,self.occupation_pair_columns=(
            np.triu_indices(particles,1)
        )

    def unpack(self,x):
        """Convert real optimizer coordinates into antisymmetric complex Z."""
        pair_count=len(self.pairs)
        x=np.asarray(x,float)
        expected=pair_count if self.real_parameters else 2*pair_count
        if x.shape!=(expected,) or not np.isfinite(x).all():
            raise ValueError('Invalid Gaussian Thouless parameters')

        # Fill only i<j, then impose Z^T=-Z exactly.
        z=np.zeros((self.modes,self.modes),complex)
        z[self.ij]=(x if self.real_parameters else
                    x[:pair_count]+1j*x[pair_count:])
        return z-z.T

    def pack(self,z):
        """Convert an antisymmetric Thouless matrix to real coordinates."""
        upper=np.asarray(z,complex)[self.ij]
        return (upper.real.copy() if self.real_parameters else
                np.concatenate((upper.real,upper.imag)))

    def fidelity(self,x):
        """Return raw fidelity between the intrinsic Gaussian and target."""
        # HFBState owns construction, normalization, Pfaffian amplitudes, and
        # the sector overlap; the objective only maps optimizer coordinates.
        state=HFBState.from_thouless(self.unpack(x))
        return state.fixed_sector_fidelity(self.target,self.occupations)

    def fidelity_state(self,state):
        """Return raw target fidelity for an already constructed vacuum."""
        z=state.thouless_matrix
        z=.5*(z-z.T)
        metric=np.eye(self.modes)+z.conj().T@z
        normalization=np.exp(-.25*np.linalg.slogdet(metric)[1])
        particles=self.occupation_array.shape[1]
        if particles:
            submatrices=z[
                self.occupation_array[:,:,None],
                self.occupation_array[:,None,:],
            ]
            values=_batch_pfaffian(submatrices)
        else:
            values=np.ones(len(self.target),complex)
        polynomial=self.target.conj()@values
        return float(abs(normalization*polynomial)**2)

    def minimize(self,x):
        """Negate fidelity for SciPy's minimization interface."""
        return -self.fidelity(x)

    def local_value_field(self,state):
        """Return fidelity and its compact analytic local ``F20`` field.

        The antisymmetric field satisfies
        ``dF = 2 Re sum_{a<m} F20[a,m] dZ_qp[a,m]*``.  All Pfaffian
        derivatives are accumulated once and reverse-contracted into the
        current quasiparticle frame, just as energy HFB constructs ``H20``.
        """
        z=state.thouless_matrix
        z=.5*(z-z.T)
        modes=self.modes
        metric=np.eye(modes)+z.conj().T@z
        inverse_metric=np.linalg.inv(metric)
        normalization=np.exp(-.25*np.linalg.slogdet(metric)[1])

        occupations=self.occupation_array
        particles=occupations.shape[1]
        coefficients=self.target.conj()
        polynomial_derivative=np.zeros((modes,modes),complex)
        if particles:
            submatrices=z[
                occupations[:,:,None],occupations[:,None,:]
            ]
            pfaffians,cofactors=_batch_pfaffian_cofactors(submatrices)
            overlap_polynomial=coefficients@pfaffians
            if cofactors.shape[1]:
                source_rows=occupations[
                    :,self.occupation_pair_rows
                ].reshape(-1)
                source_columns=occupations[
                    :,self.occupation_pair_columns
                ].reshape(-1)
                contributions=(
                    coefficients[:,None]*cofactors
                ).reshape(-1)
                np.add.at(
                    polynomial_derivative,
                    (source_rows,source_columns),
                    contributions,
                )
        else:
            overlap_polynomial=coefficients.sum()

        amplitude=normalization*overlap_polynomial
        fidelity=float(abs(amplitude)**2)

        # For a global particle-chart variation dZ,
        # dF = 2 Re sum_ij global_field_ij dZ_ij.  The second term is the
        # derivative of the exact Gaussian normalization.
        normalization_field=z.conj()@inverse_metric.T
        global_field=(
            normalization**2*overlap_polynomial.conj()*polynomial_derivative
            -.5*fidelity*normalization_field
        )

        # dZ=(U-ZV) dZ_qp* (U*)^-1.  Reverse-contract this relation once and
        # antisymmetrize to obtain all independent quasiparticle-pair entries.
        left=state.U-z@state.V
        transformed=left.T@global_field
        local_field=np.linalg.solve(state.U.conj(),transformed.T).T
        return fidelity,local_field-local_field.T

    def local_value_gradient(self,state):
        """Return fidelity and real optimizer coordinates of local ``F20``."""
        fidelity,field=self.local_value_field(state)
        upper=field[self.ij]
        gradient=(
            2*upper.real
            if self.real_parameters
            else np.concatenate((2*upper.real,2*upper.imag))
        )
        return fidelity,gradient


def maximize_gaussian_fidelity(hamiltonian,target,*,starts=8,seed=0,maxiter=1000,
                               tolerance=1e-13,gradient_tolerance=1e-6,
                               initial_parameters=None,parameter_bound=8.,
                               real_parameters=False,step_size=.1,momentum=.3):
    """Maximize raw Gaussian fidelity with local heavy-ball Thouless steps.

    The analytic overlap ``20`` field replaces the former finite-difference
    L-BFGS-B optimization. ``parameter_bound`` remains accepted for API
    compatibility but no bound is needed by the canonical local chart.

    References: D. J. Thouless, Ann. Phys. 10, 553 (1960); L. M. Robledo,
    Phys. Rev. C 79, 021302(R) (2009); B. Bally et al., Eur. Phys. J. A 57,
    69 (2021).
    """
    if (starts<1 or maxiter<1 or tolerance<=0 or gradient_tolerance<=0
            or parameter_bound<=0 or step_size<=0 or momentum<0 or momentum>=1):
        raise ValueError('Positive optimizer settings required')
    objective=GaussianFidelityObjective(
        hamiltonian,target,real_parameters=real_parameters
    )
    rng=np.random.default_rng(seed)
    attempts=[]; candidates=[]

    # Fidelity optimization is non-convex.  Alternate moderate seeds with
    # larger-norm seeds that probe nearly singular-U, Slater-like boundaries.
    for attempt in range(starts):
        if attempt==0 and initial_parameters is not None:
            x=np.asarray(initial_parameters,float)
        else:
            # Include both ordinary and large-norm starts to probe Slater-like
            # boundaries of the Thouless chart.
            scale=.35 if attempt%2==0 else 1.5
            coordinate_count=(len(objective.pairs) if real_parameters
                              else 2*len(objective.pairs))
            x=rng.normal(scale=scale,size=coordinate_count)

        state=HFBState.from_thouless(objective.unpack(x))
        velocity=np.zeros_like(x)
        eta=float(step_size)
        message='Iteration limit reached'
        for iteration in range(maxiter):
            fidelity,gradient=objective.local_value_gradient(state)
            residual=float(np.linalg.norm(gradient))
            if residual<=gradient_tolerance:
                message=(
                    'Local overlap gradient converged'
                    if fidelity>1e-14
                    else 'Zero-overlap stationary point'
                )
                break
            # Fidelity gradients are dimensionless and become small well before
            # their direction becomes uninformative. Use eta as a tangent-space
            # angular step, while backtracking determines its safe magnitude.
            proposed=(momentum*velocity+
                      eta*gradient/max(residual,1e-15))
            proposed_norm=np.linalg.norm(proposed)
            if proposed_norm>.25:
                proposed*=.25/proposed_norm
            directional_derivative=float(gradient@proposed)
            if directional_derivative<=0:
                proposed=eta*gradient/max(residual,1e-15)
                proposed_norm=np.linalg.norm(proposed)
                if proposed_norm>.25:
                    proposed*=.25/proposed_norm
                directional_derivative=float(gradient@proposed)
            accepted=False
            trial_step=proposed.copy()
            for _ in range(30):
                trial=apply_thouless_step(
                    state,trial_step,real_parameters=real_parameters
                )
                try:
                    trial_fidelity=objective.fidelity_state(trial)
                except (ValueError,np.linalg.LinAlgError):
                    trial_fidelity=-np.inf
                if trial_fidelity>=fidelity+1e-4*directional_derivative:
                    state=trial
                    velocity=trial_step
                    eta=min(.3,eta*1.1)
                    accepted=True
                    break
                trial_step*=.5
                directional_derivative*=.5
            if not accepted:
                velocity.fill(0.)
                eta*=.25
                if eta<1e-10:
                    message='Local overlap line search stalled'
                    break
        fidelity,gradient=objective.local_value_gradient(state)
        residual=float(np.linalg.norm(gradient))
        ok=bool(residual<=gradient_tolerance and fidelity>1e-14)
        parameters=objective.pack(state.thouless_matrix)
        attempts.append({'fidelity':fidelity,'converged':ok,
                         'gradient_norm':residual,'iterations':iteration+1,
                         'message':message,
                         'solver':'compact analytic F20 heavy ball',
                         'step_size':eta,'momentum':float(momentum)})
        candidates.append((fidelity,ok,residual,parameters,state))
    # Report the largest-fidelity point found, even if no attempt met the strict
    # convergence threshold; ``converged`` communicates that distinction.
    fidelity,ok,residual,x,state=max(candidates,key=lambda item:item[0])
    z=state.thouless_matrix
    z=.5*(z-z.T)
    state=HFBState.from_thouless(z)
    return GaussianFidelityResult(
        fidelity,x,z,state,ok,residual,attempts
    )


def _slater_value(orbitals,occupations,target):
    """Fidelity of an orthonormal-orbital determinant without its gradient."""
    sub=orbitals[np.asarray(occupations,int)]
    determinants=np.linalg.det(sub)
    amplitude=np.asarray(target,complex).conj()@determinants
    return float(abs(amplitude)**2)


def _slater_value_gradient(orbitals,occupations,target):
    """Fidelity and Euclidean gradient for an orthonormal-orbital determinant."""
    n=orbitals.shape[1]
    occupations=np.asarray(occupations,int)

    # A Slater determinant's coefficient on occupation I is det(C_I), where C
    # contains its occupied one-body orbitals as columns.
    sub=orbitals[occupations]
    determinants=np.linalg.det(sub)
    coefficients=target.conj()
    amplitude=coefficients@determinants
    derivative=np.zeros(orbitals.shape,complex)
    cofactors=np.empty(sub.shape,complex)
    # For nonsingular minors, det derivative = det(C) C^{-T}.  Near a singular
    # minor, form cofactors explicitly so the derivative remains defined.
    regular=abs(determinants)>1e-11
    if np.any(regular):
        cofactors[regular]=determinants[regular,None,None]*np.linalg.inv(sub[regular]).transpose(0,2,1)
    for index in np.nonzero(~regular)[0]:
        for row in range(n):
            for column in range(n):
                minor=np.delete(np.delete(sub[index],row,axis=0),column,axis=1)
                cofactors[index,row,column]=((-1)**(row+column))*np.linalg.det(minor)
    # Accumulate each submatrix cofactor back into the corresponding full
    # orbital column. Flattening the determinant and occupied-row axes reduces
    # n^2 separate indexed updates to only n updates.
    occupied_rows=occupations.reshape(-1)
    for column in range(n):
        np.add.at(
            derivative[:,column],
            occupied_rows,
            (coefficients[:,None]*cofactors[:,:,column]).reshape(-1),
        )
    value=float(abs(amplitude)**2)
    gradient=2*amplitude*derivative.conj()
    return value,gradient


def maximize_slater_fidelity(hamiltonian,target,*,starts=8,seed=0,maxiter=2000,
                             gradient_tolerance=1e-7,initial_orbitals=None,
                             real_parameters=False):
    """Optimize the number-conserving Slater boundary of Gaussian states.

    Uses Riemannian steepest ascent with QR retraction on a real or complex
    Stiefel manifold. Orbitals may mix neutron and proton modes; only total
    particle number equals the target sector's total number.
    """
    if starts<1 or maxiter<1 or gradient_tolerance<=0:
        raise ValueError('Positive Slater optimizer settings required')
    # Reuse the exact basis owned by FermiHubbardHamiltonian.
    occupations,_,_=fermionic_basis_data(hamiltonian)
    target=np.asarray(target,complex)
    if target.shape!=(len(occupations),) or np.linalg.norm(target)<1e-14:
        raise ValueError('Target must match the fixed-sector basis')
    target=target/np.linalg.norm(target)
    particles=len(occupations[0]); modes=hamiltonian.modes
    rng=np.random.default_rng(seed)
    seeds=[]
    # The determinant with largest target coefficient has rigorously nonzero
    # overlap. Put it first so even starts=1 cannot select the flat F=0 point
    # where the gradient of the squared overlap vanishes.
    dominant=occupations[int(np.argmax(abs(target)))]
    c=np.zeros((modes,particles),complex); c[list(dominant),range(particles)]=1
    seeds.append(c)
    if initial_orbitals is not None:
        c=np.asarray(initial_orbitals,complex)
        if c.shape!=(modes,particles):
            raise ValueError('Initial orbitals have wrong shape')
        if real_parameters and np.max(np.abs(c.imag),initial=0.)>1e-12:
            raise ValueError('Real Slater optimization needs real initial orbitals')
        if real_parameters:
            c=c.real
        seeds.append(np.linalg.qr(c)[0])
    while len(seeds)<starts:
        random_orbitals=rng.normal(size=(modes,particles))
        if not real_parameters:
            random_orbitals=(random_orbitals+
                             1j*rng.normal(size=(modes,particles)))
        seeds.append(np.linalg.qr(random_orbitals)[0])
    attempts=[]; candidates=[]
    for c in seeds[:starts]:
        value=0.; residual=np.inf
        for iteration in range(maxiter):
            value,g=_slater_value_gradient(c,occupations,target)
            if real_parameters:
                g=g.real

            # Project the Euclidean gradient onto the tangent space of the
            # complex Stiefel manifold C^dagger C=I.
            ctg=c.conj().T@g
            tangent=g-c@((ctg+ctg.conj().T)/2)
            residual=float(np.linalg.norm(tangent))
            if residual<=gradient_tolerance:
                break
            step=min(1.,1./max(residual,1e-12))
            accepted=False
            # Backtracking line search followed by QR retraction restores exact
            # orbital orthonormality after every trial step.
            for _ in range(30):
                trial=np.linalg.qr(c+step*tangent)[0]
                # Armijo backtracking needs only the objective value.  Avoid
                # rebuilding every determinant cofactor at each rejected step.
                trial_value=_slater_value(trial,occupations,target)
                if trial_value>=value+1e-4*step*residual**2:
                    c=trial; accepted=True; break
                step*=.5
            if not accepted:
                break
        value,g=_slater_value_gradient(c,occupations,target)
        if real_parameters:
            g=g.real
        ctg=c.conj().T@g
        residual=float(np.linalg.norm(g-c@((ctg+ctg.conj().T)/2)))
        ok=bool(residual<=gradient_tolerance and value>1e-14)
        attempts.append({'fidelity':value,'converged':ok,
                         'gradient_norm':residual,'iterations':iteration+1,
                         'message':('Converged' if ok else
                                    'Zero-overlap stationary point'
                                    if value<=1e-14 else
                                    'Iteration or line-search limit')})
        candidates.append((value,ok,residual,c.copy()))
    value,ok,residual,c=max(candidates,key=lambda item:item[0])
    return SlaterFidelityResult(value,c,ok,residual,attempts)


def maximize_best_gaussian_fidelity(
    hamiltonian,target,*,bogoliubov_starts=8,hartree_fock_starts=None,
    seed=0,bogoliubov_maxiter=1000,hartree_fock_maxiter=None,
    gradient_tolerance=1e-6,real_parameters=False,
    initial_bogoliubov_parameters=None,initial_hartree_fock_orbitals=None,
):
    """Optimize both Gaussian components and return the larger fidelity.

    Finite particle-vacuum Thouless charts do not contain nonempty Slater
    determinants because those states have singular ``U``.  A complete
    best-found pure-Gaussian search therefore needs both the paired
    Bogoliubov interior and the number-conserving Hartree-Fock boundary.
    """
    if hartree_fock_starts is None:
        hartree_fock_starts=bogoliubov_starts
    if hartree_fock_maxiter is None:
        hartree_fock_maxiter=bogoliubov_maxiter

    bogoliubov=maximize_gaussian_fidelity(
        hamiltonian,target,starts=bogoliubov_starts,seed=seed,
        maxiter=bogoliubov_maxiter,
        gradient_tolerance=gradient_tolerance,
        real_parameters=real_parameters,
        initial_parameters=initial_bogoliubov_parameters,
    )
    hartree_fock=maximize_slater_fidelity(
        hamiltonian,target,starts=hartree_fock_starts,seed=seed+1000003,
        maxiter=hartree_fock_maxiter,
        gradient_tolerance=gradient_tolerance,
        real_parameters=real_parameters,
        initial_orbitals=initial_hartree_fock_orbitals,
    )

    if hartree_fock.fidelity>bogoliubov.fidelity:
        return BestGaussianFidelityResult(
            fidelity=hartree_fock.fidelity,
            state=HFBState.from_slater(hartree_fock.orbitals),
            family='hartree_fock',
            parameters=hartree_fock.orbitals,
            converged=hartree_fock.converged,
            gradient_norm=hartree_fock.gradient_norm,
            attempts=hartree_fock.attempts,
            bogoliubov=bogoliubov,
            hartree_fock=hartree_fock,
        )
    return BestGaussianFidelityResult(
        fidelity=bogoliubov.fidelity,
        state=bogoliubov.state,
        family='bogoliubov',
        parameters=bogoliubov.parameters,
        converged=bogoliubov.converged,
        gradient_norm=bogoliubov.gradient_norm,
        attempts=bogoliubov.attempts,
        bogoliubov=bogoliubov,
        hartree_fock=hartree_fock,
    )
