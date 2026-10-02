"""Best-overlap pure fermionic Gaussian state for a fixed-sector target.

The optimizer differentiates normalized Pfaffian amplitudes analytically and
uses local canonical Thouless heavy-ball updates. It returns the best stationary
point found, not a certificate of the global maximum. Singular boundary states
such as exact non-vacuum Slater determinants are approached as limits.
"""
from dataclasses import dataclass
import numpy as np

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


def _batch_pfaffian(matrices):
    """Evaluate small antisymmetric Pfaffians with vectorized recursion."""
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
    result=np.zeros(len(matrices),complex)
    for column in range(1,size):
        retained=[index for index in range(1,size) if index!=column]
        minor=matrices[:,retained,:][:,:,retained]
        result+=((-1)**(column+1))*matrices[:,0,column]*_batch_pfaffian(minor)
    return result


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

    def local_value_gradient(self,state):
        """Return fidelity and its analytic local quasiparticle gradient.

        An infinitesimal canonical update obeys ``dU=V* dZ_qp`` and
        ``dV=U* dZ_qp``. These variations are mapped to the particle-vacuum
        Thouless matrix, where every fixed-sector coefficient is a normalized
        Pfaffian. Differentiating those Pfaffians produces the overlap analogue
        of the HFB ``H20`` field without finite differences over all coordinates.

        This combines the Thouless representation and Pfaffian amplitudes with
        the local heavy-ball strategy of B. Bally et al., Eur. Phys. J. A 57,
        69 (2021), doi:10.1140/epja/s10050-021-00369-z.
        """
        z=state.thouless_matrix
        z=.5*(z-z.T)
        modes=self.modes
        pair_rows,pair_columns=self.ij
        pair_count=len(pair_rows)
        parameter_count=(pair_count if self.real_parameters else 2*pair_count)

        dz_qp=np.zeros((parameter_count,modes,modes),complex)
        directions=np.arange(pair_count)
        dz_qp[directions,pair_rows,pair_columns]=1.
        dz_qp[directions,pair_columns,pair_rows]=-1.
        if not self.real_parameters:
            dz_qp[pair_count+directions,pair_rows,pair_columns]=1j
            dz_qp[pair_count+directions,pair_columns,pair_rows]=-1j

        # Z=V* (U*)^-1 and local dU=V* dZ_qp, dV=U* dZ_qp.
        inverse_u_conjugate=np.linalg.inv(state.U.conj())
        left=state.U-z@state.V
        dz_global=np.matmul(
            np.matmul(left[None,:,:],dz_qp.conj()),
            inverse_u_conjugate[None,:,:],
        )
        dz_global=.5*(dz_global-dz_global.transpose(0,2,1))

        metric=np.eye(modes)+z.conj().T@z
        inverse_metric=np.linalg.inv(metric)
        normalization=np.exp(-.25*np.linalg.slogdet(metric)[1])
        dmetric=(
            np.matmul(dz_global.conj().transpose(0,2,1),z)
            +np.matmul(z.conj().T[None,:,:],dz_global)
        )
        dlog_norm=-.25*np.einsum(
            'ij,pji->p',inverse_metric,dmetric,optimize=True
        ).real

        occupations=self.occupation_array
        particles=occupations.shape[1]
        coefficients=self.target.conj()
        if particles:
            submatrices=z[
                occupations[:,:,None],occupations[:,None,:]
            ]
            pfaffians=_batch_pfaffian(submatrices)
            overlap_polynomial=coefficients@pfaffians
            polynomial_derivative=np.zeros((modes,modes),complex)
            for row in range(particles-1):
                for column in range(row+1,particles):
                    retained=[index for index in range(particles)
                              if index not in (row,column)]
                    cofactors=((-1)**(row+column+1))*_batch_pfaffian(
                        submatrices[:,retained,:][:,:,retained]
                    )
                    np.add.at(
                        polynomial_derivative,
                        (occupations[:,row],occupations[:,column]),
                        coefficients*cofactors,
                    )
            derivative_polynomial=np.einsum(
                'ij,pij->p',polynomial_derivative,dz_global,optimize=True
            )
        else:
            overlap_polynomial=coefficients.sum()
            derivative_polynomial=np.zeros(parameter_count,complex)

        amplitude=normalization*overlap_polynomial
        derivative_amplitude=normalization*(
            derivative_polynomial+overlap_polynomial*dlog_norm
        )
        fidelity=float(abs(amplitude)**2)
        gradient=2*np.real(amplitude.conjugate()*derivative_amplitude)
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
                message='Local overlap gradient converged'
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
        ok=bool(residual<=gradient_tolerance)
        parameters=objective.pack(state.thouless_matrix)
        attempts.append({'fidelity':fidelity,'converged':ok,
                         'gradient_norm':residual,'iterations':iteration+1,
                         'message':message,
                         'solver':'analytic local-overlap heavy ball',
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
    derivative=np.zeros_like(orbitals)
    cofactors=np.empty_like(sub)
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
    for row in range(n):
        for column in range(n):
            # Accumulate each submatrix cofactor back into the corresponding
            # entry of the full orbital matrix.
            np.add.at(derivative[:,column],occupations[:,row],
                      coefficients*cofactors[:,row,column])
    value=float(abs(amplitude)**2)
    gradient=2*amplitude*derivative.conj()
    return value,gradient


def maximize_slater_fidelity(hamiltonian,target,*,starts=8,seed=0,maxiter=2000,
                             gradient_tolerance=1e-7,initial_orbitals=None):
    """Optimize the number-conserving Slater boundary of Gaussian states.

    Uses Riemannian steepest ascent with QR retraction on the complex Stiefel
    manifold. Orbitals may mix neutron and proton modes; only total particle
    number equals the target sector's total number.
    """
    # Reuse the exact basis owned by FermiHubbardHamiltonian.
    occupations,_,_=fermionic_basis_data(hamiltonian)
    target=np.asarray(target,complex)
    if target.shape!=(len(occupations),) or np.linalg.norm(target)<1e-14:
        raise ValueError('Target must match the fixed-sector basis')
    target=target/np.linalg.norm(target)
    particles=len(occupations[0]); modes=hamiltonian.modes
    rng=np.random.default_rng(seed)
    seeds=[]
    if initial_orbitals is not None:
        c=np.asarray(initial_orbitals,complex)
        if c.shape!=(modes,particles):
            raise ValueError('Initial orbitals have wrong shape')
        seeds.append(np.linalg.qr(c)[0])
    # Include the determinant with largest target coefficient as a deterministic
    # physically meaningful seed, then fill remaining starts randomly.
    dominant=occupations[int(np.argmax(abs(target)))]
    c=np.zeros((modes,particles),complex); c[list(dominant),range(particles)]=1
    seeds.append(c)
    while len(seeds)<starts:
        seeds.append(np.linalg.qr(rng.normal(size=(modes,particles))+
                                  1j*rng.normal(size=(modes,particles)))[0])
    attempts=[]; candidates=[]
    for c in seeds[:starts]:
        value=0.; residual=np.inf
        for iteration in range(maxiter):
            value,g=_slater_value_gradient(c,occupations,target)

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
                trial_value,_=_slater_value_gradient(trial,occupations,target)
                if trial_value>=value+1e-4*step*residual**2:
                    c=trial; accepted=True; break
                step*=.5
            if not accepted:
                break
        value,g=_slater_value_gradient(c,occupations,target)
        ctg=c.conj().T@g
        residual=float(np.linalg.norm(g-c@((ctg+ctg.conj().T)/2)))
        ok=bool(residual<=gradient_tolerance)
        attempts.append({'fidelity':value,'converged':ok,
                         'gradient_norm':residual,'iterations':iteration+1})
        candidates.append((value,ok,residual,c.copy()))
    value,ok,residual,c=max(candidates,key=lambda item:item[0])
    return SlaterFidelityResult(value,c,ok,residual,attempts)
