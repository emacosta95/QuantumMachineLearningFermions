"""Best-overlap pure fermionic Gaussian state for a fixed-sector target.

This is a non-convex optimizer. It returns the best stationary point found,
not a certificate of the global maximum. The Thouless chart contains all even
Gaussian vacua with nonzero particle-vacuum overlap; singular boundary states
such as exact non-vacuum Slater determinants are approached as limits.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

if __package__:
    from .number_projection import state_from_thouless
else:
    from number_projection import state_from_thouless


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


class GaussianFidelityObjective:
    """Maximize raw |<target|Omega(Z)>|^2 over unrestricted complex Z.

    `space` supplies the target's fixed-N,Z occupation basis and Pfaffian
    amplitudes. No number constraint, species block, energy, or projection is
    included in the objective. The determinant normalizes the complete Gaussian
    vacuum across every even particle-number sector.
    """
    def __init__(self, space, target):
        # The target coefficients must use exactly the determinant ordering of
        # NumberProjectedSpace.  Normalize here so the objective is a fidelity.
        target=np.asarray(target,complex)
        if target.shape!=(len(space.occupations),) or not np.isfinite(target).all():
            raise ValueError('Target must match the projected-space basis')
        norm=np.linalg.norm(target)
        if norm<1e-14:
            raise ValueError('Target cannot vanish')
        self.space=space
        self.target=target/norm
        # Row/column arrays selecting the independent upper-triangular entries
        # of Z; reused in the analytic normalization derivative below.
        self.ij=np.array(space.pairs).T

    def fidelity_and_gradient(self,x):
        """Return raw Gaussian fidelity and its real-coordinate gradient."""
        # Convert real optimizer coordinates to Z and obtain the target-sector
        # Pfaffian coefficients plus their analytic derivatives.
        z=self.space.unpack(x)
        amplitudes,derivative=self.space.amplitudes_and_jacobian(x)

        # Only the target's fixed-N,Z sector contributes to <target|Phi(Z)>.
        # ``amplitudes`` are still coefficients of the unnormalized exponential.
        overlap=np.vdot(self.target,amplitudes)

        # The squared norm of the complete Thouless exponential, including all
        # even particle-number sectors, is sqrt(det(I+Z^dagger Z)).
        b=np.eye(self.space.modes)+z.conj().T@z
        # Remove roundoff-level anti-Hermitian noise before Cholesky factorization.
        b=(b+b.conj().T)/2
        try:
            cholesky=np.linalg.cholesky(b)
        except np.linalg.LinAlgError as error:
            raise ValueError('Invalid Gaussian normalization matrix') from error
        logdet=2*np.log(np.diag(cholesky).real).sum()
        normalization=np.exp(.5*logdet)
        fidelity=float(abs(overlap)**2/normalization)

        # At exactly zero overlap the fidelity derivative also vanishes; avoid
        # dividing by overlap in the logarithmic derivative formulas.
        if abs(overlap)<1e-15:
            return fidelity,np.zeros_like(x)

        # Derivative of log|<target|Phi>|^2 with respect to Re(Z) and Im(Z).
        da=self.target.conj()@derivative
        log_overlap_x=2*np.real(da/overlap)
        log_overlap_y=2*np.real(1j*da/overlap)
        # Derivative of log sqrt(det B).  Antisymmetrizing the two selected
        # entries accounts for Z_ji=-Z_ij when varying one independent pair.
        c=np.linalg.solve(b,z.conj().T)
        antisymmetric_trace=c[self.ij[1],self.ij[0]]-c[self.ij[0],self.ij[1]]
        log_norm_x=np.real(antisymmetric_trace)
        log_norm_y=-np.imag(antisymmetric_trace)
        # dF = F [d log|overlap|^2 - d log(normalization)].
        gradient=fidelity*np.r_[log_overlap_x-log_norm_x,
                                log_overlap_y-log_norm_y]
        return fidelity,gradient

    def minimize(self,x):
        """Negate value and gradient for SciPy's minimization interface."""
        fidelity,gradient=self.fidelity_and_gradient(x)
        return -fidelity,-gradient


def maximize_gaussian_fidelity(space,target,*,starts=8,seed=0,maxiter=1000,
                               tolerance=1e-13,gradient_tolerance=1e-6,
                               initial_parameters=None,parameter_bound=8.):
    """Multi-start local optimization of the unrestricted Gaussian fidelity."""
    if (starts<1 or maxiter<1 or tolerance<=0 or gradient_tolerance<=0
            or parameter_bound<=0):
        raise ValueError('Positive optimizer settings required')
    objective=GaussianFidelityObjective(space,target)
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
            x=rng.normal(scale=scale,size=2*len(space.pairs))
        fit=minimize(objective.minimize,x,jac=True,method='L-BFGS-B',
                     bounds=[(-parameter_bound,parameter_bound)]*len(x),
                     options={'maxiter':maxiter,'ftol':tolerance,
                              'gtol':gradient_tolerance/10,'maxls':50})
        fidelity,gradient=objective.fidelity_and_gradient(fit.x)
        # Re-evaluate the analytic gradient at SciPy's returned point instead of
        # relying only on its success flag.
        residual=float(np.linalg.norm(gradient))
        ok=bool(fit.success and residual<=gradient_tolerance)
        attempts.append({'fidelity':fidelity,'converged':ok,
                         'gradient_norm':residual,'iterations':int(fit.nit),
                         'message':str(fit.message)})
        candidates.append((fidelity,ok,residual,fit.x))
    # Report the largest-fidelity point found, even if no attempt met the strict
    # convergence threshold; ``converged`` communicates that distinction.
    fidelity,ok,residual,x=max(candidates,key=lambda item:item[0])
    z=space.unpack(x)
    return GaussianFidelityResult(fidelity,x,z,state_from_thouless(z),ok,
                                  residual,attempts)


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


def maximize_slater_fidelity(space,target,*,starts=8,seed=0,maxiter=2000,
                             gradient_tolerance=1e-7,initial_orbitals=None):
    """Optimize the number-conserving Slater boundary of Gaussian states.

    Uses Riemannian steepest ascent with QR retraction on the complex Stiefel
    manifold. Orbitals may mix neutron and proton modes; only total particle
    number equals the target sector's total number.
    """
    target=np.asarray(target,complex)
    if target.shape!=(len(space.occupations),) or np.linalg.norm(target)<1e-14:
        raise ValueError('Target must match the fixed-sector basis')
    target=target/np.linalg.norm(target)
    particles=sum(space.targets); modes=space.modes
    rng=np.random.default_rng(seed)
    seeds=[]
    if initial_orbitals is not None:
        c=np.asarray(initial_orbitals,complex)
        if c.shape!=(modes,particles):
            raise ValueError('Initial orbitals have wrong shape')
        seeds.append(np.linalg.qr(c)[0])
    # Include the determinant with largest target coefficient as a deterministic
    # physically meaningful seed, then fill remaining starts randomly.
    dominant=space.occupations[int(np.argmax(abs(target)))]
    c=np.zeros((modes,particles),complex); c[list(dominant),range(particles)]=1
    seeds.append(c)
    while len(seeds)<starts:
        seeds.append(np.linalg.qr(rng.normal(size=(modes,particles))+
                                  1j*rng.normal(size=(modes,particles)))[0])
    attempts=[]; candidates=[]
    for c in seeds[:starts]:
        value=0.; residual=np.inf
        for iteration in range(maxiter):
            value,g=_slater_value_gradient(c,space.occupations,target)

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
                trial_value,_=_slater_value_gradient(trial,space.occupations,target)
                if trial_value>=value+1e-4*step*residual**2:
                    c=trial; accepted=True; break
                step*=.5
            if not accepted:
                break
        value,g=_slater_value_gradient(c,space.occupations,target)
        ctg=c.conj().T@g
        residual=float(np.linalg.norm(g-c@((ctg+ctg.conj().T)/2)))
        ok=bool(residual<=gradient_tolerance)
        attempts.append({'fidelity':value,'converged':ok,
                         'gradient_norm':residual,'iterations':iteration+1})
        candidates.append((value,ok,residual,c.copy()))
    value,ok,residual,c=max(candidates,key=lambda item:item[0])
    return SlaterFidelityResult(value,c,ok,residual,attempts)
