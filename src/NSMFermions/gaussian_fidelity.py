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
    from .hfb import HFBState
    from .number_projection import fermionic_basis_data
else:
    from hfb import HFBState
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


class GaussianFidelityObjective:
    """Maximize raw |<target|Omega(Z)>|^2 over unrestricted complex Z.

    ``hamiltonian`` supplies the target's fixed-N,Z determinant ordering. No
    energy minimization or projection is included in the objective: the raw
    overlap includes the probability that the intrinsic Gaussian occupies the
    target sector.
    """
    def __init__(self, hamiltonian, target):
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
        self.modes=hamiltonian.modes
        self.target=target/norm

        # Every unordered mode pair contributes one complex Z entry, stored as
        # consecutive real and imaginary blocks in the optimizer vector.
        self.ij=np.triu_indices(self.modes,1)
        self.pairs=list(zip(*self.ij))

    def unpack(self,x):
        """Convert real optimizer coordinates into antisymmetric complex Z."""
        pair_count=len(self.pairs)
        x=np.asarray(x,float)
        if x.shape!=(2*pair_count,) or not np.isfinite(x).all():
            raise ValueError('Invalid Gaussian Thouless parameters')

        # Fill only i<j, then impose Z^T=-Z exactly.
        z=np.zeros((self.modes,self.modes),complex)
        z[self.ij]=x[:pair_count]+1j*x[pair_count:]
        return z-z.T

    def fidelity(self,x):
        """Return raw fidelity between the intrinsic Gaussian and target."""
        # HFBState owns construction, normalization, Pfaffian amplitudes, and
        # the sector overlap; the objective only maps optimizer coordinates.
        state=HFBState.from_thouless(self.unpack(x))
        return state.fixed_sector_fidelity(self.target,self.occupations)

    def minimize(self,x):
        """Negate fidelity for SciPy's minimization interface."""
        return -self.fidelity(x)


def maximize_gaussian_fidelity(hamiltonian,target,*,starts=8,seed=0,maxiter=1000,
                               tolerance=1e-13,gradient_tolerance=1e-6,
                               initial_parameters=None,parameter_bound=8.):
    """Multi-start local optimization of the unrestricted Gaussian fidelity."""
    if (starts<1 or maxiter<1 or tolerance<=0 or gradient_tolerance<=0
            or parameter_bound<=0):
        raise ValueError('Positive optimizer settings required')
    objective=GaussianFidelityObjective(hamiltonian,target)
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
            x=rng.normal(scale=scale,size=2*len(objective.pairs))

        # SciPy finite-differences this scalar fidelity objective. Keeping the
        # optimizer separate from projection avoids reintroducing VAP logic.
        fit=minimize(objective.minimize,x,jac=False,method='L-BFGS-B',
                     bounds=[(-parameter_bound,parameter_bound)]*len(x),
                     options={'maxiter':maxiter,'ftol':tolerance,
                              'gtol':gradient_tolerance/10,'maxls':50})
        fidelity=objective.fidelity(fit.x)
        # L-BFGS-B reports its numerical objective gradient at the returned point.
        residual=float(np.linalg.norm(fit.jac))
        ok=bool(fit.success and residual<=gradient_tolerance)
        attempts.append({'fidelity':fidelity,'converged':ok,
                         'gradient_norm':residual,'iterations':int(fit.nit),
                         'message':str(fit.message)})
        candidates.append((fidelity,ok,residual,fit.x))
    # Report the largest-fidelity point found, even if no attempt met the strict
    # convergence threshold; ``converged`` communicates that distinction.
    fidelity,ok,residual,x=max(candidates,key=lambda item:item[0])
    z=objective.unpack(x)
    return GaussianFidelityResult(fidelity,x,z,HFBState.from_thouless(z),ok,
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
