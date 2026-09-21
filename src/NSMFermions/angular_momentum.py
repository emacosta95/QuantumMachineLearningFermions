"""J=0 angular-momentum projection for general Bogoliubov vacua.

The evaluator combines finite N,Z Fourier grids with an SO(3) Euler grid and
keeps their coherent result as a series of transformed Bogoliubov vacua. The
same series supplies transition-density energies and late determinant-basis
fidelities; no J^2 diagonalization is used as a projection method.

General J>0 projection is not implemented here: a triaxial intrinsic vacuum
requires the full P^J_MK matrix and K mixing. J=0 has only M=K=0 and is therefore
the clean first target for the Be8 0+ ground state.
"""
from dataclasses import dataclass
from math import ceil
import numpy as np
from scipy.linalg import expm

if __package__:
    from .gauge_projection import (
        GaugeProjectedEnergy,
        metropolis_vacuum_terms,
        pfaffian,
    )
    from .hfb import BogoliubovVacuumSeries, HFBState
    from .number_projection import projected_series_observables
else:
    from gauge_projection import GaugeProjectedEnergy, metropolis_vacuum_terms, pfaffian
    from hfb import BogoliubovVacuumSeries, HFBState
    from number_projection import projected_series_observables


def single_particle_angular_momentum(state_encoding):
    """Return Cartesian j generators in the repository's spherical basis.

    Each state is `(n,l,j,m,t,tz)`. Rotations preserve all labels except m.
    Ordering is taken exactly from `state_encoding`; no species blocks assumed.
    """
    # Freeze each encoded state as (n,l,j,m,t,tz) in the supplied mode order.
    states=[tuple(state) for state in state_encoding]
    # Map quantum-number tuples back to matrix indices so J_+ can connect
    # |j,m> to |j,m+1> without assuming any particular orbital ordering.
    lookup={state:index for index,state in enumerate(states)}
    # Allocate the raising operator J_+ in the one-particle basis.
    jp=np.zeros((len(states),len(states)),complex)
    # Visit every ket column and search for its m+1 partner.
    for index,(n,l,j,m,t,tz) in enumerate(states):
        # Spatial rotation changes m but preserves all other quantum numbers.
        raised=(n,l,j,m+1,t,tz)
        if raised in lookup:
            # Standard ladder-operator matrix element
            # sqrt[j(j+1)-m(m+1)].
            jp[lookup[raised],index]=np.sqrt(j*(j+1)-m*(m+1))

    # Recover the Hermitian Cartesian generators from J_+ and J_- = J_+^dagger.
    # J_x=(J_++J_-)/2.
    jx=(jp+jp.conj().T)/2
    # J_y=(J_+-J_-)/(2i).
    jy=(jp-jp.conj().T)/(2j)
    # J_z is diagonal with the magnetic quantum number m.
    jz=np.diag([s[3] for s in states]).astype(complex)
    # Every physical angular-momentum generator must be Hermitian.
    for generator in (jx,jy,jz):
        if not np.allclose(generator,generator.conj().T,atol=1e-12):
            raise ValueError('Invalid angular-momentum representation')
    # Return Cartesian generators in the order used by Euler rotations.
    return jx,jy,jz


def euler_rotation(alpha,beta,gamma,generators):
    """Single-particle active rotation exp(-iaJz)exp(-ibJy)exp(-igJz)."""
    # Only J_y and J_z enter the chosen z-y-z convention.
    _,jy,jz=generators
    # This is the z-y-z Euler convention used by the angular-momentum
    # projector.  Acting on both indices later rotates a pair matrix Z.
    first_z=expm(-1j*alpha*jz)
    middle_y=expm(-1j*beta*jy)
    final_z=expm(-1j*gamma*jz)
    # Matrix multiplication follows the operator order acting on the ket.
    return first_z@middle_y@final_z


@dataclass(frozen=True)
class J0Grid:
    """Euler quadrature and conservative finite-space frequency bounds."""

    alpha: np.ndarray
    cos_beta: np.ndarray
    beta_weights: np.ndarray
    gamma: np.ndarray
    m_bound: float
    j_bound: float

    @property
    def size(self):
        """Total number of SO(3) rotation points."""
        # The product grid contains one rotation for every index triple.
        return len(self.alpha)*len(self.cos_beta)*len(self.gamma)


def polynomial_j0_grid(state_encoding,neutron_modes,targets,grid=None,offset=.173):
    """Construct a finite-space exactness grid with polynomial point count.

    Alpha/gamma use `2*M_bound+1` periodic points, resolving all represented
    integer M,K frequencies after exact N,Z projection. Beta uses Gauss-Legendre
    with `ceil((J_bound+1)/2)` points, which integrates P_J(cos beta) through
    the conservative angular-momentum bound. User grids smaller than these
    bounds are rejected. Bounds grow at most linearly with particle count and
    single-particle j, hence the Euler-grid size is polynomial.
    """
    # Preserve the repository's single-particle mode ordering.
    states=[tuple(state) for state in state_encoding]
    # Count how many one-body modes participate in every rotation matrix.
    modes=len(states)
    # Store neutron indices explicitly.
    ns=list(neutron_modes)
    # Every mode not labeled neutron is treated as proton.
    ps=[index for index in range(modes) if index not in ns]
    if len(targets)!=2 or any(not isinstance(n,(int,np.integer)) for n in targets):
        raise ValueError('Integer N,Z targets required')
    # An odd number of half-integer fermions has half-integer total J and
    # therefore no J=0 component. Blocked odd systems need a J>0 projector.
    if sum(targets)%2:
        raise ValueError('J=0 projection requires an even total particle number')
    def bounds(indices,number):
        # M extrema follow by filling the largest/smallest available m values;
        # the sum of largest j values is a conservative upper bound on total J.
        if number<0 or number>len(indices):
            raise ValueError('Particle number outside species space')
        # Sort magnetic projections to find extremal many-body sums.
        ms=sorted(states[index][3] for index in indices)
        # Sort angular momenta downward for a conservative maximum total J.
        js=sorted((states[index][2] for index in indices),reverse=True)
        return (sum(ms[-number:]) if number else 0,
                sum(ms[:number]) if number else 0,
                sum(js[:number]))
    # Compute neutron extremal M and conservative J bounds.
    nmax,nmin,nj=bounds(ns,targets[0])
    # Compute the corresponding proton bounds.
    pmax,pmin,pj=bounds(ps,targets[1])
    # Add species projections and retain the largest represented |M|.
    m_bound=max(abs(nmax+pmax),abs(nmin+pmin))
    # Adding conservative species bounds gives a safe total-J cutoff.
    j_bound=nj+pj

    # Alpha and gamma are periodic Fourier variables resolving M and K.  Beta
    # is integrated in x=cos(beta) with Gauss-Legendre quadrature.  For J=0 the
    # Wigner D weight is constant, so these rules integrate every represented
    # angular-momentum component through the conservative bound.
    alpha_min=int(ceil(2*m_bound))+1
    gamma_min=alpha_min
    # Gauss-Legendre with L points integrates polynomials through degree 2L-1.
    beta_min=int(ceil((j_bound+1)/2))
    # Use exact finite-space defaults unless the caller supplies explicit grids.
    chosen=(alpha_min,beta_min,gamma_min) if grid is None else tuple(grid)
    if (len(chosen)!=3 or any(not isinstance(x,(int,np.integer)) for x in chosen)
            or chosen[0]<alpha_min or chosen[1]<beta_min or chosen[2]<gamma_min):
        raise ValueError(f'J=0 grid must be at least {(alpha_min,beta_min,gamma_min)}')
    # Build the shifted periodic alpha nodes.
    alpha=2*np.pi*(np.arange(chosen[0])+offset)/chosen[0]
    # Build the independent shifted periodic gamma nodes.
    gamma=2*np.pi*(np.arange(chosen[2])+offset)/chosen[2]
    # Integrate beta after changing variables from beta to x=cos(beta).
    cos_beta,weights=np.polynomial.legendre.leggauss(chosen[1])
    # Bundle nodes, weights, and theoretical bounds in one immutable object.
    return J0Grid(alpha,cos_beta,weights,gamma,m_bound,j_bound)


class ParticleNumberJ0ProjectedEnergy(GaugeProjectedEnergy):
    """Polynomial-kernel energy of P_N P_Z P_J=0 |Omega(Z)>.

    Cost per evaluation is O(Ln Lp La Lb Lg (m^4+m^3)). The default particle
    and Euler grids grow polynomially with the finite single-particle space.
    This per-evaluation statement does not guarantee polynomial global
    optimization. Singular transition kernels are reported explicitly.
    """
    def __init__(self,hamiltonian,state_encoding,neutron_modes,targets,*,
                 number_grid=None,euler_grid=None,number_offset=.137,
                 euler_offset=.173):
        # Initialize the two U(1) number grids and validate species conservation.
        super().__init__(hamiltonian,neutron_modes,targets,number_grid,number_offset)
        # Rotations and Hamiltonian tensors must share the same one-body space.
        if len(state_encoding)!=len(hamiltonian.h):
            raise ValueError('State encoding and Hamiltonian sizes differ')
        # Retain the quantum-number records for reproducibility.
        self.state_encoding=list(state_encoding)
        # Build the one-body representation of physical spatial rotations.
        self.generators=single_particle_angular_momentum(state_encoding)
        # Physical rotations must not turn a neutron orbital into a proton
        # orbital; otherwise separate N and Z projection would not commute
        # with angular-momentum projection.
        species=np.diag(self.mask.astype(float))
        if any(not np.allclose(g@species,species@g,atol=1e-12)
               for g in self.generators):
            raise ValueError('Angular-momentum generators mix neutron/proton labels')
        # Construct either the exact default Euler rule or the caller's rule.
        self.euler_grid=polynomial_j0_grid(
            state_encoding,neutron_modes,targets,euler_grid,euler_offset
        )
        # Retain the node shift as part of the reproducible series definition.
        self.euler_offset=float(euler_offset)
        # Precompute every single-particle rotation.  The scalar weight includes
        # sin(beta)d beta through the Gauss-Legendre change x=cos(beta), the
        # normalized alpha/gamma trapezoidal sums, and the J=0 Wigner weight.
        self.rotations=[]
        for alpha in self.euler_grid.alpha:
            for x,weight in zip(self.euler_grid.cos_beta,self.euler_grid.beta_weights):
                # Recover beta from the Gauss-Legendre variable x=cos(beta).
                beta=np.arccos(x)
                for gamma in self.euler_grid.gamma:
                    # Construct the one-body spatial rotation at this Euler node.
                    rotation=euler_rotation(alpha,beta,gamma,self.generators)
                    # Combine dx/2 with normalized alpha and gamma sums.
                    normalized_weight=(
                        weight/(2*len(self.euler_grid.alpha)*len(self.euler_grid.gamma))
                    )
                    # Store the rotation and its scalar quadrature coefficient.
                    self.rotations.append((rotation,normalized_weight))

    def projected_series(self, state):
        """Return P_N P_Z P_J=0|Phi> as the actual quadrature vacuum series."""
        # The series retains a complete intrinsic state, not only its densities.
        if not isinstance(state, HFBState):
            raise TypeError('state must be an HFBState')
        # Every group transformation must act on the state's one-body dimension.
        m=len(self.ham.h)
        if len(state.U)!=m:
            raise ValueError('State and projection Hamiltonian sizes differ')

        # Materialize one transformed vacuum per gauge/Euler quadrature point.
        transformations=[]
        weights=[]
        # The outer two loops are the independent neutron/proton U(1) sums.
        for i in range(self.grid[0]):
            # Convert neutron Fourier index to a shifted full-period angle.
            pn=2*np.pi*(i+self.offset)/self.grid[0]
            for j in range(self.grid[1]):
                # Convert proton Fourier index independently.
                pp=2*np.pi*(j+self.offset)/self.grid[1]
                # Apply the appropriate gauge phase to each one-body mode.
                gauge=np.exp(1j*np.where(self.mask,pn,pp))
                # Fourier character selecting exactly the requested (N,Z).
                fourier=np.exp(-1j*(pn*self.targets[0]+pp*self.targets[1]))/np.prod(self.grid)
                for rotation,euler_weight in self.rotations:
                    # Gauge phases act after the spatial rotation.  Since the
                    # generators preserve species, both operations commute.
                    transform=gauge[:,None]*rotation
                    # Retain T_q itself so the ket remains a Bogoliubov vacuum.
                    transformations.append(transform)
                    # Its scalar coefficient combines number and J=0 quadrature.
                    weights.append(fourier*euler_weight)

        # M is the product of both number grids and all three Euler grids.
        euler_shape=(
            len(self.euler_grid.alpha),
            len(self.euler_grid.cos_beta),
            len(self.euler_grid.gamma),
        )
        # No determinant basis has been introduced at this point.
        return BogoliubovVacuumSeries(
            intrinsic_state=state,
            transformations=np.asarray(transformations),
            weights=np.asarray(weights),
            number_grid=self.grid,
            euler_grid=euler_shape,
            projection='P_N P_Z P_J=0',
            number_offset=self.offset,
            euler_offset=self.euler_offset,
        )

    def metropolis_projected_series(
        self,
        state,
        samples,
        *,
        burn_in=1000,
        thinning=5,
        proposal_scale=0.12,
        seed=0,
        overlap_floor=1e-10,
    ):
        """Importance-sample Euler rotations while keeping N,Z projection exact.

        The Markov chain samples ``alpha, cos(beta), gamma``. For every retained
        rotation, the complete deterministic N,Z Fourier grid is attached. This
        prevents a finite stochastic sample from leaking into unwanted particle
        sectors, so the later fixed-N,Z basis contains the complete sampled ket.
        """
        # Map three normalized chain coordinates to one spatial rotation.
        def coordinate_to_term(coordinate):
            # The first periodic coordinate is Euler alpha.
            alpha=2*np.pi*coordinate[0]
            # Uniform x=cos(beta) supplies the SO(3) sin(beta) measure exactly.
            cos_beta=2*coordinate[1]-1
            beta=np.arccos(cos_beta)
            # The final periodic coordinate is Euler gamma.
            gamma=2*np.pi*coordinate[2]
            # Construct the physical one-body spatial rotation.
            rotation=euler_rotation(alpha,beta,gamma,self.generators)
            # J=0 has constant Wigner D weight, so its character is one.
            return rotation,1.0+0.0j

        # Alpha and gamma wrap; cos(beta)'s unit coordinate reflects.
        rotations,euler_weights,diagnostics=metropolis_vacuum_terms(
            state,
            coordinate_to_term,
            dimensions=3,
            periodic_dimensions=(True,False,True),
            samples=samples,
            burn_in=burn_in,
            thinning=thinning,
            proposal_scale=proposal_scale,
            seed=seed,
            overlap_floor=overlap_floor,
        )
        # Tensor every sampled Euler rotation with the complete exact N,Z grid.
        transformations=[]
        weights=[]
        for rotation,euler_weight in zip(rotations,euler_weights):
            # Traverse all neutron Fourier nodes for this retained rotation.
            for neutron_index in range(self.grid[0]):
                neutron_angle=(
                    2*np.pi*(neutron_index+self.offset)/self.grid[0]
                )
                # Traverse all proton Fourier nodes independently.
                for proton_index in range(self.grid[1]):
                    proton_angle=(
                        2*np.pi*(proton_index+self.offset)/self.grid[1]
                    )
                    # Gauge phases act on rows of the spatial rotation.
                    gauge=np.exp(
                        1j*np.where(self.mask,neutron_angle,proton_angle)
                    )
                    transformations.append(gauge[:,None]*rotation)
                    # Attach the exact number character and grid normalization.
                    number_character=np.exp(
                        -1j*(
                            neutron_angle*self.targets[0]
                            +proton_angle*self.targets[1]
                        )
                    )/np.prod(self.grid)
                    weights.append(euler_weight*number_character)
        # Record the physical projection labels alongside chain diagnostics.
        diagnostics['targets']=tuple(int(number) for number in self.targets)
        diagnostics['target_J']=0
        diagnostics['euler_samples']=int(samples)
        diagnostics['number_grid']=tuple(int(points) for points in self.grid)
        diagnostics['number_grid_points']=int(np.prod(self.grid))
        diagnostics['M_vacua']=len(weights)
        # M equals Euler samples times the exact deterministic number-grid size.
        return BogoliubovVacuumSeries(
            intrinsic_state=state,
            transformations=np.asarray(transformations),
            weights=np.asarray(weights),
            number_grid=self.grid,
            euler_grid=(int(samples),),
            projection='P_N P_Z P_J=0',
            number_offset=self.offset,
            sampling_method='metropolis',
            sampling_diagnostics=diagnostics,
        )

    def series_energy(self, series):
        """Evaluate energy from the same vacuum series used later for fidelity."""
        # Prevent accidental energy evaluation of a series built by another grid.
        if not isinstance(series,BogoliubovVacuumSeries):
            raise TypeError('series must be a BogoliubovVacuumSeries')
        if series.sampling_method=='quadrature':
            if series.number_grid!=tuple(self.grid):
                raise ValueError('Series number grid differs from evaluator grid')
            # Euler dimensions must also match this evaluator's stored rotations.
            expected_euler_grid=(
                len(self.euler_grid.alpha),
                len(self.euler_grid.cos_beta),
                len(self.euler_grid.gamma),
            )
            if series.euler_grid!=expected_euler_grid:
                raise ValueError('Series Euler grid differs from evaluator grid')
            # Offsets determine actual nodes even when dimensions agree.
            if (not np.isclose(series.number_offset,self.offset)
                    or not np.isclose(series.euler_offset,self.euler_offset)):
                raise ValueError('Series quadrature offsets differ from evaluator')
        # Stochastic series must still represent the same combined symmetries.
        elif series.projection!='P_N P_Z P_J=0' or series.euler_grid is None:
            raise ValueError('Metropolis series is not an N,Z,J=0 projection')
        # Transition-density kernels currently require a finite Thouless chart.
        z=series.intrinsic_state.thouless_matrix
        m=len(self.ham.h)
        # Reuse one identity matrix in every generalized-Wick solve.
        identity=np.eye(m)
        # Accumulate Hamiltonian and norm kernels with identical series weights.
        numerator=0j
        denominator=0j
        # Normalize the intrinsic ket once; unitary group rotations preserve it.
        scale=np.exp(.5*np.linalg.slogdet(identity+z.conj().T@z)[1])

        # Iterate over precisely the T_q,w_q pairs stored for later expansion.
        for transform,series_weight in zip(series.transformations,series.weights):
            # A pair-creation matrix transforms on both particle legs.
            zg=transform@z@transform.T
            # B supplies all normalized transition contractions.
            b=identity+z.conj().T@zg
            if np.linalg.cond(b)>1e12:
                raise ValueError('Singular symmetry kernel; change grid offsets or chart')
            # Solve B X=I instead of explicitly inverting B.
            inverse=np.linalg.solve(b,identity)
            # Construct normal and anomalous transition densities.
            rho=zg@inverse@z.conj().T
            kappa=zg@inverse
            creation=-inverse@z.conj().T
            # PFAPACK fixes the phase of <Phi|T_q|Phi>.
            overlap=(-1)**(m*(m+1)//2)*pfaffian(
                np.block([[zg,-identity],[identity,-z.conj()]]))/scale
            # The norm-kernel coefficient is w_q times the vacuum overlap.
            kernel_weight=series_weight*overlap
            # Evaluate one-body, normal two-body, and pairing contractions.
            kernel=(np.einsum('ij,ji->',self.ham.h,rho)
                +.5*np.einsum('ijkl,ki,lj->',self.ham.v,rho,rho)
                +.25*np.einsum('ijkl,ij,kl->',self.ham.v,creation,kappa))
            # Add this term coherently to both projected kernels.
            denominator+=kernel_weight
            numerator+=kernel_weight*kernel
        # Finite stochastic sums retain ordinary complex Monte Carlo noise.
        if series.sampling_method=='metropolis':
            if abs(denominator)<1e-13:
                raise ValueError('Metropolis N,Z,J=0 norm is unresolved')
            energy=numerator/denominator
            series.sampling_diagnostics['norm_kernel_real']=float(denominator.real)
            series.sampling_diagnostics['norm_kernel_imag']=float(denominator.imag)
            series.sampling_diagnostics['energy_imaginary']=float(energy.imag)
            return float(energy.real)
        # A deterministic exact projector has a positive real norm kernel.
        if abs(denominator.imag)>1e-7 or denominator.real<1e-13:
            raise ValueError('Vanishing or unresolved N,Z,J=0 projected norm')
        # The projected energy is the ratio of Hamiltonian and norm kernels.
        energy=numerator/denominator
        if abs(energy.imag)>1e-7:
            raise ValueError('Projection cancellation error exceeds tolerance')
        return float(energy.real)

    def energy(self,z):
        """Build the quadrature series from Z and evaluate its energy kernels."""
        # Preserve the established public API accepting a finite Thouless matrix.
        state=HFBState.from_thouless(np.asarray(z,complex))
        # Energy and later fidelity now consume the identical list of vacua.
        return self.series_energy(self.projected_series(state))


def project_state_observables(series, hamiltonian, target=None):
    """Evaluate a gauge/Euler vacuum series in a fixed determinant basis.

    Unlike the old implementation, this function never applies a projector
    obtained from J^2 diagonalization. The caller passes the exact same finite
    vacuum series used by the energy kernel; determinant amplitudes are
    introduced only for the final energy/fidelity calculation.
    """
    # Delay basis expansion and normalization until the observable boundary.
    return projected_series_observables(
        series,hamiltonian,target=target
    )
