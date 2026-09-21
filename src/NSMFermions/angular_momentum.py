"""J=0 angular-momentum projection for general Bogoliubov vacua.

The production-style evaluator combines polynomial-size N,Z Fourier grids with
an SO(3) Euler grid and uses Pfaffian/transition-density kernels. The exact
fixed-sector projector in this module is only a small-system validation backend.

General J>0 projection is not implemented here: a triaxial intrinsic vacuum
requires the full P^J_MK matrix and K mixing. J=0 has only M=K=0 and is therefore
the clean first target for the Be8 0+ ground state.
"""
from dataclasses import dataclass
from math import ceil
import numpy as np
from scipy import sparse
from scipy.linalg import expm

if __package__:
    from .gauge_projection import GaugeProjectedEnergy, pfaffian
else:
    from gauge_projection import GaugeProjectedEnergy, pfaffian


def single_particle_angular_momentum(state_encoding):
    """Return Cartesian j generators in the repository's spherical basis.

    Each state is `(n,l,j,m,t,tz)`. Rotations preserve all labels except m.
    Ordering is taken exactly from `state_encoding`; no species blocks assumed.
    """
    states=[tuple(s) for s in state_encoding]
    lookup={s:i for i,s in enumerate(states)}
    jp=np.zeros((len(states),len(states)),complex)
    for i,(n,l,j,m,t,tz) in enumerate(states):
        raised=(n,l,j,m+1,t,tz)
        if raised in lookup:
            jp[lookup[raised],i]=np.sqrt(j*(j+1)-m*(m+1))
    jx=(jp+jp.conj().T)/2
    jy=(jp-jp.conj().T)/(2j)
    jz=np.diag([s[3] for s in states]).astype(complex)
    for generator in (jx,jy,jz):
        if not np.allclose(generator,generator.conj().T,atol=1e-12):
            raise ValueError('Invalid angular-momentum representation')
    return jx,jy,jz


def euler_rotation(alpha,beta,gamma,generators):
    """Single-particle active rotation exp(-iaJz)exp(-ibJy)exp(-igJz)."""
    _,jy,jz=generators
    return expm(-1j*alpha*jz)@expm(-1j*beta*jy)@expm(-1j*gamma*jz)


@dataclass(frozen=True)
class J0Grid:
    alpha: np.ndarray
    cos_beta: np.ndarray
    beta_weights: np.ndarray
    gamma: np.ndarray
    m_bound: float
    j_bound: float

    @property
    def size(self):
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
    states=[tuple(s) for s in state_encoding]; modes=len(states)
    ns=list(neutron_modes); ps=[i for i in range(modes) if i not in ns]
    if len(targets)!=2 or any(not isinstance(n,(int,np.integer)) for n in targets):
        raise ValueError('Integer N,Z targets required')
    # An odd number of half-integer fermions has half-integer total J and
    # therefore no J=0 component. Blocked odd systems need a J>0 projector.
    if sum(targets)%2:
        raise ValueError('J=0 projection requires an even total particle number')
    def bounds(indices,number):
        if number<0 or number>len(indices):
            raise ValueError('Particle number outside species space')
        ms=sorted((states[i][3] for i in indices))
        js=sorted((states[i][2] for i in indices),reverse=True)
        return (sum(ms[-number:]) if number else 0,
                sum(ms[:number]) if number else 0,
                sum(js[:number]))
    nmax,nmin,nj=bounds(ns,targets[0]); pmax,pmin,pj=bounds(ps,targets[1])
    m_bound=max(abs(nmax+pmax),abs(nmin+pmin))
    j_bound=nj+pj
    alpha_min=gamma_min=int(ceil(2*m_bound))+1
    beta_min=int(ceil((j_bound+1)/2))
    chosen=(alpha_min,beta_min,gamma_min) if grid is None else tuple(grid)
    if (len(chosen)!=3 or any(not isinstance(x,(int,np.integer)) for x in chosen)
            or chosen[0]<alpha_min or chosen[1]<beta_min or chosen[2]<gamma_min):
        raise ValueError(f'J=0 grid must be at least {(alpha_min,beta_min,gamma_min)}')
    alpha=2*np.pi*(np.arange(chosen[0])+offset)/chosen[0]
    gamma=2*np.pi*(np.arange(chosen[2])+offset)/chosen[2]
    cos_beta,weights=np.polynomial.legendre.leggauss(chosen[1])
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
        super().__init__(hamiltonian,neutron_modes,targets,number_grid,number_offset)
        if len(state_encoding)!=len(hamiltonian.h):
            raise ValueError('State encoding and Hamiltonian sizes differ')
        self.state_encoding=list(state_encoding)
        self.generators=single_particle_angular_momentum(state_encoding)
        # Physical rotations must not turn a neutron orbital into a proton
        # orbital; otherwise separate N and Z projection would not commute
        # with angular-momentum projection.
        species=np.diag(self.mask.astype(float))
        if any(not np.allclose(g@species,species@g,atol=1e-12)
               for g in self.generators):
            raise ValueError('Angular-momentum generators mix neutron/proton labels')
        self.euler_grid=polynomial_j0_grid(state_encoding,neutron_modes,targets,
                                           euler_grid,euler_offset)
        self.rotations=[]
        for alpha in self.euler_grid.alpha:
            for x,weight in zip(self.euler_grid.cos_beta,self.euler_grid.beta_weights):
                beta=np.arccos(x)
                for gamma in self.euler_grid.gamma:
                    self.rotations.append((euler_rotation(alpha,beta,gamma,self.generators),
                                           weight/(2*len(self.euler_grid.alpha)*len(self.euler_grid.gamma))))

    def energy(self,z):
        z=np.asarray(z,complex); m=len(self.ham.h)
        if z.shape!=(m,m) or not np.isfinite(z).all() or not np.allclose(z,-z.T,atol=1e-12):
            raise ValueError('Finite antisymmetric Z required')
        identity=np.eye(m); numerator=0j; denominator=0j
        scale=np.exp(.5*np.linalg.slogdet(identity+z.conj().T@z)[1])
        for i in range(self.grid[0]):
            pn=2*np.pi*(i+self.offset)/self.grid[0]
            for j in range(self.grid[1]):
                pp=2*np.pi*(j+self.offset)/self.grid[1]
                gauge=np.exp(1j*np.where(self.mask,pn,pp))
                fourier=np.exp(-1j*(pn*self.targets[0]+pp*self.targets[1]))/np.prod(self.grid)
                for rotation,euler_weight in self.rotations:
                    transform=gauge[:,None]*rotation
                    zg=transform@z@transform.T
                    b=identity+z.conj().T@zg
                    if np.linalg.cond(b)>1e12:
                        raise ValueError('Singular symmetry kernel; change grid offsets or chart')
                    inverse=np.linalg.solve(b,identity)
                    rho=zg@inverse@z.conj().T
                    kappa=zg@inverse
                    creation=-inverse@z.conj().T
                    overlap=(-1)**(m*(m+1)//2)*pfaffian(
                        np.block([[zg,-identity],[identity,-z.conj()]]))/scale
                    weight=fourier*euler_weight*overlap
                    kernel=(np.einsum('ij,ji->',self.ham.h,rho)
                        +.5*np.einsum('ijkl,ki,lj->',self.ham.v,rho,rho)
                        +.25*np.einsum('ijkl,ij,kl->',self.ham.v,creation,kappa))
                    denominator+=weight; numerator+=weight*kernel
        if abs(denominator.imag)>1e-7 or denominator.real<1e-13:
            raise ValueError('Vanishing or unresolved N,Z,J=0 projected norm')
        energy=numerator/denominator
        if abs(energy.imag)>1e-7:
            raise ValueError('Projection cancellation error exceeds tolerance')
        return float(energy.real)


def many_body_one_body_operator(space,matrix):
    """Reference representation of sum_ij matrix_ij c_i^dag c_j."""
    matrix=np.asarray(matrix,complex)
    if matrix.shape!=(space.modes,space.modes):
        raise ValueError('One-body matrix has wrong shape')
    lookup={mask:i for i,mask in enumerate(space.masks)}
    rows=[]; cols=[]; data=[]
    for col,initial in enumerate(space.masks):
        for i,j in zip(*np.nonzero(abs(matrix)>1e-14)):
            state=initial; phase=1
            if not state&(1<<j): continue
            phase*=(-1)**bin(state&((1<<j)-1)).count('1'); state^=1<<j
            if state&(1<<i): continue
            phase*=(-1)**bin(state&((1<<i)-1)).count('1'); state^=1<<i
            rows.append(lookup[state]); cols.append(col); data.append(matrix[i,j]*phase)
    result=sparse.coo_matrix((data,(rows,cols)),shape=(len(lookup),)*2).tocsr()
    result.sum_duplicates()
    return result


@dataclass
class J0ReferenceProjector:
    projector: np.ndarray
    j2: np.ndarray
    rank: int


def exact_j0_projector(space,state_encoding,tolerance=1e-8):
    """Diagonalize J^2 in a small fixed-N,Z space for validation only."""
    generators=single_particle_angular_momentum(state_encoding)
    many=[many_body_one_body_operator(space,g) for g in generators]
    j2=sum((g@g for g in many),start=sparse.csr_matrix(space.matrix.shape)).toarray()
    j2=(j2+j2.conj().T)/2
    values,vectors=np.linalg.eigh(j2)
    selected=values<tolerance
    if not np.any(selected):
        raise ValueError('No J=0 subspace found')
    projector=vectors[:,selected]@vectors[:,selected].conj().T
    return J0ReferenceProjector(projector,j2,int(selected.sum()))


def projected_observables(vector,space,projector,target=None):
    """Normalize P0 vector and return norm, energy, and optional fidelity."""
    projected=projector.projector@np.asarray(vector,complex)
    norm=float(np.vdot(projected,projected).real)
    if norm<1e-14:
        raise ValueError('State has vanishing J=0 component')
    projected/=np.sqrt(norm)
    energy=float(np.vdot(projected,space.matrix@projected).real)
    result={'vector':projected,'j0_weight':norm,'energy':energy}
    if target is not None:
        normalized=np.asarray(target,complex)/np.linalg.norm(target)
        result['fidelity']=float(abs(np.vdot(normalized,projected))**2)
    return result
