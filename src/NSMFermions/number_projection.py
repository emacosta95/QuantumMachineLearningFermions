"""Exact-sector particle-number VAP reference backend for small model spaces.

Projected amplitudes are Pfaffians of a fully complex, species-mixed Thouless
matrix. No gauge quadrature error occurs. Explicit sector enumeration is
combinatorial: this backend is a validation reference, not a scalable kernel.
"""
from dataclasses import dataclass
from itertools import combinations
from math import comb
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

if __package__:
    from .hfb import HFBState
else:
    from hfb import HFBState


def _matchings(indices):
    if not indices:
        return [(1, ())]
    terms = []
    for j in range(1, len(indices)):
        rest = indices[1:j] + indices[j+1:]
        for sign, pairs in _matchings(rest):
            terms.append(((-1)**(j+1)*sign, ((indices[0], indices[j]),)+pairs))
    return terms


def state_from_thouless(z):
    """Normalized intrinsic vacuum for exp(sum_i<j Z_ij c_i†c_j†)|0>."""
    z = np.asarray(z, complex)
    if z.ndim != 2 or z.shape[0] != z.shape[1] or not np.isfinite(z).all():
        raise ValueError('Z must be a finite square matrix')
    if not np.allclose(z, -z.T, atol=1e-12):
        raise ValueError('Z must be antisymmetric')
    values, vectors = np.linalg.eigh(np.eye(len(z)) + z.T @ z.conj())
    u = (vectors / np.sqrt(values)) @ vectors.conj().T
    return HFBState(u, z.conj() @ u)


class NumberProjectedSpace:
    """Full fixed N,Z space; no M/J restriction or species pairing blocks.

    Hamiltonian must separately conserve N and Z. Basis occupations are sorted
    in the original mode order, fixing all fermionic phases. Resource guards
    apply before enumerating basis vectors and Pfaffian polynomials.
    """
    def __init__(self, hamiltonian, neutron_modes, targets, *, max_dimension=5000,
                 max_polynomial_terms=200000):
        self.modes = m = len(hamiltonian.h)
        ns = list(neutron_modes)
        if any(not isinstance(i, (int,np.integer)) for i in ns):
            raise ValueError('Species indices must be integers')
        if len(set(ns)) != len(ns) or any(i<0 or i>=m for i in ns):
            raise ValueError('Invalid neutron indices')
        ps = [i for i in range(m) if i not in ns]
        if len(targets)!=2 or any(not isinstance(n,(int,np.integer)) for n in targets):
            raise ValueError('Exact projection requires integer N,Z')
        if any(n<0 or n>cap for n,cap in zip(targets,[len(ns),len(ps)])):
            raise ValueError('Particle number outside model space')
        total = sum(targets)
        if total % 2:
            raise ValueError('Vacuum Thouless ansatz supports even total parity only')
        dimension = comb(len(ns),targets[0])*comb(len(ps),targets[1])
        nterms = 1
        for k in range(1,total,2):
            nterms *= k
        if dimension>max_dimension or dimension*nterms>max_polynomial_terms:
            raise ValueError('Explicit projection exceeds configured reference-backend limit')
        mask = np.zeros(m,int)
        mask[ns] = 1
        hi,hj = np.nonzero(np.abs(hamiltonian.h)>1e-12)
        vi,vj,vk,vl = np.nonzero(np.abs(hamiltonian.v)>1e-12)
        if (np.any(mask[hi]!=mask[hj]) or
            np.any(mask[vi]+mask[vj]!=mask[vk]+mask[vl])):
            raise ValueError('Hamiltonian must conserve neutron and proton numbers')
        self.targets = tuple(targets)
        self.neutron_modes = tuple(ns)
        self.occupations = [tuple(sorted(n+p)) for n in combinations(ns,targets[0])
                            for p in combinations(ps,targets[1])]
        self.masks = [sum(1<<i for i in occ) for occ in self.occupations]
        self.pairs = list(combinations(range(m),2))
        pair_index = {pair:i for i,pair in enumerate(self.pairs)}
        terms = [_matchings(occ) for occ in self.occupations]
        self.signs = np.array([[t[0] for t in row] for row in terms])
        self.term_indices = np.array([[[pair_index[p] for p in t[1]] for t in row]
                                      for row in terms],int).reshape(dimension,nterms,total//2)
        self.matrix = self._build_matrix(hamiltonian)

    def _build_matrix(self, ham):
        lookup = {mask:i for i,mask in enumerate(self.masks)}
        terms = [([(j,False),(i,True)],ham.h[i,j])
                 for i,j in zip(*np.nonzero(ham.h))]
        terms += [([(k,False),(l,False),(j,True),(i,True)],ham.v[i,j,k,l]/4)
                  for i,j,k,l in zip(*np.nonzero(ham.v))]
        rows,cols,values = [],[],[]
        for col,initial in enumerate(self.masks):
            for operations,value in terms:
                state,phase = initial,1
                for mode,create in operations:
                    if bool(state & (1<<mode)) == create:
                        break
                    phase *= (-1)**bin(state & ((1<<mode)-1)).count('1')
                    state ^= 1<<mode
                else:
                    if state in lookup:
                        rows.append(lookup[state]); cols.append(col); values.append(value*phase)
        matrix=sparse.coo_matrix((values,(rows,cols)),shape=(len(lookup),)*2).tocsr()
        matrix.sum_duplicates()
        return matrix

    def unpack(self, x):
        p=len(self.pairs)
        x=np.asarray(x,float)
        if x.shape!=(2*p,) or not np.isfinite(x).all():
            raise ValueError('Invalid real/imaginary Thouless parameters')
        z=np.zeros((self.modes,)*2,complex)
        ij=np.array(self.pairs).T
        z[ij[0],ij[1]]=x[:p]+1j*x[p:]
        return z-z.T

    def amplitudes_and_jacobian(self, x):
        p=len(self.pairs)
        self.unpack(x)  # validation
        q=np.asarray(x[:p])+1j*np.asarray(x[p:])
        factors=q[self.term_indices]
        amplitudes=np.sum(self.signs*np.prod(factors,axis=2),axis=1)
        derivative=np.zeros((len(amplitudes),p),complex)
        row=np.broadcast_to(np.arange(len(amplitudes))[:,None],self.signs.shape)
        for k in range(factors.shape[2]):
            # Product of all other factors avoids division at zero amplitudes.
            rest=np.prod(np.delete(factors,k,axis=2),axis=2)*self.signs
            np.add.at(derivative,(row,self.term_indices[:,:,k]),rest)
        return amplitudes,derivative

    def energy_and_gradient(self, x):
        a,d=self.amplitudes_and_jacobian(x)
        norm=float(np.vdot(a,a).real)
        if not np.isfinite(norm) or norm<1e-24:
            raise ValueError('Vanishing projected norm; use a paired initial state')
        ha=self.matrix@a
        energy=float(np.vdot(a,ha).real/norm)
        g=d.conj().T@(ha-energy*a)/norm
        return energy,np.r_[2*g.real,2*g.imag]


@dataclass
class VAPResult:
    energy: float
    projected_vector: np.ndarray
    intrinsic_state: HFBState
    thouless_matrix: np.ndarray
    converged: bool
    gradient_norm: float
    attempts: list


def solve_number_vap(space, *, starts=3, seed=0, maxiter=300, tolerance=1e-8,
                     gradient_tolerance=1e-5, initial_parameters=None):
    """Minimize <Phi|H P_N P_Z|Phi>/<Phi|P_N P_Z|Phi>.

    Uses exact Pfaffian-sector amplitudes and analytic gradients, with no
    pairing penalty. Physical N,Z are exact, so intrinsic average numbers need
    not be constrained. Scaling Z leaves the normalized projected state
    unchanged; returned Z is fixed to Frobenius norm sqrt(m) for reproducible
    intrinsic diagnostics. Those diagnostics are not observables of P_N P_Z Phi.
    """
    if starts<1 or maxiter<1 or tolerance<=0 or gradient_tolerance<=0:
        raise ValueError('Positive optimizer settings required')
    rng=np.random.default_rng(seed)
    attempts,candidates=[],[]
    for i in range(starts):
        x=(np.asarray(initial_parameters,float) if i==0 and initial_parameters is not None
           else rng.normal(scale=.3,size=2*len(space.pairs)))
        fit=minimize(space.energy_and_gradient,x,jac=True,method='L-BFGS-B',
                     options={'maxiter':maxiter,'ftol':tolerance,'gtol':min(1e-10,gradient_tolerance/1000),
                              'maxls':40})
        # Fix the irrelevant radial scale before measuring stationarity.
        z=space.unpack(fit.x)
        x=fit.x*np.sqrt(space.modes)/np.linalg.norm(z)
        energy,grad=space.energy_and_gradient(x)
        residual=float(np.linalg.norm(grad))
        ok=bool(fit.success and residual<=gradient_tolerance)
        attempts.append({'energy':energy,'converged':ok,'gradient_norm':residual,
                         'iterations':int(fit.nit),'message':str(fit.message)})
        candidates.append((ok,energy,x,residual))
    valid=[c for c in candidates if c[0]]
    ok,energy,x,residual=min(valid or candidates,key=lambda c:c[1])
    a,_=space.amplitudes_and_jacobian(x)
    z=space.unpack(x)
    return VAPResult(energy,a/np.linalg.norm(a),state_from_thouless(z),z,
                     ok,residual,attempts)
