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
    """Enumerate signed perfect matchings used in a Pfaffian expansion.

    For an occupied configuration ``(i, j, k, l, ...)``, the Thouless-state
    amplitude is the Pfaffian of the corresponding submatrix of Z.  Expanding
    that Pfaffian into products of independent Z entries makes its analytic
    derivative cheap during VAP.  PFAPACK is used for general numerical
    Pfaffians in ``gauge_projection``; this symbolic matching table is retained
    here specifically because the optimizer also needs derivatives.
    """
    if not indices:
        # The Pfaffian of the empty matrix is one, terminating the recursion.
        return [(1, ())]
    terms = []
    # Pair the first index with every possible partner.  Removing that pair
    # leaves a smaller Pfaffian; (-1)^(j+1) is the permutation sign.
    for j in range(1, len(indices)):
        rest = indices[1:j] + indices[j+1:]
        for sign, pairs in _matchings(rest):
            terms.append(((-1)**(j+1)*sign, ((indices[0], indices[j]),)+pairs))
    return terms


def state_from_thouless(z):
    """Build normalized U,V for exp(sum_i<j Z_ij c_i†c_j†)|0>.

    This returns the *intrinsic* Bogoliubov vacuum.  It does not perform number
    projection; use :meth:`NumberProjectedSpace.projected_state` for the
    normalized fixed-N,Z many-body vector.

    This compatibility wrapper delegates to :meth:`HFBState.from_thouless`;
    new code may call that class method directly.
    """
    return HFBState.from_thouless(z)


class NumberProjectedSpace:
    """Full fixed N,Z space; no M/J restriction or species pairing blocks.

    Hamiltonian must separately conserve N and Z. Basis occupations are sorted
    in the original mode order, fixing all fermionic phases. Resource guards
    apply before enumerating basis vectors and Pfaffian polynomials.
    """
    def __init__(self, hamiltonian, neutron_modes, targets, *, max_dimension=5000,
                 max_polynomial_terms=200000):
        self.modes = m = len(hamiltonian.h)

        # Divide the one-body basis into neutron modes and its proton
        # complement.  Mode ordering itself is otherwise unrestricted.
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

        # An unblocked Thouless vacuum contains only even total particle-number
        # parity, so an odd N+Z component is identically zero.
        if total % 2:
            raise ValueError('Vacuum Thouless ansatz supports even total parity only')

        # The explicit fixed-N,Z basis has C(m_n,N) C(m_p,Z) determinants.  A
        # 2n-particle Pfaffian contains (2n-1)!! matching terms.  Check both
        # costs before allocating the basis and derivative lookup tables.
        dimension = comb(len(ns),targets[0])*comb(len(ps),targets[1])
        nterms = 1
        for k in range(1,total,2):
            nterms *= k
        if dimension>max_dimension or dimension*nterms>max_polynomial_terms:
            raise ValueError('Explicit projection exceeds configured reference-backend limit')

        # Verify that every nonzero Hamiltonian matrix element separately
        # conserves neutron and proton number.
        mask = np.zeros(m,int)
        mask[ns] = 1
        hi,hj = np.nonzero(np.abs(hamiltonian.h)>1e-12)
        vi,vj,vk,vl = np.nonzero(np.abs(hamiltonian.v)>1e-12)
        if (np.any(mask[hi]!=mask[hj]) or
            np.any(mask[vi]+mask[vj]!=mask[vk]+mask[vl])):
            raise ValueError('Hamiltonian must conserve neutron and proton numbers')
        self.targets = tuple(targets)
        self.neutron_modes = tuple(ns)

        # Each tuple is one determinant in the projected Hilbert space.  The
        # parallel integer bit mask is used to apply creation/annihilation
        # operators efficiently when building H.
        self.occupations = [tuple(sorted(n+p)) for n in combinations(ns,targets[0])
                            for p in combinations(ps,targets[1])]
        self.masks = [sum(1<<i for i in occ) for occ in self.occupations]

        # Optimizer coordinates correspond to the independent upper-triangular
        # entries Z_ij, i<j.  Precompute how every projected amplitude depends
        # on those pair coordinates and on the matching signs.
        self.pairs = list(combinations(range(m),2))
        pair_index = {pair:i for i,pair in enumerate(self.pairs)}
        terms = [_matchings(occ) for occ in self.occupations]
        self.signs = np.array([[t[0] for t in row] for row in terms])
        self.term_indices = np.array([[[pair_index[p] for p in t[1]] for t in row]
                                      for row in terms],int).reshape(dimension,nterms,total//2)
        self.matrix = self._build_matrix(hamiltonian)

    def _build_matrix(self, ham):
        """Build H in the explicit fixed-N,Z determinant basis."""
        lookup = {mask:i for i,mask in enumerate(self.masks)}

        # Store each second-quantized Hamiltonian term as an ordered list of
        # (mode, create?) operations.  Operators act on kets from right to left.
        terms = [([(j,False),(i,True)],ham.h[i,j])
                 for i,j in zip(*np.nonzero(ham.h))]
        terms += [([(k,False),(l,False),(j,True),(i,True)],ham.v[i,j,k,l]/4)
                  for i,j,k,l in zip(*np.nonzero(ham.v))]
        rows,cols,values = [],[],[]
        for col,initial in enumerate(self.masks):
            for operations,value in terms:
                state,phase = initial,1
                for mode,create in operations:
                    # Creation on an occupied mode or annihilation on an empty
                    # mode kills the determinant.
                    if bool(state & (1<<mode)) == create:
                        break

                    # Count occupied lower-index modes to obtain the Jordan-
                    # Wigner/fermionic reordering sign, then flip occupation.
                    phase *= (-1)**bin(state & ((1<<mode)-1)).count('1')
                    state ^= 1<<mode
                else:
                    if state in lookup:
                        rows.append(lookup[state]); cols.append(col); values.append(value*phase)
        matrix=sparse.coo_matrix((values,(rows,cols)),shape=(len(lookup),)*2).tocsr()
        matrix.sum_duplicates()
        return matrix

    def many_body_matrix_error(self, fermionic_hamiltonian):
        """Compare this matrix with a built ``FermiHubbardHamiltonian``.

        The legacy class and this reference backend may order determinants
        differently.  This method derives bit masks from ``basis`` (or uses
        ``basis_bits`` in the optimized class), reorders the supplied matrix to
        ``self.masks``, and returns the maximum absolute matrix-element error.

        This validates that both representations describe the same Hamiltonian;
        it does not supply the raw h and v tensors required by gauge kernels.
        """
        other = getattr(fermionic_hamiltonian, 'hamiltonian', None)
        if other is None:
            raise ValueError('FermiHubbardHamiltonian has not been assembled')

        if hasattr(fermionic_hamiltonian, 'basis_bits'):
            other_masks = [int(mask) for mask in fermionic_hamiltonian.basis_bits]
        elif hasattr(fermionic_hamiltonian, 'basis'):
            basis = np.asarray(fermionic_hamiltonian.basis)
            if basis.ndim != 2 or basis.shape[1] != self.modes:
                raise ValueError('Fermionic basis is incompatible with this space')
            other_masks = [
                sum(int(bit) << mode for mode, bit in enumerate(row))
                for row in basis
            ]
        else:
            raise ValueError('Cannot determine FermionicHamiltonian basis order')

        if len(set(other_masks)) != len(other_masks):
            raise ValueError('FermionicHamiltonian basis contains duplicates')
        lookup = {mask: index for index, mask in enumerate(other_masks)}
        if set(lookup) != set(self.masks):
            raise ValueError('FermionicHamiltonian uses a different fixed sector')

        # Reorder rows and columns into NumberProjectedSpace determinant order.
        order = np.array([lookup[mask] for mask in self.masks])
        reordered = other[order][:, order]
        difference = reordered - self.matrix
        if sparse.issparse(difference):
            return float(np.max(np.abs(difference.data), initial=0.0))
        return float(np.max(np.abs(np.asarray(difference)), initial=0.0))

    def unpack(self, x):
        """Convert real optimizer coordinates into antisymmetric complex Z."""
        p=len(self.pairs)
        x=np.asarray(x,float)
        if x.shape!=(2*p,) or not np.isfinite(x).all():
            raise ValueError('Invalid real/imaginary Thouless parameters')
        z=np.zeros((self.modes,)*2,complex)
        ij=np.array(self.pairs).T
        z[ij[0],ij[1]]=x[:p]+1j*x[p:]
        return z-z.T

    def parameters_from_thouless(self, z):
        """Pack an antisymmetric Thouless matrix into real optimizer form."""
        z = np.asarray(z, complex)
        if (
            z.shape != (self.modes, self.modes)
            or not np.isfinite(z).all()
            or not np.allclose(z, -z.T, atol=1e-12)
        ):
            raise ValueError('Finite antisymmetric Z required')
        entries = np.array([z[i, j] for i, j in self.pairs])
        return np.r_[entries.real, entries.imag]

    def projected_state(self, z):
        """Return normalized coefficients of P_N P_Z|Phi(Z)>.

        The returned vector is ordered exactly like ``self.occupations`` and
        ``self.masks``.  Constructing it is combinatorial, so this method is for
        small-space observables and fidelities; the gauge-grid evaluator avoids
        this vector deliberately.
        """
        # HFBState owns the generic Pfaffian occupation amplitudes; this class
        # supplies the list of determinants defining the desired N,Z sector.
        return HFBState.from_thouless(z).fixed_sector_state(self.occupations)

    def projected_fidelity(self, z, target):
        """Return |<target|P_N P_Z Phi(Z)>|^2 for a normalized target."""
        state = HFBState.from_thouless(z)
        return state.fixed_sector_fidelity(
            target, self.occupations, projected=True
        )

    def amplitudes_and_jacobian(self, x):
        """Return unnormalized fixed-N,Z amplitudes and d(amplitude)/dZ."""
        p=len(self.pairs)
        self.unpack(x)  # validation
        q=np.asarray(x[:p])+1j*np.asarray(x[p:])
        # Select every Z factor appearing in every signed perfect matching.
        factors=q[self.term_indices]
        # Summing products over matchings is precisely the Pfaffian expansion
        # of the occupied Z submatrix for each basis determinant.
        amplitudes=np.sum(self.signs*np.prod(factors,axis=2),axis=1)
        derivative=np.zeros((len(amplitudes),p),complex)
        row=np.broadcast_to(np.arange(len(amplitudes))[:,None],self.signs.shape)
        for k in range(factors.shape[2]):
            # Product of all other factors avoids division at zero amplitudes.
            rest=np.prod(np.delete(factors,k,axis=2),axis=2)*self.signs
            np.add.at(derivative,(row,self.term_indices[:,:,k]),rest)
        return amplitudes,derivative

    def energy_and_gradient(self, x):
        """Return projected Rayleigh quotient and its real-coordinate gradient."""
        a,d=self.amplitudes_and_jacobian(x)
        norm=float(np.vdot(a,a).real)
        if not np.isfinite(norm) or norm<1e-24:
            raise ValueError('Vanishing projected norm; use a paired initial state')
        ha=self.matrix@a
        # E = a^dagger H a / a^dagger a.  Its complex derivative is
        # d^dagger(Ha-Ea)/norm; split it into real and imaginary coordinates.
        energy=float(np.vdot(a,ha).real/norm)
        g=d.conj().T@(ha-energy*a)/norm
        return energy,np.r_[2*g.real,2*g.imag]


@dataclass
class VAPResult:
    """Optimized intrinsic vacuum and normalized fixed-N,Z state vector."""

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

    # Multiple random starts reduce, but do not eliminate, the risk of finding
    # a local rather than global projected-energy minimum.
    rng=np.random.default_rng(seed)
    attempts,candidates=[],[]
    for i in range(starts):
        x=(np.asarray(initial_parameters,float) if i==0 and initial_parameters is not None
           else rng.normal(scale=.3,size=2*len(space.pairs)))
        fit=minimize(space.energy_and_gradient,x,jac=True,method='L-BFGS-B',
                     options={'maxiter':maxiter,'ftol':tolerance,'gtol':min(1e-10,gradient_tolerance/1000),
                              'maxls':40})
        # Fix the irrelevant radial scale before measuring stationarity.
        # Multiplying every entry of Z by the same nonzero scalar multiplies all
        # fixed-N,Z amplitudes equally and leaves their normalized vector intact.
        z=space.unpack(fit.x)
        x=fit.x*np.sqrt(space.modes)/np.linalg.norm(z)
        energy,grad=space.energy_and_gradient(x)
        residual=float(np.linalg.norm(grad))
        ok=bool(fit.success and residual<=gradient_tolerance)
        attempts.append({'energy':energy,'converged':ok,'gradient_norm':residual,
                         'iterations':int(fit.nit),'message':str(fit.message)})
        candidates.append((ok,energy,x,residual))
    valid=[c for c in candidates if c[0]]
    # Prefer the lowest-energy converged start.  If none converged, return the
    # lowest-energy attempt with converged=False so diagnostics are preserved.
    ok,energy,x,residual=min(valid or candidates,key=lambda c:c[1])
    a,_=space.amplitudes_and_jacobian(x)
    z=space.unpack(x)
    # ``projected_vector`` is the object to compare with an exact fixed-sector
    # eigenvector.  ``intrinsic_state`` is the symmetry-breaking vacuum before
    # projection and should not be used directly for projected fidelity.
    return VAPResult(energy,a/np.linalg.norm(a),state_from_thouless(z),z,
                     ok,residual,attempts)
