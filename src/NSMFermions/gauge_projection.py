"""Polynomial-cost number-projected kernels in a vacuum Thouless chart.

No occupation-basis construction. Singular transition matrices are reported,
not regularized silently. Shifted full-period Fourier grids avoid common
overlap zeros but cannot guarantee conditioning for every state.
"""
import numpy as np
from pfapack import pfaffian as pf
from scipy.optimize import minimize


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
    state vector.  That is what keeps its memory polynomial in the number of
    modes.  Use ``NumberProjectedSpace.projected_state`` when an explicit
    small-space vector is required for fidelity or other observables.
    """
    def __init__(self,hamiltonian,neutron_modes,targets,grid=None,offset=.137):
        self.ham=hamiltonian
        m=len(hamiltonian.h)

        # Mark which one-body indices are neutrons.  Every unmarked mode is
        # treated as a proton; the modes need not be stored in species blocks.
        ns=list(neutron_modes)
        if any(not isinstance(i,(int,np.integer)) for i in ns) or len(set(ns))!=len(ns) or any(i<0 or i>=m for i in ns):
            raise ValueError('Invalid neutron indices')
        self.mask=np.zeros(m,bool); self.mask[ns]=True
        caps=[len(ns),m-len(ns)]

        # The two integer targets define P_N P_Z.  An unblocked Thouless vacuum
        # contains only even total number parity, hence the parity guard.
        if len(targets)!=2 or any(not isinstance(t,(int,np.integer)) or t<0 or t>c for t,c in zip(targets,caps)):
            raise ValueError('Invalid integer particle numbers')
        if sum(targets)%2:
            raise ValueError('Even total parity required')
        self.targets=tuple(targets)

        # A species with c modes has sectors 0,...,c, so c+1 equally spaced
        # angles resolve its finite Fourier polynomial exactly.
        self.grid=tuple(c+1 for c in caps) if grid is None else tuple(grid)
        if len(self.grid)!=2 or any(not isinstance(l,(int,np.integer)) or l<=c for l,c in zip(self.grid,caps)):
            raise ValueError('Exact grid requires more points than modes of each species')
        if not np.isfinite(offset):
            raise ValueError('Offset must be finite')
        # A common fractional offset moves the quadrature away from common
        # overlap zeros without changing a complete periodic Fourier sum.
        self.offset=offset

        # Separate N and Z projectors require H to conserve both species.  The
        # labels on the incoming and outgoing legs of every nonzero term must
        # therefore match.
        labels=self.mask.astype(int)
        i,j=np.nonzero(np.abs(hamiltonian.h)>1e-12)
        a,b,c,d=np.nonzero(np.abs(hamiltonian.v)>1e-12)
        if np.any(labels[i]!=labels[j]) or np.any(labels[a]+labels[b]!=labels[c]+labels[d]):
            raise ValueError('Hamiltonian must conserve each species number')

    def energy(self,z):
        """Return <Phi|H P_N P_Z|Phi>/<Phi|P_N P_Z|Phi>.

        ``z`` is the antisymmetric pair matrix of the Thouless vacuum.  The
        double loop below is the discrete U(1)_N x U(1)_Z group integral.
        """
        z=np.asarray(z,complex)
        m=len(self.ham.h)
        if z.shape!=(m,m) or not np.isfinite(z).all() or not np.allclose(z,-z.T,atol=1e-12):
            raise ValueError('Finite antisymmetric Z required')
        # Accumulate Hamiltonian and norm kernels independently.  Their ratio
        # is the energy of the normalized projected state.
        numerator=0j; denominator=0j
        identity=np.eye(m)
        # Intrinsic Thouless norm sqrt(det(I+Z^dagger Z)); slogdet is stable
        # when the determinant spans many orders of magnitude.
        scale=np.exp(.5*np.linalg.slogdet(identity+z.conj().T@z)[1])

        # Discrete Fourier sums over neutron and proton gauge angles.
        for i in range(self.grid[0]):
            pn=2*np.pi*(i+self.offset)/self.grid[0]
            for j in range(self.grid[1]):
                pp=2*np.pi*(j+self.offset)/self.grid[1]
                # Gauge rotation multiplies a creation operator by exp(i phi).
                # Since Z creates pairs, a phase acts on each matrix index.
                phase=np.exp(1j*np.where(self.mask,pn,pp))
                zg=phase[:,None]*z*phase[None,:]

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
                overlap=(-1)**(m*(m+1)//2)*pfaffian(np.block([[zg,-identity],[identity,-z.conj()]]))/scale

                # The Fourier character selects the desired neutron and proton
                # numbers from the gauge-rotated vacuum.
                weight=overlap*np.exp(-1j*(pn*self.targets[0]+pp*self.targets[1]))

                # Generalized-Wick energy kernel: one-body, normal two-body,
                # and anomalous pairing contributions.
                kernel=(np.einsum('ij,ji->',self.ham.h,rho)
                    +.5*np.einsum('ijkl,ki,lj->',self.ham.v,rho,rho)
                    +.25*np.einsum('ijkl,ij,kl->',self.ham.v,creation,kappa))
                denominator+=weight; numerator+=weight*kernel

        # Dividing by the number of grid points gives the actual probability
        # weight of the selected N,Z sector.  It must be positive and real.
        norm=denominator/np.prod(self.grid)
        if abs(norm.imag)>1e-8 or norm.real<1e-14:
            raise ValueError('Vanishing or numerically unresolved projected norm')
        energy=numerator/denominator
        if abs(energy.imag)>1e-7:
            raise ValueError('Projection cancellation error exceeds tolerance')
        return float(energy.real)

    def unpack(self,x):
        """Map real optimizer coordinates to an antisymmetric complex Z."""
        m=len(self.ham.h); ij=np.triu_indices(m,1); p=len(ij[0])
        x=np.asarray(x,float)
        if x.shape!=(2*p,) or not np.isfinite(x).all():
            raise ValueError('Invalid complex Thouless parameters')
        # The first p entries are Re(Z_ij), the remaining p are Im(Z_ij).
        z=np.zeros((m,m),complex); z[ij]=x[:p]+1j*x[p:]
        return z-z.T

    def solve(self,*,seed=0,maxiter=200,tolerance=1e-10,initial_parameters=None):
        """One bounded local VAP run, finite-difference gradients, polynomial
        work per iteration. Returns SciPy OptimizeResult; inspect success and
        jac before using it. For larger spaces analytic gradients are needed.
        """
        m=len(self.ham.h)
        # Use either the supplied continuation point or a reproducible random
        # complex-pairing seed.  There are m(m-1) real coordinates.
        x=(np.random.default_rng(seed).normal(scale=.3,size=m*(m-1))
           if initial_parameters is None else np.asarray(initial_parameters,float))
        # L-BFGS-B finite-differences the projected kernel objective.  This is
        # useful for validation but less scalable than an analytic-gradient VAP.
        return minimize(lambda y:self.energy(self.unpack(y)),x,method='L-BFGS-B',
                        options={'maxiter':maxiter,'ftol':tolerance,'gtol':1e-6})
