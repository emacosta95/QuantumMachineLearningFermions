"""Polynomial-cost number-projected kernels in a vacuum Thouless chart.

No occupation-basis construction. Singular transition matrices are reported,
not regularized silently. Shifted full-period Fourier grids avoid common
overlap zeros but cannot guarantee conditioning for every state.
"""
import numpy as np
from scipy.optimize import minimize


def pfaffian(matrix):
    """Complex Pfaffian via pivoted antisymmetric elimination, O(n^3)."""
    a=np.array(matrix,complex,copy=True)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)%2:
        raise ValueError('Pfaffian requires even square dimension')
    if not np.isfinite(a).all() or not np.allclose(a,-a.T,atol=1e-10):
        raise ValueError('Pfaffian requires finite antisymmetric matrix')
    value=1.+0j
    for k in range(0,len(a)-1,2):
        pivot=k+1+int(np.argmax(np.abs(a[k,k+1:])))
        if a[k,pivot]==0:
            return 0j
        if pivot!=k+1:
            a[[k+1,pivot],:]=a[[pivot,k+1],:]
            a[:,[k+1,pivot]]=a[:,[pivot,k+1]]
            value=-value
        t=a[k,k+1]
        value*=t
        u=a[k,k+2:].copy(); v=a[k+1,k+2:].copy()
        a[k+2:,k+2:] += (np.outer(v,u)-np.outer(u,v))/t
    return value


class GaugeProjectedEnergy:
    """Exact full-period N,Z Fourier projection for finite mode spaces.

    Default L_n=m_n+1, L_p=m_p+1 resolves all particle-number sectors, even
    with np mixing and odd species parities. Cost per energy evaluation is
    O(L_n L_p (m^4+m^3)); dense interaction storage is O(m^4). Fixed undersized
    grids are rejected. This says nothing about global optimization complexity.
    """
    def __init__(self,hamiltonian,neutron_modes,targets,grid=None,offset=.137):
        self.ham=hamiltonian
        m=len(hamiltonian.h)
        ns=list(neutron_modes)
        if any(not isinstance(i,(int,np.integer)) for i in ns) or len(set(ns))!=len(ns) or any(i<0 or i>=m for i in ns):
            raise ValueError('Invalid neutron indices')
        self.mask=np.zeros(m,bool); self.mask[ns]=True
        caps=[len(ns),m-len(ns)]
        if len(targets)!=2 or any(not isinstance(t,(int,np.integer)) or t<0 or t>c for t,c in zip(targets,caps)):
            raise ValueError('Invalid integer particle numbers')
        if sum(targets)%2:
            raise ValueError('Even total parity required')
        self.targets=tuple(targets)
        self.grid=tuple(c+1 for c in caps) if grid is None else tuple(grid)
        if len(self.grid)!=2 or any(not isinstance(l,(int,np.integer)) or l<=c for l,c in zip(self.grid,caps)):
            raise ValueError('Exact grid requires more points than modes of each species')
        if not np.isfinite(offset):
            raise ValueError('Offset must be finite')
        self.offset=offset
        labels=self.mask.astype(int)
        i,j=np.nonzero(np.abs(hamiltonian.h)>1e-12)
        a,b,c,d=np.nonzero(np.abs(hamiltonian.v)>1e-12)
        if np.any(labels[i]!=labels[j]) or np.any(labels[a]+labels[b]!=labels[c]+labels[d]):
            raise ValueError('Hamiltonian must conserve each species number')

    def energy(self,z):
        z=np.asarray(z,complex)
        m=len(self.ham.h)
        if z.shape!=(m,m) or not np.isfinite(z).all() or not np.allclose(z,-z.T,atol=1e-12):
            raise ValueError('Finite antisymmetric Z required')
        numerator=0j; denominator=0j
        identity=np.eye(m)
        scale=np.exp(.5*np.linalg.slogdet(identity+z.conj().T@z)[1])
        for i in range(self.grid[0]):
            pn=2*np.pi*(i+self.offset)/self.grid[0]
            for j in range(self.grid[1]):
                pp=2*np.pi*(j+self.offset)/self.grid[1]
                phase=np.exp(1j*np.where(self.mask,pn,pp))
                zg=phase[:,None]*z*phase[None,:]
                b=identity+z.conj().T@zg
                if np.linalg.cond(b)>1e12:
                    raise ValueError('Singular gauge kernel; change grid offset or chart')
                inverse=np.linalg.solve(b,identity)
                rho=zg@inverse@z.conj().T
                kappa=zg@inverse
                creation=-inverse@z.conj().T
                overlap=(-1)**(m*(m+1)//2)*pfaffian(np.block([[zg,-identity],[identity,-z.conj()]]))/scale
                weight=overlap*np.exp(-1j*(pn*self.targets[0]+pp*self.targets[1]))
                kernel=(np.einsum('ij,ji->',self.ham.h,rho)
                    +.5*np.einsum('ijkl,ki,lj->',self.ham.v,rho,rho)
                    +.25*np.einsum('ijkl,ij,kl->',self.ham.v,creation,kappa))
                denominator+=weight; numerator+=weight*kernel
        norm=denominator/np.prod(self.grid)
        if abs(norm.imag)>1e-8 or norm.real<1e-14:
            raise ValueError('Vanishing or numerically unresolved projected norm')
        energy=numerator/denominator
        if abs(energy.imag)>1e-7:
            raise ValueError('Projection cancellation error exceeds tolerance')
        return float(energy.real)

    def unpack(self,x):
        m=len(self.ham.h); ij=np.triu_indices(m,1); p=len(ij[0])
        x=np.asarray(x,float)
        if x.shape!=(2*p,) or not np.isfinite(x).all():
            raise ValueError('Invalid complex Thouless parameters')
        z=np.zeros((m,m),complex); z[ij]=x[:p]+1j*x[p:]
        return z-z.T

    def solve(self,*,seed=0,maxiter=200,tolerance=1e-10,initial_parameters=None):
        """One bounded local VAP run, finite-difference gradients, polynomial
        work per iteration. Returns SciPy OptimizeResult; inspect success and
        jac before using it. For larger spaces analytic gradients are needed.
        """
        m=len(self.ham.h)
        x=(np.random.default_rng(seed).normal(scale=.3,size=m*(m-1))
           if initial_parameters is None else np.asarray(initial_parameters,float))
        return minimize(lambda y:self.energy(self.unpack(y)),x,method='L-BFGS-B',
                        options={'maxiter':maxiter,'ftol':tolerance,'gtol':1e-6})
