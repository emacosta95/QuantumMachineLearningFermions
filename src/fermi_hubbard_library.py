import itertools
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
from typing import List
from scipy.sparse.linalg import eigsh,lobpcg


def single_particle_potential(ndim:int,nparticles:int,size:float,ngrid:int,peaks:List,amplitudes:List,deviations:List):
        
        if ndim==2 and nparticles==2:
            x,y=np.mgrid[0:size:ngrid*1j,0:size:ngrid*1j]
            v:np.ndarray=0.     
            for i,peak in enumerate(peaks):
                gaussian=-1*amplitudes[i]*np.exp(-1*( ((size/2-x)**2)+((size/2-y)**2))/deviations[i])
                shift_mean_x=int((ngrid*(peak[0])/size))
                shift_mean_y=int((ngrid*(peak[1])/size))
                gaussian=np.roll(gaussian,shift=(shift_mean_x,shift_mean_y),axis=(0,1))
                v=v+gaussian
            
        return v


def coulomb_function(idx:int,jdx:int,size:float,ngrid:int,ndim:int):
    dx=size/ngrid
    
    if ndim==2:
        i1=idx % ngrid
        j1=idx // ngrid
        
        i2=jdx % ngrid
        j2=jdx // ngrid
        
        boxes=[-1,0,1]
        v=0.
        for b1x in boxes:
            for b2x in boxes:
                for b1y in boxes:
                    for b2y in boxes:
                        r=dx*np.sqrt(1+((i1+b1x*ngrid)-(i2+b2x*ngrid))**2+((j1+b1y*ngrid)-(j2+b2y*ngrid))**2)
                        v=v+0.5*1/r
                
    return v


class FemionicBasis:

    def __init__(self,size:float,ngrid:int,dim:int,nparticles) -> None:
        
        self.dx=size/ngrid
        self.size=size
        self.ngrid=ngrid
        self.dim=dim
        self.nparticles=nparticles
        self.basis=self.generate_fermi_hubbard_basis()
        
    def generate_fermi_hubbard_basis(self):
        """
        Generate the basis states for the Fermi-Hubbard model with a given number
        of lattice sites (L) and particles (N_particles).

        Parameters:
        - L: Number of lattice sites
        - N_particles: Number of particles

        Returns:
        - basis_states: List of basis states
        """

        basis_states = []
        state0=np.zeros(self.ngrid**self.dim)
        # Generate all possible combinations of particle positions
        particle_positions = list(itertools.combinations(range(self.ngrid**self.dim), self.nparticles))
        # Combine particle positions and empty sites to form basis states
        for tuple in particle_positions:
            state=state0.copy()
            for i in tuple:
                state[i]=1
            basis_states.append(state)
            
        return np.asarray(basis_states)
    
    def adag_a(self,i: int, j: int) -> np.ndarray:
        operator=lil_matrix((self.basis.shape[0],self.basis.shape[0]))
        for index,psi in enumerate(self.basis):
            new_psi = np.zeros_like(psi)
            if self.basis[index, j] != 0:
                new_basis = self.basis[index].copy()
                new_basis[j] = self.basis[index, j] - 1
                if new_basis[i] != 1:
                    new_basis[i] = new_basis[i] + 1
                    new_index = self._get_index(new_basis)
                    operator[index,new_index]=1

        return operator
    
    
    def adag_adag_a_a(self, i1: int, i2: int, j1: int, j2: int) -> np.ndarray:
        operator=lil_matrix((self.basis.shape[0],self.basis.shape[0]))
        for idx,psi in enumerate(self.basis):
            new_psi = np.zeros_like(psi)
            
            if self.basis[idx, j2] != 0:
                new_basis = self.basis[idx].copy()
                new_basis[j2] = self.basis[idx, j2] - 1
                if new_basis[j1] != 0:
                    new_basis[j1] = new_basis[j1] - 1
                    if new_basis[i2] != 1:
                        new_basis[i2] = new_basis[i2] + 1
                        if new_basis[i1] != 1:
                            new_basis[i1] = new_basis[i1] + 1

                            new_index = self._get_index(new_basis)
                            operator[idx,new_index]=1

        return operator
            
    def _get_kinetic_operator(self):
        cost=-0.5*(self.ngrid/self.size)**2
        operator=lil_matrix((self.basis.shape[0],self.basis.shape[0]))
        for idx,sigma in enumerate(self.basis):
        
            operator[idx,idx]=-2*cost
            new_sigma_y=np.zeros_like(sigma)
            new_sigma_x=np.zeros_like(sigma)
            for index in np.nonzero(sigma)[0]:
                #y direction
                x=index % self.ngrid
                y= index // self.ngrid
                new_sigma_y[x+((y+1)% self.ngrid)*self.ngrid]=1
                new_sigma_x[(x+1)%self.ngrid+(y)*self.ngrid]=1
                
            idx_x=self._get_index(new_sigma_x)
            idx_y=self._get_index(new_sigma_y)
            operator[idx,idx_x]=cost
            operator[idx,idx_y]=cost
            
        return 0.5*(operator+operator.T) 
    
    def _get_coulomb_operator(self):
        
        operator=lil_matrix((self.basis.shape[0],self.basis.shape[0]))
        for idx, sigma in enumerate(self.basis):
            two_elements_list=list(combinations(np.nonzero(sigma)[0], r=2))
            value=0.
            for pair in two_elements_list:
                idx1,idx2=pair
                value=value+coulomb_function(idx1,idx2,size=self.size,ngrid=self.ngrid,ndim=self.dim)
            operator[idx,idx]=value
            
        return operator
    
    def get_single_particle_operator(self,v:np.ndarray):
        operator=lil_matrix((self.basis.shape[0],self.basis.shape[0]))
        for index,sigma in enumerate(self.basis):
            v_value=0.
            for idx in np.nonzero(sigma)[0]:
                x=idx % self.ngrid
                y= idx// self.ngrid
                v_value=v_value+v[x,y]
            operator[index,index]=v_value
            
        return operator
            
                
                     
    
    def _get_index(self,element:np.ndarray):

        index=np.where((self.basis == element).all(axis=1))[0][0]
            
        return index
    
    def compute_density(self,psi:np.ndarray):
        psi=psi/(np.sum(psi*np.conj(psi))*self.dx**self.dim)
        density=np.einsum('s,si->i',psi*np.conj(psi),self.basis)*(self.size/self.ngrid)**self.dim
        
            
        matrix=density.reshape(self.ngrid,self.ngrid).transpose()
        return matrix
    
    def coarse_grained_initialization(self,psi0:np.ndarray,scale:int,basis0:np.ndarray):
        psi=np.zeros_like(self.basis[:,0])
        for index,sigma in enumerate(self.basis):
            sigma_big=np.zeros((self.ngrid//scale)**self.dim)
            for idx in np.nonzero(sigma)[0]:
                x=idx % self.ngrid
                y= idx // self.ngrid
                x_big=x//scale
                y_big=y//scale
                sigma_big[x_big+(self.ngrid//scale)*y_big]=1
            
            if np.sum(sigma_big)==2:
                index_big=np.where((basis0 == sigma_big).all(axis=1))[0][0]
                psi[index]=psi0[index_big]
                
        return psi