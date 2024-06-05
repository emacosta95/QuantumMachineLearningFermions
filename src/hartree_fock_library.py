import torch
import torch.nn as nn
from typing import Dict
from tqdm import trange,tqdm
import numpy as np
from scipy import sparse
from math import factorial,sqrt


def gram_schmidt(vectors):
    """ Orthonormalize a set of vectors using the Gram-Schmidt process. """
    orthonormal_vectors = []
    for v in vectors:
        w = v - sum(np.dot(v, u) * u for u in orthonormal_vectors)
        orthonormal_vectors.append(w / np.linalg.norm(w))
    return np.array(orthonormal_vectors)


class HartreeFockEnergy(nn.Module):
    
    def __init__(self,nparticles:int,size:int,basis:np.ndarray) -> None:
        super().__init__()
        
        self.hamiltonian=None
        
        self.initialize_weights(nparticles=nparticles,size=size)
        self.weights.requires_grad_(True)

        self.basis=torch.from_numpy(basis)
        
        self.nparticles=nparticles
        self.size=size
        
        
    def get_hamiltonian(self,hamiltonian:Dict):
        
        if not sparse.isspmatrix_coo(hamiltonian):
            hamiltonian = hamiltonian.tocoo()
        # here put all the checks
        
        indices = torch.tensor([hamiltonian.row, hamiltonian.col], dtype=torch.long)
        values = torch.tensor(hamiltonian.data, dtype=torch.float32)

        # Create a PyTorch sparse tensor
        shape = hamiltonian.shape
        self.hamiltonian = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        
    def get_psi(self,):
        
        hf_psi=torch.zeros(self.basis.shape[0])
        psi_singleparticle=self.weights[0]+1j*self.weights[1]
        
        
        for i,b in enumerate(self.basis):
            
            b_a=b[:self.size//2]
            b_b=b[self.size//2:]
            
            indices_a=torch.nonzero(b_a)[:,0]
            matrix_a=psi_singleparticle[:self.nparticles//2,indices_a]
            value_a=torch.linalg.det(matrix_a)

            indices_b=torch.nonzero(b_b)[:,0]
            matrix_b=psi_singleparticle[self.nparticles//2:,indices_b]
            value_b=torch.linalg.det(matrix_b)


            hf_psi[i]=value_a*value_b
            
        #print('norm=',torch.linalg.norm(hf_psi))
            
        hf_psi=hf_psi/torch.linalg.norm(hf_psi)
        return hf_psi
        
        
    def forward(self,):
        
        hf_psi=self.get_psi()
        h_psi = torch.sparse.mm(self.hamiltonian, hf_psi.unsqueeze(-1))
        energy = torch.matmul(hf_psi.unsqueeze(-1).conj().T, h_psi).sum()
        return energy
    
    def initialize_weights(self,nparticles:int,size:int):
        
        c=np.arange(nparticles)
        position=np.arange(size)
        
        psi0=np.cos(c[:,None]*np.pi*position[None,:]/size)+1j*np.sin(c[:,None]*np.pi*position[None,:]/size)
        
        #psi0=np.random.uniform(size=(nparticles,size))+1j*np.random.uniform(size=(nparticles,size))

        psi_orthonormal=gram_schmidt(psi0)
        self.weights=torch.zeros((2,nparticles,size))
        self.weights[0]=torch.from_numpy(np.real(psi_orthonormal))
        self.weights[1]=torch.from_numpy(np.imag(psi_orthonormal))
        
        
        
    


class FitHartreeFock():
    
    def __init__(self,learning_rate:int=0.01,epochs:int=1000) -> None:
        
        self.learning_rate=learning_rate
        
        self.epochs=epochs
        
    def run(self,model:nn.Module):
        
        tbar=tqdm(range(self.epochs))
        
        model.weights.requires_grad_(True)
        
        energy_history=[]
        for i in tbar:
            
            energy=model()
            energy.backward()
            
            # gradient
            with torch.no_grad():
                grad=model.weights.grad
                model.weights-=self.learning_rate*grad
                model.weights.grad.zero_()
            
            energy_history.append(energy.item())
            tbar.set_description(f'energy={energy.item():.6f}')
            tbar.refresh()
                
        return energy_history
                
                
                         
            
        
    
    
                
            
    