from itertools import combinations

from .hamiltonian_utils import FermiHubbardHamiltonian
from .nuclear_physics_utils import get_twobody_nuclearshell_model,SingleParticleState,J2operator
import numpy as np
import torch
from typing import Dict
import scipy
from .qml_models import AdaptVQEFermiHubbard
from .qml_utils.train import Fit
from .qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
from typing import Optional,List
from tqdm import tqdm
import itertools
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
from typing import List, Dict, Callable,Optional,Tuple
from scipy.sparse.linalg import  lobpcg
from itertools import product
import multiprocessing
from tqdm import tqdm, trange
from scipy.sparse import identity
class QuasiParticlesConverter():
    
    def __init__(self,):
        pass
    
    def initialize_shell(self,state_encoding:List):
        
        #### nn and pp
        couples=[]
        for a,state_a in enumerate(state_encoding):
            for b,state_b in enumerate(state_encoding):
                
                if b>a:
                    _,_,ja,ma,_,tza=state_a
                    _,_,jb,mb,_,tzb=state_b
                    if ja==jb and ma==-mb and tza==tzb:
                        couples.append([a,b])
                        
        for a,state_a in enumerate(state_encoding):
            for b,state_b in enumerate(state_encoding):
                
                if b>a:
                    _,_,ja,ma,_,tza=state_a
                    _,_,jb,mb,_,tzb=state_b
                    if ja==jb and ma==-mb and tza==-tzb:
                        couples.append([a,b])
            
        self.couples=couples



    def new_base_computation(self,base:np.ndarray):
        
        indices=np.nonzero(base)[0]
        new_base=np.zeros(len(self.couples))
        value=np.sum(base)
    
                
        list_of_token_indices=[]
        
        for i in range(new_base.shape[0]):
            
            if base[self.couples[i][0]]+base[self.couples[i][1]]!=2 :
                continue
            else:
                new_base[i]+=1
                base[self.couples[i][0]]=0
                base[self.couples[i][1]]=0
        
        if np.sum(new_base)==value//2:
            return new_base
        

    def get_the_basis_matrix_transformation(self,basis:np.ndarray):
        
        self.quasiparticle_basis=[]
        self.rest_basis=[]
        
        for i,b in enumerate(basis):
            qp_base=self.new_base_computation(base=b)
            
            if qp_base is not(None):

                self.quasiparticle_basis.append(qp_base)
            else:
                self.rest_basis.append(b)
        self.quasiparticle_basis=np.asarray(self.quasiparticle_basis)
        self.rest_basis=np.asarray(self.rest_basis)
        
        self.particles2quasiparticles=lil_matrix((self.quasiparticle_basis.shape[0],basis.shape[0]))
        self.particles2restofstates=lil_matrix((self.rest_basis.shape[0],basis.shape[0]))
        qp_idx=0
        rest_idx=0
        for i,b in enumerate(basis):
            qp_base=self.new_base_computation(base=b)
            
            if qp_base is not(None):
                self.particles2quasiparticles[qp_idx,i]=1.
                qp_idx+=1
            else:
                self.particles2restofstates[rest_idx,i]=1
                rest_idx+=1
                
                
class QuasiParticlesConverterOnlynnpp():
    
    def __init__(self,):
        pass
    
    def initialize_shell(self,state_encoding:List):
        
        #### nn and pp
        couples=[]
        phases=[]
        for a,state_a in enumerate(state_encoding):
            for b,state_b in enumerate(state_encoding):
                
                if b>a:
                    _,_,ja,ma,_,tza=state_a
                    _,_,jb,mb,_,tzb=state_b
                    if ja==jb and ma==-mb and tza==tzb:
                        couples.append([a,b])
                        phases.append((-1)**(ja-np.abs(ma)))        

        self.couples=couples
        self.phases=phases



    def new_base_computation(self,base:np.ndarray):
        
        indices=np.nonzero(base)[0]
        new_base=np.zeros(len(self.couples))
        value=np.sum(base)
    
                
        list_of_token_indices=[]
        
        for i in range(new_base.shape[0]):
            
            if base[self.couples[i][0]]+base[self.couples[i][1]]!=2 :
                continue
            else:
                new_base[i]+=1
                base[self.couples[i][0]]=0
                base[self.couples[i][1]]=0
        
        if np.sum(new_base)==value//2:
            return new_base
        

    def get_the_basis_matrix_transformation(self,basis:np.ndarray):
        
        self.quasiparticle_basis=[]
        self.rest_basis=[]
        
        for i,b in enumerate(basis):
            qp_base=self.new_base_computation(base=b.copy())
            
            if qp_base is not(None):

                self.quasiparticle_basis.append(qp_base)
            else:
                self.rest_basis.append(b.copy())
        self.quasiparticle_basis=np.asarray(self.quasiparticle_basis)
        self.rest_basis=np.asarray(self.rest_basis)
        
        self.particles2quasiparticles=lil_matrix((self.quasiparticle_basis.shape[0],basis.shape[0]))
        self.particles2restofstates=lil_matrix((self.rest_basis.shape[0],basis.shape[0]))
        qp_idx=0
        rest_idx=0
        for i,b in enumerate(basis):
            qp_base=self.new_base_computation(base=b.copy())
            
            if qp_base is not(None):
                self.particles2quasiparticles[qp_idx,i]=1.
                qp_idx+=1
            else:
                self.particles2restofstates[rest_idx,i]=1
                rest_idx+=1
                
    def phase_state_converter(self,psi:np.ndarray):
        new_psi=np.zeros_like(psi)
        for a,component in enumerate(psi):
            indices=np.nonzero(self.quasiparticle_basis[a])[0]
            phase_a=self.phases[indices]
            phase=np.prod(phase_a)
            new_psi[a]=phase*component
            
        new_psi=new_psi/np.linalg.norm(new_psi)
        return new_psi
            
            

                
class QuasiParticlesConverterWithPhase():
    
    def __init__(self,):
        pass
    
    def initialize_shell(self,state_encoding:List):
        
        #### nn and pp
        couples=[]
        phases=[]
        for a,state_a in enumerate(state_encoding):
            for b,state_b in enumerate(state_encoding):
                
                if b>a:
                    _,_,ja,ma,_,tza=state_a
                    _,_,jb,mb,_,tzb=state_b
                    if ja==jb and ma==-mb and tza==tzb:
                        couples.append([a,b])
                        phases.append((-1)**(ja-np.abs(ma)))        

        self.couples=couples
        self.phases=phases


    def new_base_computation(self,base:np.ndarray):
        
        indices=np.nonzero(base)[0]
        new_base=np.zeros(len(self.couples))
        phase_value=1.
        value=np.sum(base)
    
                
        list_of_token_indices=[]
        for i in range(new_base.shape[0]):
            if base[self.couples[i][0]]+base[self.couples[i][1]]!=2 :
                continue
            else:
                new_base[i]+=1
                base[self.couples[i][0]]=0
                base[self.couples[i][1]]=0
                phase_value*=self.phases[i]
                
        if np.sum(new_base)==value//2:
            return new_base,phase_value
        

    def get_the_basis_matrix_transformation(self,basis:np.ndarray):
        
        self.quasiparticle_basis=[]
        self.rest_basis=[]
        
        for i,b in enumerate(basis):
            items=self.new_base_computation(base=b.copy())
            if items is not(None):
                qp_base,phase_value=items
                self.quasiparticle_basis.append(qp_base)
            else:
                self.rest_basis.append(b.copy())
        self.quasiparticle_basis=np.asarray(self.quasiparticle_basis)
        self.rest_basis=np.asarray(self.rest_basis)
        
        self.particles2quasiparticles=lil_matrix((self.quasiparticle_basis.shape[0],basis.shape[0]))
        self.particles2restofstates=lil_matrix((basis.shape[0],basis.shape[0]))
        self.particles2quasiparticles_full_basis=lil_matrix((basis.shape[0],basis.shape[0]))
        
        qp_idx=0
        rest_idx=0
        for i,b in enumerate(basis):
            items=self.new_base_computation(base=b.copy())
            
            if items is not(None):
                qp_base,phase_value=items
                self.particles2quasiparticles[qp_idx,i]=phase_value
                self.particles2quasiparticles_full_basis[i,i]=phase_value
                qp_idx+=1
        self.particles2restofstates=identity(basis.shape[0])-self.particles2quasiparticles_full_basis

class HardcoreBosonsBasis:

    def __init__(
        self,basis:np.ndarray
    ) -> None:

        self.basis = basis
        
        self.size=basis.shape[-1]
        
        self.nparticles=np.sum(basis[0])
        
        self.encode = self._get_the_encode()

        self.size_a=self.size//2
        self.size_b=self.size//2
   

    

    def adag_a_matrix(self, i: int, j: int) -> np.ndarray:
        operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))    
        for index, psi in enumerate(self.basis):
            new_psi = np.zeros_like(psi)
            if self.basis[index, j] != 0:
                new_basis = self.basis[index].copy()
                new_basis[j] = self.basis[index, j] - 1
                #phase_j = np.sum(new_basis[0:j])
                if new_basis[i] != 1:
                    new_basis[i] = new_basis[i] + 1
                    #phase_i = np.sum(new_basis[0:i])
                    value=(np.einsum('i,ai->a',new_basis,self.basis))
                    value[value!=self.nparticles]=0

                    if  not(np.isclose(np.sum(value),0.)):
                                        new_index = self._get_index(new_basis)
                                        operator[new_index, index] = 1

        return operator

        
    

    def adag_adag_a_a_matrix(self, i1: int, i2: int, j1: int, j2: int) -> np.ndarray:
        operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))
        for idx, psi in enumerate(self.basis):

            if self.basis[idx, j2] != 0:
                new_basis = self.basis[idx].copy()
                new_basis[j2] = self.basis[idx, j2] - 1
                if new_basis[j1] != 0:
                    new_basis[j1] = new_basis[j1] - 1

                    if new_basis[i2] != 1:
                        new_basis[i2] = new_basis[i2] + 1

                        if new_basis[i1] != 1:
                            new_basis[i1] = new_basis[i1] + 1

                            value=(np.einsum('i,ai->a',new_basis,self.basis))
                            value[value!=self.nparticles]=0

                            if  not(np.isclose(np.sum(value),0.)):
                                                new_index = self._get_index(new_basis)
                                                operator[new_index, idx] = 1

        return operator
        
    def three_body_matrix(
        self, i1: int, i2: int, i3: int, j1: int, j2: int, j3: int
    ) -> np.ndarray:
        operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))

        
        for idx, psi in enumerate(self.basis):
            if self.basis[idx, j3] != 0:
                new_basis = self.basis[idx].copy()
                new_basis[j3] = self.basis[idx, j3] - 1
                phase_j3 = np.sum(new_basis[0:j3])
                if new_basis[j2] != 0:
                    new_basis[j2] = new_basis[j2] - 1
                    phase_j2 = np.sum(new_basis[0:j2])
                    if new_basis[j1] != 0:
                        new_basis[j1] = new_basis[j1] - 1
                        phase_j1 = np.sum(new_basis[0:j1])
                        if new_basis[i3] != 1:
                            new_basis[i3] = new_basis[i3] + 1
                            phase_i3 = np.sum(new_basis[0:i3])
                            if new_basis[i2] != 1:
                                new_basis[i2] = new_basis[i2] + 1
                                phase_i2 = np.sum(new_basis[0:i2])
                                if new_basis[i1] != 1:
                                    new_basis[i1] = new_basis[i1] + 1
                                    phase_i1 = np.sum(new_basis[0:i1])

                                    new_index = self._get_index(new_basis)
                                    operator[new_index, idx] = 1

            return operator
        else:
            print(" it does not conserve the number of Particles, Hombre! \n")

    def four_body_matrix(
        self, i1: int, i2: int, i3: int, i4: int, j1: int, j2: int, j3: int, j4: int
    ) -> np.ndarray:
        operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))

        for idx, psi in enumerate(self.basis):
            if self.basis[idx, j4] != 0:
                new_basis = self.basis[idx].copy()
                new_basis[j4] = self.basis[idx, j4] - 1
                phase_j4 = np.sum(new_basis[0:j4])
                if new_basis[j3] != 0:
                    new_basis[j3] = new_basis[j3] - 1
                    phase_j3 = np.sum(new_basis[0:j3])
                    if new_basis[j2] != 0:
                        new_basis[j2] = new_basis[j2] - 1
                        phase_j2 = np.sum(new_basis[0:j2])
                        if new_basis[j1] != 0:
                            new_basis[j1] = new_basis[j1] - 1
                            phase_j1 = np.sum(new_basis[0:j1])
                            if new_basis[i4] != 1:
                                new_basis[i4] = new_basis[i4] + 1
                                phase_i4 = np.sum(new_basis[0:i4])
                                if new_basis[i3] != 1:
                                    new_basis[i3] = new_basis[i3] + 1
                                    phase_i3 = np.sum(new_basis[0:i3])
                                    if new_basis[i2] != 1:
                                        new_basis[i2] = new_basis[i2] + 1
                                        phase_i2 = np.sum(new_basis[0:i2])
                                        if new_basis[i1] != 1:
                                            new_basis[i1] = new_basis[i1] + 1
                                            phase_i1 = np.sum(new_basis[0:i1])

                                            if new_basis in self.basis:
                                                new_index = self._get_index(new_basis)
                                                operator[new_index, idx] = 1
                                            

            return operator
        else:
            print(" it does not conserve the number of Particles, Hombre! \n")

    
    def reduced_state(self, indices: List, psi: np.ndarray):

        sub_dimension = len(indices)
        combinations = product([0, 1], repeat=sub_dimension)
        # Convert each combination into a numpy array
        basis = np.asarray([np.array(combination) for combination in combinations])

        # initialize the reduced density matrix
        density = np.zeros((basis.shape[0], basis.shape[0]))

        for density_index_d, d in enumerate(basis):
            for density_index_b, b in enumerate(basis):

                # compute the value of the reduced state for each main basis element
                value = 0
                # the nonzero check is essential for the algorithm

                # print("state_d=", d)
                # print("state_b=", b)
                for i_s, sigma in enumerate(self.basis):
                    # print("sigma=", sigma)
                    nonzero_check = True
                    for i, basis_element in enumerate(b):
                        if basis_element == 1:
                            a_value = sigma[indices[i]]
                        else:
                            a_value = 1 - sigma[indices[i]]

                        # # print(
                        # #     "a_value d",
                        # #     a_value,
                        # #     "indices=",
                        # #     indices[i],
                        # #     "basis element=",
                        # #     sigma,
                        # #     "rho indices=",
                        # #     d,
                        # #     b,
                        # #     "\n",
                        # # )

                        if a_value == 0:
                            nonzero_check = False
                            break

                    if nonzero_check:
                        for i, basis_element in enumerate(d):
                            if basis_element == 1:
                                a_value = sigma[indices[i]]
                            else:
                                a_value = 1 - sigma[indices[i]]

                            if a_value == 0:
                                nonzero_check = False
                                break

                            # print(
                            #     "a_value d",
                            #     a_value,
                            #     "indices=",
                            #     indices[i],
                            #     "basis element=",
                            #     sigma,
                            #     "rho indices=",
                            #     d,
                            #     b,
                            #     "\n",
                            # )

                    if nonzero_check:
                        value += psi[i_s] * np.conj(psi[i_s])

                # print(value)

                # print(density_index_b, density_index_d)
                density[density_index_d, density_index_b] = value

        return density

    def mutual_info(
        self,
        psi: np.ndarray,
    ):

        mutual_info = np.zeros((self.size_a + self.size_b, self.size_a + self.size_b))

        for i in range(self.size_a + self.size_b):
            for j in range(self.size_a + self.size_b):

                rho_ab = self.reduced_state(indices=[i, j], psi=psi)
                lambd, _ = np.linalg.eigh(rho_ab)
                s_ab = -1 * np.sum(np.log(lambd + 10**-20) * lambd)

                rho_a = self.reduced_state(indices=[i], psi=psi)
                lambd, _ = np.linalg.eigh(rho_a)
                s_a = -1 * np.sum(np.log(lambd + 10**-20) * lambd)

                rho_b = self.reduced_state(indices=[j], psi=psi)
                lambd, _ = np.linalg.eigh(rho_b)
                s_b = -1 * np.sum(np.log(lambd + 10**-20) * lambd)

                if i == j:
                    mutual_info[i, j] = 0.0
                else:
                    mutual_info[i, j] = -s_ab + (s_a + s_b)

        return mutual_info

    def generalized_mutual_info(self,subsets:Tuple[List],psi:np.ndarray):
        
        indices_a,indices_b=subsets
        
        s_a=self.entanglement_entropy(indices=indices_a,psi=psi)
        s_b=self.entanglement_entropy(indices=indices_b,psi=psi)
        s_ab=self.entanglement_entropy(indices=indices_a+indices_b,psi=psi)
        
        mutual_info=-s_ab + (s_a + s_b)
        return mutual_info

    def entanglement_entropy(self,indices:List,psi:np.ndarray):
        
        rho=self.reduced_state(indices=indices,psi=psi)
        lambd, _ = np.linalg.eigh(rho)
        s = -1 * np.sum(np.log(lambd + 10**-20) * lambd)
        return s
    
    def _get_the_encode(self):

        encode = {}
        for i, b in enumerate(self.basis):
            indices = np.nonzero(b)[0]
            encode[tuple(indices)] = i

        return encode

    def _get_index(self, element: np.ndarray):

        indices = np.nonzero(element)[0]
        index = self.encode[tuple(indices)]

        return index

    def charge_computation(self, initial_indices: List, final_indices: List):

        initial_tot_charge = 0
        for idx in initial_indices:
            if idx >= self.size_a:
                initial_tot_charge += 1
        final_tot_charge = 0
        for idx in final_indices:
            if idx >= self.size_a:
                final_tot_charge += 1

        return initial_tot_charge == final_tot_charge



    def set_operator_pool(
        self, operator_pool: Dict, conditions: List[Callable], nbody: str
    ):

        
        for i1 in range(self.size):
            for i2 in range(i1+1, self.size):

                if nbody == "two":
                    for i3 in range(self.size):
                        for i4 in range(i3+1, self.size):
                            

                            idxs = [i1,i2,i3,i4]
                            cond = True
                            for c in conditions:
                                cond = c(idxs) and cond
                            if cond:
                                
                                if (idxs[2],idxs[3],idxs[0],idxs[1]) in operator_pool.keys():
                                    continue
                                else:
                                    op_plus = self.adag_adag_a_a_matrix(
                                        idxs[0], idxs[1], idxs[2], idxs[3]
                                    )
                                    op_minus = self.adag_adag_a_a_matrix(
                                        idxs[3], idxs[2], idxs[1], idxs[0]
                                    )
                                    operator_pool[tuple(idxs)] = op_plus - op_minus

                                #operator_pool[(i2, i1, i3, i4)] = -(op_plus - op_minus)
                                #operator_pool[(i1, i2, i4, i3)] = -(op_plus - op_minus)
                                #operator_pool[(i1, i2, i4, i3)] = op_plus - op_minus
                            else:
                                continue

                if nbody == "one":
                    idxs = [i1, i2]
                    cond = True
                    for c in conditions:
                        cond = c(idxs) and cond

                    if cond:
                        op_plus = self.adag_a_matrix(idxs[0], idxs[1])
                        op_minus = self.adag_a_matrix(idxs[1], idxs[0])

                        operator_pool[tuple(idxs)] = op_plus - op_minus
                    else:
                        continue

        return operator_pool
