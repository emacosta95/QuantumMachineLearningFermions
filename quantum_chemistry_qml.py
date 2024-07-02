from src.hartree_fock_library import HartreeFock,HartreeFockVariational
from src.hamiltonian_utils import get_twobody_nuclearshell_model,FermiHubbardHamiltonian,SingleParticleState
import numpy as np
import torch
from typing import Dict,List
from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse




class SpinConservation():
    
    def __init__(self,size:int):
        
        self.size=size

    def spin_conservation(self,idxs:List):

        l=len(idxs)

        total_initial_spin=0.
        for idx in idxs[:l//2]:
            
            if idx < self.size:
                total_initial_spin+=1
                
        total_final_spin=0.
        for idx in idxs[l//2:]:
            
            if idx < self.size:
                total_final_spin+=1
        return total_initial_spin==total_final_spin
    
    def local_interaction(self,idxs:List):

        condition=idxs[0]==idxs[-2] and idxs[1]==idxs[-1] and (idxs[0]+self.size==idxs[1] or idxs[1]+self.size==idxs[0])
        
        return condition
    
t=1.
    
us=np.linspace(0,4,8)

size_a=4
size_b=4
nparticles_a=2
nparticles_b=2

v_ext=np.random.uniform(size=size_a)


FHHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_a,nparticles_a=nparticles_a,nparticles_b=nparticles_b)

#operator pool
spinconservation=SpinConservation(size=size_a)
operator_pool:Dict={}

operator_pool=FHHamiltonian.set_operator_pool(operator_pool=operator_pool,conditions=[spinconservation.spin_conservation],nbody='two')
operator_pool=FHHamiltonian.set_operator_pool(operator_pool=operator_pool,conditions=[spinconservation.spin_conservation],nbody='one')

kinetic_term:Dict={}
adj_matrix=np.zeros((size_a+size_b,size_a+size_b))
linear_dimension=int(np.sqrt(size_a))
for i in range(linear_dimension):
    for j in range(linear_dimension):
        
        kinetic_term[(i+j*linear_dimension,(i+1) % linear_dimension + linear_dimension*((j) % linear_dimension))]=t
        kinetic_term[(i+j*linear_dimension,(i) % linear_dimension + linear_dimension*((j+1) % linear_dimension))]=t
        
        kinetic_term[((i+1) % linear_dimension + linear_dimension*((j) % linear_dimension),i+j*linear_dimension)]=t
        kinetic_term[((i) % linear_dimension + linear_dimension*((j+1) % linear_dimension),i+j*linear_dimension)]=t
        
        
        kinetic_term[(size_a +i+j*linear_dimension,size_a+(i) % linear_dimension + linear_dimension*((j+1) % linear_dimension))]=t
        kinetic_term[(size_a +i+j*linear_dimension,size_a+(i+1) % linear_dimension + linear_dimension*((j) % linear_dimension))]=t
        
        kinetic_term[(size_a+(i+1) % linear_dimension + linear_dimension*((j) % linear_dimension),size_a+ i+j*linear_dimension)]=t
        kinetic_term[(size_a+(i) % linear_dimension + linear_dimension*((j+1) % linear_dimension),size_a+ i+j*linear_dimension)]=t
        

        # adj matrix for HF
        adj_matrix[(i+j*linear_dimension),(i) % linear_dimension + linear_dimension*((j+1) % linear_dimension)]=t
        adj_matrix[(i+j*linear_dimension),(i+1) % linear_dimension + linear_dimension*((j) % linear_dimension)]=t
        
        adj_matrix[(i+1) % linear_dimension + linear_dimension*((j) % linear_dimension),(i+j*linear_dimension)]=t
        adj_matrix[(i) % linear_dimension + linear_dimension*((j+1) % linear_dimension),(i+j*linear_dimension)]=t
        
        adj_matrix[i+j*linear_dimension,i+j*linear_dimension]=-2*t
        adj_matrix[i+j*linear_dimension+size_a,i+j*linear_dimension+size_a]=-2*t
        
        adj_matrix[i+j*linear_dimension+size_a,size_a+ (i) % linear_dimension + linear_dimension*((j+1) % linear_dimension)] = t
        adj_matrix[i+j*linear_dimension+size_a,size_a+ (i+1) % linear_dimension + linear_dimension*((j) % linear_dimension)] = t
        
        adj_matrix[size_a+(i+1) % linear_dimension + linear_dimension*((j) % linear_dimension),size_a+ i+j*linear_dimension] = t
        adj_matrix[size_a+(i) % linear_dimension + linear_dimension*((j+1) % linear_dimension),size_a+ i+j*linear_dimension] = t



for r in us:

    u=4.*r
    
    # define the local onsite potential
    twobody_matrix:Dict={}
    for i in range(size_a):
        twobody_matrix[(i,i+size_a,i+size_a,i)]=u
        twobody_matrix[(i, i + size_a, i, i + size_a)] = -u
        twobody_matrix[( i + size_a,i , i + size_a,i)] = -u
        twobody_matrix[(i+size_a, i , i , i+size_a)] = u

    # %%

    FHHamiltonian.get_external_potential(external_potential=np.append(v_ext,v_ext))
    FHHamiltonian.get_kinetic_operator(adj_matrix=kinetic_term)
    FHHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
    FHHamiltonian.get_hamiltonian()

    # print(FHHamiltonian.hamiltonian)
    # print(FHHamiltonian.twobody_operator)
    # print(FHHamiltonian.kinetic_operator+FHHamiltonian.twobody_operator-FHHamiltonian.hamiltonian)
    egs,psi0=FHHamiltonian.get_spectrum(n_states=1)

    print(egs)


    # # %% define the fit class

    # HFFit = FitHartreeFock(learning_rate=0.1, epochs=200)

    # history_hf = HFFit.run(HFE)
    # # %%
    # psi_hf = HFE.get_psi().detach().numpy()

    # print(psi_hf.conjugate().transpose().dot(FHHamiltonian.hamiltonian.dot(psi_hf)))

    # %% Hartree fock initialization
    HFclass = HartreeFock(size=size_a, nspecies=2)

    HFclass.get_hamiltonian(twobody_interaction=twobody_matrix, kinetic_term=adj_matrix,external_potential=np.append(v_ext,v_ext))

    de, history_herm, ortho_history = HFclass.selfconsistent_computation(eta=1, epochs=50)


    #%%
    print('number of operators=',len(list(operator_pool.keys())))

    psi_hf=HFclass.create_hf_psi(FHHamiltonian.basis,nparticles=nparticles_a+nparticles_b)
    print(psi_hf.conjugate().transpose() @ FHHamiltonian.hamiltonian @ psi_hf)

    # %%
    random=False

    model=AdaptVQEFermiHubbard()

    model.set_hamiltonian(FHHamiltonian.hamiltonian)
    model.set_reference_psi(psi_hf)

    model.set_operators_pool(operator_pool=operator_pool,random=random)

    #%%


    fit=Fit(method='BFGS',tolerance_opt=10**-5,e_ref=egs)

    #fit.configuration_checkpoint=configuration
    fit.configuration_checkpoint=None
    fit.init_model(model)

    #%%
    history,grad_history=fit.run()
    # # %%
    print('results=',(model.energy-egs)/egs,'\n')

    np.savez(f'data/quantum_chemistry/adaptive_vqe_results_tol_1e-05_t_{t}_u_{r:.2f}_size_{size_a}_nparticles_{nparticles_a}_{nparticles_b}',history=history,history_grad=grad_history,weights=model.weights,psi=model.psi,energy=model.energy,operators_info=model.operator_action_info,operator_action=model.operator_action,exact_energy=egs,exact_psi=psi0)