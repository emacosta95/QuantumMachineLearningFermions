#%%
from src.hartree_fock_library import HartreeFockEnergy,FitHartreeFock
from src.hamiltonian_utils import get_twobody_nuclearshell_model,FermiHubbardHamiltonian,SingleParticleState
import numpy as np
import torch
from typing import Dict,List
#from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
from src.qml_models import AdaptVQEFermiHubbard
#%%
class SpinConservation():
    
    def __init__(self,size:int):
        
        self.size=size

    def spin_conservation(self,idxs:List):

        l=len(idxs)

        total_initial_spin=0.
        for idx in idxs[:l]:
            
            if idx < self.size:
                total_initial_spin+=1
                
        total_final_spin=0.
        for idx in idxs[l:]:
            
            if idx < self.size:
                total_final_spin+=1
        
        return total_initial_spin==total_final_spin
    
    def local_interaction(self,idxs:List):
        
        condition=idxs[0]==idxs[-1] and idxs[1]==idxs[-2] and idxs[0]+self.size==idxs[1]
        
        if condition:
            print(idxs)
        
        return condition
            

#%% initialize the FH Hamiltonian
size_a=6
FHHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_a,nparticles_a=2,nparticles_b=2)

u=-3.
t=1.
v_ext=np.ones(size_a)

# define the local onsite potential
twobody_matrix:Dict={}
for i in range(size_a):
    twobody_matrix[(i,i+size_a,i+size_a,i)]=u

kinetic_term:Dict={}
for i in range(size_a):
    kinetic_term[(i,(i+1) % size_a)]=t
    kinetic_term[(size_a +i,size_a+(i+1) % size_a)]=t


#%%

FHHamiltonian.get_external_potential(external_potential=np.append(v_ext,v_ext))
FHHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
FHHamiltonian.get_kinetic_operator(adj_matrix=kinetic_term)
FHHamiltonian.get_hamiltonain()


print('TWO BODY OPERATORRR=',FHHamiltonian.twobody_operator)

egs,psi0=FHHamiltonian.get_spectrum(n_states=1)

print(egs)


# %% Hartree fock initialization

HFE=HartreeFockEnergy(nparticles=4,size=2*size_a,basis=FHHamiltonian.basis)

HFE.get_hamiltonian(hamiltonian=FHHamiltonian.hamiltonian)

# %% define the fit class

HFFit=FitHartreeFock(learning_rate=0.1,epochs=500)

HFFit.run(HFE)
# %%
psi_hf=HFE.get_psi().detach().numpy()

print(psi_hf.conjugate().transpose().dot(FHHamiltonian.hamiltonian.dot(psi_hf)))


np.save('psi_hf',psi_hf)

# old initialization works better than Hartree Fock
# min=10000
# for i,b in enumerate(FHHamiltonian.basis):
#     psi=np.zeros(FHHamiltonian.basis.shape[0])
#     psi[i]=1.    
#     value= np.conj(psi) @ FHHamiltonian.hamiltonian @ psi
#     if value<min:
#         min=value
#         psi_hf=psi

# %% Let's start the VQE. Select the operator pool
spinconservation=SpinConservation(size=size_a)
operator_pool:Dict={}

operator_pool=FHHamiltonian.set_operator_pool(operator_pool=operator_pool,conditions=[spinconservation.spin_conservation],nbody='two')

operator_pool=FHHamiltonian.set_operator_pool(operator_pool=operator_pool,conditions=[spinconservation.spin_conservation],nbody='one')
#%%
print('number of operators=',len(list(operator_pool.keys())))

# %%
random=False

model=AdaptVQEFermiHubbard()

model.set_hamiltonian(FHHamiltonian.hamiltonian)
model.set_reference_psi(psi_hf)
model.set_operators_pool(operator_pool=operator_pool,random=random)

#%%

fit=Fit(method='SLSQP',tolerance_opt=10**-7,e_ref=egs)

fit.configuration_checkpoint=configuration
fit.init_model(model)

#%%
fit.run()
# # %%
print(model.operator_action_info)
print(model.energy-egs/egs)
# %%
#AdVQE=AdaptVQEFermiHubbard()

#initialize psi0

#idxs=np.array([0,10])




# AdVQE.set_system(hamiltonian=FHHamiltonian.hamiltonian,psi0=psi_hf,operator_pool=operator_pool)

# AdVQE.set_hyperparameters(learning_rate=0.1,tolerance=10**-6)

# AdVQE.optimization()