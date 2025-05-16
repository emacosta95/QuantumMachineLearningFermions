#%%
from src.hartree_fock_library import HartreeFockEnergy,FitHartreeFock
from src.hamiltonian_utils import get_twobody_nuclearshell_model,FermiHubbardHamiltonian,SingleParticleState
import numpy as np
import torch
from typing import Dict
from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
#%%

twobody_matrix,energies=get_twobody_nuclearshell_model(file_name='data/cki')

SPS=SingleParticleState(file_name='data/cki')

print(twobody_matrix)

#%% initialize the FH Hamiltonian

FHHamiltonian=FermiHubbardHamiltonian(size_a=energies.shape[0]//2,size_b=energies.shape[0]//2,nparticles_a=2,nparticles_b=2)

FHHamiltonian.get_external_potential(external_potential=energies)
FHHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
FHHamiltonian.get_hamiltonain()

egs,psi0=FHHamiltonian.get_spectrum(n_states=1)

print(egs)


# %% Hartree fock initialization

HFE=HartreeFockEnergy(nparticles=4,size=energies.shape[0],basis=FHHamiltonian.basis)

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


#%%


InitialHamiltonian=FermiHubbardHamiltonian(size_a=energies.shape[0]//2,size_b=energies.shape[0]//2,nparticles_a=2,nparticles_b=2)

kinetic_term:Dict={}

for i in range(energies.shape[0]//2):
    for j in range(energies.shape[0]//2):
        (ni,li,ji,mi,ti,tzi)=SPS.state_encoding[i]
        (nj,lj,jj,mj,tj,tzj)=SPS.state_encoding[j]
        
        t=1.
        if np.isclose(mi,-mj) and np.isclose(ji,jj):
            kinetic_term[(i,j)]=1#np.abs(mi-mj)#+np.abs(ji-jj)
            kinetic_term[(energies.shape[0]//2 +i,energies.shape[0]//2+j)]=1#np.abs(mi-mj)#+np.abs(ji-jj)
            

InitialHamiltonian.get_kinetic_operator(adj_matrix=kinetic_term)
InitialHamiltonian.get_hamiltonain()

egs,psi_initial=InitialHamiltonian.get_spectrum(n_states=1)

# %% Let's start the VQE. Select the operator pool

operator_pool:Dict={}

operator_pool=FHHamiltonian.set_operator_pool(operator_pool=operator_pool,n_new_operators=2000,conditions=[SPS.projection_conservation],nbody='two')

operator_pool=FHHamiltonian.set_operator_pool(operator_pool=operator_pool,n_new_operators=200,conditions=[SPS.projection_conservation],nbody='one')
#%%
print(operator_pool.keys())

# %%
random=False

# AdVQE=AdaptVQEFermiHubbard()

# #initialize psi0

# #idxs=np.array([0,10])




# AdVQE.set_system(hamiltonian=FHHamiltonian.hamiltonian,psi0=psi_hf,operator_pool=operator_pool)

# AdVQE.set_hyperparameters(learning_rate=0.1,tolerance=10**-6)

# AdVQE.optimization()


model=AdaptVQEFermiHubbard()

model.set_hamiltonian(FHHamiltonian.hamiltonian)
model.set_reference_psi(psi_hf)
model.set_operators_pool(operator_pool=operator_pool,random=random)

#%%

fit=Fit(method='L-BFGS-B',tolerance_opt=10**-6)

fit.configuration_checkpoint=configuration
fit.init_model(model)

#%%
fit.run()
# %%
