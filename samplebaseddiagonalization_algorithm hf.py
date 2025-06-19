from src.hamiltonian_utils import FermiHubbardHamiltonian
from src.nuclear_physics_utils import get_twobody_nuclearshell_model,SingleParticleState,QuadrupoleOperator,J2operator
import numpy as np
import torch
from typing import Dict
import scipy
from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
from src.utils_quasiparticle_approximation import QuasiParticlesConverterOnlynnpp,QuasiParticlesConverter
from scipy.sparse import lil_matrix
file_name='data/usdb.nat'

SPS=SingleParticleState(file_name=file_name)
size_a=SPS.energies.shape[0]//2
size_b=SPS.energies.shape[0]//2

nparts=[(2,0),(4,0),(6,0),(8,0),(2,2),(4,2),(6,2),(8,2),(4,4),(6,4),(8,4),(6,6),(8,6),(10,6),(10,8)]
titles=[r'$^{18}$O',r'$^{20}$O',r'$^{22}$O',r'$^{24}$O',r'$^{20}$Ne',r'$^{22}$Ne',r'$^{24}$Ne',r'$^{26}$Ne',r'$^{24}$Mg',r'$^{26}$Mg',r'$^{28}$Mg',r'$^{28}$Si',r'$^{30}$Si',r'$^{32}$Si',r'$^{32}$Ar']

history=[]
errors=[]
n_instances_max=[]

dimensions=[]

for r,title in enumerate(titles):
    print('title=',title)
    nparticles_a=nparts[r][0]
    nparticles_b=nparts[r][1]
    
    history_sample_r=np.zeros((10**5))
    
    NSMHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
    print('size=',size_a+size_b,size_b)
    NSMHamiltonian.get_external_potential(external_potential=SPS.energies[:size_a+size_b])
    if file_name=='data/cki':
        twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)

        NSMHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
    else:
        NSMHamiltonian.twobody_operator=scipy.sparse.load_npz(f'data/nuclear_twobody_matrix/usdb_{nparticles_a}_{nparticles_b}.npz')
    NSMHamiltonian.get_hamiltonian()

    egs,psi0=NSMHamiltonian.get_spectrum(n_states=1)

    print(egs)

    print('total_m=',SPS.compute_m_exp_value(psi=psi0,basis=NSMHamiltonian.basis))
    #print('j_value=',J2Class.j_value(psi0))
    print('dimension=',NSMHamiltonian.hamiltonian.shape[0])
    dimensions.append(NSMHamiltonian.hamiltonian.shape[0])
    prob=np.conjugate(psi0[:,0])*psi0[:,0]
    
    n_instances=np.arange(2,prob.shape[0])
    
    error=1000
    i=0
    while(error> 10**-2):
        selection=np.random.choice(np.arange(prob.shape[0]),size=n_instances[i],replace=False,p=prob)
        # effective_hamiltonian=lil_matrix((n_instances[i],n_instances[i]))

        # for a,idx_a in enumerate(selection):
        #     for b,idx_b in enumerate(selection):
        #         effective_hamiltonian[a,b]=NSMHamiltonian.hamiltonian[idx_a,idx_b]
        
        effective_hamiltonian=  NSMHamiltonian.hamiltonian.tocsr()[selection,:][:,selection]
        
        effective_egs,effective_psi0=eigsh(effective_hamiltonian,k=1,which='SA')
        error=np.abs(effective_egs[0]-egs[0])/np.abs(egs[0])
        history_sample_r[i]=(error)
        
        i+=1
    history.append(history_sample_r)
    n_instances_max.append(i)
    errors.append(error)
    print('\n')
    print(n_instances[i])
    print('error=',np.abs(effective_egs[0]-egs[0])/np.abs(egs[0]),'\n')    
    
    np.savez('data/samplebaseddiagonalization_data/run_1%',errors=np.asarray(errors),history=np.asarray(history),n_instances=np.asarray(n_instances_max),titles=titles,dimensions=np.asarray(dimensions))