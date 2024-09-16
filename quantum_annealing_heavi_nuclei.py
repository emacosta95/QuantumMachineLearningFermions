from collections import Counter
from src.hamiltonian_utils import get_twobody_nuclearshell_model,FermiHubbardHamiltonian,SingleParticleState
import numpy as np
import torch
from typing import Dict
from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
import scipy

def plot_spectrum(eigenvalues):
    """
    Plot the vertical spectrum of a Hamiltonian, showing the eigenvalues as horizontal lines 
    and indicating their degeneracy.

    Parameters:
    eigenvalues (array-like): Array of eigenvalues of the Hamiltonian.
    """
    # Count the degeneracy of each eigenvalue
    degeneracy = Counter(eigenvalues)

    # Prepare data for plotting
    unique_eigenvalues = list(degeneracy.keys())
    degeneracies = list(degeneracy.values())

    # Plot the spectrum
    plt.figure(figsize=(6, 10))
    for i, (eig, deg) in enumerate(zip(unique_eigenvalues, degeneracies)):
        plt.hlines(eig, i - 0.2 * deg, i + 0.2 * deg, colors='b', linewidth=5)
        plt.text(i, eig, f'{deg}', horizontalalignment='center', verticalalignment='bottom', fontsize=24, color='r')

    # Make the plot fancy
    plt.title('Spectrum of the Hamiltonian', fontsize=16)
    plt.ylabel('Eigenvalue', fontsize=14)
    plt.xlabel('Index (degeneracy indicated by text)', fontsize=14)
    plt.xticks(range(len(unique_eigenvalues)), ['']*len(unique_eigenvalues))  # Remove x-axis ticks
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Show the plot
    plt.show()

file_name='data/usdb.nat'
SPS=SingleParticleState(file_name=file_name)
energies=SPS.energies

size_a=energies.shape[0]//2
size_b=energies.shape[0]//2
nparticles_a=6
nparticles_b=6


twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)

iso_dict={-0.5:'n',0.5:'p'}
values=np.asarray(list(twobody_matrix.values()))
print(np.average(np.abs(values)))
for key in twobody_matrix.keys():
    i,j,k,l=key
    (n,_,ja,ma,_,tza)=SPS.state_encoding[i]
    (n,_,jb,mb,_,tzb)=SPS.state_encoding[j]
    (n,_,jc,mc,_,tzc)=SPS.state_encoding[k]
    (n,_,jd,md,_,tzd)=SPS.state_encoding[l]

    print(ja,ma,iso_dict[tza]+'+'+iso_dict[tzb],jb,mb,'-->',jc,mc,iso_dict[tzc]+'+'+iso_dict[tzd],jd,md)
    print('cross section=',twobody_matrix[key],'\n')
    

average_unit_energy=np.average(np.abs(np.asarray(list(twobody_matrix.values()))))


nlevels=1
TargetHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
print('size=',size_a+size_b,size_b)
TargetHamiltonian.get_external_potential(external_potential=energies[:size_a+size_b])
print('get the two body interaction...')
#TargetHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
TargetHamiltonian.twobody_operator=scipy.sparse.load_npz(f'data/nuclear_twobody_matrix/usdb_{nparticles_a}_{nparticles_b}.npz')
TargetHamiltonian.get_hamiltonian()

print('get the eigenvalue problem...')
egs,psis=TargetHamiltonian.get_spectrum(n_states=1)
psi0=psis[:,0]
print(egs)

#print(TargetHamiltonian.twobody_operator)



Moperator = FermiHubbardHamiltonian(
    size_a=size_a,
    size_b=size_b,
    nparticles_a=nparticles_a,
    nparticles_b=nparticles_b,
    symmetries=[SPS.total_M_zero]
)



InitialHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])

kinetic_term:Dict={}
adj_matrix=np.zeros((size_a+size_b,size_a+size_b))


min_b=np.zeros(size_a+size_b)
min_b[0:5]=1
min_b[size_a:5+size_a]=1
print(min_b)
psi_index=InitialHamiltonian.encode[tuple([0,1,2,3,4,5,size_a,size_a+1,size_a+2,size_a+3,size_a+4,size_a+5])]

min_psi=np.zeros(InitialHamiltonian.basis.shape[0])

min_psi[psi_index]=1

value= value = np.conj(min_psi) @ TargetHamiltonian.hamiltonian @ min_psi


                
external_field=np.zeros(size_a+size_b)


external_field=(value/(nparticles_a+nparticles_b))*min_b

        
    
        
    

    
    #external_field[i] = SPS.energies[i]


InitialHamiltonian.get_external_potential(external_field)
InitialHamiltonian.get_hamiltonian()


es,psis=InitialHamiltonian.get_spectrum(n_states=nlevels)
einitial=es[0]
psi_initial=psis[:,0]


print('Initial Hamiltonian computed... \n')


nstep =200
tf = 20#/average_unit_energy
nlevels=10
time = np.linspace(0.0, tf, nstep)
psi = psi_initial
spectrum = np.zeros((nlevels, nstep))
probabilities=np.zeros((nlevels, nstep))
dt=time[1]-time[0]
eng_t=[]
variance_t=[]
fidelity_t=[]
fidelity_psi0_t=[]
lambd=1-time/tf
#gamma=1/(tf/2)
#lambd=np.exp(-gamma*time)
for i in trange(nstep):

    time_hamiltonian = (
        InitialHamiltonian.hamiltonian * ( lambd[i])
        + TargetHamiltonian.hamiltonian * (1-lambd[i])
    ) #+lambd[i]*(1-lambd[i]) * IntermediateHamiltonian.hamiltonian
    values, psis = eigsh(time_hamiltonian, k=nlevels, which="SA")
    psi=expm_multiply(-1j*dt*time_hamiltonian,psi)

    e_ave=psi.conjugate().transpose()@ time_hamiltonian @ psi
    e_square_ave = (
        psi.conjugate().transpose() @ time_hamiltonian @ time_hamiltonian @ psi
    )
    eng_t.append(e_ave)
    variance_t.append(e_square_ave-e_ave**2)
    spectrum[:, i] = values

    degenerate_fidelity=0.
    count=0
    for j in range(values.shape[0]):
        if np.isclose(values[j],values[0]):
            degenerate_fidelity += (
                psis[:, j].conjugate().transpose() @ psi[:]
            ) * np.conj(psis[:, j].conjugate().transpose() @ psi[:])
            count=count+1
        
        probabilities[j,i]=(
                psis[:, j].conjugate().transpose() @ psi[:]
            ) * np.conj(psis[:, j].conjugate().transpose() @ psi[:])

    fidelity=degenerate_fidelity        
    fidelity_t.append(fidelity)
    fidelity_psi0_t.append((
                psi0.conjugate().transpose() @ psi[:]
            ) * np.conj(psi0.conjugate().transpose() @ psi[:]))

eng_t=np.asarray(eng_t)
fidelity_t=np.asarray(fidelity_t)
fidelity_psi0_t=np.asarray(fidelity_psi0_t)
variance_t=np.asarray(variance_t)


np.savez('data/quantum_annealing_results/silicon_results_quantum_annealing',fidelity=fidelity_t,energy=eng_t,spectrum=spectrum,probabilities=probabilities,egs=egs,fidelity_psi0=fidelity_psi0_t)

# print('tau vs Fidelity \n')

# tfs = np.array([1,2,4,8,16,32,64])/average_unit_energy
# nsteps =10*tfs
# nlevels=2

# #gamma=1/(tf/2)
# #lambd=np.exp(-gamma*time)
# fidelities=[]
# relative_err=[]
# for a in trange(tfs.shape[0]):
#     tf=tfs[a]
#     nstep=int(nsteps[a])
#     time = np.linspace(0.0, tf, nstep)
#     psi = psi_initial
#     dt=time[1]-time[0]
#     lambd=1-time/tf
#     for i in trange(nstep):

#         time_hamiltonian = (
#             InitialHamiltonian.hamiltonian * ( lambd[i])
#             + TargetHamiltonian.hamiltonian * (1-lambd[i])
#         ) #+lambd[i]*(1-lambd[i]) * IntermediateHamiltonian.hamiltonian
#         values, psis = eigsh(time_hamiltonian, k=nlevels, which="SA")
#         psi=expm_multiply(-1j*dt*time_hamiltonian,psi)

#         e_ave=psi.conjugate().transpose()@ time_hamiltonian @ psi
#         e_square_ave = (
#             psi.conjugate().transpose() @ time_hamiltonian @ time_hamiltonian @ psi
#         )
    
#     degenerate_fidelity=0.
#     count=0
#     for j in range(values.shape[0]):
#         if np.isclose(values[j],values[0]):
#             degenerate_fidelity += (
#                 psis[:, j].conjugate().transpose() @ psi[:]
#             ) * np.conj(psis[:, j].conjugate().transpose() @ psi[:])
#             count=count+1

#     print('fidelity=',degenerate_fidelity,'relative energy error=',e_ave,'\n')
#     fidelities.append(degenerate_fidelity)
#     relative_err.append(np.abs((egs-e_ave)/egs))    


# fidelities=np.asarray(fidelities)
# relative_err=np.asarray(relative_err)

# np.savez('data/quantum_annealing_results/magnesium_qa_results_fidelity_tau',fidelity=fidelities,tau=tfs,relative_error=relative_err)
