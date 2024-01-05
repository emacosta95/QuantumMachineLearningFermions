import itertools
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
from typing import List
from scipy.sparse.linalg import eigsh,lobpcg
from src.fermi_hubbard_library import FemionicBasis,single_particle_potential,coulomb_function
import matplotlib.pyplot as plt
import time

import os

# Set the number of CPU threads
os.environ["OMP_NUM_THREADS"] = "10"  # Set to the number of threads you want
os.environ["MKL_NUM_THREADS"] = "10"


start_time=time.time()

ngrid=10
ndim=2
size=4
nparticles = 2

BasisClass= FemionicBasis(size=size,ngrid=ngrid,dim=ndim,nparticles=nparticles)

amplitudes=np.random.uniform(0.,10.,size=3)
deviations=np.random.uniform(0.3,0.6,size=3)
peaks=np.random.uniform(-2,2,size=(3,2))

v0=single_particle_potential(ndim=ndim,nparticles=nparticles,size=size,ngrid=ngrid,peaks=peaks,amplitudes=amplitudes,deviations=deviations)

# plt.imshow(v0)
# plt.colorbar()
# plt.show()

f0_operator=BasisClass._get_coulomb_operator()+BasisClass._get_kinetic_operator()

v0_operator=BasisClass.get_single_particle_operator(v0)

hamiltonian0=f0_operator+v0_operator
e,eigvector=eigsh(hamiltonian0,k=1,which='SA')
psi0=eigvector[:,0]
density0=BasisClass.compute_density(psi=psi0)
print(np.sum(density0)*(BasisClass.dx)**2)
# plt.imshow(density0)
# plt.colorbar()
# plt.show()
print(e)
basis0=BasisClass.basis.copy()

end_time=time.time()

print(end_time-start_time)

start_time=time.time()

#%% larger size with the ansatz
ngrid=20

# Set the number of CPU threads
os.environ["OMP_NUM_THREADS"] = "10"  # Set to the number of threads you want
os.environ["MKL_NUM_THREADS"] = "10"

BasisClass= FemionicBasis(size=size,ngrid=ngrid,dim=ndim,nparticles=nparticles)

v=single_particle_potential(ndim=ndim,nparticles=nparticles,size=size,ngrid=ngrid,peaks=peaks,amplitudes=amplitudes,deviations=deviations)



f_operator0=BasisClass._get_coulomb_operator()+BasisClass._get_kinetic_operator()
v_operator0=BasisClass.get_single_particle_operator(v)

hamiltonian0=f_operator0+v_operator0

psi_init=BasisClass.coarse_grained_initialization(psi0=psi0,scale=2,basis0=basis0)
print(psi_init.shape)
density_init=BasisClass.compute_density(psi=psi_init)
# plt.imshow(density_init)
# plt.show()

solution=lobpcg(hamiltonian0,X=psi_init.reshape(-1,1),largest=False,maxiter=20,verbosityLevel=1)

energy=solution[0]
psi=solution[1][:,0]

density=BasisClass.compute_density(psi=psi)

end_time=time.time()

print(end_time-start_time)

