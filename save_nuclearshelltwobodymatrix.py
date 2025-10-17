from src.NSMFermions.nuclear_physics_utils import get_twobody_nuclearshell_model,FermiHubbardHamiltonian,SingleParticleState
import scipy


file_name='data/gxpf1a'

SPS=SingleParticleState(file_name=file_name)




twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)


size_a=energies.shape[0]//2
size_b=size_a
nparticles_a=10
nparticles_b=0

TargetHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
print('size=',size_a+size_b,size_b)
TargetHamiltonian.get_external_potential_optimized(external_potential=energies[:size_a+size_b])
print('get the two body interaction...')
TargetHamiltonian.get_twobody_interaction_optimized(twobody_dict=twobody_matrix)
TargetHamiltonian.get_hamiltonian()

scipy.sparse.save_npz(f'data/nuclear_twobody_matrix/gxpf1a_{nparticles_a}_{nparticles_b}',TargetHamiltonian.twobody_operator)