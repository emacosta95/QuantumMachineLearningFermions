import argparse
import numpy as np
import scipy
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Dict, List
from scipy.sparse import lil_matrix, sparse
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.optimize import minimize

from src.hamiltonian_utils import FermiHubbardHamiltonian
from src.nuclear_physics_utils import get_twobody_nuclearshell_model, SingleParticleState
from src.hartree_fock_library import HFEnergyFunctionalNuclear
from src.qml_models import CCVQEFermiHubbard
from src.qml_utils.utils import configuration


def parse_args():
    parser = argparse.ArgumentParser(description="CC-VQE Nuclear Shell Model Simulation")
    parser.add_argument('--num_steps_hartree_fock', type=int, default=600, help='Number of Hartree-Fock steps')
    parser.add_argument('--num_parameters_vqe', type=int, default=100, help='Number of VQE parameters')
    parser.add_argument('--nparticles_a', type=int, default=4, help='Number of neutrons')
    parser.add_argument('--nparticles_b', type=int, default=2, help='Number of protons')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Tolerance for optimizer')
    parser.add_argument('--title', type=str, default='22_ne', help='Base title for saving results')
    return parser.parse_args()


def slater_determinants_combined(C_n, C_p, fock_basis):
    F, M = fock_basis.shape
    M_half = M // 2
    N_n = C_n.shape[1]
    N_p = C_p.shape[1]

    psi = torch.zeros(F, dtype=C_n.dtype)

    for i in range(F):
        occ = fock_basis[i]
        occ_n = torch.nonzero(occ[:M_half]).squeeze()
        occ_p = torch.nonzero(occ[M_half:]).squeeze() + M_half
        Cn_sub = C_n[occ_n, :]
        Cp_sub = C_p[occ_p, :]

        if Cn_sub.shape[0] != N_n or Cp_sub.shape[0] != N_p:
            continue

        psi[i] = torch.det(Cn_sub) * torch.det(Cp_sub)
    return psi


def miquel_constrainer(idxs: List[int]):
    if SPS.projection_conservation(idxs=idxs) and NSMHamiltonian.charge_computation(initial_indices=idxs[:2], final_indices=idxs[2:]):
        op = NSMHamiltonian.adag_adag_a_a_matrix(*idxs)
        non_diag_op = np.abs(op - sparse.diags(op.diagonal()))
        return not np.isclose(non_diag_op.sum(), 0.)
    return False


def miquel_constrained_3(idx: List[int]):
    size = 12
    i1, i2, j1, j2 = idx
    return (i1 < size and i2 >= size) or (i2 < size and i1 >= size)


def miquel_constrainer_2(idxs: List[int]):
    j0, tz0 = SPS.state_encoding[idxs[0]][2], SPS.state_encoding[idxs[0]][5]
    j1, tz1 = SPS.state_encoding[idxs[1]][2], SPS.state_encoding[idxs[1]][5]
    j2, tz2 = SPS.state_encoding[idxs[2]][2], SPS.state_encoding[idxs[2]][5]
    j3, tz3 = SPS.state_encoding[idxs[3]][2], SPS.state_encoding[idxs[3]][5]

    j_tot_i = np.arange(abs(j0 - j1), j0 + j1 + 1)
    j_tot_f = np.arange(abs(j2 - j3), j2 + j3 + 1)

    if tz0 == tz1:
        if j0 == j1:
            j_tot_i = [j for j in j_tot_i if j % 2 == 0]
        if j2 == j3:
            j_tot_f = [j for j in j_tot_f if j % 2 == 0]

    return bool(set(j_tot_i) & set(j_tot_f))


def main():
    args = parse_args()
    file_name = 'data/usdb.nat'

    global SPS, NSMHamiltonian

    SPS = SingleParticleState(file_name=file_name)
    size_a = size_b = SPS.energies.shape[0] // 2

    twobody_matrix, _ = get_twobody_nuclearshell_model(file_name=file_name)

    NSMHamiltonian = FermiHubbardHamiltonian(
        size_a=size_a,
        size_b=size_b,
        nparticles_a=args.nparticles_a,
        nparticles_b=args.nparticles_b,
        symmetries=[SPS.total_M_zero]
    )
    NSMHamiltonian.get_external_potential(external_potential=SPS.energies[:size_a + size_b])

    if file_name == 'data/cki':
        twobody_matrix, _ = get_twobody_nuclearshell_model(file_name=file_name)
        NSMHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
    else:
        NSMHamiltonian.twobody_operator = scipy.sparse.load_npz(
            f'data/nuclear_twobody_matrix/usdb_{args.nparticles_a}_{args.nparticles_b}.npz'
        )

    NSMHamiltonian.get_hamiltonian()
    egs, psi0 = NSMHamiltonian.get_spectrum(n_states=1)

    model = HFEnergyFunctionalNuclear(
        h_vec=torch.tensor(SPS.energies, dtype=torch.double),
        V_dict=twobody_matrix,
        num_neutrons=args.nparticles_a,
        num_protons=args.nparticles_b,
        neutron_indices=0,
        proton_indices=size_a
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for step in range(args.num_steps_hartree_fock):
        optimizer.zero_grad()
        energy = model()
        energy.backward()
        optimizer.step()
        if step % 20 == 0 or step == args.num_steps_hartree_fock - 1:
            print(f"Step {step:4d} | Energy = {energy.item():.6f}")

    psi_hf = slater_determinants_combined(model.C_n, model.C_p, torch.tensor(NSMHamiltonian.basis))
    psi_hf = psi_hf.detach().numpy()
    psi_hf /= np.linalg.norm(psi_hf)

    operator_pool: Dict = {}
    operator_pool = NSMHamiltonian.set_operator_pool(
        operator_pool=operator_pool,
        conditions=[SPS.projection_conservation, miquel_constrainer, miquel_constrainer_2],
        nbody='two'
    )

    vs, ops, keys = [], [], []
    for key, op in operator_pool.items():
        if key in twobody_matrix:
            vs.append(twobody_matrix[key])
            ops.append(op)
            keys.append(key)

    vs = np.abs(np.array(vs))
    selection = np.argsort(vs)[-300:][::-1]
    selected_vs = vs[selection]

    plt.hist(vs, bins=30)
    plt.hist(selected_vs, bins=30)
    plt.semilogy()
    plt.show()

    new_operator_pool = {keys[idx]: ops[idx] for idx in selection}

    model = CCVQEFermiHubbard()
    model.set_hamiltonian(NSMHamiltonian.hamiltonian)
    model.set_reference_psi(psi_hf, energy_gs=egs[0])
    model.set_operators_pool(operator_pool=new_operator_pool)

    res = minimize(
        model.forward,
        model.weights,
        method='BFGS',
        jac=model.backward,
        tol=args.tolerance,
        callback=model.callback,
        options={'disp': True},
    )
    model.weights = res.x
    energy = model.forward(model.weights)

    filename = f'data/ccvqe_results/{args.title}_n_parameters_{args.num_parameters_vqe}_n_hartreefock_{args.num_steps_hartree_fock}_tolerance_{args.tolerance}'
    np.savez(filename,
             history=model.history_energy,
             psi0=psi0,
             psi_hf=psi_hf,
             psi_vqe=model.compute_psi(model.weights),
             weights=model.weights,
             energy_exact=egs,
             energy_vqe=energy)


if __name__ == "__main__":
    main()