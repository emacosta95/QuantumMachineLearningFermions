import argparse
import numpy as np
import torch
import torch.optim as optim
import scipy
from typing import Dict
from src.hamiltonian_utils import FermiHubbardHamiltonian
from src.utils_quasiparticle_approximation import HardcoreBosonsBasis
from src.nuclear_physics_utils import get_twobody_nuclearshell_model, SingleParticleState
from src.qml_models import AdaptVQEFermiHubbard, NSMconstrains
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
from src.sdg_utils import NSM_SQD_circuit_ansatz
from src.fermi_hubbard_library import FemionicBasis
from src.hartree_fock_library import (
    HFEnergyFunctional,
    HFEnergyFunctionalNuclear,
    slater_determinants_combined,
    slater_determinants_only_neutrons,
)


def main(args):
    # Load single-particle states
    SPS = SingleParticleState(file_name=args.file_name)

    size_a = SPS.energies.shape[0] // 2
    size_b = SPS.energies.shape[0] // 2

    print(f"System: nparticles_a={args.nparticles_a}, nparticles_b={args.nparticles_b}")

    # Build Hamiltonian
    NSMHamiltonian = FermiHubbardHamiltonian(
        size_a=size_a,
        size_b=size_b,
        nparticles_a=args.nparticles_a,
        nparticles_b=args.nparticles_b,
        symmetries=[SPS.total_M_zero],
    )

    print("size=", size_a + size_b, size_b)
    NSMHamiltonian.get_external_potential(external_potential=SPS.energies[: size_a + size_b])

    if args.file_name == "data/cki":
        twobody_matrix, energies = get_twobody_nuclearshell_model(file_name=args.file_name)
        NSMHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
    else:
        NSMHamiltonian.twobody_operator = scipy.sparse.load_npz(
            f"data/nuclear_twobody_matrix/usdb_{args.nparticles_a}_{args.nparticles_b}.npz"
        )

    NSMHamiltonian.get_hamiltonian()

    # Spectrum
    egs, psi0 = NSMHamiltonian.get_spectrum(n_states=1)
    psi0 = psi0.reshape(-1)
    print("Ground state energy:", egs)
    print("total_m=", SPS.compute_m_exp_value(psi=psi0, basis=NSMHamiltonian.basis))
    print("dimension=", NSMHamiltonian.hamiltonian.shape[0])

    twobody_matrix, energies = get_twobody_nuclearshell_model(file_name=args.file_name)

    # Hartree-Fock model
    model = HFEnergyFunctionalNuclear(
        h_vec=torch.tensor(SPS.energies, dtype=torch.double),
        V_dict=twobody_matrix,
        num_neutrons=args.nparticles_a,
        num_protons=args.nparticles_b,
        neutron_indices=0,
        proton_indices=size_a,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for step in range(args.num_steps):
        optimizer.zero_grad()
        energy = model()
        energy.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == args.num_steps - 1:
            print(f"Step {step:4d} | Energy = {energy.item():.6f}")

    # Construct Hartree-Fock state
    if args.nparticles_b == 0:
        psi_hf = slater_determinants_only_neutrons(torch.tensor(NSMHamiltonian.basis))
    else:
        psi_hf = slater_determinants_combined(
            model.C_n, model.C_p, torch.tensor(NSMHamiltonian.basis)
        )

    psi_hf = psi_hf.detach().numpy()
    psi_hf = psi_hf / np.linalg.norm(psi_hf)

    # Operator pool
    Constrains = NSMconstrains(SPS, NSMHamiltonian=NSMHamiltonian)
    operator_pool: Dict = {}
    operator_pool = NSMHamiltonian.set_operator_pool(
        operator_pool=operator_pool,
        conditions=[
            SPS.projection_conservation,
            Constrains.miquel_constrainer_2,
            Constrains.miquel_constrainer,
            Constrains.miquel_constrainer_3,
        ],
        nbody="two",
    )
    print("number of operators in the pool=", len(operator_pool))

    # Variational quantum eigensolver
    m = NSM_SQD_circuit_ansatz(
        samples=args.samples,
        train_steps=args.train_steps,
        num_parameters=args.num_parameters,
        batches=args.batches,
        twobody_matrix=twobody_matrix,
        NSMHamiltonian=NSMHamiltonian,
    )
    m.set_save_file_name(args.save_file_name)
    m.initialize_state_hartreefock(psi_hf)
    m.create_operator_pool_twobody_symmetry(operator_pool)
    m.optimization()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDG Nuclear Shell Model Simulation")

    parser.add_argument("--file_name", type=str, default="data/usdb.nat", help="Input file for single-particle states")
    parser.add_argument("--save_file_name", type=str, default="data/sdg/usdb_8_2_200_samples_10_steps_300_parameters_1000_batches", help="Input save file for the run")
    parser.add_argument("--nparticles_a", type=int, default=8, help="Number of neutrons")
    parser.add_argument("--nparticles_b", type=int, default=2, help="Number of protons")
    parser.add_argument("--num_steps", type=int, default=600, help="Number of Hartree-Fock optimization steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument("--log_interval", type=int, default=20, help="Logging interval for optimization")

    # VQE parameters
    parser.add_argument("--samples", type=int, default=200, help="Number of samples in ansatz")
    parser.add_argument("--train_steps", type=int, default=10, help="Training steps for circuit optimization")
    parser.add_argument("--num_parameters", type=int, default=300, help="Number of variational parameters")
    parser.add_argument("--batches", type=int, default=1000, help="Number of batches for training")

    args = parser.parse_args()
    main(args)
