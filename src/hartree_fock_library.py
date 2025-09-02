import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import trange, tqdm
import numpy as np
from scipy import sparse, linalg
from math import factorial, sqrt
import matplotlib.pyplot as plt


# def gram_schmidt(matrix):
#     Q, R = torch.linalg.qr(matrix)
#     return Q

def slater_determinants_combined(C_n, C_p, fock_basis):
    """
    C_n: [M_half, N_n]  -- neutron orbitals
    C_p: [M_half, N_p]  -- proton orbitals
    fock_basis: [F, M]  -- full occupation basis (neutrons + protons)

    Returns:
        psi: [F]  -- Slater determinant amplitudes
    """
    F, M = fock_basis.shape
    M_half = M // 2
    N_n = C_n.shape[1]
    N_p = C_p.shape[1]

    psi = torch.zeros(F, dtype=C_n.dtype)

    for i in range(F):
        occ = fock_basis[i]  # [M]

        occ_n = torch.nonzero(occ[:M_half]).squeeze()
        occ_p = torch.nonzero(occ[M_half:]).squeeze()+M_half

        Cn_sub = C_n[occ_n, :]  # shape [N_n, N_n]
        Cp_sub = C_p[occ_p, :]  # shape [N_p, N_p]

        if Cn_sub.shape[0] != N_n or Cp_sub.shape[0] != N_p:
            # Skip invalid configurations (e.g., wrong number of particles)
            continue

        det_n = torch.det(Cn_sub)
        det_p = torch.det(Cp_sub)
        psi[i] = det_n * det_p


    return psi  # [F]

def slater_determinants_only_neutrons(C_n, fock_basis):
    """
    C_n: [M_half, N_n]  -- neutron orbitals
    C_p: [M_half, N_p]  -- proton orbitals
    fock_basis: [F, M]  -- full occupation basis (neutrons + protons)

    Returns:
        psi: [F]  -- Slater determinant amplitudes
    """
    F, M = fock_basis.shape
    M_half = M // 2
    N_n = C_n.shape[1]


    psi = torch.zeros(F, dtype=C_n.dtype)

    for i in range(F):
        occ = fock_basis[i]  # [M]

        occ_n = torch.nonzero(occ[:M_half]).squeeze()


        Cn_sub = C_n[occ_n, :]  # shape [N_n, N_n]




        det_n = torch.det(Cn_sub)

        psi[i] = det_n 

    return psi  # [F]



def gram_schmidt(V):
    """
    Perform Gram-Schmidt orthogonalization on the set of vectors V.
    
    Parameters:
        V (numpy.ndarray): A 2D numpy array where each column is a vector.
        
    Returns:
        numpy.ndarray: A 2D numpy array where each column is an orthonormal vector.
    """
    # Number of vectors
    num_vectors = V.shape[1]
    # Dimension of each vector
    dim = V.shape[0]
    
    # Initialize an empty array for the orthogonal vectors
    Q = np.zeros((dim, num_vectors))
    
    for i in range(num_vectors):
        # Start with the original vector
        q = V[:, i]
        
        # Subtract the projection of q onto each of the previously calculated orthogonal vectors
        for j in range(i):
            q = q - np.dot(Q[:, j], V[:, i]) * Q[:, j]
        
        # Normalize the resulting vector
        Q[:, i] = q / np.linalg.norm(q)
    
    return Q


class HartreeFock(nn.Module):

    def __init__(self, size: int, nspecies: int) -> None:
        super().__init__()

        self.hamiltonian = None

        self.size = nspecies * size
        self.initialize_weights(size=self.size)

        self.nspecies = nspecies

        self.kinetic_matrix = None
        self.twobody_matrix = None
        self.external_matrix = None

    def get_hamiltonian(
        self,
        twobody_interaction: Optional[Dict] = None,
        kinetic_term: Optional[np.ndarray] = None,
        external_potential: Optional[np.ndarray] = None,
    ):

        if twobody_interaction is not None:

            self.twobody_matrix = np.zeros((self.size, self.size, self.size, self.size))

            for item in twobody_interaction.items():
                (i1, i2, i3, i4), value = item
                self.twobody_matrix[i1, i2, i3, i4] = value

        if kinetic_term is not None:

            self.kinetic_matrix = kinetic_term

        if external_potential is not None:

            self.external_matrix = np.eye(self.size)
            self.external_matrix = np.einsum(
                "ij,j->ij", self.external_matrix, external_potential
            )


    def selfconsistent_computation(self, epochs=1000, eta=0.01):
        des = []
        herm_history = []
        orthogonality_history = []
        eigen_old = -1000
        tbar = tqdm(range(epochs))
        for i in tbar:

            effective_hamiltonian = 0.0

            if self.twobody_matrix is not None:

                effective_two_body_term = (1 / 8) * np.einsum(
                    "ijkl,ja,la->ik",
                    self.twobody_matrix,
                    self.weights.conjugate(),
                    self.weights,
                )
                # print('twobody matrix=',self.twobody_matrix)
                # print('effective_two_body_term=',effective_two_body_term)
                effective_hamiltonian += effective_two_body_term
            if self.kinetic_matrix is not None:
                effective_hamiltonian += self.kinetic_matrix
            if self.external_matrix is not None:
                effective_hamiltonian += self.external_matrix

            ishermcheck = np.average(
                np.abs(
                    effective_hamiltonian
                    - np.einsum("ij->ji", effective_hamiltonian).conjugate()
                )
            )
            
            herm_history.append(ishermcheck)
            if not (np.isclose(0, ishermcheck)):
                print("effective Hamiltonian not Hermitian \n")

            eigen, new_weights = np.linalg.eigh(effective_hamiltonian)
            
            #new_weights=np.einsum('ij,ja->ia',effective_hamiltonian,self.weights) 
            new_weights=gram_schmidt(new_weights)
            #self.weights = self.weights / np.linalg.norm(self.weights, axis=0)[None, :]

            #eigen=np.einsum('ia,ij,ja->a',np.conj(self.weights),effective_hamiltonian,self.weights)
            
            #new_weights = new_weights / np.linalg.norm(new_weights, axis=0)[None, :]

            de = np.average(np.abs(eigen_old - eigen))
            eigen_old = eigen
            self.weights = self.weights * (1 - eta) + eta * new_weights

            #self.weights = gram_schmidt(self.weights)

            
            isortho = np.average(
                np.abs(
                    np.einsum("ia,ja->ij", self.weights.conj(), self.weights)
                    - np.eye(self.size)
                )
            )

            orthogonality_history.append(isortho)
            tbar.set_description(f"de={de:.15f}")
            tbar.refresh()
            des.append(eigen)

        return (
            np.asarray(des),
            np.asarray(herm_history),
            np.asarray(orthogonality_history),
        )

    def compute_energy(
        self,
    ):
        effective_hamiltonian = 0.0

        if self.twobody_matrix is not None:

            effective_two_body_term = (1 / 8) * np.einsum(
                "ijkl,ja,la->ik",
                self.twobody_matrix,
                self.weights.conjugate(),
                self.weights,
            )
            effective_hamiltonian += effective_two_body_term
        if self.kinetic_matrix is not None:
            effective_hamiltonian += self.kinetic_matrix
        if self.external_matrix is not None:
            effective_hamiltonian += self.external_matrix

        return np.einsum(
            "ia,ij,ja->",
            self.weights.conj(),
            effective_hamiltonian,
            self.weights,
        )

    def initialize_weights(self, size: int):

        # put some conditions
        # self.weights = np.random.uniform(size=(size, size))
        # self.weights = self.weights / np.linalg.norm(self.weights, axis=0)[None, :]
        self.weights = np.eye(size)

    def create_hf_psi(self, basis: np.ndarray, nparticles_a: int,nparticles_b:int):

        psi = np.zeros(basis.shape[0])
        jdx=np.append(np.arange(nparticles_a),np.arange(nparticles_b)+basis.shape[1]//2)
        jdx=list(jdx)
        print(jdx)
        matrix=np.zeros((nparticles_a+nparticles_b,nparticles_a+nparticles_b))
        for i, b in enumerate(basis):
            idx = np.nonzero(b)[0]
            matrix = self.weights[idx,:nparticles_a+nparticles_b]
            coeff = np.linalg.det(matrix)
            psi[i] = coeff

        psi = psi / np.linalg.norm(psi)

        return psi


class HartreeFockVariational(nn.Module):

    def __init__(self, size: int, nspecies: int, mu: float = 10) -> None:
        super().__init__()

        self.hamiltonian = None

        self.size = size
        # self.initialize_weights(size=self.size)

        self.nspecies = nspecies

        self.weights = None
        self.kinetic_matrix = None
        self.twobody_matrix = None
        self.external_matrix = None

        self.mu = mu

    def get_hamiltonian(
        self,
        twobody_interaction: Optional[Dict] = None,
        kinetic_term: Optional[np.ndarray] = None,
        external_potential: Optional[np.ndarray] = None,
    ):

        if twobody_interaction is not None:

            self.twobody_matrix = torch.zeros(
                (self.size, self.size, self.size, self.size), dtype=torch.complex64
            )

            for item in twobody_interaction.items():
                (i1, i2, i3, i4), value = item
                self.twobody_matrix[i1, i2, i3, i4] = value

        if kinetic_term is not None:

            self.kinetic_matrix = kinetic_term

        if external_potential is not None:

            self.external_matrix = torch.eye(self.size, dtype=torch.complex64)
            self.external_matrix = torch.einsum(
                "ij,j->ij", self.external_matrix, external_potential
            )

    def forward(self, psi: torch.tensor):

        effective_hamiltonian = 0.0

        if self.twobody_matrix is not None:
            effective_two_body_term = (1 / 8) * torch.einsum(
                "ijkl,ja,la->ik",
                self.twobody_matrix,
                psi.conj(),
                psi,
            )
            effective_hamiltonian += effective_two_body_term
        if self.kinetic_matrix is not None:
            effective_hamiltonian += self.kinetic_matrix
        if self.external_matrix is not None:
            effective_hamiltonian += self.external_matrix

        self.effective_hamiltonian = effective_hamiltonian

        energy = torch.einsum("ia,ij,ja->", psi.conj(), effective_hamiltonian, psi)
        normalization_constrain = torch.mean(
            torch.abs(torch.eye(self.size) - torch.einsum("ia,ja->ij", psi.conj(), psi))
        )
        # print(normalization_constrain.item())

        return energy + self.mu * normalization_constrain, normalization_constrain

    def train(self, epochs=1000, eta=0.01):
        des = []
        herm_history = []
        orthogonality_history = []
        tbar = tqdm(range(epochs))

        psi = self.initialize_weights(size=self.size)

        for i in tbar:
            self.weights.requires_grad_(True)
            # self.weights = (
            #     self.weights / torch.linalg.norm(self.weights, axis=0)[None, :]
            # )
            psi = self.weights[0] + 1j * self.weights[1]
            psi = psi / torch.linalg.norm(psi, dim=0)[None, :]

            energy, norm_constrain = self.forward(psi)

            ishermcheck = torch.mean(
                torch.abs(
                    self.effective_hamiltonian
                    - torch.einsum("ij->ji", self.effective_hamiltonian).conj()
                )
            )
            herm_history.append(ishermcheck.detach().numpy())
            if not (np.isclose(0, ishermcheck.detach().numpy())):
                print("effective Hamiltonian not Hermitian \n")

            energy.backward()
            with torch.no_grad():

                grad_energy = self.weights.grad

                self.weights -= eta * (grad_energy)  # + 2 * mu * self.weights)
                self.weights.grad.zero_()

            self.eigen = torch.einsum(
                "ia,ij,ja->a", psi.conj(), self.effective_hamiltonian, psi
            )

            # self.weights = gram_schmidt(self.weights)

            isortho = torch.mean(
                torch.abs(
                    torch.einsum("ia,ja->ij", psi.conj(), psi) - torch.eye(self.size)
                )
            )

            orthogonality_history.append(isortho.clone().detach().numpy())
            tbar.set_description(
                f"energy={energy.item():.15f}, norm constrain={norm_constrain.item():.15f}"
            )
            tbar.refresh()
            des.append(self.eigen.clone().detach().numpy())

        return (
            np.asarray(des),
            np.asarray(herm_history),
            np.asarray(orthogonality_history),
        )

    def initialize_weights(self, size: int):

        # put some conditions
        # self.weights = np.random.uniform(size=(size, size))
        # self.weights = self.weights / np.linalg.norm(self.weights, axis=0)[None, :]
        self.weights = torch.cat(
            (torch.eye(size).unsqueeze(0), torch.zeros((size, size)).unsqueeze(0)),
            dim=0,
        )
        self.weights.requires_grad_(True)

        return self.weights[0] + 1j * self.weights[1]

    def compute_psi(self):
        psi = self.weights[0] + 1j * self.weights[1]
        psi = psi / torch.linalg.norm(psi, dim=0)[None, :]
        idx = np.argsort(self.eigen.detach().numpy())

        return psi.detach().numpy()[idx]

    def create_hf_psi(self, basis: np.ndarray, nparticles_a: int,nparticles_b:int):

        psi = np.zeros(basis.shape[0])

        orbitals = self.compute_psi()
        for i, b in enumerate(basis):
            
            jdx=np.append(np.arange(nparticles_a),np.arange(nparticles_b)+basis.shape[1]//2)
            idx = np.nonzero(b)[0]
            matrix = orbitals[idx, jdx]
            coeff = np.linalg.det(matrix)
            psi[i] = coeff

        psi = psi / np.linalg.norm(psi)

        return psi


class HFEnergyFunctional(nn.Module):
    def __init__(self, h_vec, V_dict, num_particles):
        super().__init__()
        self.h = h_vec  # shape [M]
        self.M = h_vec.shape[0]
        self.N = num_particles

        # Convert V_dict → dense tensor [M, M, M, M]

        self.V_tensor = torch.zeros((self.M, self.M, self.M, self.M), dtype=h_vec.dtype)
        for (a, b, c, d), val in V_dict.items():
            self.V_tensor[a, b, c, d] = val

        # Learnable parameter A → orthonormal C via QR
        A_init = torch.randn(self.M, self.N)
        self.A = nn.Parameter(A_init)

    def forward(self):
        # Reparametrize with QR → orthonormal orbitals
        C, _ = torch.linalg.qr(self.A)
        self.C = C.clone()  # optional: save for external use

        rho = C @ C.T  # Density matrix: [M, M]

        # One-body term: sum_a h_a * rho_aa
        E1 = torch.dot(self.h.to(rho.dtype), torch.diagonal(rho))

        # Two-body term: 0.5 * sum_abcd V_abcd * rho_ca * rho_db
        E2 = 0.5 * torch.einsum('abcd,ca,db->', self.V_tensor.to(rho.dtype), rho, rho)

        return E1 + E2
    
    
class HFEnergyFunctionalNuclear(nn.Module):
    def __init__(self, h_vec, V_dict, num_neutrons, num_protons, neutron_indices, proton_indices):
        super().__init__()
        self.h = h_vec  # [M]
        self.M = h_vec.shape[0]
        self.Nn = num_neutrons
        self.Np = num_protons

        self.proton_idx = proton_indices
        if num_protons!=0:
            self.V_tensor = torch.zeros((self.M, self.M, self.M, self.M), dtype=h_vec.dtype)
            for (a, b, c, d), val in V_dict.items():
                self.V_tensor[a, b, c, d] = val
        else:
            self.V_tensor = torch.zeros((self.M//2, self.M//2, self.M//2, self.M//2), dtype=h_vec.dtype)
            for (a, b, c, d), val in V_dict.items():
                if a<self.M//2 and a<self.M//2 and b<self.M//2 and c<self.M//2  and d<self.M//2: 
                    self.V_tensor[a, b, c, d] = val
        self.A_n = nn.Parameter(torch.randn(self.proton_idx, self.Nn,dtype=h_vec.dtype))
        if num_protons!=0:
            self.A_p = nn.Parameter(torch.randn(self.proton_idx, self.Np,dtype=h_vec.dtype))

    def forward(self):
        C_n_local, _ = torch.linalg.qr(self.A_n)
        if self.Np!=0:
            C_p_local, _ = torch.linalg.qr(self.A_p)

            C_n = torch.zeros((self.M, self.Nn), dtype=C_n_local.dtype, device=C_n_local.device)
        else:
             C_n_local=C_n_local[ :self.M//2]
             C_n = torch.zeros((self.M//2, self.Nn), dtype=C_n_local.dtype, device=C_n_local.device)
        if self.Np!=0:
            C_p = torch.zeros((self.M, self.Np), dtype=C_p_local.dtype, device=C_p_local.device)
        
        
        C_n[:self.proton_idx, :] = C_n_local
        if self.Np!=0:
            C_p[self.proton_idx:, :] = C_p_local

            rho_p = C_p @ C_p.T

        rho_n = C_n @ C_n.T
        if self.Np!=0:
            
            E1 = torch.dot(self.h, torch.diagonal(rho_n + rho_p))
            E2 = (
            0.5 * torch.einsum('abcd,ca,db->', self.V_tensor, rho_n, rho_n) +
            0.5 * torch.einsum('abcd,ca,db->', self.V_tensor, rho_p, rho_p) +
            torch.einsum('abcd,ca,db->', self.V_tensor, rho_n, rho_p)
            )
            self.C_p=C_p.clone()
        
        
        else:
            E1 = torch.dot(self.h[:self.M//2], torch.diagonal(rho_n ))
            E2 = (
            0.5 * torch.einsum('abcd,ca,db->', self.V_tensor, rho_n, rho_n) 
            )
            
        self.C_n=C_n.clone()        

        
        return E1 + E2