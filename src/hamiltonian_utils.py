import numpy as np
import itertools
from itertools import combinations
from src.cg_utils import ClebschGordan, SelectCG
import matplotlib.pyplot as plt
from tqdm import trange,tqdm
from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from scipy.sparse.linalg import eigsh

from typing import List, Dict, Tuple, Text, Optional,Callable


class FermiHubbardHamiltonian(FemionicBasis):

    def __init__(
        self, size_a: int, size_b: int, nparticles_a: int, nparticles_b: int,symmetries:Optional[List[Callable]]=None
    ) -> None:

        super().__init__(size_a, size_b, nparticles_a, nparticles_b)

        self.kinetic_operator = None
        self.external_potential = None
        self.twobody_operator = None

        self.hamiltonian = None

        self.dim_hilbert_space = self.basis.shape[0]
        
        self.basis = self.generate_fermi_hubbard_basis(symmetries)
        self.encode = self._get_the_encode()

    def get_kinetic_operator(
        self, hopping_term: Optional[float] = None, adj_matrix: Optional[Dict] = None
    ):

        if adj_matrix is None:

            adj_matrix = {}
            # spin down (proton)
            for i in range(self.size_a - 1):

                adj_matrix[(i, i + 1)] = hopping_term

            # spin up (neutron)
            for i in range(self.size_a, self.size_a + self.size_b - 1):

                adj_matrix[(i, i + 1)] = hopping_term

        operator = 0.0
        for element in adj_matrix.items():

            (i, j), value = element
            operator = operator + value * self.adag_a_matrix(i=i, j=j)

        self.kinetic_operator = operator #+ operator.transpose().conjugate()

    def get_external_potential(self, external_potential: np.ndarray):

        operator = 0.0
        for i, v in enumerate(external_potential):

            operator = operator + v * self.adag_a_matrix(i=i, j=i)

        self.external_potential = operator

    def get_twobody_interaction(self, twobody_dict: Dict):

        matrix_keys = twobody_dict.keys()
        matrix_values = list(twobody_dict.values())
        ham_int=0.
        tbar=tqdm(enumerate(matrix_keys))
        for q, indices in tbar:
            i1, i2, i3, i4 = indices
            
            if any(idx> self.size_a+self.size_b-1 for idx in indices):
                continue
            value = matrix_values[q]

            ham_int = (
                ham_int
                + (value * (self.adag_adag_a_a_matrix(i1=i1, i2=i2, j1=i4, j2=i3)))
                / 4
            )
            tbar.refresh()

        self.twobody_operator = ham_int

    def get_hamiltonian(
        self,
    ):

        self.hamiltonian = 0.0
        if self.kinetic_operator is not None:
            self.hamiltonian = self.kinetic_operator.copy()
        if self.external_potential is not None:
            self.hamiltonian = self.hamiltonian + self.external_potential.copy()
        if self.twobody_operator is not None:
            self.hamiltonian = self.hamiltonian + self.twobody_operator.copy()

        # we can add all the double check for the hamiltonian

    def get_spectrum(self, n_states: int):

        e, states = eigsh(self.hamiltonian, k=n_states, which="SA")

        return e, states
    
    def generate_fermi_hubbard_basis(self,symmetries:Optional[List[Callable]]=None):
        combinations_list = []
        for indices_part1 in list(combinations(range(self.size_a), self.nparticles_a)):
            for indices_part2 in list(
                combinations(range(self.size_b), self.nparticles_b)
            ):
                base = [0] * (self.size_a + self.size_b)
                for idx in indices_part1:
                    base[idx] = 1
                for idx in indices_part2:
                    # because the second subsystem is related to the other species
                    base[idx + self.size_a] = 1
                combinations_list.append(base)
                
        basis=np.asarray(combinations_list)
        if symmetries is not None:
            basis_with_symmetry=[]
            for b in basis:
                idxs=np.nonzero(b)[0]
                full_cond=True
                for sym in symmetries:
                    cond=sym(idxs)
                    full_cond=cond*full_cond
                
                if full_cond:
                    #print(idxs)
                    basis_with_symmetry.append(b)
            
            basis=np.asarray(basis_with_symmetry)
        
        return basis
