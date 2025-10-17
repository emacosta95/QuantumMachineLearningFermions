import numpy as np
import itertools
from itertools import combinations
from .cg_utils import ClebschGordan, SelectCG
import matplotlib.pyplot as plt
from tqdm import trange,tqdm
from .fermi_hubbard_library import FemionicBasis,FermionicBasisOptimized
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Text, Optional,Callable
from joblib import Parallel, delayed
import sys, time
from tqdm_joblib import tqdm_joblib
from src.NSMFermions.fermi_hubbard_library import build_mask_mapping


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

        self.masks, self.mask2index = build_mask_mapping(self.basis)
        
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
        for i in range(external_potential.shape[0]):

            operator = operator + external_potential[i] * self.adag_a_matrix(i=i, j=i)

        self.external_potential = operator

    def get_twobody_interaction(self, twobody_dict: Dict):

        matrix_keys = twobody_dict.keys()
        matrix_values = list(twobody_dict.values())
        ham_int=0.
        tbar=tqdm(enumerate(matrix_keys))
        for q, indices in tbar:
            i1, i2, i3, i4 = indices
            
            #if any(idx> self.size_a+self.size_b-1 for idx in indices):
            #    continue
            value = matrix_values[q]

            ham_int = (
                ham_int
                + (value * (self.adag_adag_a_a_matrix(i1=i1, i2=i2, j1=i4, j2=i3)))
                / 4
            )
            tbar.refresh()

        self.twobody_operator = ham_int


    def get_twobody_interaction_optimized(self, twobody_dict):
        """
        Build the two-body Hamiltonian efficiently using the precomputed
        adag_adag_a_a_matrix_optimized kernel with bitmask lookup.
        """
        N = self.basis.shape[0]  # total Hilbert-space dimension

        row_accum = []
        col_accum = []
        data_accum = []

        matrix_items = list(twobody_dict.items())

        print(f"Building two-body operator with {len(matrix_items)} terms...")

        for (i1, i2, i3, i4), value in tqdm(matrix_items):
            # Generate term from Numba-optimized kernel
            term = self.adag_adag_a_a_matrix_optimized(
                i1=i1, i2=i2, j1=i4, j2=i3
            ).tocoo()


            # Accumulate contributions, scaled by 1/4
            row_accum.append(term.row)
            col_accum.append(term.col)
            data_accum.append(term.data * (value / 4.0))
        # Concatenate all term arrays in one shot

        if len(data_accum) == 0:
            print("Warning: no nonzero two-body terms generated.")
            self.twobody_operator = coo_matrix((N, N)).tocsr()
            return

        rows = np.concatenate(row_accum)
        cols = np.concatenate(col_accum)
        data = np.concatenate(data_accum)

        # Final sparse matrix build
        self.twobody_operator = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

        print(f"✅ Two-body operator built: shape={self.twobody_operator.shape}, nnz={self.twobody_operator.nnz}")

    
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

    def get_external_potential_optimized(self, external_potential: np.ndarray):

        operator = 0.0
        for i in range(external_potential.shape[0]):

            operator = operator + external_potential[i] * self.adag_a_matrix_optimized(i=i, j=i).tocsr()

        self.external_potential = operator
        
        
        

class FermiHubbardHamiltonianOptimized(FermionicBasisOptimized):
    def __init__(
        self, size_a: int, size_b: int, nparticles_a: int, nparticles_b: int,
        symmetries: Optional[List[Callable]] = None
    ):
        # Initialize bit-based basis from optimized class
        super().__init__(size_a, nparticles_a, size_b, nparticles_b)
        
        # Apply symmetries if provided
        if symmetries is not None:
            keep_indices = []
            for idx in range(len(self.basis_bits)):
                vec = self.get_bitstring(idx)
                keep = all(sym(np.nonzero(vec)[0]) for sym in symmetries)
                if keep:
                    keep_indices.append(idx)
            self.basis_bits = self.basis_bits[keep_indices]
            # Rebuild bit->index map
            self.bit2index = self._make_bit2index(self.basis_bits)
        
        self.dim_hilbert_space = len(self.basis_bits)
        self.kinetic_operator = None
        self.external_potential = None
        self.twobody_operator = None
        self.hamiltonian = None

    def get_kinetic_operator(self, hopping_term: float, adj_matrix: Optional[Dict] = None):
        if adj_matrix is None:
            adj_matrix = {}
            # spin down / subsystem A
            for i in range(self.nsites_a - 1):
                adj_matrix[(i, i + 1)] = hopping_term
            # spin up / subsystem B
            for i in range(self.nsites_a, self.nsites_a + self.nsites_b - 1):
                adj_matrix[(i, i + 1)] = hopping_term

        operator = 0.0
        for (i, j), value in adj_matrix.items():
            operator += value * self.adag_a_matrix_optimized(i, j)
        self.kinetic_operator = operator

    def get_external_potential(self, external_potential: np.ndarray):
        operator = 0.0
        for i in range(len(external_potential)):
            operator += external_potential[i] * self.adag_a_matrix_optimized(i, i)
        self.external_potential = operator

    def get_twobody_interaction(self, twobody_dict: Dict):
        row_accum = []
        col_accum = []
        data_accum = []

        for (i1, i2, j1, j2), value in tqdm(twobody_dict.items()):
            term = self.adag_adag_a_a_matrix_optimized(i1, i2, j1, j2).tocoo()
            row_accum.append(term.row)
            col_accum.append(term.col)
            data_accum.append(term.data * value / 4.0)

        if len(data_accum) == 0:
            self.twobody_operator = coo_matrix((len(self.basis_bits), len(self.basis_bits))).tocsr()
            return

        rows = np.concatenate(row_accum)
        cols = np.concatenate(col_accum)
        data = np.concatenate(data_accum)
        self.twobody_operator = coo_matrix((data, (rows, cols)), shape=(len(self.basis_bits), len(self.basis_bits))).tocsr()

    def get_hamiltonian(self):
        self.hamiltonian = 0.0
        if self.kinetic_operator is not None:
            self.hamiltonian = self.kinetic_operator.copy()
        if self.external_potential is not None:
            self.hamiltonian += self.external_potential.copy()
        if self.twobody_operator is not None:
            self.hamiltonian += self.twobody_operator.copy()

    def get_spectrum(self, n_states: int):
        e, states = eigsh(self.hamiltonian, k=n_states, which="SA")
        return e, states