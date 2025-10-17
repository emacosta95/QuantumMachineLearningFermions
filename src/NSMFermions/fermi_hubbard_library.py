import itertools
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
from typing import List, Callable,Optional,Tuple
from scipy.sparse.linalg import eigsh, lobpcg
from itertools import product
import multiprocessing
from tqdm import tqdm, trange
from scipy.sparse import coo_matrix
from numba import njit
from numba import njit, int64, uint64, float64
from numba.typed import Dict
from numba import types
from src.NSMFermions.numba_utils import _adag_a_loop_numba_bits,_count_bits,_phase_for_annihilation,_adag_adag_a_a_loop_numba_bits

class FemionicBasis:

    def __init__(
        self, size_a: int, size_b: int, nparticles_a: int, nparticles_b: int
    ) -> None:

        self.size_a = size_a
        self.size_b = size_b
        self.nparticles_a = nparticles_a
        self.nparticles_b = nparticles_b

        self.basis = self.generate_fermi_hubbard_basis()
        
        self.masks, self.mask2index = build_mask_mapping(self.basis)

        self.encode = self._get_the_encode()

    def generate_fermi_hubbard_basis_old(self):
        """
        Generate the basis states for the Fermi-Hubbard model with a given number
        of lattice sites (L) and particles (N_particles).

        Parameters:
        - L: Number of lattice sites
        - N_particles: Number of particles

        Returns:
        - basis_states: List of basis states
        """

        basis_states = []
        state0 = np.zeros(self.size)
        # Generate all possible combinations of particle positions
        particle_positions = list(
            itertools.combinations(range(self.size), self.nparticles)
        )
        # Combine particle positions and empty sites to form basis states
        for tuple in particle_positions:
            state = state0.copy()
            for i in tuple:
                state[i] = 1
            basis_states.append(state)

        return np.asarray(basis_states)

    def generate_fermi_hubbard_basis(self):
        combinations_list = []
        #print(combinations(range(self.nparticles_a), self.size_a))
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
        
        return basis

    def adag_a_matrix(self, i: int, j: int) -> np.ndarray:
        operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))
        charge_conservation = self.charge_computation([i], [j])
        if charge_conservation:
            
            for index, psi in enumerate(self.basis):
                new_psi = np.zeros_like(psi)
                if self.basis[index, j] != 0:
                    new_basis = self.basis[index].copy()
                    new_basis[j] = self.basis[index, j] - 1
                    phase_j = np.sum(new_basis[0:j])
                    if new_basis[i] != 1:
                        new_basis[i] = new_basis[i] + 1
                        phase_i = np.sum(new_basis[0:i])
                        new_index = self._get_index(new_basis)
                        operator[new_index, index] = (-1) ** (phase_i + phase_j)

            return operator

        else:
            print("It does not conserve the number of Particles, Hombre! \n")

    def adag_a(self, i: int, j: int, psi: np.ndarray) -> np.ndarray:
        charge_conservation = self.charge_computation([i], [j])

        if charge_conservation:
            indices = np.nonzero(psi)[0]
            new_psi = np.zeros_like(psi)
            for index in indices:
                if self.basis[index, j] != 0:
                    new_basis = self.basis[index].copy()
                    new_basis[j] = self.basis[index, j] - 1
                    phase_j = np.sum(new_basis[0:j])
                    if new_basis[i] != 1:
                        new_basis[i] = new_basis[i] + 1
                        phase_i = np.sum(new_basis[0:i])
                        new_index = self._get_index(new_basis)
                        new_psi[new_index] = (-1) ** (phase_i + phase_j) * psi[index]

            return new_psi

        else:
            print("It does not conserve the number of Particles, Hombre! \n")

    def adag_adag_a_a_matrix(self, i1: int, i2: int, j1: int, j2: int) -> np.ndarray:
        operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))

        charge_conservation = self.charge_computation([i1, i2], [j1, j2])

        # print(i1, i2, j1, j2, initial_phase, final_phase)

        if charge_conservation:
            for idx, psi in enumerate(self.basis):
                if self.basis[idx, j2] != 0:
                    new_basis = self.basis[idx].copy()
                    new_basis[j2] = self.basis[idx, j2] - 1
                    phase_j2 = np.sum(new_basis[0:j2])
                    if new_basis[j1] != 0:
                        new_basis[j1] = new_basis[j1] - 1
                        phase_j1 = np.sum(new_basis[0:j1])
                        if new_basis[i2] != 1:
                            new_basis[i2] = new_basis[i2] + 1
                            phase_i2 = np.sum(new_basis[0:i2])
                            if new_basis[i1] != 1:
                                new_basis[i1] = new_basis[i1] + 1
                                phase_i1 = np.sum(new_basis[0:i1])

                                new_index = self._get_index(new_basis)
                                operator[new_index, idx] = (-1) ** (
                                    phase_j2 + phase_j1 + phase_i1 + phase_i2
                                )

            return operator
        else:
            print(" it does not conserve the number of Particles, Hombre! \n")

    def three_body_matrix(
        self, i1: int, i2: int, i3: int, j1: int, j2: int, j3: int
    ) -> np.ndarray:
        operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))

        charge_conservation = self.charge_computation([i1, i2, i3], [j1, j2, j3])

        # print(i1, i2, j1, j2, initial_phase, final_phase)

        if charge_conservation:
            for idx, psi in enumerate(self.basis):
                if self.basis[idx, j3] != 0:
                    new_basis = self.basis[idx].copy()
                    new_basis[j3] = self.basis[idx, j3] - 1
                    phase_j3 = np.sum(new_basis[0:j3])
                    if new_basis[j2] != 0:
                        new_basis[j2] = new_basis[j2] - 1
                        phase_j2 = np.sum(new_basis[0:j2])
                        if new_basis[j1] != 0:
                            new_basis[j1] = new_basis[j1] - 1
                            phase_j1 = np.sum(new_basis[0:j1])
                            if new_basis[i3] != 1:
                                new_basis[i3] = new_basis[i3] + 1
                                phase_i3 = np.sum(new_basis[0:i3])
                                if new_basis[i2] != 1:
                                    new_basis[i2] = new_basis[i2] + 1
                                    phase_i2 = np.sum(new_basis[0:i2])
                                    if new_basis[i1] != 1:
                                        new_basis[i1] = new_basis[i1] + 1
                                        phase_i1 = np.sum(new_basis[0:i1])

                                        new_index = self._get_index(new_basis)
                                        operator[new_index, idx] = (-1) ** (
                                            phase_j2
                                            + phase_j1
                                            + phase_i1
                                            + phase_i2
                                            + phase_j3
                                            + phase_i3
                                        )

            return operator
        else:
            print(" it does not conserve the number of Particles, Hombre! \n")

    def four_body_matrix(
        self, i1: int, i2: int, i3: int, i4: int, j1: int, j2: int, j3: int, j4: int
    ) -> np.ndarray:
        operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))

        charge_conservation = self.charge_computation(
            [i1, i2, i3, i4], [j1, j2, j3, j4]
        )

        # print(i1, i2, j1, j2, initial_phase, final_phase)

        if charge_conservation:
            for idx, psi in enumerate(self.basis):
                if self.basis[idx, j4] != 0:
                    new_basis = self.basis[idx].copy()
                    new_basis[j4] = self.basis[idx, j4] - 1
                    phase_j4 = np.sum(new_basis[0:j4])
                    if new_basis[j3] != 0:
                        new_basis[j3] = new_basis[j3] - 1
                        phase_j3 = np.sum(new_basis[0:j3])
                        if new_basis[j2] != 0:
                            new_basis[j2] = new_basis[j2] - 1
                            phase_j2 = np.sum(new_basis[0:j2])
                            if new_basis[j1] != 0:
                                new_basis[j1] = new_basis[j1] - 1
                                phase_j1 = np.sum(new_basis[0:j1])
                                if new_basis[i4] != 1:
                                    new_basis[i4] = new_basis[i4] + 1
                                    phase_i4 = np.sum(new_basis[0:i4])
                                    if new_basis[i3] != 1:
                                        new_basis[i3] = new_basis[i3] + 1
                                        phase_i3 = np.sum(new_basis[0:i3])
                                        if new_basis[i2] != 1:
                                            new_basis[i2] = new_basis[i2] + 1
                                            phase_i2 = np.sum(new_basis[0:i2])
                                            if new_basis[i1] != 1:
                                                new_basis[i1] = new_basis[i1] + 1
                                                phase_i1 = np.sum(new_basis[0:i1])

                                                new_index = self._get_index(new_basis)
                                                operator[new_index, idx] = (-1) ** (
                                                    phase_j2
                                                    + phase_j1
                                                    + phase_i1
                                                    + phase_i2
                                                    + phase_j3
                                                    + phase_i3
                                                    + phase_i4
                                                    + phase_j4
                                                )

            return operator
        else:
            print(" it does not conserve the number of Particles, Hombre! \n")

    def adag_adag_a_a(
        self, i1: int, i2: int, j1: int, j2: int, psi: np.ndarray
    ) -> np.ndarray:
        indices = np.nonzero(psi)[0]
        new_psi = np.zeros_like(psi)

        # condition for p n  -> p n without violating the N particles
        # IT DOES NOT WORK UP TO NOW

        charge_conservation = self.charge_computation([i1, i2], [j1, j2])

        if charge_conservation:
            for idx in indices:

                if self.basis[idx, j2] != 0:
                    new_basis = self.basis[idx].copy()
                    new_basis[j2] = 0
                    phase_j2 = np.sum(new_basis[0:j2])
                    if new_basis[j1] != 0:
                        new_basis[j1] = new_basis[j1] - 1
                        phase_j1 = np.sum(new_basis[0:j1])
                        if new_basis[i2] != 1:
                            new_basis[i2] = new_basis[i2] + 1
                            phase_i2 = np.sum(new_basis[0:i2])
                            if new_basis[i1] != 1:
                                new_basis[i1] = new_basis[i1] + 1
                                phase_i1 = np.sum(new_basis[0:i1])

                                #print(new_basis)
                                new_index = self._get_index(new_basis)
                                new_psi[new_index] = (-1) ** (
                                    phase_j2 + phase_j1 + phase_i1 + phase_i2
                                ) * psi[idx]

            return new_psi
        else:
            print(" it does not conserve the number of particles, Hombre! \n")
            return psi

    def reduced_state(self, indices: List, psi: np.ndarray):

        sub_dimension = len(indices)
        combinations = product([0, 1], repeat=sub_dimension)
        # Convert each combination into a numpy array
        basis = np.asarray([np.array(combination) for combination in combinations])

        # initialize the reduced density matrix
        density = np.zeros((basis.shape[0], basis.shape[0]))

        for density_index_d, d in enumerate(basis):
            for density_index_b, b in enumerate(basis):

                # compute the value of the reduced state for each main basis element
                value = 0
                # the nonzero check is essential for the algorithm

                # print("state_d=", d)
                # print("state_b=", b)
                for i_s, sigma in enumerate(self.basis):
                    # print("sigma=", sigma)
                    nonzero_check = True
                    for i, basis_element in enumerate(b):
                        if basis_element == 1:
                            a_value = sigma[indices[i]]
                        else:
                            a_value = 1 - sigma[indices[i]]

                        # # print(
                        # #     "a_value d",
                        # #     a_value,
                        # #     "indices=",
                        # #     indices[i],
                        # #     "basis element=",
                        # #     sigma,
                        # #     "rho indices=",
                        # #     d,
                        # #     b,
                        # #     "\n",
                        # # )

                        if a_value == 0:
                            nonzero_check = False
                            break

                    if nonzero_check:
                        for i, basis_element in enumerate(d):
                            if basis_element == 1:
                                a_value = sigma[indices[i]]
                            else:
                                a_value = 1 - sigma[indices[i]]

                            if a_value == 0:
                                nonzero_check = False
                                break

                            # print(
                            #     "a_value d",
                            #     a_value,
                            #     "indices=",
                            #     indices[i],
                            #     "basis element=",
                            #     sigma,
                            #     "rho indices=",
                            #     d,
                            #     b,
                            #     "\n",
                            # )

                    if nonzero_check:
                        value += psi[i_s] * np.conj(psi[i_s])

                # print(value)

                # print(density_index_b, density_index_d)
                density[density_index_d, density_index_b] = value

        return density

    def mutual_info(
        self,
        psi: np.ndarray,
    ):

        mutual_info = np.zeros((self.size_a + self.size_b, self.size_a + self.size_b))

        for i in range(self.size_a + self.size_b):
            for j in range(self.size_a + self.size_b):

                rho_ab = self.reduced_state(indices=[i, j], psi=psi)
                lambd, _ = np.linalg.eigh(rho_ab)
                s_ab = -1 * np.sum(np.log(lambd + 10**-20) * lambd)

                rho_a = self.reduced_state(indices=[i], psi=psi)
                lambd, _ = np.linalg.eigh(rho_a)
                s_a = -1 * np.sum(np.log(lambd + 10**-20) * lambd)

                rho_b = self.reduced_state(indices=[j], psi=psi)
                lambd, _ = np.linalg.eigh(rho_b)
                s_b = -1 * np.sum(np.log(lambd + 10**-20) * lambd)

                if i == j:
                    mutual_info[i, j] = 0.0
                else:
                    mutual_info[i, j] = -s_ab + (s_a + s_b)

        return mutual_info

    def generalized_mutual_info(self,subsets:Tuple[List],psi:np.ndarray):
        
        indices_a,indices_b=subsets
        
        s_a=self.entanglement_entropy(indices=indices_a,psi=psi)
        s_b=self.entanglement_entropy(indices=indices_b,psi=psi)
        s_ab=self.entanglement_entropy(indices=indices_a+indices_b,psi=psi)
        
        mutual_info=-s_ab + (s_a + s_b)
        return mutual_info

    def entanglement_entropy(self,indices:List,psi:np.ndarray):
        
        rho=self.reduced_state(indices=indices,psi=psi)
        lambd, _ = np.linalg.eigh(rho)
        s = -1 * np.sum(np.log(lambd + 10**-20) * lambd)
        return s
    
    def _get_the_encode(self):

        encode = {}
        for i, b in enumerate(self.basis):
            indices = np.nonzero(b)[0]
            encode[tuple(indices)] = i
    
        # # You can also precompute prefix sums if basis never changes:
        # self.prefix_sums = np.cumsum(self.basis, axis=1)    
        #         # Convert encode dictionary to numeric arrays (Numba cannot use dicts)
        # max_len = max(len(k) for k in encode.keys())
        # self.encode_keys = -np.ones((len(encode), max_len), dtype=np.int64)
        # self.encode_vals = np.zeros(len(encode), dtype=np.int64)
        # for idx, (k, v) in enumerate(encode.items()):
        #     self.encode_keys[idx, :len(k)] = np.array(k, dtype=np.int64)
        #     self.encode_vals[idx] = v

        return encode

    def _get_index(self, element: np.ndarray):

        indices = np.nonzero(element)[0]
        index = self.encode[tuple(indices)]

        return index

    def charge_computation(self, initial_indices: List, final_indices: List):

        initial_tot_charge = 0
        for idx in initial_indices:
            if idx >= self.size_a:
                initial_tot_charge += 1
        final_tot_charge = 0
        for idx in final_indices:
            if idx >= self.size_a:
                final_tot_charge += 1

        return initial_tot_charge == final_tot_charge

    def set_operator_pool(
        self, operator_pool: Dict, conditions: List[Callable], nbody: str
    ):
        # count=0
        # while (count<n_new_operators):
        #     if nbody=='one':
        #         idxs=np.random.randint(0,self.size_a+self.size_b,size=(2))

        #     if nbody=='two':
        #         idxs=np.random.randint(0,self.size_a+self.size_b,size=(4))

        #     full_condition=True
        #     for cond in conditions:
        #         logic_statement=cond(idxs)
        #         full_condition=full_condition and logic_statement

        #     if full_condition:

        #         if nbody=='one':
        #             op_plus = self.adag_a_matrix(idxs[0], idxs[1])
        #             op_minus = self.adag_a_matrix( idxs[1], idxs[0])

        #         if nbody=='two':
        #             op_plus = self.adag_adag_a_a_matrix(idxs[0], idxs[1],idxs[2],idxs[3])
        #             op_minus = self.adag_adag_a_a_matrix( idxs[3], idxs[2],idxs[1],idxs[0])

        #         operator_pool[tuple(idxs)]=op_plus-op_minus
        #         count=count+1

        # translator for Antonio's encoding


        for i1 in range(self.size_a + self.size_b):
            for i2 in range(i1+1, self.size_a + self.size_b):

                if nbody == "two":
                    for i3 in range(self.size_a + self.size_b):
                        for i4 in range(i3+1, self.size_a + self.size_b):
                            

                            idxs = [i1, i2, i3, i4]
                            cond = True
                            for c in conditions:
                                cond = c(idxs) and cond
                            if cond:
                                
                                if (idxs[2],idxs[3],idxs[0],idxs[1]) in operator_pool.keys():
                                    continue
                                else:
                                    op_plus = self.adag_adag_a_a_matrix(
                                        idxs[0], idxs[1], idxs[2], idxs[3]
                                    )
                                    op_minus = self.adag_adag_a_a_matrix(
                                        idxs[3], idxs[2], idxs[1], idxs[0]
                                    )
                                    operator_pool[tuple(idxs)] = op_plus - op_minus

                                #operator_pool[(i2, i1, i3, i4)] = -(op_plus - op_minus)
                                #operator_pool[(i1, i2, i4, i3)] = -(op_plus - op_minus)
                                #operator_pool[(i1, i2, i4, i3)] = op_plus - op_minus
                            else:
                                continue

                if nbody == "one":
                    idxs = [i1, i2]
                    cond = True
                    for c in conditions:
                        cond = c(idxs) and cond

                    if cond:
                        op_plus = self.adag_a_matrix(idxs[0], idxs[1])
                        op_minus = self.adag_a_matrix(idxs[1], idxs[0])

                        operator_pool[tuple(idxs)] = op_plus - op_minus
                    else:
                        continue

        return operator_pool


    def adag_adag_a_a_matrix_optimized(self, i1: int, i2: int, j1: int, j2: int) -> coo_matrix:
        """
        Fully optimized version using numba Dict lookup.
        """
        n_states, _ = self.basis.shape

        if not self.charge_computation([i1, i2], [j1, j2]):
            print("Does not conserve particles.")
            return coo_matrix((n_states, n_states))

        rows, cols, data = _adag_adag_a_a_loop_numba_with_dict(
            self.basis,
            i1, i2, j1, j2,
            self.masks,
            self.mask2index
        )
        

        return coo_matrix((data, (rows, cols)), shape=(n_states, n_states))

    def adag_a_matrix_optimized(self, i, j):
        rows, cols, data = _adag_a_loop_numba(
            self.basis,        # basis array
            i, j,
            self.masks,
            self.mask2index
        )
        from scipy.sparse import coo_matrix
        return coo_matrix((data, (rows, cols)), shape=(self.basis.shape[0], self.basis.shape[0]))




def build_mask_mapping(basis: np.ndarray, allow_overwrite: bool = False):
    """
    Build uint64 bitmask per basis row and a fast numba dictionary for O(1) lookup.
    Validates that no index exceeds n_states.

    Parameters
    ----------
    basis : ndarray[int8 or uint8], shape (n_basis, n_sites)
        0/1 occupation vectors.
    allow_overwrite : bool
        If True, allow building even if the basis length changes relative to previously
        saved mapping (useful if deliberately rebuilding). Default False only matters
        if you keep a previous mapping in memory — typical usage doesn't need it.
    """
    n_basis, n_sites = basis.shape
    if n_sites > 64:
        raise ValueError("Bitmask version only supports n_sites <= 64. Use multiword mask or combinadic ranking.")

    # Ensure dtype compactness: uint8 is good for packing and speed
    if basis.dtype != np.uint8:
       basis = basis.astype(np.uint8)

    masks = np.zeros(n_basis, dtype=np.uint64)
    for i in range(n_basis):
        m = np.uint64(0)
        # pack bits: site j -> bit j
        for j in range(n_sites):
            if basis[i, j]:
                m |= np.uint64(1) << np.uint64(j)
        masks[i] = m

    mask2index = Dict.empty(key_type=types.uint64, value_type=types.int64)
    for i, m in enumerate(masks):
        # defensive: if same mask appears multiple times (shouldn't happen for unique basis),
        # last write wins — but we warn.
        if m in mask2index:
            # warn about duplicates (should not occur in valid occupation-basis)
            print(f"Warning: duplicate mask for index {i}; previous index {mask2index[m]} will be overwritten.")
        mask2index[m] = i

    # Validation: ensure mapping values are within [0, n_basis-1]
    vals = [mask2index[k] for k in mask2index.keys()]
    max_val = max(vals) if vals else -1
    if max_val >= n_basis:
        raise RuntimeError(f"mask2index contains index {max_val} >= n_basis {n_basis}. "
                           "This means mapping was built against a different basis. "
                           "Rebuild mapping after finalizing self.basis.")

    return masks, mask2index


# ============================================================
# === Numba helper functions for phase + packing bits      ===
# ============================================================

@njit(inline='always')
def sign_from_phase(phase: int) -> float64:
    # much faster than (-1)**phase
    return 1.0 if (phase & 1) == 0 else -1.0


@njit(inline='always')
def pack_row_to_mask(row: np.ndarray) -> uint64:
    m = np.uint64(0)
    for j in range(row.shape[0]):
        if row[j]:
            m |= np.uint64(1) << np.uint64(j)
    return m


# ============================================================
# === Numba core kernel with O(1) lookup using numba.Dict   ===
# ============================================================

@njit
def _adag_adag_a_a_loop_numba_with_dict(
    basis: np.ndarray,
    i1: int, i2: int, j1: int, j2: int,
    masks: np.ndarray,
    mask2index: Dict
):
    """
    Numba-optimized kernel for constructing adag adag a a matrix elements
    using O(1) lookup via a numba typed.Dict.

    Returns rows, cols, data arrays for sparse COO construction.
    """
    n_basis, n_sites = basis.shape
    rows = np.empty(n_basis, dtype=np.int64)
    cols = np.empty(n_basis, dtype=np.int64)
    data = np.empty(n_basis, dtype=np.float64)
    count = 0

    for idx in range(n_basis):
        psi = basis[idx]

        # Skip if creation/annihilation violates occupancy
        if psi[j1]==0 or psi[j2]==0:
            continue
        
        if i1!=j1 and i1!=j2 and psi[i1]==1:
            continue
        
        if i2!=j1 and i2!=j2 and psi[i2]==1:
            continue
        
        
        
        # if psi[j2] == 0 or psi[j1] == 0 or psi[i2] == 1 or psi[i1] == 1:
        #     continue

        # Copy and apply operators
        phase = 0
        new_basis = psi.copy()
        if j2 > 0:
            phase += np.sum(psi[ :j2 ])
        new_basis[j2] = 0
        if j1 > 0:
            phase += np.sum(new_basis[ :j1 ])
        new_basis[j1] = 0
        if i2 > 0:
            phase += np.sum(new_basis[: i2 ])
        new_basis[i2] = 1
        if i1>0:
            phase += np.sum(new_basis[: i1 ])
        new_basis[i1] = 1


        # Bitpack and lookup new index
        m = pack_row_to_mask(new_basis)
        new_index = mask2index.get(m, -1)



        if new_index >= 0:
            rows[count] = new_index
            cols[count] = idx
            data[count] = (-1) ** phase
            count += 1

    return rows[:count], cols[:count], data[:count]


# ============================================================
# === Integration into your FemionicBasis class             ===
# ============================================================

@njit
def _adag_a_loop_numba(
    basis: np.ndarray,
    i: int, j: int,
    masks: np.ndarray,
    mask2index: Dict
):
    """
    Numba-optimized kernel for constructing a single creation/annihilation matrix element
    <basis_new | a^dag_i a_j | basis>.
    
    Returns rows, cols, data arrays for sparse COO construction.
    """
    n_basis, n_sites = basis.shape
    rows = np.empty(n_basis, dtype=np.int64)
    cols = np.empty(n_basis, dtype=np.int64)
    data = np.empty(n_basis, dtype=np.float64)
    count = 0

    for idx in range(n_basis):
        psi = basis[idx]

        # Skip if annihilation is not allowed
        if psi[j] == 0:
            continue

        # Skip if creation is not allowed (avoid double occupancy)
        if i != j and psi[i] == 1:
            continue

        # Copy and apply operators
        new_basis = psi.copy()
         # Compute fermionic phase
        phase = 0
        if j > 0:
            phase += np.sum(psi[: j ])
        new_basis[j] = 0
        if i > 0:
            phase += np.sum(new_basis[: i ])
        new_basis[i] = 1

        # Bitpack and lookup new index
        m = 0
        for s in range(n_sites):
            if new_basis[s]:
                m |= 1 << s
        new_index = mask2index.get(m, -1)

        if new_index >= 0:
            rows[count] = new_index
            cols[count] = idx
            data[count] = (-1) ** phase
            count += 1

    return rows[:count], cols[:count], data[:count]



import numpy as np
from numba.typed import Dict
from numba import njit, int64
from src.NSMFermions.numba_utils import generate_bit_basis_numba

class FermionicBasisOptimized:
    def __init__(self, nsites_a, nparticles_a, nsites_b, nparticles_b):
        self.nsites_a = nsites_a
        self.nparticles_a = nparticles_a
        self.nsites_b = nsites_b
        self.nparticles_b = nparticles_b
        
        # Generate basis as bitstrings
        self.basis_bits = generate_bit_basis_numba(
            nsites_a, nparticles_a, nsites_b, nparticles_b
        )
        
        # Create bit → index dictionary
        self.bit2index = self._make_bit2index(self.basis_bits)
        

        
    @staticmethod
    def _make_bit2index(basis_bits):
        """
        Create a numba-compatible dictionary mapping bit → index.
        """
        bit2index = Dict.empty(key_type=int64, value_type=int64)
        for idx, b in enumerate(basis_bits):
            bit2index[b] = idx
        return bit2index

    def get_bitstring(self, index: int) -> np.ndarray:
        """
        Given an index in the basis, return the corresponding occupation vector
        as a 0/1 array.
        """
        b = self.basis_bits[index]
        n_sites_total = self.nsites_a + self.nsites_b
        vec = np.zeros(n_sites_total, dtype=np.int64)
        for s in range(n_sites_total):
            if (b >> s) & 1:
                vec[s] = 1
        return vec
    
    def adag_a_matrix_optimized(self, i: int, j: int) -> coo_matrix:
        """Compute <basis_new | a^†_i a_j | basis> using bit basis."""
        rows, cols, data = _adag_a_loop_numba_bits(
            self.basis_bits, i, j, self.bit2index, self.nsites_a + self.nsites_b
        )
        return coo_matrix((data, (rows, cols)), shape=(len(self.basis_bits), len(self.basis_bits)))

    def adag_adag_a_a_matrix_optimized(self, i1: int, i2: int, j1: int, j2: int) -> coo_matrix:
        """Compute <basis_new | a^†_i a^†_j a_k a_l | basis> using bit basis."""
        rows, cols, data = _adag_adag_a_a_loop_numba_bits(
            self.basis_bits, i1, i2, j1, j2, self.bit2index, self.nsites_a + self.nsites_b
        )
        return coo_matrix((data, (rows, cols)), shape=(len(self.basis_bits), len(self.basis_bits)))

# ===============================
# === Numba kernels (bitwise) ===
# ===============================

