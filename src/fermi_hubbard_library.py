import itertools
from itertools import combinations
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse as sparse
from typing import List, Dict, Callable,Optional
from scipy.sparse.linalg import eigsh, lobpcg
from itertools import product
import multiprocessing
from tqdm import tqdm, trange


def single_particle_potential(
    ndim: int,
    nparticles: int,
    size: float,
    ngrid: int,
    peaks: List,
    amplitudes: List,
    deviations: List,
):

    if ndim == 2 and nparticles == 2:
        x, y = np.mgrid[0 : size : ngrid * 1j, 0 : size : ngrid * 1j]
        v: np.ndarray = 0.0
        for i, peak in enumerate(peaks):
            gaussian = (
                -1
                * amplitudes[i]
                * np.exp(
                    -1 * (((size / 2 - x) ** 2) + ((size / 2 - y) ** 2)) / deviations[i]
                )
            )
            shift_mean_x = int((ngrid * (peak[0]) / size))
            shift_mean_y = int((ngrid * (peak[1]) / size))
            gaussian = np.roll(
                gaussian, shift=(shift_mean_x, shift_mean_y), axis=(0, 1)
            )
            v = v + gaussian

    return v


def coulomb_function(idx: int, jdx: int, size: float, ngrid: int, ndim: int):
    dx = size / ngrid

    if ndim == 2:
        i1 = idx % ngrid
        j1 = idx // ngrid

        i2 = jdx % ngrid
        j2 = jdx // ngrid

        boxes = [-1, 0, 1]
        v = 0.0
        for b1x in boxes:
            for b2x in boxes:
                for b1y in boxes:
                    for b2y in boxes:
                        r = dx * np.sqrt(
                            1
                            + ((i1 + b1x * ngrid) - (i2 + b2x * ngrid)) ** 2
                            + ((j1 + b1y * ngrid) - (j2 + b2y * ngrid)) ** 2
                        )
                        v = v + 0.5 * 1 / r

    return v


def unique_combinations(lists):
    seen_values = set()
    for combination in product(*lists):
        if len(set(combination)) == len(combination) and combination not in seen_values:
            seen_values.add(combination)
            yield combination


def define_kinetic_element(
    sigma: np.ndarray,
    indices: int,
    nparticles: int,
    ngrid: int,
    cost: float,
    get_index: Callable,
    operator: sparse.csr_matrix,
):

    indices_and_variations = []
    for idx in indices:
        x = idx % ngrid
        y = idx // ngrid
        # print('x=',x)
        # print('y=',y)
        idx_x = (x + 1) % ngrid + ngrid * y
        idx_y = x + ngrid * ((y + 1) % ngrid)
        indices_and_variations.append([(idx, -2 * cost), (idx_x, cost), (idx_y, cost)])

    # permutation of all the combinations
    combinations = unique_combinations(indices_and_variations)

    for combo in combinations:
        operator_value = 0.0
        new_sigma = np.zeros_like(sigma)
        for element in combo:
            combo_index = element[0]
            value = element[1]
            operator_value = operator_value + value
            new_sigma[combo_index] = 1.0
        if int(np.sum(new_sigma)) == nparticles:
            old_index = get_index(sigma)
            new_index = get_index(new_sigma)

            # for jdx in np.nonzero(new_sigma)[0]:
            #     print('x_new=',jdx % ngrid)
            #     print('y_new=',jdx // ngrid)

            operator[old_index, new_index] = operator_value

    return operator


class FemionicBasis:

    def __init__(
        self, size_a: int, size_b: int, nparticles_a: int, nparticles_b: int
    ) -> None:

        self.size_a = size_a
        self.size_b = size_b
        self.nparticles_a = nparticles_a
        self.nparticles_b = nparticles_b

        self.basis = self.generate_fermi_hubbard_basis()

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
        print(combinations(range(self.nparticles_a), self.size_a))
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

        charge_conservation = self.__charge_computation([i], [j])
        if charge_conservation:
            operator = lil_matrix((self.basis.shape[0], self.basis.shape[0]))
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
        charge_conservation = self.__charge_computation([i], [j])

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

        charge_conservation = self.__charge_computation([i1, i2], [j1, j2])

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

        charge_conservation = self.__charge_computation([i1, i2, i3], [j1, j2, j3])

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

        charge_conservation = self.__charge_computation(
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

        charge_conservation = self.__charge_computation([i1, i2], [j1, j2])

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

                                print(new_basis)
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

    def _get_the_encode(self):

        encode = {}
        for i, b in enumerate(self.basis):
            indices = np.nonzero(b)[0]
            encode[tuple(indices)] = i

        return encode

    def _get_index(self, element: np.ndarray):

        indices = np.nonzero(element)[0]
        index = self.encode[tuple(indices)]

        return index

    def __charge_computation(self, initial_indices: List, final_indices: List):

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

        for i1 in range(self.size_a + self.size_b):
            for i2 in range(i1, self.size_a + self.size_b):

                if nbody == "two":
                    for i3 in range(self.size_a + self.size_b):
                        for i4 in range(i3, self.size_a + self.size_b):

                            idxs = [i1, i2, i3, i4]
                            cond = True
                            for c in conditions:
                                cond = c(idxs) and cond
                            if cond:
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
