import numpy as np
import itertools
from itertools import combinations
from .cg_utils import ClebschGordan, SelectCG
import matplotlib.pyplot as plt
from tqdm import trange,tqdm
from .fermi_hubbard_library import FemionicBasis
from .hamiltonian_utils import FermiHubbardHamiltonian
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy
from scipy import sparse
from typing import List, Dict, Tuple, Text, Optional,Callable,ClassVar






class SingleParticleState:

    def __init__(self, file_name: str) -> None:
        """Class that defines the single particle states related to the nuclear interaction saved in the file

        Args:
            file_name (str): name of the .txt file with the coupling of the nuclear interaction (in the coupled basis)
        """

        with open(file_name, "r") as file:
            lines = file.readlines()

        # we save the strings related to the single
        # particle states
        
        labels = lines[1].split()
        
        # we encode the single particle energy values
        
        energy_values = [float(element) for element in lines[2].split()]
        
        # we save the map between the nucleon modes and the single particle
        # quantum numbers
        
        self.state_encoding: List = []
        self.energies: List = []
        
        for i_z in [1 / 2, -1 / 2]: # loop over the isospin projection
            for i, label in enumerate(labels[2:]): # loop over the label (single particle state in strings)
                n = int(label[1]) # these positions are defined by the structure of the .txt file (e.g cki or usdb.nat)
                l = int(label[0])
                two_j = int(label[-2:]) # the j value is encoded as 2j
                two_m_range = -1 * two_j + np.arange(0, 2 * two_j + 2, 2) # we define the range of possible m values
                
                # explicit the m value (projection of the total angular)
                # momentum
                for two_m in two_m_range:
                    self.state_encoding.append((n, l, two_j / 2, two_m / 2, 1 / 2, i_z))
                    self.energies.append(energy_values[i])

        # convert in a np.ndarray
        self.energies = np.asarray(self.energies)

    def get_index(self, state: Tuple) -> int:
        """it provides the nucleon mode index of a given single particle state

        Args:
            state (Tuple): a tuple of the form (n,l,j,m,t,t_z)

        Returns:
            int: the corresponding index of the nucleon mode
        """

        idx = self.state_encoding.index(state)
        return idx

    def projection_conservation(self, idxs: List[int]) -> bool:
        """ conservation law of a state amplitude <psi_f |psi_i> with respect to M and I_z

        Args:
            idxs (List[int]): list of nucleon modes in both psi_i and psi_f

        Returns:
            bool: True if both M and I_z is conserved, False if it is not 
        """
        
        # divide the length because 
        # the number of nucleon modes is equal in both
        # psi_i and psi_f
        nbody = len(idxs) // 2

        # initialize the global values related to psi_f
        t_z_final = 0.0
        m_final = 0
        
        for idx in idxs[:nbody]: # the loop for psi_f

            # extract the nucleon mode quantum numbers
            (n, l, j, m, t, t_z) = self.state_encoding[idx]
            
            # collect the global values
            t_z_final += t_z
            m_final += m

        # initialize the global values related to psi_i
        t_z_initial = 0.0
        m_initial = 0
        for idx in idxs[nbody:]: # loop for psi_i

            # extract the nucleon mode quantum numbers
            (n, l, j, m, t, t_z) = self.state_encoding[idx]
            # collect the global values
            t_z_initial += t_z
            m_initial += m

        # the condition
        condition = (m_initial == m_final) and (t_z_final == t_z_initial)

        return condition

    def compute_m_exp_value(self, psi: np.ndarray, basis: np.ndarray)-> float:
        """Compute the M expectation value <psi|M|psi>

        Args:
            psi (np.ndarray): many-body state
            basis (np.ndarray): basis in which |psi> is written

        Returns:
            float: <psi|M|psi>
        """
        
        # initialize the global value
        m_ave = 0.0
        
        # run over all the basis element of |psi>=\sum_i c_i |i>
        for basis_index, b in enumerate(basis):

            # extrapolate the nucleon modes from the basis element |i> == |a,b,c,d>
            # idxs=(a,b,c,d)
            idxs = np.nonzero(b)[0]
            
            # run over the nucleon modes
            for idx in idxs:
                
                # collect the m value corresponding to the corresponding nucleon mode
                (n, l, j, m, t, t_z) = self.state_encoding[idx]
                # average <M>=\sum_i |c_i|^2 \sum_a(i) m_a(i)
                m_ave = m_ave + m * psi[basis_index].conjugate() * psi[basis_index]

        return  m_ave 
    
    def total_M_zero(self,idxs:np.ndarray)->bool:
        """Restrict the basis to the M=0 subspace

        Args:
            idxs (np.ndarray): set of orbitals occupied in a many-body basis element

        Returns:
            bool: True if M=0 in the basis element
        """
        
        total_m=0.
        for idx in idxs:
            (n, l, j, m, t, t_z) = self.state_encoding[idx]
            total_m+=m
            
        return np.isclose(total_m,0.)


# 
def krond(tuple_a:Tuple[int], tuple_b:Tuple[int])-> int:  # Kronecker delta
    """routine for computing the kronecker delta of a reduced single particle state (n,l,j)

    Args:
        tuple_a (Tuple[int]): reduced single particle state a (n_a,l_a,j_a)
        tuple_b (Tuple[int]): reduced single particle state b (n_b, l_b, j_b)

    Returns:
        int: 1 if state_a==state_b, 0 if state_a!=state_b 
    """
    krond = 0
    if (
        tuple_a[0] == tuple_b[0]
        and tuple_a[1] == tuple_b[1]
        and tuple_a[2] == tuple_b[2]
    ):
        krond = 1
    return krond


def scattering_matrix_reader(file_name: str) -> Tuple[Dict]:
    """Extrapolate the scattering matrix from the .txt file of the nuclear interaction. 
    ❗❗ N.B: this routine depends on the format of the .txt file of the nuclear interaction. 
    It may not work with different format❗❗

    Args:
        file_name (str): .txt file related to the nuclear interaction

    Returns:
        Tuple[Dict]: a Tuple of dictionaries with both the total J and total I values and
        the values of the nuclear interaction, with the coupled state as key
    """
    
    # read the .txt file
    with open(file_name, "r") as file:
        lines = file.readlines()

    # extract the txt file into general information of the interaction 
    # the entries of the interaction and the related labels (reduced state + J + I) corresponding to the 
    # values of the interaction
    matrix_info: List = []
    matrix_entries: List = []
    matrix_entries_string: List = []

    for i in range(4, len(lines), 3):
        float_line_1 = [(element) for element in lines[i].split()]
        float_line_2 = [float(element) for element in lines[i + 1].split()]
        float_line_3 = [float(element) for element in lines[i + 2].split()]
        

        line_2 = [(element) for element in lines[i + 1].split()]
        line_3 = [(element) for element in lines[i + 2].split()]
        matrix_info.append(float_line_1)
        matrix_entries.append(float_line_2)
        matrix_entries.append(float_line_3)
        matrix_entries_string.append(line_2)
        matrix_entries_string.append(line_3)

    # initia
    j_tot_i_tot: Dict = {}
    scattering_values: Dict = {}
    #    get the matrix entries as function of J and I
    for i in range(len(matrix_info)):
        # we range over the total J and total I, therefore we need lists
        tot_iso_range = np.arange(int(matrix_info[i][0]), int(matrix_info[i][1]) + 1)
        tot_j_range = np.arange(int(matrix_info[i][-2]), int(matrix_info[i][-1]) + 1)
        
        # principal quantum number (radial quantum number)
        n1f = int(matrix_info[i][2][1]) # the encoding depends on the structure of the .txt file (e.g: cki, usdb.nat)
        n2f = int(matrix_info[i][3][1])
        n1i = int(matrix_info[i][4][1])
        n2i = int(matrix_info[i][5][1])

        # orbital angular momentum
        l1f = int(matrix_info[i][2][0])
        l2f = int(matrix_info[i][3][0])
        l1i = int(matrix_info[i][4][0])
        l2i = int(matrix_info[i][5][0])

        # total angular momentum (2j, we need to divide)
        j1f = int(matrix_info[i][2][-2:]) / 2
        j2f = int(matrix_info[i][3][-2:]) / 2
        j1i = int(matrix_info[i][4][-2:]) / 2
        j2i = int(matrix_info[i][5][-2:]) / 2

        # initialize the dict (J,I)_{labels}
        j_tot_i_tot[
            (n1f, l1f, j1f), (n2f, l2f, j2f), (n1i, l1i, j1i), (n2i, l2i, j2i)
        ] = (tot_j_range, tot_iso_range)

        # add all the permutations 
        if (n1i, l1i, j1i) != (n2i, l2i, j2i):
            j_tot_i_tot[
                (n1f, l1f, j1f), (n2f, l2f, j2f), (n2i, l2i, j2i), (n1i, l1i, j1i)
            ] = (tot_j_range, tot_iso_range)

        if (n1f, l1f, j1f) != (n2f, l2f, j2f):
            j_tot_i_tot[
                (n2f, l2f, j2f), (n1f, l1f, j1f), (n1i, l1i, j1i), (n2i, l2i, j2i)
            ] = (tot_j_range, tot_iso_range)

        if (n1f, l1f, j1f) != (n2f, l2f, j2f) and (n1i, l1i, j1i) != (n2i, l2i, j2i):
            j_tot_i_tot[
                (n2f, l2f, j2f), (n1f, l1f, j1f), (n2i, l2i, j2i), (n1i, l1i, j1i)
            ] = (tot_j_range, tot_iso_range)
        
        # initialize the dict V_{labels}
        for matrix_jdx, j_tot in enumerate(tot_j_range): # matrix_jdx and matrix_idx are internal indices of the coupled matrix
            for matrix_idx, i_tot in enumerate(tot_iso_range): # becuase V is a matrix of J and I ->  V(I,J)_{labels}

                scattering_values[
                    (
                        (n1f, l1f, j1f),
                        (n2f, l2f, j2f),
                        (n1i, l1i, j1i),
                        (n2i, l2i, j2i),
                        j_tot,
                        i_tot,
                    )
                ] = matrix_entries[2 * i + matrix_idx][matrix_jdx]

                # add all the permutations
                if (n1i, l1i, j1i) != (n2i, l2i, j2i):
                    scattering_values[
                        (n1f, l1f, j1f),
                        (n2f, l2f, j2f),
                        (n2i, l2i, j2i),
                        (n1i, l1i, j1i),
                        j_tot,
                        i_tot,
                    ] = (-1) ** int(j2i + j1i + j_tot + i_tot) * matrix_entries[
                        2 * i + matrix_idx
                    ][
                        matrix_jdx
                    ]
                    

                if (n1f, l1f, j1f) != (n2f, l2f, j2f):
                    scattering_values[
                        (n2f, l2f, j2f),
                        (n1f, l1f, j1f),
                        (n1i, l1i, j1i),
                        (n2i, l2i, j2i),
                        j_tot,
                        i_tot,
                    ] = (-1) ** int(j2f + j1f + j_tot + i_tot) * matrix_entries[
                        2 * i + matrix_idx
                    ][
                        matrix_jdx
                    ]

                if (n1f, l1f, j1f) != (n2f, l2f, j2f) and (n1i, l1i, j1i) != (
                    n2i,
                    l2i,
                    j2i,
                ):
                    scattering_values[
                        (n2f, l2f, j2f),
                        (n1f, l1f, j1f),
                        (n2i, l2i, j2i),
                        (n1i, l1i, j1i),
                        j_tot,
                        i_tot,
                    ] = (-1) ** int(
                        j2f + j1f + 2 * j_tot + 2 * i_tot + j1i + j2i
                    ) * matrix_entries[
                        2 * i + matrix_idx
                    ][
                        matrix_jdx
                    ]


    return j_tot_i_tot, scattering_values


def compute_nuclear_twobody_matrix(
    spg:ClassVar, j_tot_i_tot: Dict, scattering_values: Dict
) -> Dict:
    """Compute the nuclear interaction matrix from the coupled to the uncoupled basis (or single particle basis)

    Args:
        spg (SingleParticleState): single particle state class
        j_tot_i_tot (Dict): J/I dictionary related to the single particle state class in the coupled basis
        scattering_values (Dict): interaction dictionary related to the single particle state class (and interation file) in the coupled basis

    Returns:
        Dict: dictionary of the nuclear interaction in the single particle basis
    """

    matrix: Dict = {}
    print('Computing the matrix, pls wait... (u_u) \n')
    # run over all the nucleon modes of the two-body interaction
    for i in trange(len(spg.state_encoding)):
        for j in range(i, len(spg.state_encoding)):
            for l in range(len(spg.state_encoding)):
                for m in range(l, len(spg.state_encoding)):
                    
                    # extrapolate the quantum number
                    (ni, li, ji, mi, ii, izi) = spg.state_encoding[i]
                    (nj, lj, jj, mj, ij, izj) = spg.state_encoding[j]
                    (nl, ll, jl, ml, il, izl) = spg.state_encoding[l]
                    (nm, lm, jm, mm, im, izm) = spg.state_encoding[m]


                    #### check conservation laws
                    if (-1) ** (li + lj) != (-1) ** (ll + lm):
                        continue
                    if mi + mj != ml + mm:
                        continue
                    if izi + izj != izl + izm:
                        continue

                    # if the reduced state is in the dict, 🔄✅
                    if (
                        (ni, li, ji),
                        (nj, lj, jj),
                        (nl, ll, jl),
                        (nm, lm, jm),
                    ) in j_tot_i_tot.keys():
                        
                        # compute both the J-range and I-range
                        j_tot_range, i_tot_range = j_tot_i_tot[
                            (ni, li, ji), (nj, lj, jj), (nl, ll, jl), (nm, lm, jm)
                        ]

                        # initialize the value of the matrix element of
                        # the interaction
                        value = 0.0
                        
                        # run over the J and I range
                        for j_tot in j_tot_range:
                            for i_tot in i_tot_range:
                                # compute all the CB coefficients
                                # for the basis transformation (see Suhonen's book)
                                cg_final_list = ClebschGordan(
                                    ji,
                                    jj,
                                    j_tot,
                                )
                                cg_initial_list = ClebschGordan(
                                    jl,
                                    jm,
                                    j_tot,
                                )
                                cg_iso_initial_list = ClebschGordan(
                                    1 / 2,
                                    1 / 2,
                                    i_tot,
                                )

                                cg_iso_final_list = ClebschGordan(
                                    1 / 2,
                                    1 / 2,
                                    i_tot,
                                )

                                # get the matrix value in the coupled basis
                                matrix_outcome = scattering_values[
                                    (
                                        (ni, li, ji),
                                        (nj, lj, jj),
                                        (nl, ll, jl),
                                        (nm, lm, jm),
                                        j_tot,
                                        i_tot,
                                    )
                                ]

                                # get the CG coefficient specifing the m values 
                                cg_initial = SelectCG(
                                    cg_initial_list,
                                    jl,
                                    ml,
                                    jm,
                                    mm,
                                    j_tot,
                                    ml+mm,
                                )
                                cg_final = SelectCG(
                                    cg_final_list, ji, mi, jj, mj, j_tot, mi+mj
                                )
                                cg_iso_initial = SelectCG(
                                    cg_iso_initial_list,
                                    1 / 2,
                                    izl,
                                    1 / 2,
                                    izm,
                                    i_tot,
                                    izl+izm,
                                )
                                cg_iso_final = SelectCG(
                                    cg_iso_final_list,
                                    1 / 2,
                                    izi,
                                    1 / 2,
                                    izj,
                                    i_tot,
                                    izi+izj,
                                )

                                # compute the JT phase
                                phaseJT = (-1.0) ** (j_tot + i_tot)
                                
                                # compute the normalization 
                                nij = np.sqrt(
                                    1.0
                                    - krond((ni, li, ji), (nj, lj, jj))
                                    * phaseJT
                                ) / (1.0 + krond((ni, li, ji), (nj, lj, jj)))
                                nlm = np.sqrt(
                                    1.0
                                    - krond((nl, ll, jl), (nm, lm, jm))
                                    * phaseJT
                                ) / (1.0 + krond((nl, ll, jl), (nm, lm, jm)))

                                # to avoid divergences
                                if (abs(nij) == 0.0) or (abs(nlm) == 0.0):
                                    continue

                                # compute the corresponding interaction value in the
                                # single particle basis
                                value = (
                                    value
                                    + cg_initial
                                    * cg_final
                                    * cg_iso_initial
                                    * cg_iso_final
                                    * matrix_outcome
                                    / (nij * nlm)
                                )

                        # compute all the possible permutation of the matrix elements
                        if value != 0:
                            matrix[(i, j, l, m)] = value  # regular value
                            if m != l:
                                matrix[(i, j, m, l)] = -1 * value  # asymmetric final
                            if j != i:
                                matrix[(j, i, l, m)] = -1 * value  # asymmetric initial
                            matrix[(j, i, m, l)] = value  # double asymmetric

                
    # add the permutated elements in the dictionary
    old_matrix_keys = list(matrix.keys())
    for key in old_matrix_keys:
        i, j, l, m = key
        matrix[(m, l, j, i)] = matrix[key]




    return matrix


def get_twobody_nuclearshell_model(file_name: str)->Tuple[Dict]:
    """provides the nuclear interaction dictionary in the single particle basis and the single particle class with respect to
    the .txt file of the nuclear interaction

    Args:
        file_name (str): .txt file of the nuclear interaction

    Returns:
        Tuple[Dict]: dict of the nuclear shell model interaction with respect to the single particle basis and the single particle class
    """
    
    # read the dictionaries in the coupled basis
    j_tot_i_tot, scattering_values = scattering_matrix_reader(file_name=file_name)
    # initialize the single particle class
    SPG = SingleParticleState(file_name=file_name)

    # get the dict interaction with respect to the single particle basis
    twobody_matrix= compute_nuclear_twobody_matrix(
        spg=SPG, j_tot_i_tot=j_tot_i_tot, scattering_values=scattering_values
    )

    return twobody_matrix, SPG.energies



def write_j_square_twobody_file(filename:str):
    """Write a .txt file of the J^2 OFF-DIAGONAL matrix elements in the coupled basis following the nuclear interaction .txt file

    Args:
        filename (str): .txt nuclear interaction file
    """
    
    # open the file interaction file
    file=open(filename)

    # get the title
    title=file.readline()
    # make the J^2 title
    jsquare_title='J^2'+title

    # split and copy the info about the sps from the .txt input file to the output file
    singleparticlestate_info=file.readline().strip().split()
    singleparticlestate_number=int(singleparticlestate_info[1])
    states=singleparticlestate_info[-singleparticlestate_number:]

    # encode the 2j states in the coupled basis on a list
    double_j_states=[]
    for state in states:        
        double_j=int(state[-1])
        double_j_states.append(double_j)
    
    # open the output J^2 file
    fileJ2=open(filename+'_j2','w')
    fileJ2.write('   J^2 ' + title )
    fileJ2.write('%i %i '  % (1, len(states)))
    for state in states:
        fileJ2.write('%s '% state)
    fileJ2.write('\n')
    fileJ2.write('0. 0. 0. \n')
    fileJ2.write('0. 0. 0. \n')
    
    # loop over the single particle states in the coupled basis
    for a in range(singleparticlestate_number):
        for b in range(singleparticlestate_number):
            double_ja=double_j_states[a]
            double_jb=double_j_states[b]
            totalj=np.arange(np.abs(double_ja-double_jb),np.abs(double_ja+double_jb)+2,2)/2
            # write the indices of the matrix elements in a txt format with J, I range and the corresponding reduced state
            fileJ2.write('%2i %2i %s %s %s %s %i %i \n'  % (0, 1, states[a], states[b], states[a], states[b], totalj[0], totalj[-1]))

            # write the value of the matrix element
            for t in [0,1]:
                for j in totalj:
                    # is this a constrain (???) -> it looks so in Antonio's code
                    tpj = t + j
                    if ( tpj % 2 == 0 and double_ja == double_jb ):
                        j_square_value = 0
                    else:
                        # compute the off-diagonal element
                        j_square_value=j*(j+1)-double_ja*(double_ja/2+1)/2-double_jb*(double_jb/2+1)/2
                    fileJ2.write(' %f '  %  j_square_value )
                fileJ2.write('\n')
            
    fileJ2.close()
      
      
class J2operator(FermiHubbardHamiltonian):
    
    def __init__(self, size_a:int, size_b:int, nparticles_a:int, nparticles_b:int, single_particle_states:List,j_square_filename:str,symmetries:Optional[List[Callable]] = None):
        """ initialize the J^2 class

        Args:
            size_a (int): number of the neutron single particle states
            size_b (int): number of the proton single particle states
            nparticles_a (int): number of neutrons
            nparticles_b (int): number of protons
            single_particle_states (List): list of the single particle states (both neutron and proton)
            j_square_filename (str): .txt of the J^2 off diagonal values in the coupled basis
            symmetries (List[Callable], optional): symmetries of the many-body basis
        """
        super().__init__(size_a, size_b, nparticles_a, nparticles_b, symmetries)
        
        
        self.__get_single_particle_term(single_particle_states=single_particle_states)
        self.__get_twobody_matrix(j_square_filename=j_square_filename)
        self.get_hamiltonian()
    
    def __get_single_particle_term(self,single_particle_states:List):
        """internal routine. Compute the diagonal term of J^2 in the single particle basis

        Args:
            single_particle_states (List): list of single particle states
        """
        diag_j=np.zeros(len(single_particle_states))
        label=[]
        for i in range(diag_j.shape[0]):
            n,l,j,m,_,tz=single_particle_states[i]
            # compute the diagonal term
            diag_j[i]=j*(j+1)
            # initialize the single particle operator relate to J^2_diagonal
            self.get_external_potential(diag_j)
        
    def __get_twobody_matrix(self,j_square_filename:str):
        """internal routine. Compute the J^2 off-diagonal dictionary with respect to the single particle state

        Args:
            j_square_filename (str): .txt of the off-diagonal term of J^2 with respect to the coupled basis
        """
        
        # get the J^2 off-diagonal matrix
        matrix_j,_=get_twobody_nuclearshell_model(file_name=j_square_filename)
        # compute the two-body interaction of the J^2 off-diagonal
        self.get_twobody_interaction(twobody_dict=matrix_j)    
        
    def j2_operator(self):
        """get the J^2 operator as a Scipy.sparse matrix in the many-body basis

        Returns:
            scipy.sparse.csr_matrix: sparse matrix in the many-body basis
        """
        return self.hamiltonian
    
    def j_value(self,psi:np.ndarray)->float:
        """compute the corresponding j-value with respect of a many-body state

        Args:
            psi (np.ndarray): many-body state

        Returns:
            float: j-value
        """
        
        # compute <psi| J^2| psi>
        j2=psi.transpose().conjugate().dot(self.j2_operator().dot(psi))
        
        # compute j-value from J^2
        jvalue=0.5 * ( np.sqrt(4.0 * j2 + 1) - 1 )
        return jvalue
    
    


# class QuadrupoleOperator(FermiHubbardHamiltonian):
    
#     def __init__(self, size_a, size_b, nparticles_a, nparticles_b, symmetries = None,single_particle_basis:List=None,operator_symmetries:List=None):
#         super().__init__(size_a, size_b, nparticles_a, nparticles_b, symmetries)
        
#         self.single_particle_basis=single_particle_basis
        
#         self.__load_quadrupole_matrix()
#         self.__get_manybody_operator(operator_symmetries=operator_symmetries)
        
#     def __load_quadrupole_matrix(self,):
        
#         self.quadrupole_reduced_matrix_dictionary={}
        
#         # nlj_a nlj_b structure
#         self.quadrupole_reduced_matrix_dictionary[(0,1,3/2),(0,1,3/2)]=-1.410
#         self.quadrupole_reduced_matrix_dictionary[(0,1,3/2),(0,1,1/2)]=-1.410
#         self.quadrupole_reduced_matrix_dictionary[(0,1,1/2),(0,1,3/2)]=1.410
        
#         self.quadrupole_reduced_matrix_dictionary[(0,2,5/2),(0,2,5/2)]=-2.585
#         self.quadrupole_reduced_matrix_dictionary[(0,2,5/2),(1,0,1/2)]=-2.185
#         self.quadrupole_reduced_matrix_dictionary[(0,2,5/2),(0,2,3/2)]=-1.293
        
#         self.quadrupole_reduced_matrix_dictionary[(1,0,1/2),(0,2,5/2)]=1.293
#         self.quadrupole_reduced_matrix_dictionary[(1,0,1/2),(0,2,3/2)]=-1.784
        
#         self.quadrupole_reduced_matrix_dictionary[(0,2,3/2),(0,2,5/2)]=1.293
#         self.quadrupole_reduced_matrix_dictionary[(0,2,3/2),(1,0,1/2)]=1.784
#         self.quadrupole_reduced_matrix_dictionary[(0,2,3/2),(0,2,3/2)]=-1.975
        
        
#     def __get_the_quadrupole_matrix(self,mu:int):
        
#         state_encoding=self.single_particle_basis
        
#         self.quadrupole_matrix={}
#         for idx_a,a in enumerate(state_encoding):
#             for idx_b,b in enumerate(state_encoding):
#                 na,la,ja,ma,ta,tza=a
#                 nb,lb,jb,mb,tb,tzb=b
                
                
#                 # get the Clebasch Gordan coefficient related to the transition a-b
#                 # (we have to double check if the W-E theorem returns the order of the variables
#                 # in the C-G in this way)
#                 if ja >= np.abs(jb-2) and ja <= np.abs(jb+2):

#                     cg_term_j=ClebschGordan(j1=jb,j2=2,J=ja)
#                     cg_value=SelectCG(cg_term_j,j1=jb,m1=mb,j2=2,m2=mu,J=ja,M=ma)
#                 else:
#                     cg_value=0.
                
                    
#                 if tza==tzb:
#                     if ((na,la,ja),(nb,lb,jb)) in self.quadrupole_reduced_matrix_dictionary.keys():
#                         quadrupole_value= self.quadrupole_reduced_matrix_dictionary[(na,la,ja),(nb,lb,jb)]*cg_value/np.sqrt(2*ja+1)
#                 else:
#                     quadrupole_value=0.
                
#                 if np.abs(quadrupole_value)<10*-10:
#                     continue
#                 else:
#                     self.quadrupole_matrix[(idx_a,idx_b)]=quadrupole_value
                    
            
                
#     def __get_manybody_operator(self,operator_symmetries:List):
        
#         self.quadrupole_operator={} # it is gonna be a dictionary
#         self.quadrupole_matrices={} # same for the matrix in the single particle basis
#         mus=[-2,-1,0,1,2]
        
#         for mu in mus:
#             self.__get_the_quadrupole_matrix(mu=mu)
#             self.quadrupole_matrices[mu]=self.quadrupole_matrix
            
#             qab=0.
#             for a in range(self.size_a+self.size_b):
#                 for b in range(self.size_a+self.size_b):
#                     if (a,b) in self.quadrupole_matrices[mu].keys():
                        
#                         full_cond=True
#                         if operator_symmetries is not None:
#                             idxs=[a,b]
#                             for sym in operator_symmetries:
#                                 cond=sym(idxs)
#                                 full_cond=cond*full_cond
#                         if full_cond:
#                             qab+=self.quadrupole_matrices[mu][(a,b)]*self.adag_a_matrix(i=a,j=b)
#             self.quadrupole_operator[mu]=qab
            
    
#     def deformation_value(self,psi:np.ndarray):
        
#         tot_value=0.
#         for _,op in self.quadrupole_operator.items():
#             print('op=',op)
#             value=psi.transpose().conjugate().dot(op.dot(psi))
#             tot_value+=value**2
            
#         return np.sqrt(tot_value)
            