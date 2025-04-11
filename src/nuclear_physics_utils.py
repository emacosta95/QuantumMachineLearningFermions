import numpy as np
import itertools
from itertools import combinations
from src.cg_utils import ClebschGordan, SelectCG
import matplotlib.pyplot as plt
from tqdm import trange,tqdm
from src.fermi_hubbard_library import FemionicBasis
from src.hamiltonian_utils import FermiHubbardHamiltonian
import numpy as np
from scipy.sparse.linalg import eigsh

from typing import List, Dict, Tuple, Text, Optional,Callable






class SingleParticleState:

    def __init__(self, file_name: str) -> None:

        with open(file_name, "r") as file:
            lines = file.readlines()

        labels = lines[1].split()
        energy_values = [float(element) for element in lines[2].split()]
        
        self.state_encoding: List = []
        self.energies: List = []
        #self.energies_dictionary:Dict={}
        for i_z in [1 / 2, -1 / 2]:
            for i, label in enumerate(labels[2:]):
                n = int(label[1])
                l = int(label[0])
                two_j = int(label[-1])
                two_m_range = -1 * two_j + np.arange(0, 2 * two_j + 2, 2)
                # we put the n=1 restriction
                # if n==1 or (n==2 and two_j==3):
                for two_m in two_m_range:
                    self.state_encoding.append((n, l, two_j / 2, two_m / 2, 1 / 2, i_z))
                    self.energies.append(energy_values[i])
                    #self.energies_dictionary[(n, l, two_j / 2, two_m / 2, 1 / 2, i_z)]=energy_values[i]

        self.energies = np.asarray(self.energies)

    def get_index(self, state: Tuple):

        idx = self.state_encoding.index(state)
        return idx

    def projection_conservation(self, idxs: List[int]):

        nbody = len(idxs) // 2

        t_z_final = 0.0
        m_final = 0
        for idx in idxs[:nbody]:

            (n, l, j, m, t, t_z) = self.state_encoding[idx]
            t_z_final += t_z
            m_final += m

        t_z_initial = 0.0
        m_initial = 0
        for idx in idxs[nbody:]:

            (n, l, j, m, t, t_z) = self.state_encoding[idx]
            t_z_initial += t_z
            m_initial += m

        condition = (m_initial == m_final) and (t_z_final == t_z_initial)

        return condition

    def compute_m_exp_value(self, psi: np.ndarray, basis: np.ndarray):

        j_ave = 0.0
        m_ave = 0.0
        for basis_index, b in enumerate(basis):

            idxs = np.nonzero(b)[0]
            for idx in idxs:

                (n, l, j, m, t, t_z) = self.state_encoding[idx]
                m_ave = m_ave + m * psi[basis_index].conjugate() * psi[basis_index]

        return  m_ave 

    def total_M_zero(self,idxs:np.ndarray):
        
        total_m=0.
        for idx in idxs:
            (n, l, j, m, t, t_z) = self.state_encoding[idx]
            total_m+=m
            
        return np.isclose(total_m,0.)
    
    # def single_particle_energy(self,basis:np.ndarray):
    #     spe=[]
    #     for b in basis:
    #         idxs=np.nonzero(b)[0]
    #         eng=0.
    #         for idx in idxs:
    #             key = self.state_encoding[idx]
    #             eng+=self.energies_dictionary[key]
    #         spe.append(eng)
        
    #     return np.asarray(spe)

def krond(tuple_a, tuple_b):  # Kronecker delta
    krond = 0
    if (
        tuple_a[0] == tuple_b[0]
        and tuple_a[1] == tuple_b[1]
        and tuple_a[2] == tuple_b[2]
    ):
        krond = 1
    return krond


def scattering_matrix_reader(file_name: str) -> Tuple[Dict]:
    with open(file_name, "r") as file:
        lines = file.readlines()

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

    j_tot_i_tot: Dict = {}
    scattering_values: Dict = {}
    #    get the matrix entries as function of J and I
    for i in range(len(matrix_info)):
        tot_iso_range = np.arange(int(matrix_info[i][0]), int(matrix_info[i][1]) + 1)
        # print('iso range=',tot_iso_range,'\n')
        tot_j_range = np.arange(int(matrix_info[i][-2]), int(matrix_info[i][-1]) + 1)
        
        # principal quantum number (radial quantum number)
        n1f = int(matrix_info[i][2][1])
        n2f = int(matrix_info[i][3][1])
        n1i = int(matrix_info[i][4][1])
        n2i = int(matrix_info[i][5][1])

        # orbital angular momentum
        l1f = int(matrix_info[i][2][0])
        l2f = int(matrix_info[i][3][0])
        l1i = int(matrix_info[i][4][0])
        l2i = int(matrix_info[i][5][0])

        # total angular momentum
        j1f = int(matrix_info[i][2][-1]) / 2
        j2f = int(matrix_info[i][3][-1]) / 2
        j1i = int(matrix_info[i][4][-1]) / 2
        j2i = int(matrix_info[i][5][-1]) / 2

        j_tot_i_tot[
            (n1f, l1f, j1f), (n2f, l2f, j2f), (n1i, l1i, j1i), (n2i, l2i, j2i)
        ] = (tot_j_range, tot_iso_range)

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
        # we must to take into account all the combinations!
        for matrix_jdx, j_tot in enumerate(tot_j_range):
            for matrix_idx, i_tot in enumerate(tot_iso_range):

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
    spg, j_tot_i_tot: Dict, scattering_values: Dict
) -> Dict:

    matrix: Dict = {}
    print('Computing the matrix, pls wait... (u_u) \n')

    for i in trange(len(spg.state_encoding)):
        for j in range(i, len(spg.state_encoding)):
            for l in range(len(spg.state_encoding)):
                for m in range(l, len(spg.state_encoding)):

                    (ni, li, ji, mi, ii, izi) = spg.state_encoding[i]
                    (nj, lj, jj, mj, ij, izj) = spg.state_encoding[j]
                    (nl, ll, jl, ml, il, izl) = spg.state_encoding[l]
                    (nm, lm, jm, mm, im, izm) = spg.state_encoding[m]

                    if (-1) ** (li + lj) != (-1) ** (ll + lm):
                        continue
                    if mi + mj != ml + mm:
                        continue
                    if izi + izj != izl + izm:
                        continue

                    if (
                        (ni, li, ji),
                        (nj, lj, jj),
                        (nl, ll, jl),
                        (nm, lm, jm),
                    ) in j_tot_i_tot.keys():
                        # WE NEED TO ADD ALL THE ANTISYMMETRIES BEFORE

                        j_tot_range, i_tot_range = j_tot_i_tot[
                            (ni, li, ji), (nj, lj, jj), (nl, ll, jl), (nm, lm, jm)
                        ]

                        value = 0.0
                        value_j = 0.0
                        for j_tot in j_tot_range:
                            for i_tot in i_tot_range:
                                
                                #phij, phlm = (-1) ** int(ji + jj + j_tot + i_tot), (-1) ** int(jl + jm + j_tot + i_tot)

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

                                phaseJT = (-1.0) ** (j_tot + i_tot)
                                # normalization
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

                                if (abs(nij) == 0.0) or (abs(nlm) == 0.0):
                                    continue

                                value = (
                                    value
                                    + cg_initial
                                    * cg_final
                                    * cg_iso_initial
                                    * cg_iso_final
                                    * matrix_outcome
                                    / (nij * nlm)
                                )


                        if value != 0:
                            matrix[(i, j, l, m)] = value  # regular value
                            if m != l:
                                matrix[(i, j, m, l)] = -1 * value  # asymmetric final
                            if j != i:
                                matrix[(j, i, l, m)] = -1 * value  # asymmetric initial
                            matrix[(j, i, m, l)] = value  # double asymmetric

                        


                            # matrix[(l,m,j,i)]=value   # inversion

    old_matrix_keys = list(matrix.keys())
    for key in old_matrix_keys:
        i, j, l, m = key
        matrix[(m, l, j, i)] = matrix[key]




    return matrix


def get_twobody_nuclearshell_model(file_name: str):

    j_tot_i_tot, scattering_values = scattering_matrix_reader(file_name=file_name)
    SPG = SingleParticleState(file_name=file_name)
    twobody_matrix= compute_nuclear_twobody_matrix(
        spg=SPG, j_tot_i_tot=j_tot_i_tot, scattering_values=scattering_values
    )

    return twobody_matrix, SPG.energies



def write_j_square_twobody_file(filename:str):
    
    file=open(filename)

    title=file.readline()
    jsquare_title='J^2'+title

    singleparticlestate_info=file.readline().strip().split()

    singleparticlestate_number=int(singleparticlestate_info[1])

    states=singleparticlestate_info[-singleparticlestate_number:]

    double_j_states=[]

    for state in states:
        
        double_j=int(state[-1])
        double_j_states.append(double_j)
        
    fileJ2=open(filename+'_j2','w')
    fileJ2.write('   J^2 ' + title )
    fileJ2.write('%i %i '  % (1, len(states)))
    for state in states:
        fileJ2.write('%s '% state)
    fileJ2.write('\n')
    fileJ2.write('0. 0. 0. \n')
    fileJ2.write('0. 0. 0. \n')
    for a in range(singleparticlestate_number):
        for b in range(singleparticlestate_number):
            double_ja=double_j_states[a]
            double_jb=double_j_states[b]
            totalj=np.arange(np.abs(double_ja-double_jb),np.abs(double_ja+double_jb)+2,2)/2
            fileJ2.write('%2i %2i %s %s %s %s %i %i \n'  % (0, 1, states[a], states[b], states[a], states[b], totalj[0], totalj[-1]))
            
            print('J=',totalj)
            for t in [0,1]:
                for j in totalj:
                    # is this a constrain (???)
                    tpj = t + j
                    if ( tpj % 2 == 0 and double_ja == double_jb ):
                        j_square_value = 0
                    else:
                        j_square_value=j*(j+1)-double_ja*(double_ja/2+1)/2-double_jb*(double_jb/2+1)/2
                    fileJ2.write(' %f '  %  j_square_value )
                fileJ2.write('\n')
            
    fileJ2.close()
      
      
class J2operator(FermiHubbardHamiltonian):
    
    def __init__(self, size_a, size_b, nparticles_a, nparticles_b, single_particle_states:List,j_square_filename:str,symmetries = None):
        super().__init__(size_a, size_b, nparticles_a, nparticles_b, symmetries)
        
        
        self.__get_single_particle_term(single_particle_states=single_particle_states)
        self.__get_twobody_matrix(j_square_filename=j_square_filename)
        self.get_hamiltonian()
    
    def __get_single_particle_term(self,single_particle_states:List):
        diag_j=np.zeros(len(single_particle_states))
        label=[]
        for i in range(diag_j.shape[0]):
            n,l,j,m,_,tz=single_particle_states[i]
            diag_j[i]=j*(j+1)
            self.get_external_potential(diag_j)
        
    def __get_twobody_matrix(self,j_square_filename:str):
        matrix_j,_=get_twobody_nuclearshell_model(file_name=j_square_filename)
        self.get_twobody_interaction(twobody_dict=matrix_j)    
        
    def j2_operator(self):
        return self.hamiltonian
    
    def j_value(self,psi:np.ndarray):
        
        j2=psi.transpose().conjugate().dot(self.j2_operator().dot(psi))
        print(j2.shape)
        jvalue=0.5 * ( np.sqrt(4.0 * j2 + 1) - 1 )
        return jvalue
    
    


class QuadrupoleOperator(FermiHubbardHamiltonian):
    
    def __init__(self, size_a, size_b, nparticles_a, nparticles_b, symmetries = None,single_particle_basis:List=None,operator_symmetries:List=None):
        super().__init__(size_a, size_b, nparticles_a, nparticles_b, symmetries)
        
        self.single_particle_basis=single_particle_basis
        
        self.__load_quadrupole_matrix()
        self.__get_manybody_operator(operator_symmetries=operator_symmetries)
        
    def __load_quadrupole_matrix(self,):
        
        self.quadrupole_reduced_matrix_dictionary={}
        
        # nlj_a nlj_b structure
        self.quadrupole_reduced_matrix_dictionary[(0,1,3/2),(0,1,3/2)]=-1.410
        self.quadrupole_reduced_matrix_dictionary[(0,1,3/2),(0,1,1/2)]=-1.410
        self.quadrupole_reduced_matrix_dictionary[(0,1,1/2),(0,1,3/2)]=1.410
        
        self.quadrupole_reduced_matrix_dictionary[(0,2,5/2),(0,2,5/2)]=-2.585
        self.quadrupole_reduced_matrix_dictionary[(0,2,5/2),(1,0,1/2)]=-2.185
        self.quadrupole_reduced_matrix_dictionary[(0,2,5/2),(0,2,3/2)]=-1.293
        
        self.quadrupole_reduced_matrix_dictionary[(1,0,1/2),(0,2,5/2)]=1.293
        self.quadrupole_reduced_matrix_dictionary[(1,0,1/2),(0,2,3/2)]=-1.784
        
        self.quadrupole_reduced_matrix_dictionary[(0,2,3/2),(0,2,5/2)]=1.293
        self.quadrupole_reduced_matrix_dictionary[(0,2,3/2),(1,0,1/2)]=1.784
        self.quadrupole_reduced_matrix_dictionary[(0,2,3/2),(0,2,3/2)]=-1.975
        
        
    def __get_the_quadrupole_matrix(self,mu:int):
        
        state_encoding=self.single_particle_basis
        
        self.quadrupole_matrix={}
        for idx_a,a in enumerate(state_encoding):
            for idx_b,b in enumerate(state_encoding):
                na,la,ja,ma,ta,tza=a
                nb,lb,jb,mb,tb,tzb=b
                
                
                # get the Clebasch Gordan coefficient related to the transition a-b
                # (we have to double check if the W-E theorem returns the order of the variables
                # in the C-G in this way)
                if ja >= np.abs(jb-2) and ja <= np.abs(jb+2):

                    cg_term_j=ClebschGordan(j1=jb,j2=2,J=ja)
                    cg_value=SelectCG(cg_term_j,j1=jb,m1=mb,j2=2,m2=mu,J=ja,M=ma)
                else:
                    cg_value=0.
                
                    
                if tza==tzb:
                    if ((na,la,ja),(nb,lb,jb)) in self.quadrupole_reduced_matrix_dictionary.keys():
                        quadrupole_value= self.quadrupole_reduced_matrix_dictionary[(na,la,ja),(nb,lb,jb)]*cg_value/np.sqrt(2*ja+1)
                else:
                    quadrupole_value=0.
                
                if np.abs(quadrupole_value)<10*-10:
                    continue
                else:
                    self.quadrupole_matrix[(idx_a,idx_b)]=quadrupole_value
                    
            
                
    def __get_manybody_operator(self,operator_symmetries:List):
        
        self.quadrupole_operator={} # it is gonna be a dictionary
        self.quadrupole_matrices={} # same for the matrix in the single particle basis
        mus=[-2,-1,0,1,2]
        
        for mu in mus:
            self.__get_the_quadrupole_matrix(mu=mu)
            self.quadrupole_matrices[mu]=self.quadrupole_matrix
            
            qab=0.
            for a in range(self.size_a+self.size_b):
                for b in range(self.size_a+self.size_b):
                    if (a,b) in self.quadrupole_matrices[mu].keys():
                        
                        full_cond=True
                        if operator_symmetries is not None:
                            idxs=[a,b]
                            for sym in operator_symmetries:
                                cond=sym(idxs)
                                full_cond=cond*full_cond
                        if full_cond:
                            qab+=self.quadrupole_matrices[mu][(a,b)]*self.adag_a_matrix(i=a,j=b)
            self.quadrupole_operator[mu]=qab
            
    
    def deformation_value(self,psi:np.ndarray):
        
        tot_value=0.
        for _,op in self.quadrupole_operator.items():
            print('op=',op)
            value=psi.transpose().conjugate().dot(op.dot(psi))
            tot_value+=value**2
            
        return np.sqrt(tot_value)
            