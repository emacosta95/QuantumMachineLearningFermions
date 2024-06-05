import numpy as np

from src.cg_utils import ClebschGordan,SelectCG
import matplotlib.pyplot as plt
from tqdm import trange
from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from scipy.sparse.linalg import eigsh

from typing import List,Dict,Tuple,Text,Optional

class SingleParticleState():

    def __init__(self,file_name:str) -> None:

        with open(file_name, "r") as file:
            lines = file.readlines()

        labels=lines[1].split()    
        energy_values = [float(element) for element in lines[2].split()]
        print(energy_values)

        self.state_encoding:List=[]
        self.energies:List=[]
        for i_z in [1/2,-1/2]:
            for i,label in enumerate(labels[2:]):
                n=int(label[0])
                l=int(label[1])
                two_j=int(label[-1])
                two_m_range=-1*two_j+np.arange(0,2*two_j+2,2)
                # we put the n=1 restriction
                # if n==1 or (n==2 and two_j==3):
                for two_m in two_m_range:
                    self.state_encoding.append((n,l,two_j/2,two_m/2,1/2,i_z))
                    self.energies.append(energy_values[i])

        self.energies=np.asarray(self.energies)

    def get_index(self,state:Tuple):

        idx=self.state_encoding.index(state)
        return idx
    
    
    def projection_conservation(self,idxs:List[int]):
        
        nbody=len(idxs)//2
        
        t_z_final=0.
        m_final=0
        for idx in idxs[:nbody]:
            
            (n,l,j,m,t,t_z)=self.state_encoding[idx]
            t_z_final+=t_z
            m_final+=m
            
        t_z_initial=0.
        m_initial=0
        for idx in idxs[nbody:]:
            
            (n,l,j,m,t,t_z)=self.state_encoding[idx]
            t_z_initial+=t_z
            m_initial+=m
            
        condition= (m_initial==m_final) and (t_z_final==t_z_initial)
        
        return condition
        
    def compute_j_m_exp_value(self,psi:np.ndarray,basis:np.ndarray):
        
        j_ave=0.
        m_ave=0.
        for basis_index,b in enumerate(basis):
            
            idxs=np.nonzero(b)[0]
            for idx in idxs:
                
                (n,l,j,m,t,t_z)=self.state_encoding[idx]
                j_ave+=j*psi[basis_index].conjugate()*psi[basis_index]
                m_ave+=m*psi[basis_index].conjugate()*psi[basis_index]
            
        return j_ave,m_ave


def krond(tuple_a, tuple_b):  # Kronecker delta
    krond = 0
    if (
        tuple_a[0] == tuple_b[0]
        and tuple_a[1] == tuple_b[1]
        and tuple_a[2] == tuple_b[2]
    ):
        krond = 1
    return krond



def scattering_matrix_reader(file_name:str)-> Tuple[Dict]:
    with open(file_name, "r") as file:
        lines = file.readlines()

 

    matrix_info:List=[]
    matrix_entries:List=[]
    matrix_entries_string:List=[]

    for i in range(4,len(lines),3):
        print(lines[i])
        print(lines[i+1])
        print(lines[i+2])
        float_line_1=[(element) for element in lines[i].split()]
        float_line_2 = [float(element) for element in lines[i+1].split()]
        float_line_3 = [float(element) for element in lines[i+2].split()]

        line_2=[(element) for element in lines[i+1].split()]
        line_3 = [(element) for element in lines[i + 2].split()]
        matrix_info.append(float_line_1)
        matrix_entries.append(float_line_2)
        matrix_entries.append(float_line_3)
        matrix_entries_string.append(line_2)
        matrix_entries_string.append(line_3)
        
    
    j_tot_i_tot:Dict={}
    scattering_values:Dict={}
#    get the matrix entries as function of J and I
    for i in range(len(matrix_info)):
        tot_iso_range = np.arange(int(matrix_info[i][0]), int(matrix_info[i][1]) + 1)
        # print('iso range=',tot_iso_range,'\n')
        tot_j_range = np.arange(int(matrix_info[i][-2]), int(matrix_info[i][-1]) + 1)
        n1f = int(matrix_info[i][2][0])
        n2f = int(matrix_info[i][3][0])
        n1i = int(matrix_info[i][4][0])
        n2i = int(matrix_info[i][5][0])

        l1f = int(matrix_info[i][2][1])
        l2f = int(matrix_info[i][3][1])
        l1i = int(matrix_info[i][4][1])
        l2i = int(matrix_info[i][5][1])

        # print('j range=',tot_j_range,'\n')
        j1f = int(matrix_info[i][2][-1])/2
        j2f = int(matrix_info[i][3][-1])/2
        j1i = int(matrix_info[i][4][-1])/2
        j2i = int(matrix_info[i][5][-1])/2

        j_tot_i_tot[(n1f,l1f,j1f),(n2f,l2f,j2f),(n1i,l1i,j1i),(n2i,l2i,j2i)]=(tot_j_range,tot_iso_range)

        if (n1i,l1i,j1i)!=(n2i,l2i,j2i):
            j_tot_i_tot[
                (n1f, l1f, j1f),
                (n2f, l2f, j2f),
                (n2i, l2i, j2i),(n1i, l1i, j1i)
            ] = (tot_j_range, tot_iso_range)

        if (n1f,l1f,j1f)!=(n2f,l2f,j2f):
            j_tot_i_tot[
                (n2f, l2f, j2f),(n1f, l1f, j1f), (n1i, l1i, j1i), (n2i, l2i, j2i)
            ] = (tot_j_range, tot_iso_range)

        if (n1f, l1f, j1f) != (n2f, l2f, j2f) and (n1i, l1i, j1i) != (n2i, l2i, j2i):
            j_tot_i_tot[
                (n2f, l2f, j2f),(n1f, l1f, j1f), (n2i, l2i, j2i), (n1i, l1i, j1i)
            ] = (tot_j_range, tot_iso_range)
        # si deve fare qui sto lavoro di anti symmetria IMPORTANTISSIMO
        # matrix_entries=np.asarray(matrix_entries)
        for matrix_jdx, j_tot in enumerate(tot_j_range):
            for matrix_idx, i_tot in enumerate(tot_iso_range):

                scattering_values[((n1f,l1f,j1f),(n2f,l2f,j2f),(n1i,l1i,j1i),(n2i,l2i,j2i),j_tot,i_tot)]=matrix_entries[2 * i + matrix_idx][matrix_jdx]

                if (n1i,l1i,j1i)!=(n2i,l2i,j2i):
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
                    print(
                        (-1) ** int(j2i + j1i + j_tot + i_tot)
                        * matrix_entries[2 * i + matrix_idx][matrix_jdx]
                    )

                if (n1f,l1f,j1f)!=(n2f,l2f,j2f):
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

                if (n1f, l1f, j1f) != (n2f, l2f, j2f) and (n1i, l1i, j1i) != (n2i, l2i, j2i):
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
            
        
    return j_tot_i_tot,scattering_values



def compute_nuclear_twobody_matrix(spg,j_tot_i_tot:Dict,scattering_values:Dict)->Dict:
    
    
    
    
    matrix:Dict={}
    for i in trange(len(spg.state_encoding)):
        for j in range(i,len(spg.state_encoding)):
            for l in range(len(spg.state_encoding)):
                for m in range(l,len(spg.state_encoding)):

                    (ni,li,ji,mi,ii,izi)=spg.state_encoding[i]
                    (nj, lj, jj, mj, ij, izj) = spg.state_encoding[j]
                    (nl, ll, jl, ml, il, izl) = spg.state_encoding[l]
                    (nm, lm, jm, mm, im, izm) = spg.state_encoding[m]


                    if ((-1) ** (li+lj) != (-1) ** (ll+lm) ):
                        continue
                    if ( mi + mj != ml + mm ):
                        continue 
                    if ( izi + izj != izl + izm ):
                        continue 



                    if ((ni,li,ji),(nj,lj,jj),(nl,ll,jl),(nm,lm,jm)) in j_tot_i_tot.keys():
                        # WE NEED TO ADD ALL THE ANTISYMMETRIES BEFORE


                        j_tot_range,i_tot_range=j_tot_i_tot[(ni,li,ji),(nj,lj,jj),(nl,ll,jl),(nm,lm,jm)]

                        value=0.

                        for j_tot in j_tot_range:
                            for i_tot in i_tot_range:

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
                                    1/2,
                                    1/2,
                                    i_tot,
                                )

                                cg_iso_final_list = ClebschGordan(
                                    1 / 2,
                                    1 / 2,
                                    i_tot,
                                )

                                matrix_outcome=scattering_values[((ni, li, ji),
                                            (nj, lj, jj),
                                            (nl, ll, jl),
                                            (nm, lm, jm),
                                            j_tot,
                                            i_tot,)
                                        ]


                                for m_tot in np.linspace(-j_tot, j_tot, int(2 * j_tot) + 1):
                                    for iz_tot in np.linspace(-i_tot,i_tot,int(2*i_tot)+1):

                                        # cg values
                                        cg_initial=SelectCG(
                                                cg_initial_list,
                                                jl,
                                                ml,
                                                jm,
                                                mm,
                                                j_tot,
                                                m_tot,
                                            )
                                        cg_final=SelectCG(
                                                cg_final_list, ji, mi, jj, mj, j_tot, m_tot
                                            )
                                        cg_iso_initial=SelectCG(
                                                cg_iso_initial_list,
                                                1 / 2,
                                                izl,
                                                1 / 2,
                                                izm,
                                                i_tot,
                                                iz_tot,
                                            )
                                        cg_iso_final = SelectCG(
                                            cg_iso_final_list,
                                            1 / 2,
                                            izi,
                                            1 / 2,
                                            izj,
                                            i_tot,
                                            iz_tot,
                                        )

                                        phaseJT =  (-1.0) ** (j_tot + i_tot)
                                        # normalization
                                        nij = np.sqrt(
                                            1.0
                                            - krond((ni, li, ji), (nj, lj, jj)) * phaseJT
                                        ) / (1.0 + krond((ni, li, ji), (nj, lj, jj)))
                                        nlm = np.sqrt(
                                            1.0
                                            - krond((nl, ll, jl), (nm, lm, jm)) * phaseJT
                                        ) / (1.0 + krond((nl, ll, jl), (nm, lm, jm)))




                                        if ( (abs(nij) == 0.0) or (abs(nlm) == 0.0)):
                                            continue  

                                        value = (
                                            value
                                            +cg_initial
                                            * cg_final
                                            * cg_iso_initial
                                            * cg_iso_final
                                            * matrix_outcome/(nij*nlm))

                        if value!=0:         
                            matrix[(i,j,l,m)]=value #regular value
                            if m!=l:
                                matrix[(i,j,m,l)]=-1*value #asymmetric final
                            if j!=i:
                                matrix[(j,i,l,m)]=-1*value #asymmetric initial
                            matrix[(j, i, m, l)] = value  # double asymmetric

                            # matrix[(l,m,j,i)]=value   # inversion

    old_matrix_keys=list(matrix.keys())                

    for key in old_matrix_keys:
        i,j,l,m=key
        matrix[(m,l,j,i)]=matrix[key]
        
    return matrix


def get_twobody_nuclearshell_model(file_name:str):
    
    j_tot_i_tot,scattering_values=scattering_matrix_reader(file_name=file_name)
    SPG=SingleParticleState(file_name=file_name)
    twobody_matrix=compute_nuclear_twobody_matrix(spg=SPG,j_tot_i_tot=j_tot_i_tot,scattering_values=scattering_values)
    
    return twobody_matrix,SPG.energies

class FermiHubbardHamiltonian(FemionicBasis):
    
    def __init__(self, size_a: int, size_b: int, nparticles_a: int, nparticles_b: int) -> None:
        
        super().__init__(size_a, size_b, nparticles_a, nparticles_b)
        
        self.kinetic_operator=None
        self.external_potential=None
        self.twobody_operator=None
        
        self.hamiltonian=None
        
        self.dim_hilbert_space=self.basis.shape[0]
        
    def get_kinetic_operator(self,hopping_term:Optional[float]=None,adj_matrix:Optional[Dict]=None):
        
        if adj_matrix is None:
            
            adj_matrix={}
            # spin down (proton)
            for i in range(self.size_a-1):
                
                adj_matrix[(i,i+1)]=hopping_term
            
            # spin up (neutron)
            for i in range(self.size_a,self.size_a+self.size_b-1):
                
                adj_matrix[(i,i+1)]=hopping_term
            
        operator=0.
        for element in adj_matrix.items():
            
            (i,j),value=element
            operator=operator+value*self.adag_a_matrix(i=i,j=j)
            
        self.kinetic_operator=operator+operator.transpose().conjugate()

    def get_external_potential(self,external_potential:np.ndarray):
                
        operator=0.
        for i,v in enumerate(external_potential):
            
            operator=operator+v*self.adag_a_matrix(i=i,j=i)
            
        self.external_potential=operator               
    
    def get_twobody_interaction(self,twobody_dict:Dict):
                

        matrix_keys=twobody_dict.keys()
        matrix_values=list(twobody_dict.values())
        for q, indices in enumerate(matrix_keys):
            i1, i2, i3, i4 = indices

            value = matrix_values[q]
            #print(i1,i2,i3,i4,value)
            if q == 0:

                ham_int = (value * self.adag_adag_a_a_matrix(
                    i1=i1, i2=i2, j1=i4, j2=i3
                ))/4

            else:
                ham_int = ham_int + (value * (self.adag_adag_a_a_matrix(
                    i1=i1, i2=i2, j1=i4, j2=i3
                )))/4
        
        self.twobody_operator=ham_int  
        
        
    def get_hamiltonain(self,):
        
        self.hamiltonian=0.
        if self.kinetic_operator is not None:
            self.hamiltonian=self.kinetic_operator.copy()
        if self.external_potential is not None:
            self.hamiltonian=self.hamiltonian+self.external_potential.copy()
        if self.twobody_operator is not None:
            self.hamiltonian=self.hamiltonian+self.twobody_operator.copy()
            
        # we can add all the double check for the hamiltonian
        
    def get_spectrum(self,n_states:int):
        
        e,states=eigsh(self.hamiltonian, k=n_states, which="SA")
        
        return e,states