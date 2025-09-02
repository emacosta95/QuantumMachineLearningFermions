from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict
from scipy.linalg import expm
import scipy
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize


class AdaptVQEFermiHubbard:

    def __init__(
        self,
    ) -> None:

        # physics hyperparameters
        self.hamiltonian: np.ndarray = None
        self.psi0: np.ndarray = None
        self.operator_action = []
        self.operator_action_info = []
        self.exact_energy=0.

        # pool operators
        self.operator_pool: Dict = None

        # commutator operators
        self.commutator_pool: List = None

        # optimization hyperparameter
        self.grad_tolerance: float = 1000
        self.weights: np.ndarray = None

        # energy
        self.energy = 0.0
        self.grad = 0.0
        self.psi = 0.0
        
        self.gradient_selected=[]

        # histories
        self.history_energy = []
        self.history_grad = []
        
        # computational counter
        self.total_operation_miquel=0

    def set_hamiltonian(
        self,
        hamiltonian: np.ndarray,
    ):

        # check all the options for the hamiltonian, like is herm so on and so forth

        self.hamiltonian = hamiltonian

    def set_reference_psi(self, reference_psi: np.ndarray,energy_gs:np.ndarray):

        # we can add preparations or other methods
        self.psi0 = reference_psi
        self.exact_energy=energy_gs

    def set_operators_pool(self, operator_pool: Dict, random: bool = False):

        # set some double check property of the operator pool

        self.operator_pool: Dict = operator_pool

        self.random = random

    def __select_new_operator(self):

        max = -1000
        psi = self.compute_psi(self.weights)
        sigma = self.hamiltonian @ psi

        self.grad_tolerance = 0.0
        values: List = []
        for key, op in self.operator_pool.items():
            value = 2 * np.abs(np.real(sigma.conjugate().transpose().dot(op.dot(psi))))
            values.append(value)
            self.grad_tolerance += value**2
            # print("value=", value, list(self.onebody_operator_pool.keys())[i], "\n")
            if value > max:
                max = np.abs(value)
                selected_operator = op
                selected_key = key
            
        #print('maximum value=',max)
        self.gradient_selected.append(max)
        self.__update_weights()

        # self.operator_action.append(selected_operator)
        # self.operator_action_info.append(selected_key)
        self.__update_operators(
            selected_operator=selected_operator,
            selected_key=selected_key,
            values=values,
        )

        self.grad_tolerance = np.sqrt(self.grad_tolerance)

    def __select_random_new_operator(self):

        self.__update_weights()

        idx = np.random.randint(0, len(list(self.operator_pool.values())))
        selected_key = list(self.operator_pool.keys())[idx]
        selected_operator = list(self.operator_pool.values())[idx]

        self.operator_action.append(selected_operator)
        self.operator_action_info.append(selected_key)

    def model_preparation(
        self,
    ):

        if self.random:

            self.__select_random_new_operator()

        else:
            self.__select_new_operator()

    def compute_psi(self, weights: np.ndarray) -> np.ndarray:
        psi = self.psi0.copy()
        if weights is not (None):
            for i, w in enumerate(weights):
                # print(np.conj(expm(self.weights[i] * op).T) @ expm(self.weights[i] * op))
                psi = scipy.sparse.linalg.expm_multiply(
                    w * self.operator_action[i], psi
                )
                # psi = psi / np.linalg.norm(psi)

        return psi

    def backward(self, weights: np.ndarray):

        psi = self.compute_psi(weights=weights)
        # print(psi)
        # energy value
        # energy = np.conj(psi.T) @ self.hamiltonian @ psi

        sigma = self.hamiltonian @ psi

        # print('sigma here=',sigma)
        n_tot = len(self.operator_action)
        grad: np.ndarray = np.zeros(n_tot)
        for i in range(n_tot):
            grad[n_tot - 1 - i] = 2 * np.real(
                np.conj(sigma.T) @ self.operator_action[n_tot - 1 - i] @ psi
            )
            sigma = scipy.sparse.linalg.expm_multiply(
                -1 * weights[n_tot - 1 - i] * self.operator_action[n_tot - 1 - i], sigma
            )
            psi = scipy.sparse.linalg.expm_multiply(
                -1 * weights[n_tot - 1 - i] * self.operator_action[n_tot - 1 - i], psi
            )

        self.grad = grad

        return grad

    def __update_weights(
        self,
    ):
        if self.operator_action == []:
            self.weights = np.zeros(1)
        else:
            self.weights = np.append(self.weights, 0.0)

    def __update_operators(
        self, selected_operator: np.ndarray, selected_key: str, values: List
    ):

        # conditions and constrains for the selection
        if len(self.operator_action) != 0:
            if selected_operator is self.operator_action[-1]:
                values = np.asarray(values)
                idx_max = np.argmax(values)
                values[idx_max] = -(10**23)
                new_idx_max = np.argmax(values)

                selected_operator = list(self.operator_pool.values())[new_idx_max]
                selected_key = list(self.operator_pool.keys())[new_idx_max]

        self.operator_action.append(selected_operator)
        self.operator_action_info.append(selected_key)

    def forward(self, weights):

        psi = self.compute_psi(weights)
        psi.transpose().conj() @ self.hamiltonian @ psi
        # print(f"energy value={self.energy:.3f} \n")
        self.energy = psi.transpose().conj() @ self.hamiltonian @ psi
        self.total_operation_miquel+=len(self.operator_action_info)
        return self.energy

    def callback(self,*args):

        self.history_grad.append(self.grad)
        self.history_energy.append(self.energy)
        #print('energy value=',self.energy,'\n')
        
        
        #self.total_operation_metric+=1
        #print(f'total operations={self.total_operation_metric} \n')




class CCVQEFermiHubbard:

    def __init__(
        self,
    ) -> None:

        # physics hyperparameters
        self.hamiltonian: np.ndarray = None
        self.psi0: np.ndarray = None
        self.operator_action = []
        self.operator_action_info = []
        self.exact_energy=0.

        # pool operators
        self.operator_pool: Dict = None

        # commutator operators
        self.commutator_pool: List = None

        # optimization hyperparameter
        self.grad_tolerance: float = 1000
        self.weights: np.ndarray = None

        # energy
        self.energy = 0.0
        self.grad = 0.0
        self.psi = 0.0
        
        self.gradient_selected=[]

        # histories
        self.history_energy = []
        self.history_grad = []
        
        # computational counter
        self.total_operation_miquel=0

    def set_hamiltonian(
        self,
        hamiltonian: np.ndarray,
    ):

        # check all the options for the hamiltonian, like is herm so on and so forth

        self.hamiltonian = hamiltonian

    def set_reference_psi(self, reference_psi: np.ndarray,energy_gs:np.ndarray):

        # we can add preparations or other methods
        self.psi0 = reference_psi
        self.exact_energy=energy_gs

    def set_operators_pool(self, operator_pool: Dict, random: bool = False):

        # set some double check property of the operator pool

        self.operator_pool: Dict = operator_pool
        
        for op in self.operator_pool.values():
            self.operator_action.append(op)
        
        self.weights=np.zeros(len(self.operator_pool))

        self.random = random

    
    def __select_random_new_operator(self):

        self.__update_weights()

        idx = np.random.randint(0, len(list(self.operator_pool.values())))
        selected_key = list(self.operator_pool.keys())[idx]
        selected_operator = list(self.operator_pool.values())[idx]

        self.operator_action.append(selected_operator)
        self.operator_action_info.append(selected_key)

    def model_preparation(
        self,
    ):

        if self.random:

            self.__select_random_new_operator()

        else:
            self.__select_new_operator()

    def compute_psi(self, weights: np.ndarray) -> np.ndarray:
        psi = self.psi0.copy()
        if weights is not (None):
            for i, w in enumerate(weights):
                # print(np.conj(expm(self.weights[i] * op).T) @ expm(self.weights[i] * op))
                psi = scipy.sparse.linalg.expm_multiply(
                    w * self.operator_action[i], psi
                )
                # psi = psi / np.linalg.norm(psi)

        return psi

    def backward(self, weights: np.ndarray):

        psi = self.compute_psi(weights=weights)
        # print(psi)
        # energy value
        # energy = np.conj(psi.T) @ self.hamiltonian @ psi

        sigma = self.hamiltonian @ psi

        # print('sigma here=',sigma)
        n_tot = len(self.operator_action)
        grad: np.ndarray = np.zeros(n_tot)
        for i in range(n_tot):
            grad[n_tot - 1 - i] = 2 * np.real(
                np.conj(sigma.T) @ self.operator_action[n_tot - 1 - i] @ psi
            )
            sigma = scipy.sparse.linalg.expm_multiply(
                -1 * weights[n_tot - 1 - i] * self.operator_action[n_tot - 1 - i], sigma
            )
            psi = scipy.sparse.linalg.expm_multiply(
                -1 * weights[n_tot - 1 - i] * self.operator_action[n_tot - 1 - i], psi
            )

        self.grad = grad

        return grad

    def forward(self, weights):

        psi = self.compute_psi(weights)
        psi.transpose().conj() @ self.hamiltonian @ psi
        # print(f"energy value={self.energy:.3f} \n")
        self.energy = psi.transpose().conj() @ self.hamiltonian @ psi
        self.total_operation_miquel+=len(self.operator_action_info)

        return self.energy

    def callback(self,*args):

        self.history_grad.append(self.grad)
        self.history_energy.append(self.energy)
        print('energy value=',self.energy,'\n',flush=True)
        
        
        #self.total_operation_metric+=1
        #print(f'total operations={self.total_operation_metric} \n')



class NSMconstrains:
    def __init__(self,SPS,NSMHamiltonian):
        self.SPS=SPS
        self.NSMHamiltonian=NSMHamiltonian
        
    def miquel_constrainer(self,idxs:List[int]):

        if self.SPS.projection_conservation(idxs=idxs):
            if self.NSMHamiltonian.charge_computation(initial_indices=idxs[:2],final_indices=idxs[2:]):
                op=self.NSMHamiltonian.adag_adag_a_a_matrix(idxs[0],idxs[1],idxs[2],idxs[3])
                diag_op = sparse.diags(op.diagonal())

                non_diag_op =np.abs( op - diag_op)
                if not(np.isclose(non_diag_op.sum(),0.)):
                    condition=True
                else:
                    condition=False
            
            else:
                condition=False
        else:
            condition=False
                    
        return condition


    def miquel_constrainer_2(self,idxs:List[int]):
        _,_,j0,_,_,tz0=self.SPS.state_encoding[idxs[0]]
        _,_,j1,_,_,tz1=self.SPS.state_encoding[idxs[1]]
        _,_,j2,_,_,tz2=self.SPS.state_encoding[idxs[2]]
        _,_,j3,_,_,tz3=self.SPS.state_encoding[idxs[3]]
        
        j_tot_i = np.arange(start=int(np.abs(j0 - j1)), stop=int(j0 + j1) + 1)  # Include j0 + j1
        j_tot_f = np.arange(start=int(np.abs(j2 - j3)), stop=int(j2 + j3) + 1)  # Include j2 + j3
        #print(j_tot_i,j0,j1)
        if tz0==tz1:
            if j0==j1:
                j_tot_i=[j for j in j_tot_i if j % 2==0 ]
                #print('i=',j_tot_i,j0,j1)
            if j2==j3:
                j_tot_f=[j for j in j_tot_f if j % 2==0 ]
                #print('f=',j_tot_f,j2,j3,'\n')
            if set(j_tot_i) & set(j_tot_f):
                condition=True
            else:
                
                condition=False
        else:
        
            if set(j_tot_i) & set(j_tot_f):
                condition=True
            else:

                condition=False


                
        return condition

    def miquel_constrainer_3(self,idxs:List[int]):
        condition=False
        p=np.random.uniform(0,1)
        if self.SPS.projection_conservation(idxs=idxs):
            if p<1:
                condition=True
                    
        return condition