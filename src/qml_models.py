from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict
from scipy.linalg import expm
import scipy
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

        # histories
        self.history_energy = []
        self.history_grad = []

    def set_hamiltonian(
        self,
        hamiltonian: np.ndarray,
    ):

        # check all the options for the hamiltonian, like is herm so on and so forth

        self.hamiltonian = hamiltonian

    def set_reference_psi(self, reference_psi: np.ndarray):

        # we can add preparations or other methods
        self.psi0 = reference_psi

    def set_operators_pool(self, operator_pool: Dict, random: bool = False):

        # set some double check property of the operator pool

        self.operator_pool: Dict = operator_pool

        self.random = random

    def __select_new_operator(self):

        max = -1000
        psi = self.__compute_psi(self.weights)
        sigma = self.hamiltonian @ psi

        self.grad_tolerance = 0.0
        values: List = []
        for key, op in self.operator_pool.items():
            value = 2 * np.real(sigma.conjugate().transpose().dot(op.dot(psi)))
            values.append(value)
            self.grad_tolerance += value**2
            # print("value=", value, list(self.onebody_operator_pool.keys())[i], "\n")
            if np.abs(value) > max:
                max = np.abs(value)
                selected_operator = op
                selected_key = key

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

    def __compute_psi(self, weights: np.ndarray) -> np.ndarray:
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

        psi = self.__compute_psi(weights=weights)
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

        psi = self.__compute_psi(weights)
        psi.transpose().conj() @ self.hamiltonian @ psi
        # print(f"energy value={self.energy:.3f} \n")
        self.energy = psi.transpose().conj() @ self.hamiltonian @ psi
        return self.energy

    def callback(self,*args):

        self.history_grad.append(self.grad)
        self.history_energy.append(self.energy)
        print('energy value=',self.energy,'\n')



class QAOAFermiHubbard:

    def __init__(
        self,
    ) -> None:

        # physics hyperparameters
        self.hamiltonian: np.ndarray = None
        self.psi0: np.ndarray = None
        self.hamiltonian_driving = None

        
        
        # optimization hyperparameter
        self.grad_tolerance: float = 1000
        self.weights: np.ndarray = None
        self.nstep=None

        # energy
        self.energy = 0.0
        self.grad = 0.0
        self.psi = 0.0

        # histories
        self.history_energy = []
        self.history_grad = []

    def set_hamiltonian(
        self,
        hamiltonian: np.ndarray,hamiltonian_driving:np.ndarray
    ):

        # check all the options for the hamiltonian, like is herm so on and so forth

        self.hamiltonian = hamiltonian
        self.hamiltonian_driving=hamiltonian_driving

    def set_reference_psi(self, reference_psi: np.ndarray):

        # we can add preparations or other methods
        self.psi0 = reference_psi

    def set_weights(self, total_step:int,initialization_type:str='ones',tf:float=None):

        # set some double check property of the operator pool
        if initialization_type=='ones':
            self.weights=np.ones((2*total_step))
            
        if initialization_type=='annealing':
            self.weights=np.zeros((2*total_step))
            self.weights[0:total_step]=np.linspace(0,tf,total_step)/tf
            self.weights[total_step:]=1-np.linspace(0,tf,total_step)/tf
            
        self.nstep=total_step
            

    def model_preparation(
        self,
    ):

        return None
    def __compute_psi(self,weights:np.ndarray) -> np.ndarray:

        
        psi = self.psi0.copy()
        #print(weights.shape,'weights shapeee')
        for i in range(self.nstep):
            # print(np.conj(expm(weights[i] * op).T) @ expm(weights[i] * op))
            psi = scipy.sparse.linalg.expm_multiply(
                -1j*weights[i] * self.hamiltonian_driving, psi
            )
            psi = scipy.sparse.linalg.expm_multiply(
                -1j*weights[i+self.nstep] * self.hamiltonian, psi
            )
            
            # psi = psi / np.linalg.norm(psi)

        return psi

    def backward(self,weights:np.ndarray ):

        # minimize reshape everytime the weights, we should put it on a callback


        psi = self.__compute_psi(weights=weights)
        # print(psi)
        # energy value
        # energy = np.conj(psi.T) @ self.hamiltonian @ psi

        sigma = self.hamiltonian @ psi

        # print('sigma here=',sigma)
        n_tot = 2*self.nstep
        grad: np.ndarray = np.zeros((n_tot))
        action_operator=[self.hamiltonian_driving,self.hamiltonian]
        for i in range(n_tot//2):
            # update immediately state psi
            psi = scipy.sparse.linalg.expm_multiply(
                1j * weights[n_tot - 1 - i] * self.hamiltonian, psi
            )
            psi = scipy.sparse.linalg.expm_multiply(
                1j * weights[n_tot - 1 - i-self.nstep] * self.hamiltonian_driving, psi
            )
            
            # add the exp contribution of the acting operator
            a_state=scipy.sparse.linalg.expm_multiply(
                -1j * weights[n_tot - 1 - i-self.nstep] * self.hamiltonian_driving, psi
            )
            b_state=scipy.sparse.linalg.expm_multiply(
                1j * weights[n_tot - 1 - i] * self.hamiltonian, sigma
            )
            
            # compute the gradient
            for a in range(2):
                grad[n_tot - 1 - i-(1-a)*self.nstep] = 2 * np.imag(
                    np.conj(b_state.T) @ action_operator[a] @ a_state
                )
                
            #update sigma state
            sigma = scipy.sparse.linalg.expm_multiply(
                1j * weights[n_tot - 1 - i] * self.hamiltonian, sigma
            )
            sigma = scipy.sparse.linalg.expm_multiply(
                1j * weights[n_tot - 1 - i-self.nstep] * self.hamiltonian_driving, sigma
            )
            
            

        self.grad = grad


        return grad



    def forward(self, weights):

        psi = self.__compute_psi(weights)
        psi.transpose().conj() @ self.hamiltonian @ psi
        # print(f"energy value={self.energy:.3f} \n")
        self.energy = (psi.transpose().conj() @ self.hamiltonian @ psi).real[0,0]
        return self.energy

    def callback(self,*args):

        self.history_grad.append(self.grad)
        self.history_energy.append(self.energy)
        
        print('energy value=',self.energy,'\n')



# class WeightPickAdaptVQEFermiHubbard:

#     def __init__(
#         self,
#     ) -> None:

#         # physics hyperparameters
#         self.hamiltonian: np.ndarray = None
#         self.psi0: np.ndarray = None
#         self.operator_action = []
#         self.operator_action_info = []

#         # pool operators
#         self.operator_pool: Dict = None

#         # commutator operators
#         self.commutator_pool: List = None

#         # optimization hyperparameter
#         self.grad_tolerance: float = 1000
#         self.weights: np.ndarray = None

#         # energy
#         self.energy = 0.0
#         self.grad = 0.0
#         self.psi = 0.0

#         # histories
#         self.history_energy = []
#         self.history_grad = []

#     def set_hamiltonian(
#         self,
#         hamiltonian: np.ndarray,
#     ):

#         # check all the options for the hamiltonian, like is herm so on and so forth

#         self.hamiltonian = hamiltonian

#     def set_reference_psi(self, reference_psi: np.ndarray):

#         # we can add preparations or other methods
#         self.psi0 = reference_psi

#     def set_operators_pool(self, operator_pool: Dict, random: bool = False,n_weights_selected:int):

#         # set some double check property of the operator pool

#         self.operator_pool: Dict = operator_pool

#         self.random = random
        
#         self.n_weights_selected=n_weights_selected

#     def __select_new_operator(self):

#         max = -1000
#         psi = self.__compute_psi(self.weights)
#         sigma = self.hamiltonian @ psi

#         self.grad_tolerance = 0.0
#         values: List = []
#         for key, op in self.operator_pool.items():
#             value = 2 * np.real(sigma.conjugate().transpose().dot(op.dot(psi)))
#             values.append(value)
#             self.grad_tolerance += value**2
#             # print("value=", value, list(self.onebody_operator_pool.keys())[i], "\n")
#             if np.abs(value) > max:
#                 max = np.abs(value)
#                 selected_operator = op
#                 selected_key = key

#         self.__update_weights()

#         # self.operator_action.append(selected_operator)
#         # self.operator_action_info.append(selected_key)
#         self.__update_operators(
#             selected_operator=selected_operator,
#             selected_key=selected_key,
#             values=values,
#         )

#         self.grad_tolerance = np.sqrt(self.grad_tolerance)

#     def __select_random_new_operator(self):

#         self.__update_weights()

#         idx = np.random.randint(0, len(list(self.operator_pool.values())))
#         selected_key = list(self.operator_pool.keys())[idx]
#         selected_operator = list(self.operator_pool.values())[idx]

#         self.operator_action.append(selected_operator)
#         self.operator_action_info.append(selected_key)

#     def model_preparation(
#         self,
#     ):

#         if self.random:

#             self.__select_random_new_operator()

#         else:
#             self.__select_new_operator()

#     def __compute_psi(self, weights: np.ndarray) -> np.ndarray:
#         psi = self.psi0.copy()
#         if weights is not (None):

#             for i, w in enumerate(weights):
#                 # print(np.conj(expm(self.weights[i] * op).T) @ expm(self.weights[i] * op))
#                 psi = scipy.sparse.linalg.expm_multiply(
#                     w * self.operator_action[i], psi
#                 )
#                 # psi = psi / np.linalg.norm(psi)

#         return psi

#     def backward(self, selected_weights: np.ndarray):
#         # we need to figure out how we can update just the selected weights
#         weights=self.from_selected2weights(selected_weights)

#         psi = self.__compute_psi(weights=weights)
#         # print(psi)
#         # energy value
#         # energy = np.conj(psi.T) @ self.hamiltonian @ psi

#         sigma = self.hamiltonian @ psi

#         # print('sigma here=',sigma)
#         n_tot = len(self.operator_action)
#         proper_index=0
#         grad: np.ndarray = np.zeros(len(self.selected_weights_indices))
#         for i in range(n_tot):
#             if n_tot-1-i in self.selected_weights_indices:    
#                 grad[proper_index] = 2 * np.real(
#                     np.conj(sigma.T) @ self.operator_action[n_tot - 1 - i] @ psi
#                 )
#                 proper_index+=1
#                 # we update just the weights that we have selected
#             sigma = scipy.sparse.linalg.expm_multiply(
#                 -1 * weights[n_tot - 1 - i] * self.operator_action[n_tot - 1 - i], sigma
#             )
#             psi = scipy.sparse.linalg.expm_multiply(
#                 -1 * weights[n_tot - 1 - i] * self.operator_action[n_tot - 1 - i], psi
#             )

#         self.grad = grad

#         return grad

#     def __update_weights(
#         self,
#     ):
#         if self.operator_action == []:
#             self.weights = np.zeros(1)
#         else:
#             self.weights = np.append(self.weights, 0.0)

#     def __update_operators(
#         self, selected_operator: np.ndarray, selected_key: str, values: List
#     ):

#         # conditions and constrains for the selection
#         if len(self.operator_action) != 0:
#             if selected_operator is self.operator_action[-1]:
#                 values = np.asarray(values)
#                 idx_max = np.argmax(values)
#                 values[idx_max] = -(10**23)
#                 new_idx_max = np.argmax(values)

#                 selected_operator = list(self.operator_pool.values())[new_idx_max]
#                 selected_key = list(self.operator_pool.keys())[new_idx_max]

#         self.operator_action.append(selected_operator)
#         self.operator_action_info.append(selected_key)

#     def forward(self, weights):

#         psi = self.__compute_psi(weights)
#         psi.transpose().conj() @ self.hamiltonian @ psi
#         # print(f"energy value={self.energy:.3f} \n")
#         self.energy = psi.transpose().conj() @ self.hamiltonian @ psi
#         return self.energy

#     def callback(self,*args):

#         self.history_grad.append(self.grad)
#         self.history_energy.append(self.energy)
#         print('energy value=',self.energy,'\n')
        
        
#     def select_weights(self,):
        
#         self.selected_weights_indices=np.random.sample(np.arange(self.weights.shape[0]),self.n_weights_selected)

    #def selected