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
        self.epochs: int = None
        self.learning_rate: float = None

        # energy
        self.energy = 0.0
        self.psi=0.

    def set_hyperparameters(
        self, learning_rate: float, tolerance: float = None, epochs: int = None
    ):

        self.learning_rate = learning_rate
        if tolerance == None:
            self.epochs = epochs
        else:
            self.tolerance = tolerance

    def set_hamiltonian(
        self,
        hamiltonian: np.ndarray,
        ):
        
        # check all the options for the hamiltonian, like is herm so on and so forth

        self.hamiltonian = hamiltonian
        
    def set_reference_psi(self,reference_psi:np.ndarray):
        
        # we can add preparations or other methods
        self.psi0=reference_psi
        
    def set_operators_pool(self,operator_pool: Dict):
        
        
        # set some double check property of the operator pool
        
        self.operator_pool: Dict = operator_pool
        


    def __select_new_operator(self):

        max = -1000
        psi=self.__compute_psi(self.weights)
        sigma = self.hamiltonian @ psi
        
        self.grad_tolerance=0.
        values:List=[]
        for key, op in self.operator_pool.items():
            value = 2*np.real(sigma.transpose().dot(op.dot( psi)))
            values.append(value)
            self.grad_tolerance+=value**2
            #print("value=", value, list(self.onebody_operator_pool.keys())[i], "\n")
            if np.abs(value) > max:
                max = np.abs(value)
                selected_operator = op
                selected_key = key
                
        self.__update_weights()
        
        self.__update_operators()
        
        self.grad_tolerance=np.sqrt(self.grad_tolerance)
        
    def model_preparation(self,):
        self.__select_new_operator()
        
    def __compute_psi(self,weights:np.ndarray) -> np.ndarray:
        psi = self.psi0.copy()
        if weights is not(None):
            
            for i, w in enumerate(weights):
                # print(np.conj(expm(self.weights[i] * op).T) @ expm(self.weights[i] * op))
                psi = scipy.sparse.linalg.expm_multiply(w * self.operator_action[i],psi) 
                # psi = psi / np.linalg.norm(psi)

        return psi

    def backward(
        self,weights:np.ndarray
    ):

        psi = self.__compute_psi(weights=weights)

        # energy value
        #energy = np.conj(psi.T) @ self.hamiltonian @ psi

        sigma = self.hamiltonian @ psi
        n_tot = len(self.operator_action)
        grad: np.ndarray = np.zeros(n_tot)
        for i in range(n_tot):
            grad[n_tot - 1 - i] = 2 * np.real(
                np.conj(sigma.T) @ self.operator_action[n_tot - 1 - i] @ psi
            )
            sigma = (
                scipy.sparse.linalg.expm_multiply(
                    -1
                    * weights[n_tot - 1 - i]
                    * self.operator_action[n_tot - 1 - i]
                ,
                 sigma)
            )
            psi = (
                scipy.sparse.linalg.expm_multiply(
                    -1
                    * weights[n_tot - 1 - i]
                    * self.operator_action[n_tot - 1 - i]
                , psi)
            )



        return grad
    
    
    def __update_weights(self,):
        if self.operator_action == []:
            self.weights = np.zeros(1)
        else:
            self.weights = np.append(self.weights, 0.0)
            
    def __update_operators(self,selected_operator:np.ndarray,selected_key:str,values:List):
        
        
        # conditions and constrains for the selection
        if selected_operator== self.operator_action[-1]:
            values=np.asarray(values)
            idx_max=np.argmax(values)
            values[idx_max]=-10**23
            new_idx_max=np.argmax(values)
            
            selected_operator=list(self.operator_pool.values())[new_idx_max]    
            selected_key=list(self.operator_pool.keys())[new_idx_max]    
        
        self.operator_action.append(selected_operator)
        self.operator_action_info.append(selected_key)
        
    def forward(
        self,weights
    ):

        psi = self.__compute_psi(weights)
        psi.transpose().conj() @ self.hamiltonian @ psi
        #print(f"energy value={self.energy:.3f} \n")
        return psi.transpose().conj() @ self.hamiltonian @ psi


            
