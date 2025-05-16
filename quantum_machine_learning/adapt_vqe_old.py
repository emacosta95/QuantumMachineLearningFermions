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
        self.twobody_operator_pool: Dict = None
        self.onebody_operator_pool: Dict = None

        # commutator operators
        self.twobody_commutator_pool: List = None
        self.onebody_commutator_pool: List = None

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

    def set_system(
        self,
        hamiltonian: np.ndarray,
        psi0: np.ndarray,
        operator_pool: Dict,
    ):

        self.hamiltonian = hamiltonian
        self.psi0 = psi0
        self.operator_pool = operator_pool
        
    def __select_new_operator(self):

        max = -1000
        psi=self.__compute_psi(self.weights)
        sigma = self.hamiltonian @ psi
        
        self.grad_tolerance=0.
        for key, obop in self.operator_pool.items():
            value = 2*np.real(sigma.conjugate().transpose().dot(obop.dot( psi)))
            self.grad_tolerance+=value**2
            #print("value=", value, list(self.onebody_operator_pool.keys())[i], "\n")
            if np.abs(value) > max:
                max = np.abs(value)
                selected_operator = obop
                selected_key = key
                
                
        
        if self.operator_action == []:
            self.weights = np.zeros(1)
        else:
            self.weights = np.append(self.weights, 0.0)
        self.operator_action.append(selected_operator)
        self.operator_action_info.append(selected_key)
        
        self.grad_tolerance=np.sqrt(self.grad_tolerance)
        print('grad tolerance=',self.grad_tolerance)
        
    def __compute_psi(self,weights:np.ndarray) -> np.ndarray:
        psi = self.psi0.copy()
        if weights is not(None):
            
            for i, w in enumerate(weights):
                # print(np.conj(expm(self.weights[i] * op).T) @ expm(self.weights[i] * op))
                psi = scipy.sparse.linalg.expm_multiply(w * self.operator_action[i],psi) 
                # psi = psi / np.linalg.norm(psi)

        return psi

    def __compute_gradient(
        self,weights:np.ndarray
    ):

        psi = self.__compute_psi(weights=weights)

        # energy value
        #energy = np.conj(psi.T) @ self.hamiltonian @ psi

        sigma = self.hamiltonian @ psi
        n_tot = len(self.operator_action)
        grad: np.ndarray = np.zeros(n_tot)
        for i in range(n_tot):
            
            grad[n_tot-1-i]=np.real(sigma.transpose().conjugate().dot(self.operator_action[n_tot - 1 - i].dot( psi)))
            sigma = (
                scipy.sparse.linalg.expm_multiply(
                    -1
                    * self.weights[n_tot - 1 - i]
                    * self.operator_action[n_tot - 1 - i]
                ,
                 sigma)
            )
            psi = (
                scipy.sparse.linalg.expm_multiply(
                    -1
                    * self.weights[n_tot - 1 - i]
                    * self.operator_action[n_tot - 1 - i]
                , psi)
            )

        return grad
    
    def compute_energy_functional(
        self,weights:np.ndarray
    ):

        psi = self.__compute_psi(weights)
        psi.transpose().conj() @ self.hamiltonian @ psi
        #print(f"energy value={self.energy:.3f} \n")
        return psi.transpose().conj() @ self.hamiltonian @ psi

    def optimization(
        self,
    ):

        #while energy_gap > self.tolerance:
        while(self.grad_tolerance>self.tolerance):
            self.__select_new_operator()
            #gradient, energy = self.__compute_gradient(psi)

            # classical optimization

            # while de > self.tolerance:
            #     # update
            #     self.weights = self.weights - self.learning_rate * gradient
            #     psi=self.__compute_psi()
            #     gradient, energy = self.__compute_gradient(psi=psi)
            #     de = np.abs(energy_old_classical_opt - energy)
            #     energy_old_classical_opt = energy
            #     print("e,gradient=", energy, np.average(gradient), "\n")
            
            # optimization algorithm
            res=minimize(self.compute_energy_functional, self.weights, args=(), method='BFGS', jac=self.__compute_gradient, tol=10**-8, callback=None, options=None)

            self.weights=res.x
            self.energy=self.compute_energy_functional(self.weights)
            self.psi=self.__compute_psi(self.weights)
            print('Optimization Success=',res.success)
            print('energy=',self.energy)
            print('gradient=',self.__compute_gradient(self.weights),'\n')
            
