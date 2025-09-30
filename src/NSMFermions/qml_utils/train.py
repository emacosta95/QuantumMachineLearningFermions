from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict, Callable, Optional
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
from tqdm import trange


class Fit:

    def __init__(self, method: str, tolerance_opt: float, e_ref: float = None,tolerance_adapt=None) -> None:

        self.method: str = method
        self.tolerance = tolerance_opt
        self.tolerance_adapt=tolerance_adapt
        self.model = None

        self.configuration_checkpoint: Callable = None

    def init_model(self, model):
        self.model = model

    def run(self, epochs: Optional[int] = None):

        energy_history = []
        gradient_history = []
        e_old = -(10**25)
        de = 100
        tot_op=0

        if epochs is None:

            while de > self.tolerance_adapt:

                self.model.model_preparation()
                
                # optimization algorithm
                res = minimize(
                    self.model.forward,
                    self.model.weights,
                    method=self.method,
                    #jac=self.model.backward,
                    options={'ftol':self.tolerance,'gtol':10**-3},
                    callback=self.configuration_checkpoint,
                )
                self.model.weights = res.x
                energy = self.model.forward(self.model.weights)
                grad_energy = self.model.backward(self.model.weights)

                de = np.abs(energy - e_old)
                e_old = energy

                
                translator=np.array([3,2,1,0,5,4,9,8,7,6,11,10])
                # for i in range(12):
                #     print(SPS.state_encoding[i],i,'\n')
                    
                #print('Operator:',translator[self.model.operator_action_info[-1][0]],translator[self.model.operator_action_info[-1][1]],translator[self.model.operator_action_info[-1][2]],translator[self.model.operator_action_info[-1][3]],)
                print("Optimization Success=", res.success)
                print('weights=',self.model.weights)
                print(f"energy={energy:.5f}")
                print(f"energy={self.model.exact_energy:.5f}")
                print(f"de={de:.9f}")
                #print(f"average gradient={np.average(np.abs(grad_energy)):.15f} ")
                #print(f"grad tolerance={self.model.grad_tolerance:.15f} ")
                print(f'TOT_OPERATION_METRIC={self.model.total_operation_miquel}')
                print(f'LAYERS=',len(self.model.operator_action_info),)
                print('gradient selected=',self.model.gradient_selected[-1],'\n')
                #print('operator_action=',self.model.operator_action_info)
                energy_history.append(energy)
                gradient_history.append(np.average(np.abs(grad_energy)))

        else:
            for i in trange(epochs):
                self.model.model_preparation()

                # optimization algorithm
                res = minimize(
                    self.model.forward,
                    self.model.weights,
                    method=self.method,
                    jac=self.model.backward,
                    tol=self.tolerance,
                    gtol=10**-3,
                    callback=self.configuration_checkpoint,
                    options=None,
                )
                self.model.weights = res.x
                energy = self.model.forward(self.model.weights)
                grad_energy = self.model.backward(self.model.weights)

                print("Optimization Success=", res.success)
                print(f"energy={energy:.5f}")
                print(f"average gradient={np.average(np.abs(grad_energy)):.15f} \n")
                print(f"grad tolerance={self.model.grad_tolerance:.15f} \n")

                energy_history.append(energy)
                gradient_history.append(np.average(np.abs(grad_energy)))

        return energy_history, gradient_history

    def run_gradient_descent(
        self,
    ):

        while self.model.grad_tolerance > self.tolerance:

            self.model.model_preparation()

            # optimization algorithm
            # res=minimize(self.model.forward, self.model.weights, args=(), method=self.method, jac=self.model.backward, tol=self.tolerance, callback=None, options=None)
            grad = 1000
            while np.average(np.abs(grad)) > self.tolerance:
                grad = self.model.backward(self.model.weights)

                self.model.weights -= grad * 0.1
                energy = self.model.forward(self.model.weights)

                print(f"energy={energy:.5f}, gs energy={self.model.exact_energy:.5f}")
                print(f"average gradient={np.average(np.abs(grad)):.15f} \n")
                print(f"grad tolerance={self.model.grad_tolerance:.15f} \n")
