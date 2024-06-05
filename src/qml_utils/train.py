from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict, Callable
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize







class Fit():
    
    def __init__(self, method:str,tolerance_opt:float,e_ref:float=None) -> None:
        
        
        self.method:str=method
        self.tolerance=tolerance_opt
        
        
        self.model=None
        
        
        self.configuration_checkpoint:Callable=None
    

        
    def init_model(self,model):
        self.model=model
        
        
    def run(self,):
            while(self.model.grad_tolerance>self.tolerance):
                
                self.model.model_preparation()

                
                # optimization algorithm
                res=minimize(self.model.forward, self.model.weights, args=(), method=self.method, jac=self.model.backward, tol=self.tolerance, callback=None, options=None)
                self.model.weights=res.x
                energy=self.model.forward(self.model.weights)
                grad_energy=self.model.backward(self.model.weights)
                
                print('Optimization Success=',res.success)
                print(f'energy={energy:.5f}')
                print(f'average gradient={np.average(np.abs(grad_energy)):.15f} \n')
                print(f'grad tolerance={self.model.grad_tolerance:.15f} \n')
                
    
    def run_gradient_descent(self,):
        
        while(self.model.grad_tolerance>self.tolerance):
                
                self.model.model_preparation()

                
                # optimization algorithm
                #res=minimize(self.model.forward, self.model.weights, args=(), method=self.method, jac=self.model.backward, tol=self.tolerance, callback=None, options=None)
                grad=1000
                while(np.average(np.abs(grad))>self.tolerance):
                    grad=self.model.backward(self.model.weights)
                    
                    self.model.weights-=grad*0.1
                    energy=self.model.forward(self.model.weights)
                    
                    print(f'energy={energy:.5f}')
                    print(f'average gradient={np.average(np.abs(grad)):.15f} \n')
                    print(f'grad tolerance={self.model.grad_tolerance:.15f} \n')
                    
        

    
        