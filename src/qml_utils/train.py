from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict, Callable
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize







class fit():
    
    def __init__(self, method,tolerance_opt:float) -> None:
        
        
        self.method:str=method
        self.tolerance=tolerance_opt
        
        
        self.model=None
        
        
        self.configuration_checkpoint:Callable=None
    
    def set_checkpoint(self,checkpoint:Callable):
        
        self.configuration_checkpoint=checkpoint
        
    def init_model(self,model):
        self.model=model
        
        
    def run(self,):
            while(self.model.grad_tolerance>self.tolerance):
                
                self.model.model_preparation()
                
                # optimization algorithm
                res=minimize(model.energy, self.weights, args=(), method=self.method, jac=model.backward, tol=self.tolerance, callback=None, options=None)

                energy=self.model.forward()
                grad_energy=self.model.backward()
                
                self.configuration_checkpoint(res,energy,grad_energy)
                
            
        
    
    
        