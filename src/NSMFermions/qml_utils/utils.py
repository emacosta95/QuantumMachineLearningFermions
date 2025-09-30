from ..fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict, Callable
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize



def configuration(res,energy,grad_energy):
    
    print('Optimization Success=',res.success)
    print(f'energy={energy:.5f}')
    print(f'average gradient={np.average(grad_energy):.5f} \n')