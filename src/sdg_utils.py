from src.hamiltonian_utils import FermiHubbardHamiltonian
from src.utils_quasiparticle_approximation import HardcoreBosonsBasis
from src.nuclear_physics_utils import get_twobody_nuclearshell_model,SingleParticleState
import numpy as np
import torch
from typing import Dict
import scipy
from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy.optimize import minimize
from scipy import sparse
from src.hartree_fock_library import HFEnergyFunctional,HFEnergyFunctionalNuclear
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from scipy.optimize import dual_annealing
from typing import Optional




class NSM_SQD_circuit_ansatz:

    def __init__(self,twobody_matrix:Dict, NSMHamiltonian, samples = 20, batches =100, train_steps = 10,num_parameters=30):



        self.NSMHamiltonian=NSMHamiltonian
        self.twobody_matrix=twobody_matrix
        ####
        self.H = NSMHamiltonian.hamiltonian # hamiltonian as sparse matrix
        self.basis = NSMHamiltonian.basis # basis
        ###

        self.l = len(self.basis[0]) # lunghezza un elemento base
        self.psi = np.zeros(len(self.basis))
        self.E = 0.
        self.E_old=1E10
        self.prob=None
        self.psi_batches=None
        self.variance=None
        self.operator_pool=None
        self.operator_pool_list=None
        self.weights=None
  
        
        self.seed = 1024

        self.samples = samples # number of samples for sqd energy estimation
        self.batches = batches # different batches for sqd to make an average of the energy
        self.train_steps = train_steps # steps in optimization
        self.num_parameters = num_parameters # number of parameters in the ansatz
        
        self.save_file_name=None
        self.history = []
        self.history_variance=[]
        
        
    def set_save_file_name(self,save_file_name):
        self.save_file_name=save_file_name
        self.counts=0
        return None

    def initialize_state_minslater(self): # ref state is the element of the basis with the minimum energy
        self.psi = np.zeros(len(self.basis))  
        diag = self.H.diagonal()
        min_state_idx = diag.argmin()
        self.psi[min_state_idx] = 1. 
        self.psi /= np.linalg.norm(self.psi)
        self.E = diag.min()
        return None

    def initialize_state_random(self,ns): # ref state is the element of the basis with the minimum energy
        self.psi = np.zeros(len(self.basis)) 
        diag = self.H.diagonal()
        min_indices = np.argsort(diag)[:ns]
        self.psi[min_indices] = 1. / np.sqrt(ns)
        self.psi /= np.linalg.norm(self.psi)
        self.E = diag[min_indices].mean()
        return None

    def initialize_state_hartreefock(self,psi_hf): # ref state is the element of the basis with the minimum energy
        self.psi_initial=psi_hf.copy()
        self.psi = psi_hf
        self.E = self.psi.transpose().conj() @ self.H @ self.psi
        return None

    def create_operator_pool_twobody_symmetry(self,original_operator_pool): # nella pool two body op che rispettano i miquel constraints
        
        self.operator_pool=original_operator_pool
        vs, ops, keys = [], [], []
        for key, op in self.operator_pool.items():
            if key in self.twobody_matrix:
                vs.append(self.twobody_matrix[key])
                ops.append(op)
                keys.append(key)

        vs = np.abs(np.array(vs))
        selection = np.argsort(vs)[-self.num_parameters:][::-1]
        selected_vs = vs[selection]


        new_operator_pool = {keys[idx]: ops[idx] for idx in selection}
        self.operator_pool = new_operator_pool
        self.operator_pool_list = list(self.operator_pool.values())
        np.random.seed(self.seed) 
        self.weights = np.zeros(len(self.operator_pool))#np.random.uniform(-1.3, 1.3, size=len(self.operator_pool))
        return None

    def forward(self, weights): # compute psi
        psi = self.psi.copy()
        if self.weights is not(None):
            
            for i, w in enumerate(weights):
                # print(np.conj(expm(self.weights[i] * op).T) @ expm(self.weights[i] * op))
                psi = scipy.sparse.linalg.expm_multiply( (0. + 1j) * w * self.operator_pool_list[i],psi) 
                psi = psi / np.linalg.norm(psi)

        return psi

    def SQD(self, weights): #output è energia stimata con SQD (compute energy functional)
        psi = self.forward(weights)
        prob = (np.conjugate(psi) * psi).real # .real needed to change data type from complex to float
        self.prob=prob.copy()
        Ham = self.H.copy()
        np.random.seed(self.seed) # seed con cui calcolo il vettore random i cui elementi sono i seed dei diversi batch
        seeds = np.random.randint(1, 30002, size=self.batches)
        e = np.zeros(self.batches)
        self.psi_batches=np.zeros((self.batches,self.psi.shape[0]))
        for k,s in enumerate(seeds):
            np.random.seed(s)
            #print('non zero prob=',np.nonzero(prob)[0].shape[0])

            
            if np.nonzero(prob)[0].shape[0]<self.samples:
                raw_selection = np.random.choice(len(self.basis), size=self.samples * 5, replace=True, p=prob)
                # oversample to increase chance of enough unique values
                selection = []
                seen = set()
                for idx in raw_selection:
                    if idx not in seen:
                        seen.add(idx)
                        selection.append(idx)
                    if len(selection) == self.samples:
                        break

                # In case not enough unique indices are collected, pad with random ones
                if len(selection) < self.samples:
                    remaining = list(set(range(len(self.basis))) - seen)
                    np.random.shuffle(remaining)
                    selection.extend(remaining[: self.samples - len(selection)])

                selection = np.array(selection)

            else:
                
                raw_selection = np.random.choice(len(self.basis), size=self.samples * 5, replace=True, p=prob)
                # oversample to increase chance of enough unique values
                selection = []
                seen = set()
                for idx in raw_selection:
                    if idx not in seen:
                        seen.add(idx)
                        selection.append(idx)
                    if len(selection) == self.samples:
                        break

                # In case not enough unique indices are collected, pad with random ones
                if len(selection) < self.samples:
                    remaining = list(set(range(len(self.basis))) - seen)
                    np.random.shuffle(remaining)
                    selection.extend(remaining[: self.samples - len(selection)])

                selection = np.array(selection)

            if len(selection)!=1:    
                effective_hamiltonian=  Ham.tocsr()[selection,:][:,selection]

                value,eighvec = eigsh(effective_hamiltonian, k=1, which="SA", maxiter=int(1E6), ) # Diagonalization of reduced H
                e[k]=value[0]
                self.psi_batches[k,selection]=eighvec[:,0]
            
            else:
                e[k]=Ham[selection[0],selection[0]]
            
        self.variance=np.std(e)
        return np.average(e)
    
    def cobyla_callback(self, x):
        # Callback function for COBYLA (optional, for printing intermediate results)
        self.history.append(self.SQD(x))
        self.history_variance.append(self.variance)
        print(f"Current COBYLA weights: {np.linalg.norm(x)}, SQD: {self.SQD(x)} Variance energy {self.variance}")
        if self.counts % 100 ==0:
            if self.E_old > self.SQD(x):
                np.savez(self.save_file_name,weights=self.weights,energy=self.E,variance=self.variance,psi=self.psi,history=self.history,history_variance=self.history_variance,prob=self.prob,psis=self.psi_batches,psi_initial=self.psi_initial)
                self.E_old=self.E
        self.counts += 1  
    def optimization(self):

                    
        for i in range(self.train_steps):
            # COBYLA optimization
            res2 = minimize(self.SQD, x0=self.weights, method='COBYLA', options={'disp':True, 'maxiter': 100000},callback=self.cobyla_callback)

            self.weights = res2.x.copy()  # update weights
            self.E = self.SQD(self.weights)
            self.psi = self.forward(self.weights)

            print(f'\n--- Training Step {i+1} ---')
            print('Optimization Success =', res2.success)
            print('Energy (SQD) =', self.E)
            print('Message =', res2.message)
            print('Current weights =', self.weights)
            print('Number of function evaluations =', res2.nfev)


    def optimization_annealing(self):
        """
        Optimization routine using dual annealing (simulated annealing variant).
        """

        # Define bounds for parameters
        # (scale according to expected operator weight range, e.g. [-2, 2])
        bounds = [(-2.0, 2.0)] * len(self.weights)

        for i in range(self.train_steps):
            print(f"\n--- Simulated Annealing Step {i+1} ---")

            res = dual_annealing(
                self.SQD,
                bounds=bounds,
                #maxiter=200,       # number of annealing iterations
                #maxfun=5000,       # max evaluations
                seed=self.seed+i,  # different seed per step for exploration
                callback=self.annealing_callback,
            )

            # Update weights and state
            self.weights = res.x
            self.E = self.SQD(self.weights)
            self.psi = self.forward(self.weights)

            print("Optimization Success =", res.success)
            print("Energy (SQD) =", self.E)
            print("Message =", res.message)
            print("Current weights =", self.weights)
            print("Number of function evaluations =", res.nfev)

    def annealing_callback(self, x, f, context):
        # Append values to history (use f, not recomputing SQD)
        self.history.append(f)
        self.history_variance.append(self.variance)

        # Log every iteration
        print(f"Step  | ||weights||={np.linalg.norm(x):.4f} | Energy={f:.6f} | Variance={self.variance:.6f}", flush=True)

        # Periodic saving
        if self.counts % 100 == 0:
            if self.E_old > f:
                np.savez(self.save_file_name,
                        weights=self.weights,
                        energy=self.E,
                        variance=self.variance,
                        psi=self.psi,
                        history=self.history,
                        history_variance=self.history_variance,
                        prob=self.prob,
                        psis=self.psi_batches,
                        psi_initial=self.psi_initial)
                self.E_old = f

        # Update counts and current weights
        self.counts += 1
        self.weights = x.copy()