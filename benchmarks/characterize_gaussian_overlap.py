"""Diagnose why the two CKI Be8 Slater determinants have small overlap."""
import json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from cki_be8 import ROOT, legacy_definitions
from typing import List, Dict, Tuple, Optional, Callable, ClassVar


def generators(states):
    m=len(states); jp=np.zeros((m,m),complex); tp=np.zeros((m,m),complex)
    lookup={tuple(s):i for i,s in enumerate(states)}
    for i,(n,l,j,mj,t,tz) in enumerate(states):
        raised=(n,l,j,mj+1,t,tz)
        if raised in lookup:
            jp[lookup[raised],i]=np.sqrt(j*(j+1)-mj*(mj+1))
        raised_t=(n,l,j,mj,t,tz+1)
        if raised_t in lookup:
            tp[lookup[raised_t],i]=np.sqrt(t*(t+1)-tz*(tz+1))
    jx=(jp+jp.conj().T)/2; jy=(jp-jp.conj().T)/(2j)
    jz=np.diag([s[3] for s in states])
    tx=(tp+tp.conj().T)/2; ty=(tp-tp.conj().T)/(2j)
    tz=np.diag([s[5] for s in states])
    return (jx,jy,jz),(tx,ty,tz)


def squared_quantum_number(c,generators):
    rho=c@c.conj().T; identity=np.eye(len(rho)); total=0.
    for a in generators:
        mean=np.trace(rho@a)
        total+=(mean.conjugate()*mean+np.trace(rho@a@(identity-rho)@a)).real
    return float(total)


def rotation(angles,generators):
    x,y,z=generators
    return expm(-1j*angles[0]*z)@expm(-1j*angles[1]*y)@expm(-1j*angles[2]*z)


def main():
    collapsed=np.load(ROOT/'benchmarks/results/cki_be8_state.npz')
    best=np.load(ROOT/'benchmarks/results/cki_be8_best_gaussian_state.npz')
    rho=collapsed['V'].conj()@collapsed['V'].T
    _,ch=np.linalg.eigh(rho); ch=ch[:,-4:]
    cg=best['slater_orbitals']
    singular=np.linalg.svd(ch.conj().T@cg,compute_uv=False)
    ns=dict(globals(),trange=range)
    legacy_definitions('nuclear_physics_utils.py',['SingleParticleState'],ns)
    states=ns['SingleParticleState'](str(ROOT/'data/cki')).state_encoding
    jgen,tgen=generators(states)
    def fidelity(angles,use_j=True,use_t=True):
        r=np.eye(12,dtype=complex)
        if use_j:r=rotation(angles[:3],jgen)@r
        if use_t:r=rotation(angles[-3:],tgen)@r
        return float(abs(np.linalg.det(ch.conj().T@r@cg))**2)
    rng=np.random.default_rng(18); aligned={}; best_parameters={}
    for name,use_j,use_t,n in [('J',True,False,3),('T',False,True,3),('JT',True,True,6)]:
        fits=[]
        for _ in range(30):
            x=rng.uniform(-np.pi,np.pi,n)
            if n==3:
                full=(lambda y:np.r_[y,np.zeros(3)]) if use_j else (lambda y:np.r_[np.zeros(3),y])
            else: full=lambda y:y
            fits.append(minimize(lambda y:-fidelity(full(y),use_j,use_t),x,method='BFGS',
                                 options={'maxiter':300,'gtol':1e-10}))
        winner=min(fits,key=lambda fit:fit.fun)
        aligned[name]=float(-winner.fun)
        best_parameters[name]=winner.x.tolist()
    angles=np.asarray(best_parameters['J'])
    aligned_singular=np.linalg.svd(ch.conj().T@rotation(angles,jgen)@cg,compute_uv=False)
    result={'principal_cosines':singular.tolist(),
        'principal_angles_degrees':np.degrees(np.arccos(np.clip(singular,0,1))).tolist(),
        'determinant_fidelity_product':float(np.prod(singular**2)),
        'best_symmetry_aligned_fidelity':aligned,
        'best_rotation_parameters':best_parameters,
        'J_aligned_principal_cosines':aligned_singular.tolist(),
        'J_aligned_principal_angles_degrees':np.degrees(np.arccos(np.clip(aligned_singular,0,1))).tolist(),
        'collapsed_J2':squared_quantum_number(ch,jgen),
        'closest_gaussian_J2':squared_quantum_number(cg,jgen),
        'collapsed_T2':squared_quantum_number(ch,tgen),
        'closest_gaussian_T2':squared_quantum_number(cg,tgen)}
    out=ROOT/'benchmarks/results/cki_be8_gaussian_overlap_diagnostic.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
