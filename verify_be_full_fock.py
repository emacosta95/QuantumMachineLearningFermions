"""Independent quasiparticle-vacuum check in all 2048 even Fock states."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "NSMFermions"))
if sys.platform == "win32" and (ROOT / ".hfb-test-deps").exists():
    sys.path.insert(0, str(ROOT / ".hfb-test-deps"))
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, expm_multiply
from nuclear_workflow import load_nuclear_interaction
from angular_momentum import single_particle_angular_momentum

interaction, eps, sp, _ = load_nuclear_interaction("cki", repository_root=ROOT)
modes = len(eps)
even = [mask for mask in range(1 << modes) if mask.bit_count() % 2 == 0]
odd = [mask for mask in range(1 << modes) if mask.bit_count() % 2 == 1]
even_index = {mask: i for i, mask in enumerate(even)}
odd_index = {mask: i for i, mask in enumerate(odd)}

def annihilator(mode, source, destination):
    rows, columns, values = [], [], []
    for column, mask in enumerate(source):
        if mask & (1 << mode):
            rows.append(destination[mask ^ (1 << mode)])
            columns.append(column)
            values.append((-1) ** ((mask & ((1 << mode)-1)).bit_count()))
    return coo_matrix((values, (rows, columns)), shape=(len(destination), len(source)), dtype=complex).tocsr()

A = [annihilator(i, even, odd_index) for i in range(modes)]
B = [annihilator(i, odd, even_index) for i in range(modes)]
pair = {(i,j): B[j] @ A[i] for i in range(modes) for j in range(modes) if i != j}
diagonal = np.array([sum(eps[i] for i in range(modes) if mask & (1 << i)) for mask in even])
H = diags(diagonal, dtype=complex).tocsr()
for (i,j,k,l), value in interaction.items():
    H += .25 * value * (pair[i,j].conj().T @ pair[k,l])
jx, jy, jz = single_particle_angular_momentum(sp.state_encoding)
jp = jx + 1j * jy
Jplus = csr_matrix(H.shape, dtype=complex)
for i,j in zip(*np.nonzero(jp)):
    Jplus += jp[i,j] * (A[i].conj().T @ A[j])
magnetic = np.array([sum(sp.state_encoding[i][3] for i in range(modes) if mask & (1 << i)) for mask in even])
Jy = (Jplus-Jplus.conj().T)/(2j)
J2 = Jplus.conj().T@Jplus + diags(magnetic*(magnetic+1))

def project_j0(psi):
    # Eleven Fourier nodes cover all M in the full p-shell Fock space.
    angles = 2*np.pi*(np.arange(11)+.173)/11
    periodic = np.exp(-1j*angles[:,None]*magnetic).mean(axis=0)
    x, weights = np.polynomial.legendre.leggauss(10)
    rotated = sum(w/2 * expm_multiply(-1j*np.arccos(c)*Jy, periodic*psi)
                  for c,w in zip(x,weights))
    result = periodic*rotated
    weight = float(np.vdot(result,result).real)
    return result/np.sqrt(weight), weight

with (ROOT/'results/be_energy_audit.pkl').open('rb') as stream:
    saved = pickle.load(stream)
rows=[]
for mass, entry in saved.items():
    indices = [even_index[sum(1 << int(i) for i in occ)] for occ in entry['occupations']]
    ground = np.zeros(len(even),complex)
    ground[indices] = entry['ground']
    for method in ('hf','hfb'):
        solution = entry[method]['solution']
        state = solution.state
        Q = csr_matrix(H.shape,dtype=complex)
        for j in range(modes):
            beta = sum(state.U[i,j].conjugate()*A[i] + state.V[i,j].conjugate()*B[i].conj().T for i in range(modes))
            Q += beta.conj().T@beta
        _, vectors = eigsh(Q,k=1,which='SA',tol=1e-12,v0=np.random.default_rng(18).normal(size=len(even)))
        psi = vectors[:,0]
        direct_energy = float(np.vdot(psi,H@psi).real)
        library = state.stable_normalized_occupation_amplitudes(entry['occupations'])
        independent = psi[indices]
        phase = np.vdot(independent,library)
        phase /= abs(phase)
        amplitude_error = float(np.max(abs(library-phase*independent)))
        projected, j0_weight = project_j0(psi)
        nzj = np.zeros_like(projected)
        nzj[indices] = projected[indices]
        nzj /= np.linalg.norm(nzj)
        row = dict(mass=mass,method=method,quasiparticle_residual=float(np.linalg.norm(Q@psi)),
                   intrinsic_energy=direct_energy,energy_error=direct_energy-solution.energy,
                   amplitude_max_error=amplitude_error,
                   intrinsic_J2=float(np.vdot(psi,J2@psi).real),
                   J0_weight=j0_weight,J0_only_energy=float(np.vdot(projected,H@projected).real),
                   NZJ0_energy=float(np.vdot(nzj,H@nzj).real),
                   J0_residual=float(np.linalg.norm(J2@projected)),
                   raw_fidelity=float(abs(np.vdot(ground,psi))**2))
        rows.append(row)
        print(row,flush=True)
        (ROOT/'results/be_full_fock_audit.json').write_text(json.dumps(rows,indent=2))
