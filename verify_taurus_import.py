"""Import collaborator TAURUS vacua and compare with mass-scaled USDB."""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src/NSMFermions'))
if sys.platform == 'win32':
    sys.path.insert(0, str(ROOT / '.hfb-test-deps'))
import numpy as np
from scipy.io import FortranFile
from hfb import HFBState, HFBHamiltonian
from nuclear_workflow import load_nuclear_interaction, build_fermionic_hamiltonian
from number_projection import exact_ground_state


def read_taurus(path, encoding):
    # TAURUS module_basis: protons first, shells as listed in .sho, m descending.
    # Our encoding has the same species blocks, but m ascending within shells.
    labels = path.with_name('usdb.sho').read_text().splitlines()[1].split()[2:]
    taurus_modes = []
    for species in (0, 1):
        for label in labels:
            label = label.zfill(4)
            n, l, two_j = int(label[0]), int(label[1]), int(label[2:])
            for two_m in range(two_j, -two_j - 1, -2):
                taurus_modes.append((species, n, l, two_j, two_m))
    permutation = [taurus_modes.index((i // (len(encoding)//2), n, l,
                                      int(2*j), int(2*m)))
                   for i, (n, l, j, m, _, _) in enumerate(encoding)]
    with FortranFile(path, 'r') as f:
        header = f.read_record(np.int32)
        shells = f.read_record(np.int32)
        label = int(f.read_record(np.int64)[0])
        u = f.read_record(np.float64)
        v = f.read_record(np.float64)
    dim = len(encoding)
    assert u.size == v.size == dim**2
    # module_wavefunctions writes ((U(j,i), j=1,dim), i=1,dim).
    u = u.reshape((dim, dim), order='F')[permutation]
    v = v.reshape((dim, dim), order='F')[permutation]
    assert np.linalg.norm(u.T@u + v.T@v - np.eye(dim)) < 1e-9
    assert np.linalg.norm(u.T@v + v.T@u) < 1e-9
    return HFBState(u, v), label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masses', nargs='+', type=int, default=[24])
    args = parser.parse_args()
    outdir = ROOT / 'results/taurus_import_audit'
    outdir.mkdir(exist_ok=True)
    # Require identical single-particle energies and coupled matrix elements.
    # Descriptive title and mass metadata differ between file formats; scaling
    # is applied explicitly below using the physical total mass.
    source = ROOT / f'results/collaborator_20261016/results/{args.masses[0]}Ne/usdb.sho'
    def interaction_content(path):
        lines = path.read_text().splitlines()
        return [line.split() for i, line in enumerate(lines)
                if i not in (0, 3) and line.strip()]
    assert interaction_content(ROOT/'data/usdb.nat') == interaction_content(source)
    base, eps, sp, _ = load_nuclear_interaction('usdb')
    rows = []
    for mass in args.masses:
        directory = ROOT / f'results/collaborator_20261016/results/{mass}Ne'
        assert (directory/'usdb.sho').read_bytes() == source.read_bytes()
        state, label = read_taurus(directory/'final_wf.bin', sp.state_encoding)
        factor = (18/mass)**0.3
        interaction = {key: value*factor for key, value in base.items()}
        ham = HFBHamiltonian(np.diag(eps), interaction)
        energy = float(ham.energy(state))
        logs = {}
        for path in directory.glob('out*.txt'):
            content = path.read_text()
            printed_label = re.findall(r'Label of the state:\s*(\d+)', content)
            printed_energy = re.findall(r'^Full H\s+.*', content, re.M)
            logs[path.name] = dict(label=int(printed_label[-1]),
                                  energy=float(printed_energy[-1].split()[-1]))
        row = dict(mass=mass, mass_factor=factor, binary_label=label,
                   imported_energy=energy, taurus_logs=logs,
                   protons=float(np.trace(state.rho[:12,:12]).real),
                   neutrons=float(np.trace(state.rho[12:,12:]).real),
                   proton_pairing_norm=float(np.linalg.norm(state.kappa[:12,:12])),
                   neutron_pairing_norm=float(np.linalg.norm(state.kappa[12:,12:])))
        print(row, flush=True)
        # A label identifies which output belongs to the binary, if multiple exist.
        matched = [log for log in logs.values() if log['label'] == label]
        if not matched or abs(energy-matched[0]['energy']) > 2e-6:
            rows.append(row)
            (outdir/'summary.json').write_text(json.dumps(rows, indent=2))
            raise ValueError('Imported energy does not match the corresponding TAURUS log')
        exact = build_fermionic_hamiltonian(interaction, eps, (mass-18, 2),
                                           symmetries=[sp.total_M_zero])
        ground_energy, ground = exact_ground_state(exact)
        amplitudes = state.stable_normalized_occupation_amplitudes(exact.occupations)
        fidelity = float(abs(np.vdot(ground, amplitudes))**2)
        occupations_nz = [p+n for p in itertools.combinations(range(12), 2)
                          for n in itertools.combinations(range(12,24), mass-18)]
        amplitudes_nz = state.stable_normalized_occupation_amplitudes(occupations_nz)
        weight_nz = float(np.vdot(amplitudes_nz, amplitudes_nz).real)
        row.update(exact_energy=float(ground_energy), raw_fidelity=fidelity,
                   raw_infidelity=1-fidelity, number_sector_weight=weight_nz,
                   number_projected_fidelity=fidelity/weight_nz,
                   number_m0_projected_fidelity=fidelity/float(np.vdot(amplitudes, amplitudes).real))
        np.savez(outdir/f'Ne{mass}.npz', U=state.U, V=state.V,
                 occupations=exact.occupations, ground=ground,
                 ground_energy=ground_energy, mass_factor=factor)
        rows.append(row)
        (outdir/'summary.json').write_text(json.dumps(rows, indent=2))
        print(row, flush=True)


if __name__ == '__main__':
    main()
