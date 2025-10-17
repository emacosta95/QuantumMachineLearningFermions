import numpy as np
from numba.typed import List
from numba.typed import Dict
from numba import njit, types
from numba import  int64
import numpy as np

from math import comb  # import outside

@njit
def generate_combinations(n, k):
    n_comb = comb(n, k)  # now OK
    result = np.zeros((n_comb, k), dtype=np.int64)
    if k == 0:
        return result
    # first combination
    for i in range(k):
        result[0, i] = i
    idx = 0
    while True:
        # generate next combination
        for i in range(k-1, -1, -1):
            if result[idx, i] != i + n - k:
                break
        else:
            break  # finished
        next_comb = result[idx].copy()
        next_comb[i] += 1
        for j in range(i+1, k):
            next_comb[j] = next_comb[j-1] + 1
        idx += 1
        if idx < n_comb:
            result[idx] = next_comb
        else:
            break
    return result

@njit
def generate_bit_basis_numba(nsites_a, nparticles_a, nsites_b, nparticles_b):
    combos_a = generate_combinations(nsites_a, nparticles_a)
    combos_b = generate_combinations(nsites_b, nparticles_b)

    n_basis = combos_a.shape[0] * combos_b.shape[0]
    basis_bits = np.zeros(n_basis, dtype=np.int64)
    count = 0
    for ia in range(combos_a.shape[0]):
        bits_a = 0
        for idx in range(nparticles_a):
            bits_a |= 1 << combos_a[ia, idx]
        for ib in range(combos_b.shape[0]):
            bits_b = 0
            for idx in range(nparticles_b):
                bits_b |= 1 << combos_b[ib, idx]
            basis_bits[count] = bits_a | (bits_b << nsites_a)
            count += 1
    return basis_bits

@njit
def _count_bits(x: int) -> int:
    """Count number of 1s in integer x."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

@njit
def _phase_for_annihilation(bits: int, site: int) -> int:
    """Compute fermionic phase for annihilation at `site`."""
    mask = (1 << site) - 1
    return _count_bits(bits & mask)

@njit
def _adag_a_loop_numba_bits(basis_bits, i, j, bit2index: Dict, n_sites):
    n_basis = len(basis_bits)
    rows = np.empty(n_basis, dtype=np.int64)
    cols = np.empty(n_basis, dtype=np.int64)
    data = np.empty(n_basis, dtype=np.float64)
    count = 0

    for idx in range(n_basis):
        b = basis_bits[idx]

        # Annihilation check
        if (b >> j) & 1 == 0:
            continue
        # Creation check
        if i != j and ((b >> i) & 1):
            continue

        # Compute phase and apply operators
        phase = _phase_for_annihilation(b, j)
        b_new = b & ~(1 << j)       # annihilate
        phase += _phase_for_annihilation(b_new, i)
        b_new |= (1 << i)           # create

        new_idx = bit2index.get(b_new, -1)
        if new_idx >= 0:
            rows[count] = new_idx
            cols[count] = idx
            data[count] = 1.0 if phase % 2 == 0 else -1.0
            count += 1

    return rows[:count], cols[:count], data[:count]

@njit
def _adag_adag_a_a_loop_numba_bits(basis_bits, i1, i2, j1, j2, bit2index: Dict, n_sites):
    n_basis = len(basis_bits)
    rows = np.empty(n_basis, dtype=np.int64)
    cols = np.empty(n_basis, dtype=np.int64)
    data = np.empty(n_basis, dtype=np.float64)
    count = 0

    for idx in range(n_basis):
        b = basis_bits[idx]

        # Skip if any annihilation not allowed
        if ((b >> j1) & 1 == 0) or ((b >> j2) & 1 == 0):
            continue
        # Skip if any creation not allowed
        if (i1 != j1 and i1 != j2 and ((b >> i1) & 1)) or (i2 != j1 and i2 != j2 and ((b >> i2) & 1)):
            continue

        # Apply operators stepwise
        phase = 0
        b_new = b

        # annihilate j2 then j1
        phase += _phase_for_annihilation(b_new, j2)
        b_new &= ~(1 << j2)
        phase += _phase_for_annihilation(b_new, j1)
        b_new &= ~(1 << j1)

        # create i2 then i1
        phase += _phase_for_annihilation(b_new, i2)
        b_new |= 1 << i2
        phase += _phase_for_annihilation(b_new, i1)
        b_new |= 1 << i1

        new_idx = bit2index.get(b_new, -1)
        if new_idx >= 0:
            rows[count] = new_idx
            cols[count] = idx
            data[count] = 1.0 if phase % 2 == 0 else -1.0
            count += 1

    return rows[:count], cols[:count], data[:count]