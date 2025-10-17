import numpy as np
from numba import njit
from numba.typed import List

@njit
def generate_combinations(n, k):
    """
    Generate all k-combinations of range(n) as lists of indices.
    Returns a List of Lists.
    """
    if k == 0:
        res = List()
        res.append(List())
        return res

    if n < k:
        return List()

    res = List()
    for i in range(n - k + 1):
        sub_combos = generate_combinations(n - i - 1, k - 1)
        for sc in sub_combos:
            new_combo = List()
            new_combo.append(i)
            for x in sc:
                new_combo.append(x + i + 1)
            res.append(new_combo)
    return res

@njit
def generate_bit_basis_numba(nsites_a, nparticles_a, nsites_b, nparticles_b):
    """
    Generate Fermi-Hubbard bit basis as integers.
    """
    basis_bits = List()

    combos_a = generate_combinations(nsites_a, nparticles_a)
    combos_b = generate_combinations(nsites_b, nparticles_b)

    for indices_a in combos_a:
        bits_a = 0
        for idx in indices_a:
            bits_a |= 1 << idx

        for indices_b in combos_b:
            bits_b = 0
            for idx in indices_b:
                bits_b |= 1 << idx

            basis_bits.append(bits_a | (bits_b << nsites_a))

    return np.array(basis_bits, dtype=np.int64)
