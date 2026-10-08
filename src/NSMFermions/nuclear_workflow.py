"""Small, dependency-light helpers for shell-model HF/HFB workflows.

The original research drivers kept these utilities in ``benchmarks``.  They
live in the library now so tutorials and tests do not depend on benchmark
scripts or their saved outputs.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse


SOURCE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _optimized_two_body_kernel(
    basis, i1, i2, j1, j2, masks, mask2index
):
    """Apply a two-body operator to a determinant basis using bit masks."""
    masks = np.asarray(masks, dtype=np.uint64)
    empty_int = np.empty(0, dtype=np.int64)
    empty_float = np.empty(0, dtype=float)
    if i1 == i2 or j1 == j2 or not len(masks):
        return empty_int, empty_int.copy(), empty_float

    bits = [
        np.uint64(1) << np.uint64(index)
        for index in (i1, i2, j1, j2)
    ]
    bit_i1, bit_i2, bit_j1, bit_j2 = bits

    valid = (masks & bit_j2) != 0
    transformed = masks & ~bit_j2
    valid &= (transformed & bit_j1) != 0
    transformed &= ~bit_j1
    valid &= (transformed & bit_i2) == 0
    transformed |= bit_i2
    valid &= (transformed & bit_i1) == 0
    transformed |= bit_i1

    columns = np.flatnonzero(valid).astype(np.int64, copy=False)
    if not len(columns):
        return empty_int, empty_int.copy(), empty_float

    candidate_rows = np.fromiter(
        (mask2index.get(int(mask), -1) for mask in transformed[columns]),
        dtype=np.int64,
        count=len(columns),
    )
    retained = candidate_rows >= 0
    rows = candidate_rows[retained]
    columns = columns[retained]
    if not len(columns):
        return empty_int, empty_int.copy(), empty_float

    selected = np.asarray(basis[columns], dtype=np.int8)
    phase = np.sum(selected[:, :j2], axis=1, dtype=np.int64)
    phase += np.sum(selected[:, :j1], axis=1, dtype=np.int64)
    phase -= int(j2 < j1)
    phase += np.sum(selected[:, :i2], axis=1, dtype=np.int64)
    phase -= int(j2 < i2) + int(j1 < i2)
    phase += np.sum(selected[:, :i1], axis=1, dtype=np.int64)
    phase -= int(j2 < i1) + int(j1 < i1)
    phase += int(i2 < i1)
    data = np.where(phase & 1, -1.0, 1.0)
    return rows, columns, data


def _progress(iterable):
    """Print occasional progress without requiring a progress-bar package."""
    total = len(iterable)
    stride = max(1, total // 10)
    for index, item in enumerate(iterable, 1):
        if index == 1 or index == total or index % stride == 0:
            print(f"  two-body terms: {index}/{total}", flush=True)
        yield item


def _legacy_definitions(filename, names, namespace):
    """Load selected legacy definitions without importing optional ML code."""
    tree = ast.parse(
        (SOURCE_DIRECTORY / filename).read_text(encoding="utf-8")
    )
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise ValueError("Legacy definition missing")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, filename, "exec"), namespace)


def load_nuclear_interaction(name, *, repository_root=None):
    """Load CKI or USDB in the single-particle basis.

    Returns ``(interaction, one_body_energies, encoding, path)``.
    """
    normalized = str(name).lower()
    if normalized not in {"cki", "usdb"}:
        raise ValueError("Interaction must be 'cki' or 'usdb'")

    namespace = dict(globals(), trange=range)
    _legacy_definitions(
        "cg_utils.py",
        [
            "CG",
            "ClebschGordan",
            "SelectCG",
            "CreateInitialCGList",
            "CalcInitialValues",
            "DivCalc",
            "CgJM",
        ],
        namespace,
    )
    _legacy_definitions(
        "nuclear_physics_utils.py",
        [
            "SingleParticleState",
            "krond",
            "scattering_matrix_reader",
            "compute_nuclear_twobody_matrix",
            "get_twobody_nuclearshell_model",
        ],
        namespace,
    )

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else REPOSITORY_ROOT
    )
    path = root / ("data/cki" if normalized == "cki" else "data/usdb.nat")
    interaction, energies = namespace["get_twobody_nuclearshell_model"](
        str(path)
    )
    encoding = namespace["SingleParticleState"](str(path)).state_encoding
    return interaction, np.asarray(energies), encoding, path


def nucleus_from_mass(interaction_name, mass, species_modes):
    """Return the label and ``(valence neutrons, valence protons)``."""
    normalized = str(interaction_name).lower()
    if normalized == "cki":
        valence_neutrons = int(mass) - 6
        label = f"Be{int(mass)}"
    elif normalized == "usdb":
        valence_neutrons = int(mass) - 18
        label = f"Ne{int(mass)}"
    else:
        raise ValueError("Interaction must be 'cki' or 'usdb'")
    if valence_neutrons < 0 or valence_neutrons > int(species_modes):
        raise ValueError("Nucleus lies outside the selected valence space")
    return label, (valence_neutrons, 2)


def build_fermionic_hamiltonian(
    interaction, eps, particles=(2, 2), *, symmetries=None
):
    """Assemble the exact fixed-particle shell-model Hamiltonian."""
    namespace = dict(globals())

    def build_mask_mapping(basis):
        masks = np.array(
            [
                sum(int(bit) << mode for mode, bit in enumerate(row))
                for row in basis
            ],
            dtype=np.uint64,
        )
        return masks, {int(mask): index for index, mask in enumerate(masks)}

    namespace["build_mask_mapping"] = build_mask_mapping
    namespace["_adag_adag_a_a_loop_numba_with_dict"] = (
        _optimized_two_body_kernel
    )
    namespace["tqdm"] = _progress
    namespace["lil_matrix"] = sparse.lil_matrix
    namespace["coo_matrix"] = sparse.coo_matrix
    namespace["combinations"] = itertools.combinations

    _legacy_definitions("fermi_hubbard_library.py", ["FemionicBasis"], namespace)
    _legacy_definitions(
        "hamiltonian_utils.py", ["FermiHubbardHamiltonian"], namespace
    )
    hamiltonian_class = namespace["FermiHubbardHamiltonian"]

    if len(eps) % 2:
        raise ValueError("Expected equal proton and neutron mode blocks")
    species_modes = len(eps) // 2
    fermionic = hamiltonian_class(
        species_modes,
        species_modes,
        particles[1],
        particles[0],
        symmetries=symmetries,
    )
    fermionic.get_external_potential(np.asarray(eps))
    fermionic.get_twobody_interaction_optimized(interaction)
    fermionic.get_hamiltonian()
    return fermionic
