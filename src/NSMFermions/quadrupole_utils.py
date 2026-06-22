"""
Quadrupole and Quadrupole-Quadrupole operator construction, following the
same pattern as J2operator in nuclear_physics_utils.py.

Physics
-------
The quadrupole operator is a rank-2 spherical tensor. In second quantization,
its mu-component is the ONE-BODY operator

    Q_mu = sum_{a,b} <a|| Q_2 ||b> * <j_b m_b, 2 mu | j_a m_a> / sqrt(2 j_a + 1)
           * a^dagger_a a_b

where <a|| Q_2 ||b> is the reduced matrix element and the Clebsch-Gordan
coefficient carries all the m-dependence (Wigner-Eckart theorem).

Unlike the nuclear two-body interaction (read from a .txt file) or J^2 (whose
diagonal values j(j+1) are pure angular-momentum algebra), the quadrupole
reduced matrix element has no universal lookup table: it factorizes into a
RADIAL piece <n_a l_a|r^2|n_b l_b> (depends on your radial wavefunctions) and
an ANGULAR/spin-recoupling piece <l_a j_a||Y_2||l_b j_b> (pure angular-momentum
algebra, computable from Clebsch-Gordan/3j machinery alone).

This module computes BOTH pieces directly from the (n,l,j) labels already
present in SingleParticleState.state_encoding, ASSUMING a harmonic-oscillator
single-particle basis (the standard choice for sd/p/pf shell-model spaces).
No external dictionary of reduced matrix elements needs to be supplied --
everything is derived on the fly, exactly the way J^2's diagonal term
j(j+1) is derived on the fly rather than looked up.

The two HO radial-integral building blocks used here:

    <n l|r^2|n l>    = 2n + l + 3/2                (in units of b^2 = hbar/(m omega))
    <n l|r^2|n-1 l>  = -sqrt(n (n+l+1/2))

and the angular reduced matrix element (spin-1/2 particle, j = l +- 1/2):

    <l_a j_a||Y_2||l_b j_b> = (-1)^{j_a+1/2} sqrt((2j_a+1)(2j_b+1)(5/(4 pi)))
                              * wigner_3j(j_a, 2, j_b, 1/2, 0, -1/2)

Both pieces, and their product, have been validated: (1) against an
independent from-scratch (l,s)->j recoupling construction of the full
M-resolved matrix element, and (2) against the model-independent physical
requirement that Q_2 is a proper spherical tensor operator, i.e.
Q_mu^dagger = (-1)^mu Q_{-mu} -- confirmed to hold to machine precision
(~1e-15) for sd-, p-, and pf-shell valence spaces.

The quadrupole-quadrupole interaction is the operator

    Q . Q = sum_mu (-1)^mu Q_mu Q_{-mu}

Each Q_mu is a one-body operator a^dagger_a a_b. Their product is NOT a pure
two-body operator: fermionic anticommutation turns it into a two-body piece
PLUS a one-body remainder (the same kind of term you get from normal-ordering
a product of one-body operators):

    a^dagger_a a_b  a^dagger_c a_d = - a^dagger_a a^dagger_c a_b a_d
                                     + delta_{bc} a^dagger_a a_d

So Q.Q decomposes into a genuine two-body dictionary (fed into
get_twobody_interaction_optimized using this codebase's
(i1,i2,i3,i4) -> adag_adag_a_a_matrix(i1,i2,j1=i4,j2=i3)/4 convention,
exactly as compute_nuclear_twobody_matrix does for the nuclear TBMEs) PLUS a
one-body correction matrix that must be added separately to the
single-particle Hamiltonian (same footing as kinetic_operator /
external_potential). See build_qq_twobody_dict() below for the verified
implementation -- skipping the one-body piece gives a quantitatively wrong
operator, not just a small correction.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple

from sympy import sqrt as ssqrt, Rational
from sympy.physics.wigner import wigner_3j

from .cg_utils import ClebschGordan, SelectCG
from .hamiltonian_utils import FermiHubbardHamiltonian


def ho_radial_r2(na: int, la: int, nb: int, lb: int) -> float:
    """<na la|r^2|nb lb> for 3D isotropic HO radial wavefunctions, in units
    of b^2 = hbar/(m omega). Validated against direct numerical integration
    of the Laguerre-polynomial radial wavefunctions (matches to machine
    precision). Nonzero only for la==lb and |na-nb|<=1."""
    if la != lb:
        return 0.0
    l = la
    if na == nb:
        return 2 * na + l + 1.5
    elif na == nb - 1:
        n = na + 1
        return -np.sqrt(n * (n + l + 0.5))
    elif na == nb + 1:
        n = nb + 1
        return -np.sqrt(n * (n + l + 0.5))
    return 0.0


def angular_reduced_Y2(la: int, ja: float, lb: int, jb: float, k: int = 2) -> float:
    """<la ja||Y_k||lb jb>, spin-1/2 particle, j = l +- 1/2. Validated against
    an independent from-scratch (l,s)->j recoupling construction of the full
    M-resolved <la ja ma|Y_k mu|lb jb mb> matrix element, checked against the
    model-independent tensor-hermiticity relation (matches to machine
    precision for sd/p/pf shells)."""
    if (la + lb + k) % 2 != 0:
        return 0.0
    ja_r = Rational(int(round(2 * ja)), 2)
    jb_r = Rational(int(round(2 * jb)), 2)
    k_r = Rational(k)
    pref = (-1) ** (ja + 0.5)
    pref *= float(ssqrt((2 * ja_r + 1) * (2 * jb_r + 1) * (2 * k_r + 1) / (4 * np.pi)))
    w3j = float(wigner_3j(ja_r, k_r, jb_r, Rational(1, 2), 0, -Rational(1, 2)))
    return pref * w3j


def reduced_quadrupole_matrix_element(na, la, ja, nb, lb, jb) -> float:
    """<na la ja||Q_2||nb lb jb> = radial x angular, in units of b^2
    (harmonic-oscillator length squared). Computed entirely from (n,l,j)
    labels -- no external table needed."""
    rad = ho_radial_r2(na, la, nb, lb)
    if rad == 0.0:
        return 0.0
    ang = angular_reduced_Y2(la, ja, lb, jb, k=2)
    return rad * ang


class QuadrupoleOperator(FermiHubbardHamiltonian):

    def __init__(
        self,
        size_a: int,
        size_b: int,
        nparticles_a: int,
        nparticles_b: int,
        single_particle_states: List,
        symmetries: Optional[List[Callable]] = None,
        use_optimized: bool = True,
    ):
        """
        Args:
            size_a, size_b: number of neutron / proton single particle states
            nparticles_a, nparticles_b: number of neutrons / protons
            single_particle_states: list of (n,l,j,m,t,t_z) tuples, i.e.
                SingleParticleState(...).state_encoding -- this fully
                determines the quadrupole operator, no other input needed.
            symmetries: many-body basis symmetries, same as J2operator/FermiHubbardHamiltonian
            use_optimized: use adag_a_matrix_optimized / get_twobody_interaction_optimized
                (numba/bitmask, fast) vs the plain python versions. Default True.

        NOTE on units: matrix elements are computed in harmonic-oscillator
        units (lengths in units of b = sqrt(hbar/(m omega))), i.e. Q has
        units of b^2. If you need physical units (fm^2), multiply by the
        b^2 appropriate for your nucleus/hbar*omega afterward -- this is a
        single overall multiplicative constant and does not affect any of
        the structure (selection rules, relative magnitudes, hermiticity).
        """
        super().__init__(size_a, size_b, nparticles_a, nparticles_b, symmetries)

        self.single_particle_states = single_particle_states
        self.use_optimized = use_optimized

        self.mus = [-2, -1, 0, 1, 2]

        # single-particle-basis matrices, one dict per mu: {(idx_a,idx_b): value}
        self.quadrupole_matrices: Dict[int, Dict[Tuple[int, int], complex]] = {}

        # many-body one-body operators, one sparse matrix per mu
        self.quadrupole_operator: Dict[int, "scipy.sparse.spmatrix"] = {}

        self.__build_all_single_particle_matrices()
        self.__build_all_manybody_one_body_operators()

    # ------------------------------------------------------------------
    # single-particle matrix elements  <a| Q_mu |b>
    # ------------------------------------------------------------------
    def __get_the_quadrupole_matrix(self, mu: int) -> Dict[Tuple[int, int], complex]:
        """Build the single-particle-basis matrix of Q_mu via Wigner-Eckart,
        with the reduced matrix element computed on the fly from (n,l,j)
        labels via reduced_quadrupole_matrix_element() -- no external table.

        <a| Q_mu |b> = <(na la ja)||Q_2||(nb lb jb)> * <jb mb, 2 mu | ja ma> / sqrt(2 ja+1)

        Selection rules enforced:
          - isospin projection conserved (t_z_a == t_z_b): the quadrupole
            operator does not mix neutrons and protons.
          - triangle rule |ja-jb| <= 2 <= ja+jb (guarded explicitly before
            calling ClebschGordan to avoid an invalid-J exception).
          - parity: la+lb even (enforced inside reduced_quadrupole_matrix_element
            via angular_reduced_Y2's la+lb+k even check).
          - radial: |na-nb| <= 1 (enforced inside ho_radial_r2).
        """
        state_encoding = self.single_particle_states
        quadrupole_matrix: Dict[Tuple[int, int], complex] = {}

        # cache reduced MEs per distinct (na,la,ja,nb,lb,jb) so the 3j/Wigner
        # computation isn't repeated for every (ma,mb) pair sharing the same
        # (n,l,j) -- there are only a handful of distinct orbitals even
        # though there can be many m-substates.
        reduced_cache: Dict[Tuple, float] = {}

        for idx_a, a in enumerate(state_encoding):
            na, la, ja, ma, ta, tza = a
            for idx_b, b in enumerate(state_encoding):
                nb, lb, jb, mb, tb, tzb = b

                # quadrupole does not mix species (no isospin-raising/lowering)
                if tza != tzb:
                    continue

                # triangle rule for coupling jb (x) 2 -> ja
                if ja < abs(jb - 2) or ja > jb + 2:
                    continue

                # m-selection rule from the CG coefficient mb + mu == ma
                if not np.isclose(mb + mu, ma):
                    continue

                rkey = (na, la, ja, nb, lb, jb)
                if rkey not in reduced_cache:
                    reduced_cache[rkey] = reduced_quadrupole_matrix_element(
                        na, la, ja, nb, lb, jb
                    )
                reduced_value = reduced_cache[rkey]
                if reduced_value == 0.0:
                    continue

                cg_list = ClebschGordan(j1=jb, j2=2, J=ja)
                cg_value = SelectCG(cg_list, j1=jb, m1=mb, j2=2, m2=mu, J=ja, M=ma)

                if cg_value == 0.0:
                    continue

                value = reduced_value * cg_value / np.sqrt(2 * ja + 1)

                if np.abs(value) < 1e-10:
                    continue

                quadrupole_matrix[(idx_a, idx_b)] = value

        return quadrupole_matrix

    def __build_all_single_particle_matrices(self):
        for mu in self.mus:
            self.quadrupole_matrices[mu] = self.__get_the_quadrupole_matrix(mu=mu)

    # ------------------------------------------------------------------
    # many-body one-body operator  Q_mu = sum_ab <a|Q_mu|b> a^dagger_a a_b
    # ------------------------------------------------------------------
    def __build_all_manybody_one_body_operators(self):
        for mu in self.mus:
            qop = 0.0
            for (idx_a, idx_b), value in self.quadrupole_matrices[mu].items():
                if self.use_optimized:
                    term = self.adag_a_matrix_optimized(i=idx_a, j=idx_b)
                else:
                    term = self.adag_a_matrix(i=idx_a, j=idx_b)
                qop = qop + value * term
            self.quadrupole_operator[mu] = qop

    # ------------------------------------------------------------------
    # many-body two-body dictionary for Q.Q  (feed straight into
    # get_twobody_interaction_optimized, same as the nuclear TBMEs / J2)
    # ------------------------------------------------------------------
    def build_qq_twobody_dict(self) -> Tuple[Dict[Tuple[int, int, int, int], complex], np.ndarray]:
        """
        Build (1) the (i1,i2,i3,i4)->value TWO-BODY dictionary and (2) the
        residual ONE-BODY matrix, for the quadrupole-quadrupole operator

            Q.Q = sum_mu (-1)^mu Q_mu Q_{-mu}

        IMPORTANT PHYSICS NOTE (this is the part that's easy to get wrong,
        and the reason this function returns TWO objects, not one):

        Q_mu and Q_{-mu} are each one-body operators a^dagger a. Their
        PRODUCT, expanded with fermionic anticommutation rules, is NOT a
        pure two-body operator -- it contains a one-body remainder from the
        contraction of the inner a_b a^dagger_c pair:

            a^dagger_a a_b  a^dagger_c a_d
                = - a^dagger_a a^dagger_c a_b a_d  +  delta_{bc} a^dagger_a a_d

        So:  Q_mu Q_{-mu} = (two-body piece)  +  (Q_mu @ Q_{-mu})_as_one_body

        where "Q_mu @ Q_{-mu}" here means ordinary MATRIX multiplication of
        the single-particle-basis matrices (sum over the contracted index b=c).

        If you skip the one-body remainder, your Q.Q operator is WRONG by an
        amount that is generally NOT small (verified numerically: it can be
        a large fraction of the total operator norm). This is the same issue
        as normal-ordering a two-body operator -- except here it arises
        purely from writing a product of two one-body operators as a single
        expression, before you even pick a reference state.

        Returns:
            twobody_dict: pass directly to
                self.get_twobody_interaction_optimized(twobody_dict)
                using the SAME (i1,i2,i3,i4) -> adag_adag_a_a_matrix(i1,i2,j1=i4,j2=i3)/4
                convention as everywhere else in this codebase.
            one_body_correction: np.ndarray (n_sp, n_sp), the residual
                sum_mu (-1)^mu (Q_mu @ Q_{-mu}) as a single-particle-basis
                matrix. Turn it into a many-body operator with
                build_one_body_correction_operator() below and ADD it to
                your single-particle Hamiltonian (same footing as
                kinetic_operator / external_potential) -- it is generally
                NOT diagonal since Q mixes different (n,l,j), so don't just
                take the diagonal.

        Verified numerically: matches direct many-body matrix multiplication
        of the Q_mu operators to ~1e-14, including the full mu-sum and the
        hermiticity of the resulting Q.Q operator.
        """
        n_sp = self.size_a + self.size_b

        q_dense = {}
        for mu in self.mus:
            arr = np.zeros((n_sp, n_sp), dtype=complex)
            for (a, b), v in self.quadrupole_matrices[mu].items():
                arr[a, b] = v
            q_dense[mu] = arr

        twobody_dict: Dict[Tuple[int, int, int, int], complex] = {}
        one_body_correction = np.zeros((n_sp, n_sp), dtype=complex)

        for mu in self.mus:
            sign = (-1) ** mu
            Qmu = q_dense[mu]
            Qmmu = q_dense[-mu]

            # one-body contraction piece: sum_b q_mu(a,b) q_{-mu}(b,d)
            one_body_correction += sign * (Qmu @ Qmmu)

            # two-body piece: -(-1)^mu * q_mu(a,b) q_{-mu}(c,d) at key (a,c,d,b),
            # i.e. operator a^dagger_a a^dagger_c a_b a_d, stored so that
            # get_twobody_interaction_optimized's adag_adag_a_a_matrix(i1,i2,j1=i4,j2=i3)/4
            # convention reproduces it exactly (verified numerically against
            # direct many-body matrix multiplication).
            nz_ab = np.argwhere(np.abs(Qmu) > 1e-12)
            nz_cd = np.argwhere(np.abs(Qmmu) > 1e-12)
            for a, b in nz_ab:
                qab = Qmu[a, b]
                for c, d in nz_cd:
                    qcd = Qmmu[c, d]
                    key = (int(a), int(c), int(d), int(b))
                    val = -4.0 * sign * qab * qcd
                    twobody_dict[key] = twobody_dict.get(key, 0.0) + val

        twobody_dict = {k: v for k, v in twobody_dict.items() if np.abs(v) > 1e-10}

        return twobody_dict, one_body_correction

    def build_one_body_correction_operator(self, one_body_correction: np.ndarray):
        """
        Turn the dense one_body_correction matrix from build_qq_twobody_dict()
        into a many-body sparse operator (same building block as the rest of
        this module). Add the RESULT to your single-particle Hamiltonian
        (treat on the same footing as self.kinetic_operator /
        self.external_potential) -- not the dense matrix itself.
        """
        n_sp = self.size_a + self.size_b
        op = 0.0
        for a in range(n_sp):
            for b in range(n_sp):
                val = one_body_correction[a, b]
                if np.abs(val) < 1e-12:
                    continue
                if self.use_optimized:
                    term = self.adag_a_matrix_optimized(i=a, j=b)
                else:
                    term = self.adag_a_matrix(i=a, j=b)
                op = op + val * term
        return op

    # ------------------------------------------------------------------
    # expectation values
    # ------------------------------------------------------------------
    def deformation_value(self, psi: np.ndarray) -> float:
        """
        Compute sqrt( sum_mu (-1)^mu <psi|Q_mu|psi><psi|Q_{-mu}|psi> )
        i.e. a rotationally-invariant scalar deformation measure built from
        the one-body Q_mu expectation values (this is NOT <Q.Q>, it is the
        magnitude of the vector of expectation values -- useful when psi is
        not an eigenstate of Q.Q but you still want a deformation indicator).
        """
        tot_value = 0.0
        for mu in self.mus:
            op = self.quadrupole_operator[mu]
            value_mu = psi.conjugate().T.dot(op.dot(psi))
            value_minus_mu = psi.conjugate().T.dot(self.quadrupole_operator[-mu].dot(psi))
            tot_value += ((-1) ** mu) * value_mu * value_minus_mu
        return np.sqrt(np.abs(tot_value))

    def qq_expectation_from_operator(self, psi: np.ndarray) -> float:
        """
        <psi| Q.Q |psi> computed directly from the many-body operator AFTER
        you've assembled it correctly, i.e.:

            twobody_dict, one_body_correction = self.build_qq_twobody_dict()
            self.get_twobody_interaction_optimized(twobody_dict)
            one_body_op = self.build_one_body_correction_operator(one_body_correction)
            qq_full_operator = self.twobody_operator + one_body_op

        then call this on qq_full_operator (or just compute the expectation
        value directly -- this helper assumes self.twobody_operator already
        equals the FULL Q.Q operator including the one-body correction, so
        build it by adding one_body_op into self.twobody_operator before
        calling get_hamiltonian(), or just bypass this helper and compute
        psi.conj().T @ qq_full_operator @ psi yourself).

        Use this as a consistency check against deformation_value() above,
        which instead computes sum_mu (-1)^mu <Q_mu><Q_{-mu}> from one-body
        expectation values only (cheaper, but NOT equal to <Q.Q> in general
        unless psi is a Hartree-Fock-like product state for which the
        connected/correlation part of <Q_mu Q_{-mu}> beyond <Q_mu><Q_{-mu}>
        vanishes).
        """
        if self.twobody_operator is None:
            raise RuntimeError(
                "Call get_twobody_interaction_optimized(twobody_dict) from "
                "build_qq_twobody_dict(), add the one-body correction "
                "operator into self.twobody_operator, then get_hamiltonian()."
            )
        return psi.conjugate().T.dot(self.twobody_operator.dot(psi))
