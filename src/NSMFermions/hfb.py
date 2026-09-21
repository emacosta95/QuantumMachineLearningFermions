"""Complex, species-unrestricted HFB reference implementation (NumPy/SciPy).

H = h_ij c_i^dag c_j + v_ijkl c_i^dag c_j^dag c_l c_k / 4.
rho_ij = <c_j^dag c_i>, kappa_ij = <c_j c_i>.
No species, reality, or time-reversal blocks are imposed. The vacuum-based
solver currently covers even total number parity only, not blocked odd states.
"""

from dataclasses import dataclass
import numpy as np
from pfapack import pfaffian as pf
from scipy.linalg import expm, null_space
from scipy.optimize import minimize


@dataclass
class HFBState:
    """Bogoliubov amplitudes defining an even-parity quasiparticle vacuum.

    The convention used throughout this module is

        beta = U^dagger c + V^dagger c^dagger.

    Both arrays have shape ``(modes, modes)``.  They are not separately
    unitary; together they form the canonical transformation in Nambu space.
    ``Z`` optionally stores the particle-vacuum Thouless chart used to create
    the state.  It is useful for occupation amplitudes and symmetry projection.
    """

    U: np.ndarray
    V: np.ndarray
    Z: np.ndarray = None

    @classmethod
    def from_thouless(cls, z):
        """Construct a normalized Bogoliubov vacuum from antisymmetric Z."""
        z = np.asarray(z, complex)
        if z.ndim != 2 or z.shape[0] != z.shape[1] or not np.isfinite(z).all():
            raise ValueError("Z must be a finite square matrix")
        if not np.allclose(z, -z.T, atol=1e-12):
            raise ValueError("Z must be antisymmetric")

        # U=(I+Z^T Z*)^-1/2 and V=Z*U satisfy the canonical relations and
        # annihilate exp(1/2 c^dag Z c^dag)|0>.  The eigendecomposition evaluates
        # the positive Hermitian inverse square root without explicitly inverting.
        metric = np.eye(len(z)) + z.T @ z.conj()
        values, vectors = np.linalg.eigh(metric)
        u = (vectors / np.sqrt(values)) @ vectors.conj().T
        return cls(u, z.conj() @ u, Z=z.copy())

    @classmethod
    def from_slater(cls, orbitals):
        """Construct the singular-U Bogoliubov vacuum of a Slater determinant.

        ``orbitals`` has shape ``(modes, particles)`` and contains orthonormal
        occupied orbitals.  Occupied-orbital creation operators are hole-like
        quasiparticle annihilators; the orthogonal complement supplies the
        ordinary particle-like quasiparticles.
        """
        # Convert lists or real arrays to the complex matrix used by HFB algebra.
        occupied = np.asarray(orbitals, complex)
        # A Slater state is specified by one finite two-dimensional orbital matrix.
        if occupied.ndim != 2 or not np.isfinite(occupied).all():
            raise ValueError("Slater orbitals must be a finite matrix")
        # Read the one-body dimension and number of occupied orbitals from C.
        modes, particles = occupied.shape
        # Canonical creation operators require C^dagger C = I.
        if particles > modes or not np.allclose(
            occupied.conj().T @ occupied, np.eye(particles), atol=1e-10
        ):
            raise ValueError("Slater orbitals must have orthonormal columns")

        # Complete the occupied columns with an orthonormal empty-orbital basis.
        empty = null_space(occupied.conj().T)
        # Allocate the particle-like block of the Bogoliubov transformation.
        u = np.zeros((modes, modes), complex)
        # Allocate the hole-like block of the Bogoliubov transformation.
        v = np.zeros((modes, modes), complex)
        # beta_h = d_h^dagger for occupied orbitals and beta_p = d_p for empty
        # orbitals.  This makes every beta annihilate the Slater determinant.
        v[:, :particles] = occupied.conj()
        u[:, particles:] = empty
        # HFBState validation and observables now use the same canonical U,V API.
        return cls(u, v)

    @property
    def slater_orbitals(self):
        """Return occupied orbitals when this state is an exact Slater state."""
        # A number-conserving Slater determinant has kappa=0 and rho^2=rho.
        rho = self.rho
        if np.linalg.norm(self.kappa) > 1e-8 or not np.allclose(
            rho @ rho, rho, atol=1e-8
        ):
            raise ValueError("Bogoliubov state is not an unpaired Slater state")
        # Hermitize roundoff before extracting natural occupations and orbitals.
        hermitian_rho = (rho + rho.conj().T) / 2
        occupations, orbitals = np.linalg.eigh(hermitian_rho)
        # Eigenvalue one identifies an occupied natural orbital.
        selected = occupations > 0.5
        if not np.all(
            (occupations[selected] > 1 - 1e-8)
        ) or not np.all(occupations[~selected] < 1e-8):
            raise ValueError("One-body density is not an idempotent Slater density")
        # Return only the occupied columns; their phases do not affect fidelity.
        return orbitals[:, selected]

    @property
    def thouless_matrix(self):
        """Return Z in |Phi> proportional to exp(1/2 c^dag Z c^dag)|0>.

        If the state was created directly from a Thouless matrix, return that
        stored chart.  Otherwise recover ``Z = V* (U*)^-1``.  A singular U means
        the vacuum has zero overlap with the particle vacuum, so this chart does
        not exist even though the Bogoliubov state itself remains valid.
        """
        if self.Z is not None:
            z = np.asarray(self.Z, complex)
        else:
            if np.linalg.cond(self.U) > 1e12:
                raise ValueError(
                    "Singular U: no particle-vacuum Thouless chart exists"
                )
            # Solve on transposed matrices instead of explicitly forming U^-1.
            z = np.linalg.solve(
                self.U.conj().T, self.V.conj().T
            ).T
        if not np.allclose(z, -z.T, atol=1e-10):
            raise ValueError("Recovered Thouless matrix is not antisymmetric")
        return z

    def occupation_amplitude(self, occupied_modes, *, normalized=False):
        """Return the Pfaffian coefficient of one occupation-basis determinant.

        For an even ordered set I of occupied modes, the coefficient in the
        unnormalized Thouless exponential is ``Pf(Z[I,I])``.  Odd occupations
        have zero amplitude in an unblocked even-parity vacuum.  Set
        ``normalized=True`` to include the norm of the full intrinsic vacuum.
        """
        occupied = tuple(int(i) for i in occupied_modes)
        modes = len(self.U)
        if len(set(occupied)) != len(occupied) or any(
            i < 0 or i >= modes for i in occupied
        ):
            raise ValueError("Occupied modes must be distinct valid indices")
        if len(occupied) % 2:
            return 0j

        # Sort the creation operators into the repository's canonical mode
        # order.  The caller normally supplies sorted determinant occupations.
        if occupied != tuple(sorted(occupied)):
            raise ValueError("Occupied modes must be in increasing order")
        # A finite particle-vacuum Thouless chart covers paired vacua with
        # nonzero vacuum overlap.  A nonempty HF determinant has singular U;
        # calculate that boundary state's coefficients as orbital determinants.
        try:
            z = self.thouless_matrix
        except ValueError as chart_error:
            try:
                orbitals = self.slater_orbitals
            except ValueError:
                raise chart_error
            if len(occupied) != orbitals.shape[1]:
                # A fixed-number Slater determinant vanishes in every other sector.
                return 0j
            # The coefficient of determinant I is the occupied-orbital minor det C_I.
            return complex(np.linalg.det(orbitals[list(occupied), :]))

        if occupied:
            submatrix = z[np.ix_(occupied, occupied)]
            amplitude = pf.pfaffian(
                submatrix, overwrite_a=False, method="P"
            )
        else:
            # By definition, the Pfaffian of the empty matrix is one.
            amplitude = 1.0 + 0j

        if normalized:
            # ||exp(1/2 c^dag Z c^dag)|0>||^2 = sqrt(det(I+Z^dag Z)).
            # Therefore the ket normalization factor is det(...)^(-1/4).
            log_determinant = np.linalg.slogdet(
                np.eye(modes) + z.conj().T @ z
            )[1]
            amplitude *= np.exp(-0.25 * log_determinant)
        return amplitude

    def occupation_amplitudes(self, occupations, *, normalized=False):
        """Return Pfaffian coefficients for a sequence of determinants."""
        return np.array([
            self.occupation_amplitude(occupied, normalized=normalized)
            for occupied in occupations
        ])

    def fixed_sector_state(self, occupations):
        """Return this vacuum normalized within a selected determinant sector.

        ``occupations`` defines both which determinants are retained and their
        coefficient ordering.  For example, passing all determinants with fixed
        neutron and proton numbers constructs normalized P_N P_Z|Phi>.
        """
        amplitudes = self.occupation_amplitudes(occupations)
        norm = float(np.vdot(amplitudes, amplitudes).real)
        if not np.isfinite(norm) or norm < 1e-24:
            raise ValueError("State has vanishing weight in selected sector")
        return amplitudes / np.sqrt(norm)

    def fixed_sector_weight(self, occupations):
        """Return the probability of a determinant sector in normalized |Phi>."""
        amplitudes = self.occupation_amplitudes(occupations, normalized=True)
        return float(np.vdot(amplitudes, amplitudes).real)

    def fixed_sector_fidelity(self, target, occupations, *, projected=False):
        """Return fidelity with a target supported on a determinant sector.

        With ``projected=False`` this is the raw overlap with the normalized
        intrinsic Gaussian vacuum and therefore includes the sector probability.
        With ``projected=True`` the selected sector is normalized first, giving
        the fidelity after projection onto that sector.
        """
        target = np.asarray(target, complex)
        if target.shape != (len(occupations),):
            raise ValueError("Target vector has wrong sector dimension")
        target_norm = np.linalg.norm(target)
        if not np.isfinite(target_norm) or target_norm == 0:
            raise ValueError("Target vector must have finite nonzero norm")
        target = target / target_norm
        amplitudes = (
            self.fixed_sector_state(occupations)
            if projected
            else self.occupation_amplitudes(occupations, normalized=True)
        )
        return float(abs(np.vdot(target, amplitudes)) ** 2)

    @property
    def rho(self):
        """Return the normal one-body density rho_ij = <c_j^dagger c_i>."""
        # With the convention above,
        # rho_ij = sum_k V_ik^* V_jk = (V^* V^T)_ij.
        return self.V.conj() @ self.V.T

    @property
    def kappa(self):
        """Return the anomalous density kappa_ij = <c_j c_i>."""
        # kappa_ij = sum_k V_ik^* U_jk = (V^* U^T)_ij.  The canonical
        # relations make this matrix antisymmetric, as required for fermions.
        return self.V.conj() @ self.U.T

    def canonical_error(self):
        """Measure violation of the fermionic canonical relations.

        This is zero exactly when the full Nambu-space Bogoliubov matrix is
        unitary.  A value near machine precision is expected for states made
        by :func:`state_from_parameters`.
        """
        m = len(self.U)
        # {beta_i, beta_j^dagger} = delta_ij gives the normalization residual.
        normalization_error = np.linalg.norm(
            self.U.conj().T @ self.U
            + self.V.conj().T @ self.V
            - np.eye(m)
        )
        # {beta_i, beta_j} = 0 gives the anomalous residual.
        anomalous_error = np.linalg.norm(
            self.U.T @ self.V + self.V.T @ self.U
        )
        # Report the worse of the two Frobenius-norm residuals.
        return max(
            normalization_error,
            anomalous_error,
        )


@dataclass
class BogoliubovVacuumSeries:
    """Weighted sum of one-body transformed Bogoliubov vacua.

    A symmetry projector is represented without choosing a many-body basis as

        |Psi> = sum_q weight[q] T[q] |Phi>.

    ``transformations[q]`` is the one-body unitary associated with gauge and/or
    Euler point ``q``. Determinant amplitudes are evaluated only when
    :meth:`occupation_amplitudes` is called, which is the boundary between the
    polynomial vacuum-series representation and an explicit Fock-space basis.
    """

    # Intrinsic HFB/HF vacuum to which every group transformation is applied.
    intrinsic_state: HFBState
    # One-body matrices T_q in the convention Z_q = T_q Z T_q^T.
    transformations: np.ndarray
    # Complex Fourier/quadrature coefficient multiplying every transformed ket.
    weights: np.ndarray
    # User-selected neutron/proton or subsystem-A/subsystem-B grid dimensions.
    number_grid: tuple
    # User-selected (alpha, beta, gamma) grid, or None for number projection only.
    euler_grid: object = None
    # Human-readable description retained in saved benchmark metadata.
    projection: str = ""
    # Fractional shift of both number Fourier grids.
    number_offset: float = 0.137
    # Fractional shift of alpha/gamma Euler grids, if present.
    euler_offset: object = None

    def __post_init__(self):
        """Validate that every series term acts on the intrinsic one-body space."""
        # Projection series are meaningful only for complete Bogoliubov states.
        if not isinstance(self.intrinsic_state, HFBState):
            raise TypeError("intrinsic_state must be an HFBState")
        # Convert all transforms to one dense array with shape (terms,modes,modes).
        self.transformations = np.asarray(self.transformations, dtype=complex)
        # Convert all coefficients to one complex vector in the same term order.
        self.weights = np.asarray(self.weights, dtype=complex)
        # Read the expected one-body dimension from the intrinsic U matrix.
        modes = len(self.intrinsic_state.U)
        # Every coefficient must correspond to exactly one square transformation.
        expected_shape = (len(self.weights), modes, modes)
        if self.transformations.shape != expected_shape:
            raise ValueError("Projection transforms and weights have incompatible shapes")
        # NaN or infinite quadrature data would invalidate every observable.
        if not np.isfinite(self.transformations).all() or not np.isfinite(self.weights).all():
            raise ValueError("Projection series must contain finite data")
        # Store grid metadata as immutable ordinary integers.
        self.number_grid = tuple(int(points) for points in self.number_grid)
        if self.euler_grid is not None:
            self.euler_grid = tuple(int(points) for points in self.euler_grid)
        # Store finite scalar offsets so series metadata are self-contained.
        self.number_offset = float(self.number_offset)
        if not np.isfinite(self.number_offset):
            raise ValueError("Number-grid offset must be finite")
        if self.euler_offset is not None:
            self.euler_offset = float(self.euler_offset)
            if not np.isfinite(self.euler_offset):
                raise ValueError("Euler-grid offset must be finite")

    @property
    def number_of_vacua(self):
        """Number M of transformed Bogoliubov vacua in the finite series."""
        # There is one vacuum for each stored group-quadrature transformation.
        return len(self.weights)

    def occupation_amplitudes(self, occupations):
        """Expand the projected series in selected occupation configurations.

        This is intentionally the first operation that introduces an explicit
        determinant basis. Finite-Z vacua use Pfaffians; exact Slater limits use
        occupied-orbital minors with phases propagated by the same T_q matrices.
        """
        # Freeze determinant ordering because it must match the Hamiltonian rows.
        occupations = tuple(tuple(int(mode) for mode in row) for row in occupations)
        # Allocate the coefficient vector of the unnormalized projected state.
        projected = np.zeros(len(occupations), dtype=complex)

        try:
            # A finite Thouless chart gives phase-consistent Pfaffian amplitudes.
            z = self.intrinsic_state.thouless_matrix
        except ValueError as chart_error:
            try:
                # Singular-U HF states are represented by occupied orbitals instead.
                orbitals = self.intrinsic_state.slater_orbitals
            except ValueError:
                # Preserve the original chart error for a genuinely unsupported state.
                raise chart_error

            # Apply every gauge/Euler transformation to the same phased orbitals.
            for weight, transform in zip(self.weights, self.transformations):
                # A one-body unitary maps occupied columns as C_q = T_q C.
                rotated_orbitals = transform @ orbitals
                # Add the determinant minor for every requested configuration.
                for index, occupied in enumerate(occupations):
                    if len(occupied) == rotated_orbitals.shape[1]:
                        projected[index] += weight * np.linalg.det(
                            rotated_orbitals[list(occupied), :]
                        )
            return projected

        # All unitary group rotations preserve det(I+Z^dagger Z), so compute the
        # normalized intrinsic vacuum coefficient once for the complete series.
        norm_matrix = np.eye(len(z)) + z.conj().T @ z
        log_determinant = np.linalg.slogdet(norm_matrix)[1]
        vacuum_amplitude = np.exp(-0.25 * log_determinant)

        # Each quadrature term remains a Bogoliubov vacuum with Z_q=T_q Z T_q^T.
        for weight, transform in zip(self.weights, self.transformations):
            # Rotate both creation-operator indices of the Thouless matrix.
            rotated_z = transform @ z @ transform.T
            # Evaluate only the configurations requested by the eventual observable.
            for index, occupied in enumerate(occupations):
                # Unblocked even vacua have no odd-total-particle coefficients.
                if len(occupied) % 2:
                    continue
                # The empty determinant has Pfaffian one by definition.
                if not occupied:
                    coefficient = 1.0 + 0.0j
                else:
                    # Extract the principal antisymmetric matrix for configuration I.
                    submatrix = rotated_z[np.ix_(occupied, occupied)]
                    # PFAPACK supplies its phase-consistent complex Pfaffian.
                    coefficient = pf.pfaffian(
                        submatrix, overwrite_a=False, method="P"
                    )
                # Accumulate w_q <I|T_q|Phi> into the projected-state coefficient.
                projected[index] += weight * vacuum_amplitude * coefficient
        return projected


def state_from_parameters(x, modes):
    """Exponentiate an arbitrary complex antisymmetric pairing generator.

    Unlike a vacuum Thouless inverse, this representation permits singular U.
    All mode pairs, including neutron-proton pairs, are parameterized.
    This is used to optimize the HFB energy over independent real parameters.
    """
    # A complex antisymmetric modes x modes matrix has modes*(modes-1)/2
    # independent complex entries.  ``ij`` is a tuple of row and column arrays
    # selecting those entries strictly above the diagonal.
    ij = np.triu_indices(modes, 1)
    p = len(ij[0])

    # Store the real and imaginary parts consecutively in one real optimizer
    # vector.  Its required length is 2*p = modes*(modes-1).
    x = np.asarray(x, dtype=float)
    if x.shape != (2 * p,) or not np.isfinite(x).all():
        raise ValueError("Expected modes*(modes-1) finite real parameters")

    # Fill the independent upper-triangular entries, then reflect them with a
    # minus sign.  No complex conjugation is used: z^T = -z, not z^dagger = -z.
    z = np.zeros((modes, modes), complex)
    z[ij] = x[:p] + 1j * x[p:]
    z -= z.T

    # Embed z in a 2*modes dimensional particle-hole (Nambu) generator.  The
    # antisymmetry of z makes this block matrix anti-Hermitian.
    zero = np.zeros_like(z)
    generator = np.block([[zero, z.conj()], [z, zero]])

    # Exponentiating an anti-Hermitian generator produces a unitary canonical
    # transformation.  This avoids forming the potentially undefined U^{-1}
    # that appears in vacuum Thouless coordinates.
    w = expm(generator)

    # The first modes columns of w are stacked as [U; V].  The remaining
    # columns are their particle-hole partners and need not be stored.
    return HFBState(w[:modes, :modes], w[modes:, :modes])


class HFBHamiltonian:
    """Validated one- plus antisymmetrized two-body Hamiltonian.

    Parameters
    ----------
    h : array_like, shape (modes, modes)
        Hermitian one-body matrix ``h[i, j]`` multiplying ``c_i^dagger c_j``.
    interaction : mapping or array_like
        Antisymmetrized matrix elements ``v[i, j, k, l]`` multiplying
        ``c_i^dagger c_j^dagger c_l c_k / 4``.  A mapping may be used to
        specify a sparse set of entries, but all required permutations must
        already be present; this class does not generate them automatically.
    """

    def __init__(self, h, interaction):
        # Copy the one-body input so later modifications by the caller cannot
        # silently change the Hamiltonian held by this object.
        self.h = np.array(h, dtype=complex, copy=True)
        if self.h.ndim != 2 or self.h.shape[0] != self.h.shape[1]:
            raise ValueError("h must be square")
        m = len(self.h)

        # Internally the interaction is always stored as a dense rank-four
        # tensor.  Dictionary input is useful when only a few elements are
        # nonzero; unspecified entries remain zero.
        self.v = np.zeros((m,) * 4, complex)
        if isinstance(interaction, dict):
            for indices, value in interaction.items():
                if len(indices) != 4 or any(i < 0 or i >= m for i in indices):
                    raise ValueError("Interaction index out of range")
                self.v[indices] = value
        else:
            self.v = np.array(interaction, dtype=complex, copy=True)

        # The one- and two-body tensors must use the same number of modes.
        if self.v.shape != (m,) * 4:
            raise ValueError("Interaction must have shape (m,m,m,m)")

        # NaNs or infinities would make both the energy and optimizer output
        # unreliable, so reject them before checking tensor symmetries.
        if not np.isfinite(self.h).all() or not np.isfinite(self.v).all():
            raise ValueError("Hamiltonian must be finite")

        # Enforce, in order:
        #   h_ij = h_ji^*,
        #   v_ijkl = -v_jikl,
        #   v_ijkl = -v_ijlk,
        #   v_ijkl = v_klij^*.
        # The middle two identities encode fermionic antisymmetry within the
        # creation and annihilation index pairs; the last is Hermiticity.
        for a, b in [
            (self.h, self.h.conj().T),
            (self.v, -self.v.swapaxes(0, 1)),
            (self.v, -self.v.swapaxes(2, 3)),
            (self.v, self.v.transpose(2, 3, 0, 1).conj()),
        ]:
            if not np.allclose(a, b, atol=1e-10, rtol=1e-10):
                raise ValueError("Hamiltonian violates Hermiticity or antisymmetry")

    def energy(self, state):
        """Evaluate the HFB expectation value using Wick's theorem."""
        # Use short local names because the following index contractions are
        # the mathematical HFB energy formula written directly in einsum form.
        r, k = state.rho, state.kappa

        # One-body term: sum_ij h_ij rho_ji.
        one_body = np.einsum("ij,ji->", self.h, r)

        # Normal two-body contraction.  The antisymmetrized interaction already
        # contains the direct-minus-exchange combination, giving the 1/2 factor.
        normal_two_body = 0.5 * np.einsum("ijkl,ki,lj->", self.v, r, r)

        # Pairing contraction.  The Hamiltonian convention contains 1/4 and
        # contracts v_ijkl with kappa_ij^* kappa_kl.
        pairing = 0.25 * np.einsum(
            "ijkl,ij,kl->", self.v, k.conj(), k
        )

        # A valid Hermitian Hamiltonian and canonical state give a real result.
        # Keep a small tolerance for roundoff from dense complex contractions.
        e = one_body + normal_two_body + pairing
        if abs(e.imag) > 1e-8:
            raise ValueError("Non-real HFB energy")
        return float(e.real)


@dataclass
class HFBResult:
    """Best constrained solution and diagnostics from :func:`solve_hfb`.

    ``numbers`` contains the final ``[<N>, <Z>]`` expectation values.
    ``attempts`` records convergence information for every random start.
    ``chemical_potentials`` contains the neutron and proton Lagrange
    multipliers, or NaNs when the constraint Jacobian is rank deficient.
    ``stationarity_error`` measures the part of the energy gradient that
    cannot be represented by the two number-constraint gradients.
    """

    state: HFBState
    energy: float
    numbers: np.ndarray
    converged: bool
    message: str
    parameters: np.ndarray
    attempts: list
    chemical_potentials: np.ndarray
    stationarity_error: float


def solve_hfb(
    hamiltonian,
    neutron_modes,
    targets,
    *,
    starts=3,
    seed=0,
    maxiter=300,
    tolerance=1e-8,
    initial_parameters=None,
):
    """Minimize E subject to <N>=targets[0], <Z>=targets[1] using SLSQP.

    Equality constraints implement the Lagrange-multiplier problem, without a
    finite particle-number penalty. Finite-difference gradients and dense matrix
    exponentials make this a reference solver, not a production nuclear solver.
    Multiple paired starts reduce trapping; global optimality is not guaranteed.
    """
    # Establish which single-particle modes count as neutrons.  The complement
    # of this mask is treated as the proton subspace.
    m = len(hamiltonian.h)
    indices = np.asarray(neutron_modes, dtype=int)
    if (
        indices.ndim != 1
        or len(set(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= m)
    ):
        raise ValueError("neutron_modes must contain distinct valid indices")
    mask = np.zeros(m, bool)
    mask[indices] = True

    # Validate the requested average neutron and proton numbers against the
    # corresponding single-particle capacities.
    targets = np.asarray(targets, float)
    capacities = np.array([mask.sum(), (~mask).sum()])
    if (
        targets.shape != (2,)
        or not np.isfinite(targets).all()
        or np.any(targets < 0)
        or np.any(targets > capacities)
    ):
        raise ValueError("Invalid neutron/proton targets")
    if np.any(capacities == 0):
        raise ValueError("Reference solver requires both species in the space")
    if starts < 1 or maxiter < 1 or tolerance <= 0:
        raise ValueError("Positive starts, maxiter and tolerance required")

    def numbers(state):
        """Compute <N> and <Z> by summing diagonal occupations."""
        occupation = state.rho.diagonal().real
        return np.array([occupation[mask].sum(), occupation[~mask].sum()])

    def constraint(x):
        """Return equality-constraint residuals [<N>-N0, <Z>-Z0]."""
        return numbers(state_from_parameters(x, m)) - targets

    def objective(x):
        """Map optimizer coordinates to the physical HFB energy."""
        return hamiltonian.energy(state_from_parameters(x, m))

    # Run several randomized starts because the constrained HFB landscape is
    # non-convex.  A caller-supplied initial point is used for the first start.
    rng = np.random.default_rng(seed)
    attempts, candidates = [], []
    for attempt in range(starts):
        x = (
            np.array(initial_parameters, float)
            if attempt == 0 and initial_parameters is not None
            else rng.normal(scale=0.5 / np.sqrt(m), size=m * (m - 1))
        )
        fit = minimize(
            objective,
            x,
            method="SLSQP",
            constraints={"type": "eq", "fun": constraint},
            options={"maxiter": maxiter, "ftol": tolerance},
        )

        # Recompute the physical state and number residual from the returned
        # parameters rather than relying only on the optimizer success flag.
        state = state_from_parameters(fit.x, m)
        residual = float(np.max(np.abs(constraint(fit.x))))
        ok = bool(fit.success and residual < max(1e-7, 10 * tolerance))
        attempts.append(
            {
                "converged": ok,
                "energy": float(fit.fun),
                "number_error": residual,
                "message": str(fit.message),
            }
        )
        candidates.append((ok, residual, fit, state))

    # Among feasible runs choose the lowest energy.  If none converged, return
    # the run closest to satisfying the number constraints for diagnostics.
    feasible = [c for c in candidates if c[0]]
    best = (
        min(feasible, key=lambda c: c[2].fun)
        if feasible
        else min(candidates, key=lambda c: c[1])
    )
    ok, _, fit, state = best

    # Recover multipliers in grad(E) = lambda_n grad(N) + lambda_p grad(Z).
    # Central finite differences are adequate here because this is a compact
    # reference implementation and the optimizer itself is finite-difference
    # based.  Each row of ``jacobian`` is the gradient of one constraint.
    step = 1e-5
    directions = np.eye(len(fit.x)) * step
    gradient = np.array(
        [(objective(fit.x + d) - objective(fit.x - d)) / (2 * step) for d in directions]
    )
    jacobian = np.array(
        [
            (constraint(fit.x + d) - constraint(fit.x - d)) / (2 * step)
            for d in directions
        ]
    ).T
    multipliers = np.linalg.lstsq(jacobian.T, gradient, rcond=None)[0]

    # The remaining component of grad(E) tangent to the constraint surface is
    # a post-optimization stationarity diagnostic.
    stationarity = float(np.linalg.norm(gradient - jacobian.T @ multipliers))

    # At rank-deficient constraints (e.g. an HF limit), chemical potentials
    # are not uniquely determined by these first derivatives.
    if np.linalg.matrix_rank(jacobian, tol=1e-7) < 2:
        multipliers[:] = np.nan

    # Require both optimizer feasibility and sufficiently small stationarity.
    ok = bool(ok and stationarity < max(1e-5, 10 * np.sqrt(tolerance)))
    # Package both the physical solution and enough numerical information to
    # decide whether it is trustworthy without inspecting SciPy's raw result.
    return HFBResult(
        state=state,
        energy=float(fit.fun),
        numbers=numbers(state),
        converged=ok,
        message=str(fit.message),
        parameters=fit.x,
        attempts=attempts,
        chemical_potentials=multipliers,
        stationarity_error=stationarity,
    )
