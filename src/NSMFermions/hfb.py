"""Structure-preserving HF/HFB implementation (NumPy/SciPy).

H = h_ij c_i^dag c_j + v_ijkl c_i^dag c_j^dag c_l c_k / 4.
rho_ij = <c_j^dag c_i>, kappa_ij = <c_j c_i>.
No species or time-reversal blocks are imposed. HFB uses constrained local
Thouless gradients and heavy-ball updates following the TAURUS method, with
real or complex variational manifolds. The vacuum solver covers even total
number parity only, not blocked odd states.
"""

from dataclasses import dataclass
import numpy as np
from pfapack import pfaffian as pf
from scipy.linalg import expm, null_space
from scipy.optimize import minimize


class ProjectionGridWarning(UserWarning):
    """Warn that a user-selected grid lacks a finite-space exactness guarantee."""


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

    @staticmethod
    def _particle_hole_excitation_sign(reference, excitation):
        """Sign of a particle-hole excitation relative to ``|reference>``.

        For a reference determinant R, particle-hole creation operators are
        d_i^dagger=c_i for i in R and d_i^dagger=c_i^dagger otherwise.  This
        routine applies their ordered product and returns the fermionic sign
        relating the result to the canonical physical determinant ordering.
        """
        reference = set(int(index) for index in reference)
        occupied = set(reference)
        sign = 1
        # Operators written in increasing order act on the ket right-to-left.
        for mode in reversed(tuple(sorted(int(index) for index in excitation))):
            if sum(index < mode for index in occupied) % 2:
                sign = -sign
            if mode in reference:
                if mode not in occupied:
                    raise ValueError("Invalid repeated hole excitation")
                occupied.remove(mode)
            else:
                if mode in occupied:
                    raise ValueError("Invalid repeated particle excitation")
                occupied.add(mode)
        return sign, occupied

    def _reference_chart_amplitudes(self, occupations):
        """Normalized amplitudes in a nonsingular determinant-reference chart.

        A particle-vacuum Thouless chart fails when ``U`` is singular, even for
        a valid partially paired HFB vacuum.  Particle-hole conjugating the
        modes occupied in a determinant R replaces the vacuum reference by
        ``|R>``.  We choose the requested-sector determinant maximizing
        ``|det(U_R)|`` and expand the same physical state in that stable chart.

        The returned vector is defined up to one global phase, which cannot
        affect a single-state fidelity or a normalized fixed-sector state.
        It must not be used to phase independent terms of a projected vacuum
        series; those terms use their dedicated phase-tracking implementation.
        """
        configurations = [tuple(int(index) for index in row) for row in occupations]
        if not configurations:
            return np.empty(0, complex)
        if any(tuple(sorted(row)) != row for row in configurations):
            raise ValueError("Occupied modes must be in increasing order")
        particle_counts = {len(row) for row in configurations}
        if len(particle_counts) != 1:
            raise ValueError("Reference-chart amplitudes require one particle sector")

        # |det(U_R)| is the squared-overlap scale of the R-reference chart.
        # Maximizing it selects a determinant on which the HFB vacuum has a
        # numerically resolvable coefficient.
        best_reference = None
        best_log_determinant = -np.inf
        best_u = None
        best_v = None
        for reference in configurations:
            u_reference = self.U.copy()
            v_reference = self.V.copy()
            if reference:
                rows = np.asarray(reference, dtype=int)
                u_reference[rows] = self.V[rows]
                v_reference[rows] = self.U[rows]
            sign, log_determinant = np.linalg.slogdet(u_reference)
            if sign != 0 and log_determinant > best_log_determinant:
                best_reference = reference
                best_log_determinant = float(log_determinant)
                best_u = u_reference
                best_v = v_reference
        if best_reference is None or np.linalg.cond(best_u) > 1e12:
            raise ValueError(
                "No nonsingular particle-hole Thouless chart was found in "
                "the requested determinant sector"
            )

        z = np.linalg.solve(best_u.conj().T, best_v.conj().T).T
        antisymmetry_error = np.linalg.norm(z + z.T)
        if antisymmetry_error > 1e-8 * max(1.0, np.linalg.norm(z)):
            raise ValueError("Reference-chart Thouless matrix is not antisymmetric")
        z = 0.5 * (z - z.T)
        log_metric = np.linalg.slogdet(
            np.eye(len(z)) + z.conj().T @ z
        )[1]
        normalization = np.exp(-0.25 * log_metric)
        reference_set = set(best_reference)
        amplitudes = np.empty(len(configurations), complex)
        for position, occupied in enumerate(configurations):
            excitation = tuple(sorted(reference_set.symmetric_difference(occupied)))
            sign, result = self._particle_hole_excitation_sign(
                best_reference, excitation
            )
            if result != set(occupied):
                raise ValueError("Particle-hole excitation produced wrong determinant")
            if excitation:
                submatrix = z[np.ix_(excitation, excitation)]
                coefficient = pf.pfaffian(
                    submatrix, overwrite_a=False, method="P"
                )
            else:
                coefficient = 1.0 + 0.0j
            amplitudes[position] = sign * normalization * coefficient
        return amplitudes

    def stable_normalized_occupation_amplitudes(self, occupations):
        """Normalized single-vacuum amplitudes with a stable chart fallback.

        The ordinary particle-vacuum chart is retained whenever it is valid.
        Singular or numerically non-antisymmetric recovery switches to a
        determinant-reference particle-hole chart.  Both routes describe the
        identical normalized HFB vacuum up to an irrelevant global phase.
        """
        try:
            return self.occupation_amplitudes(occupations, normalized=True)
        except ValueError as particle_chart_error:
            try:
                return self._reference_chart_amplitudes(occupations)
            except ValueError:
                raise particle_chart_error

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
    # Construction rule: deterministic quadrature or Metropolis importance sampling.
    sampling_method: str = "quadrature"
    # Optional acceptance-rate and effective-sample-size diagnostics.
    sampling_diagnostics: object = None
    # Whether the number grid meets the state-independent finite-space bound.
    number_grid_guaranteed_exact: bool = True
    # Corresponding exactness marker for an Euler grid, or None when absent.
    euler_grid_guaranteed_exact: object = None
    # Minimum grids that provide the finite-space guarantees, for diagnostics.
    minimum_number_grid: object = None
    minimum_euler_grid: object = None

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
        self.number_grid_guaranteed_exact = bool(
            self.number_grid_guaranteed_exact
        )
        if self.euler_grid_guaranteed_exact is not None:
            self.euler_grid_guaranteed_exact = bool(
                self.euler_grid_guaranteed_exact
            )
        if self.minimum_number_grid is not None:
            self.minimum_number_grid = tuple(
                int(points) for points in self.minimum_number_grid
            )
        if self.minimum_euler_grid is not None:
            self.minimum_euler_grid = tuple(
                int(points) for points in self.minimum_euler_grid
            )
        # Store finite scalar offsets so series metadata are self-contained.
        self.number_offset = float(self.number_offset)
        if not np.isfinite(self.number_offset):
            raise ValueError("Number-grid offset must be finite")
        if self.euler_offset is not None:
            self.euler_offset = float(self.euler_offset)
            if not np.isfinite(self.euler_offset):
                raise ValueError("Euler-grid offset must be finite")
        # Limit the marker to the two supported series-construction algorithms.
        if self.sampling_method not in ("quadrature", "metropolis"):
            raise ValueError("Unknown projection-series sampling method")

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
            # Matrix exponentials in the larger sd-shell representation can
            # leave roundoff-level symmetric noise above PFAPACK's strict
            # assertion threshold. Restore the defining Z_q^T=-Z_q identity.
            rotated_z = 0.5 * (rotated_z - rotated_z.T)
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


def state_from_parameters(x, modes, *, real_parameters=False):
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

    # A real Bogoliubov vacuum uses one coordinate per antisymmetric pair.
    # The unrestricted complex chart stores the real and imaginary parts
    # consecutively and therefore uses twice as many real coordinates.
    x = np.asarray(x, dtype=float)
    expected = p if real_parameters else 2 * p
    if x.shape != (expected,) or not np.isfinite(x).all():
        raise ValueError(
            f"Expected {expected} finite Bogoliubov parameters"
        )

    # Fill the independent upper-triangular entries, then reflect them with a
    # minus sign.  No complex conjugation is used: z^T = -z, not z^dagger = -z.
    z = np.zeros((modes, modes), complex)
    z[ij] = x if real_parameters else x[:p] + 1j * x[p:]
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


def state_parameter_derivatives(x, modes, *, real_parameters=False):
    """Return ``state, dU/dx, dV/dx`` for the exponential HFB chart.

    The derivative uses the exact divided-difference representation of the
    Frechet derivative of the matrix exponential.  The anti-Hermitian Nambu
    generator is normal, so one Hermitian eigendecomposition supplies every
    parameter direction at once.

    References
    ----------
    N. J. Higham, *Functions of Matrices*, SIAM (2008), Secs. 3.1-3.2.
    A. H. Al-Mohy and N. J. Higham, SIAM J. Matrix Anal. Appl. 30,
    1639-1657 (2009), doi:10.1137/080716426.
    """
    ij = np.triu_indices(modes, 1)
    pairs = len(ij[0])
    x = np.asarray(x, dtype=float)
    parameter_count = pairs if real_parameters else 2 * pairs
    if x.shape != (parameter_count,) or not np.isfinite(x).all():
        raise ValueError(
            f"Expected {parameter_count} finite Bogoliubov parameters"
        )

    z = np.zeros((modes, modes), complex)
    z[ij] = x if real_parameters else x[:pairs] + 1j * x[pairs:]
    z -= z.T
    zero = np.zeros_like(z)
    generator = np.block([[zero, z.conj()], [z, zero]])

    # i*G is Hermitian when G is anti-Hermitian. If iG=Q diag(mu) Q^dagger,
    # then G=Q diag(-i*mu) Q^dagger.
    mu, eigenvectors = np.linalg.eigh(1j * generator)
    eigenvalues = -1j * mu
    exponentials = np.exp(eigenvalues)
    adjoint = eigenvectors.conj().T
    transformation = (eigenvectors * exponentials) @ adjoint

    # Build dG/dx for all real and imaginary antisymmetric coordinates.
    dz = np.zeros((parameter_count, modes, modes), complex)
    directions = np.arange(pairs)
    dz[directions, ij[0], ij[1]] = 1.0
    dz[directions, ij[1], ij[0]] = -1.0
    if not real_parameters:
        dz[pairs + directions, ij[0], ij[1]] = 1j
        dz[pairs + directions, ij[1], ij[0]] = -1j
    dgenerator = np.zeros(
        (parameter_count, 2 * modes, 2 * modes), dtype=complex
    )
    dgenerator[:, :modes, modes:] = dz.conj()
    dgenerator[:, modes:, :modes] = dz

    # L_exp(G,E)_ab=f[lambda_a,lambda_b] E_ab in the eigenbasis.
    left = eigenvalues[:, None]
    right = eigenvalues[None, :]
    denominator = left - right
    numerator = exponentials[:, None] - exponentials[None, :]
    divided_difference = np.empty_like(denominator)
    separated = np.abs(denominator) > 1e-12
    divided_difference[separated] = (
        numerator[separated] / denominator[separated]
    )
    divided_difference[~separated] = np.exp(
        0.5 * (left + right)[~separated]
    )
    derivative_eigenbasis = np.matmul(
        np.matmul(adjoint[None, :, :], dgenerator),
        eigenvectors[None, :, :],
    )
    derivative_eigenbasis *= divided_difference[None, :, :]
    derivatives = np.matmul(
        np.matmul(eigenvectors[None, :, :], derivative_eigenbasis),
        adjoint[None, :, :],
    )

    state = HFBState(
        transformation[:modes, :modes],
        transformation[modes:, :modes],
    )
    return (
        state,
        derivatives[:, :modes, :modes],
        derivatives[:, modes:, :modes],
    )


def hfb_energy_number_jacobian(
    x, hamiltonian, neutron_modes, *, real_parameters=False
):
    """Analytic energy gradient and particle-number constraint Jacobian.

    This differentiates the repository's Wick-contracted energy exactly through
    the exponential Bogoliubov chart.  The density variations are the standard
    HFB variations described in P. Ring and P. Schuck, *The Nuclear Many-Body
    Problem*, Springer (1980), Chs. 7-8.  The matrix-exponential derivative is
    evaluated by :func:`state_parameter_derivatives`.
    """
    modes = len(hamiltonian.h)
    neutron_mask = np.zeros(modes, bool)
    neutron_mask[np.asarray(neutron_modes, dtype=int)] = True
    state, du, dv = state_parameter_derivatives(
        x, modes, real_parameters=real_parameters
    )
    u, v = state.U, state.V
    drho = (
        np.matmul(dv.conj(), v.T)
        + np.matmul(v.conj()[None, :, :], dv.transpose(0, 2, 1))
    )
    dkappa = (
        np.matmul(dv.conj(), u.T)
        + np.matmul(v.conj()[None, :, :], du.transpose(0, 2, 1))
    )
    rho, kappa = state.rho, state.kappa

    one_body = np.einsum("ij,pji->p", hamiltonian.h, drho, optimize=True)
    normal_left = 0.5 * np.einsum(
        "ijkl,lj->ki", hamiltonian.v, rho, optimize=True
    )
    normal_right = 0.5 * np.einsum(
        "ijkl,ki->lj", hamiltonian.v, rho, optimize=True
    )
    normal = (
        np.einsum("ki,pki->p", normal_left, drho, optimize=True)
        + np.einsum("lj,plj->p", normal_right, drho, optimize=True)
    )
    pairing_left = 0.25 * np.einsum(
        "ijkl,kl->ij", hamiltonian.v, kappa, optimize=True
    )
    pairing_right = 0.25 * np.einsum(
        "ijkl,ij->kl", hamiltonian.v, kappa.conj(), optimize=True
    )
    pairing = (
        np.einsum("ij,pij->p", pairing_left, dkappa.conj(), optimize=True)
        + np.einsum("kl,pkl->p", pairing_right, dkappa, optimize=True)
    )
    energy_gradient_complex = one_body + normal + pairing
    if np.max(np.abs(energy_gradient_complex.imag), initial=0.0) > 2e-8:
        raise ValueError("Analytic HFB energy gradient is not real")
    energy_gradient = energy_gradient_complex.real

    diagonal = np.diagonal(drho, axis1=1, axis2=2).real
    number_jacobian = np.vstack((
        diagonal[:, neutron_mask].sum(axis=1),
        diagonal[:, ~neutron_mask].sum(axis=1),
    ))
    numbers = np.array([
        state.rho.diagonal().real[neutron_mask].sum(),
        state.rho.diagonal().real[~neutron_mask].sum(),
    ])
    return (
        hamiltonian.energy(state),
        numbers,
        energy_gradient,
        number_jacobian,
        state,
    )


def _antisymmetric_coordinates(matrix, *, real_parameters=False):
    """Pack an antisymmetric matrix into real optimizer coordinates."""
    matrix = np.asarray(matrix, complex)
    pairs = np.triu_indices(len(matrix), 1)
    upper = matrix[pairs]
    if real_parameters:
        return upper.real.copy()
    return np.concatenate((upper.real, upper.imag))


def _antisymmetric_matrix(coordinates, modes, *, real_parameters=False):
    """Unpack real coordinates into a complex antisymmetric matrix."""
    pairs = np.triu_indices(modes, 1)
    pair_count = len(pairs[0])
    coordinates = np.asarray(coordinates, float)
    expected = pair_count if real_parameters else 2 * pair_count
    if coordinates.shape != (expected,) or not np.isfinite(coordinates).all():
        raise ValueError(f"Expected {expected} finite Thouless coordinates")
    matrix = np.zeros((modes, modes), complex)
    matrix[pairs] = (
        coordinates
        if real_parameters
        else coordinates[:pair_count] + 1j * coordinates[pair_count:]
    )
    matrix -= matrix.T
    return matrix


def apply_thouless_step(state, coordinates, *, real_parameters=False):
    """Apply one canonical quasiparticle Thouless rotation to ``state``.

    The update is local to the current quasiparticle vacuum.  Consequently it
    needs only one Nambu-space exponential per optimization iteration, rather
    than one Frechet derivative for every global chart coordinate.
    """
    modes = len(state.U)
    z = _antisymmetric_matrix(
        coordinates, modes, real_parameters=real_parameters
    )
    zero = np.zeros_like(z)
    local = expm(np.block([[zero, z.conj()], [z, zero]]))
    transformation = np.block([
        [state.U, state.V.conj()],
        [state.V, state.U.conj()],
    ]) @ local
    return HFBState(
        transformation[:modes, :modes],
        transformation[modes:, :modes],
    )


def hfb_local_gradient(
    state, hamiltonian, neutron_modes, *, real_parameters=False
):
    """Return energy, numbers and their local Thouless derivatives.

    This is the ``H20`` gradient used by gradient-based HFB solvers.  The
    tangent relation ``dU=V* dZ, dV=U* dZ`` evaluates all independent
    directions with dense BLAS contractions and avoids differentiating a
    global matrix exponential.

    References
    ----------
    P. Ring and P. Schuck, *The Nuclear Many-Body Problem*, Springer (1980),
    Chs. 7-8.
    B. Bally et al., Eur. Phys. J. A 57, 69 (2021),
    doi:10.1140/epja/s10050-021-00369-z (TAURUS heavy-ball method).
    """
    modes = len(state.U)
    neutron_mask = np.zeros(modes, bool)
    neutron_mask[np.asarray(neutron_modes, dtype=int)] = True
    pair_rows, pair_columns = np.triu_indices(modes, 1)
    pair_count = len(pair_rows)
    parameter_count = pair_count if real_parameters else 2 * pair_count

    dz = np.zeros((parameter_count, modes, modes), complex)
    directions = np.arange(pair_count)
    dz[directions, pair_rows, pair_columns] = 1.0
    dz[directions, pair_columns, pair_rows] = -1.0
    if not real_parameters:
        dz[pair_count + directions, pair_rows, pair_columns] = 1j
        dz[pair_count + directions, pair_columns, pair_rows] = -1j

    du = np.matmul(state.V.conj()[None, :, :], dz)
    dv = np.matmul(state.U.conj()[None, :, :], dz)
    drho = (
        np.matmul(dv.conj(), state.V.T)
        + np.matmul(state.V.conj()[None, :, :], dv.transpose(0, 2, 1))
    )
    dkappa = (
        np.matmul(dv.conj(), state.U.T)
        + np.matmul(state.V.conj()[None, :, :], du.transpose(0, 2, 1))
    )
    rho, kappa = state.rho, state.kappa

    one_body = np.einsum("ij,pji->p", hamiltonian.h, drho, optimize=True)
    normal_left = 0.5 * np.einsum(
        "ijkl,lj->ki", hamiltonian.v, rho, optimize=True
    )
    normal_right = 0.5 * np.einsum(
        "ijkl,ki->lj", hamiltonian.v, rho, optimize=True
    )
    normal = (
        np.einsum("ki,pki->p", normal_left, drho, optimize=True)
        + np.einsum("lj,plj->p", normal_right, drho, optimize=True)
    )
    pairing_left = 0.25 * np.einsum(
        "ijkl,kl->ij", hamiltonian.v, kappa, optimize=True
    )
    pairing_right = 0.25 * np.einsum(
        "ijkl,ij->kl", hamiltonian.v, kappa.conj(), optimize=True
    )
    pairing = (
        np.einsum("ij,pij->p", pairing_left, dkappa.conj(), optimize=True)
        + np.einsum("kl,pkl->p", pairing_right, dkappa, optimize=True)
    )
    energy_gradient_complex = one_body + normal + pairing
    if np.max(np.abs(energy_gradient_complex.imag), initial=0.0) > 2e-8:
        raise ValueError("Local HFB energy gradient is not real")

    diagonal = np.diagonal(drho, axis1=1, axis2=2).real
    number_jacobian = np.vstack((
        diagonal[:, neutron_mask].sum(axis=1),
        diagonal[:, ~neutron_mask].sum(axis=1),
    ))
    occupations = rho.diagonal().real
    numbers = np.array([
        occupations[neutron_mask].sum(), occupations[~neutron_mask].sum()
    ])
    return (
        hamiltonian.energy(state),
        numbers,
        energy_gradient_complex.real,
        number_jacobian,
    )


def _hfb_local_number_jacobian(state, neutron_modes, *, real_parameters=False):
    """Evaluate only N, Z and their local gradients for constraint repair."""
    modes = len(state.U)
    neutron_mask = np.zeros(modes, bool)
    neutron_mask[np.asarray(neutron_modes, dtype=int)] = True
    rows, columns = np.triu_indices(modes, 1)
    pair_count = len(rows)
    parameter_count = pair_count if real_parameters else 2 * pair_count
    dz = np.zeros((parameter_count, modes, modes), complex)
    directions = np.arange(pair_count)
    dz[directions, rows, columns] = 1.0
    dz[directions, columns, rows] = -1.0
    if not real_parameters:
        dz[pair_count + directions, rows, columns] = 1j
        dz[pair_count + directions, columns, rows] = -1j
    dv = np.matmul(state.U.conj()[None, :, :], dz)
    drho = (
        np.matmul(dv.conj(), state.V.T)
        + np.matmul(state.V.conj()[None, :, :], dv.transpose(0, 2, 1))
    )
    diagonal = np.diagonal(drho, axis1=1, axis2=2).real
    jacobian = np.vstack((
        diagonal[:, neutron_mask].sum(axis=1),
        diagonal[:, ~neutron_mask].sum(axis=1),
    ))
    occupation = state.rho.diagonal().real
    values = np.array([
        occupation[neutron_mask].sum(), occupation[~neutron_mask].sum()
    ])
    return values, jacobian


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


def hartree_fock_energy_gradient(orbitals, hamiltonian):
    """Return the HF energy, orbital gradient and equivalent HFB state.

    ``orbitals`` contains orthonormal occupied one-body states.  The pairing
    tensor is therefore identically zero.  The Fock matrix is obtained by
    differentiating the same antisymmetrized Wick energy used by
    :class:`HFBHamiltonian`; see P. Ring and P. Schuck, *The Nuclear Many-Body
    Problem*, Springer (1980), Chs. 3 and 7.
    """
    orbitals = np.asarray(orbitals, complex)
    modes = len(hamiltonian.h)
    if (
        orbitals.ndim != 2
        or orbitals.shape[0] != modes
        or not np.isfinite(orbitals).all()
        or not np.allclose(
            orbitals.conj().T @ orbitals,
            np.eye(orbitals.shape[1]),
            atol=1e-10,
        )
    ):
        raise ValueError("HF orbitals must have finite orthonormal columns")
    state = HFBState.from_slater(orbitals)
    rho = state.rho

    # If dE=sum_ab G_ab d(rho_ab), the Hermitian Fock matrix satisfying
    # dE=Tr(F d rho) is F=G^T.  Keeping both Wick contractions explicit avoids
    # assuming additional interaction symmetries beyond those validated above.
    normal_left = 0.5 * np.einsum(
        "ijkl,lj->ki", hamiltonian.v, rho, optimize=True
    )
    normal_right = 0.5 * np.einsum(
        "ijkl,ki->lj", hamiltonian.v, rho, optimize=True
    )
    fock = hamiltonian.h + normal_left.T + normal_right.T
    fock = 0.5 * (fock + fock.conj().T)
    gradient = 2.0 * fock @ orbitals
    return hamiltonian.energy(state), gradient, fock, state


def solve_hartree_fock(
    hamiltonian,
    neutron_modes,
    targets,
    *,
    starts=8,
    seed=0,
    maxiter=500,
    tolerance=1e-8,
    initial_orbitals=None,
):
    """Optimize species-conserving HF determinants with multiple starts.

    Neutron and proton orbitals are optimized on separate complex Grassmann
    manifolds.  Thus ``kappa=0`` and integer N,Z are exact by construction,
    while the real parameter count is reduced from ``modes*(modes-1)`` in the
    unrestricted HFB chart to
    ``2*(N*(d_n-N) + Z*(d_p-Z))`` physical orbital-rotation coordinates.

    Every iteration uses the analytic Fock gradient, projects it onto the two
    Grassmann tangent spaces, and restores orthonormality with QR retraction.
    The first seed occupies the lowest one-body orbitals of each species;
    remaining starts use independent random neutron/proton configurations.
    This follows the manifold optimization framework of P.-A. Absil,
    R. Mahony and R. Sepulchre, *Optimization Algorithms on Matrix
    Manifolds*, Princeton University Press (2008), Chs. 3-4.
    """
    modes = len(hamiltonian.h)
    neutron = np.asarray(neutron_modes, dtype=int)
    if (
        neutron.ndim != 1
        or len(set(neutron)) != len(neutron)
        or np.any(neutron < 0)
        or np.any(neutron >= modes)
    ):
        raise ValueError("neutron_modes must contain distinct valid indices")
    neutron_mask = np.zeros(modes, bool)
    neutron_mask[neutron] = True
    proton = np.nonzero(~neutron_mask)[0]
    requested = np.asarray(targets, float)
    integer_targets = np.rint(requested).astype(int)
    capacities = np.array([len(neutron), len(proton)])
    if (
        requested.shape != (2,)
        or not np.isfinite(requested).all()
        or not np.allclose(requested, integer_targets, atol=1e-12)
        or np.any(integer_targets < 0)
        or np.any(integer_targets > capacities)
    ):
        raise ValueError("HF requires valid integer neutron/proton targets")
    if starts < 1 or maxiter < 1 or tolerance <= 0:
        raise ValueError("Positive starts, maxiter and tolerance required")
    neutron_number, proton_number = integer_targets
    particles = int(neutron_number + proton_number)

    def assemble(neutron_orbitals, proton_orbitals):
        combined = np.zeros((modes, particles), complex)
        combined[neutron, :neutron_number] = neutron_orbitals
        combined[proton, neutron_number:] = proton_orbitals
        return combined

    def split_and_orthonormalize(orbitals):
        supplied = np.asarray(orbitals, complex)
        if supplied.shape != (modes, particles):
            raise ValueError("initial_orbitals has the wrong shape")
        neutron_block = supplied[neutron, :neutron_number]
        proton_block = supplied[proton, neutron_number:]
        forbidden = supplied.copy()
        forbidden[neutron, :neutron_number] = 0
        forbidden[proton, neutron_number:] = 0
        if np.linalg.norm(forbidden) > 1e-10:
            raise ValueError("initial HF orbitals must not mix species")
        return (
            np.linalg.qr(neutron_block)[0][:, :neutron_number],
            np.linalg.qr(proton_block)[0][:, :proton_number],
        )

    def lowest_one_body_seed(indices, occupied):
        if occupied == 0:
            return np.empty((len(indices), 0), complex)
        block = hamiltonian.h[np.ix_(indices, indices)]
        _, vectors = np.linalg.eigh(block)
        return vectors[:, :occupied]

    rng = np.random.default_rng(seed)
    seeds = []
    if initial_orbitals is not None:
        supplied_array = np.asarray(initial_orbitals)
        if supplied_array.ndim == 2:
            supplied_seeds = [initial_orbitals]
        elif isinstance(initial_orbitals, (list, tuple)):
            supplied_seeds = list(initial_orbitals)
        else:
            raise ValueError(
                "initial_orbitals must be one matrix or a sequence of matrices"
            )
        seeds.extend(split_and_orthonormalize(item) for item in supplied_seeds)
    seeds.append((
        lowest_one_body_seed(neutron, neutron_number),
        lowest_one_body_seed(proton, proton_number),
    ))
    while len(seeds) < starts:
        random_neutron = (
            rng.normal(size=(len(neutron), neutron_number))
            + 1j * rng.normal(size=(len(neutron), neutron_number))
        )
        random_proton = (
            rng.normal(size=(len(proton), proton_number))
            + 1j * rng.normal(size=(len(proton), proton_number))
        )
        seeds.append((
            np.linalg.qr(random_neutron)[0][:, :neutron_number],
            np.linalg.qr(random_proton)[0][:, :proton_number],
        ))

    attempts, candidates = [], []
    # Below 1e-6 the dense Wick contractions and QR retractions can reach their
    # line-search resolution before the formal gradient target.  This floor is
    # still stringent on the nuclear energy scale and avoids labeling a stable
    # determinant unconverged solely because of roundoff-level backtracking.
    gradient_tolerance = max(tolerance, 1e-6)
    for neutron_orbitals, proton_orbitals in seeds[:starts]:
        message = "Iteration limit reached"
        previous_gradient = None
        previous_direction = None
        previous_norm_squared = None
        next_step = 0.25
        for iteration in range(maxiter):
            orbitals = assemble(neutron_orbitals, proton_orbitals)
            energy, gradient, _, state = hartree_fock_energy_gradient(
                orbitals, hamiltonian
            )
            gradient_neutron = gradient[neutron, :neutron_number]
            gradient_proton = gradient[proton, neutron_number:]
            tangent_neutron = gradient_neutron - neutron_orbitals @ (
                neutron_orbitals.conj().T @ gradient_neutron
            )
            tangent_proton = gradient_proton - proton_orbitals @ (
                proton_orbitals.conj().T @ gradient_proton
            )
            residual = float(np.sqrt(
                np.linalg.norm(tangent_neutron) ** 2
                + np.linalg.norm(tangent_proton) ** 2
            ))
            if residual <= gradient_tolerance:
                message = "Grassmann gradient converged"
                break
            if previous_gradient is None:
                direction_neutron = -tangent_neutron
                direction_proton = -tangent_proton
            else:
                # Projecting an old tangent vector onto the new tangent space
                # is a first-order vector transport for QR retraction.  The
                # non-negative Polak-Ribiere coefficient gives a Riemannian
                # nonlinear conjugate-gradient step and restarts automatically
                # when conjugacy is lost.
                old_gradient_neutron = previous_gradient[0] - neutron_orbitals @ (
                    neutron_orbitals.conj().T @ previous_gradient[0]
                )
                old_gradient_proton = previous_gradient[1] - proton_orbitals @ (
                    proton_orbitals.conj().T @ previous_gradient[1]
                )
                old_direction_neutron = previous_direction[0] - neutron_orbitals @ (
                    neutron_orbitals.conj().T @ previous_direction[0]
                )
                old_direction_proton = previous_direction[1] - proton_orbitals @ (
                    proton_orbitals.conj().T @ previous_direction[1]
                )
                numerator = float(np.real(
                    np.vdot(
                        tangent_neutron,
                        tangent_neutron - old_gradient_neutron,
                    )
                    + np.vdot(
                        tangent_proton,
                        tangent_proton - old_gradient_proton,
                    )
                ))
                beta = max(0.0, numerator / max(previous_norm_squared, 1e-30))
                direction_neutron = (
                    -tangent_neutron + beta * old_direction_neutron
                )
                direction_proton = (
                    -tangent_proton + beta * old_direction_proton
                )
            directional_derivative = float(np.real(
                np.vdot(tangent_neutron, direction_neutron)
                + np.vdot(tangent_proton, direction_proton)
            ))
            if directional_derivative >= -1e-12 * residual ** 2:
                direction_neutron = -tangent_neutron
                direction_proton = -tangent_proton
                directional_derivative = -(residual ** 2)
            step = min(1.0, next_step)
            accepted = False
            for _ in range(30):
                trial_neutron = np.linalg.qr(
                    neutron_orbitals + step * direction_neutron
                )[0][:, :neutron_number]
                trial_proton = np.linalg.qr(
                    proton_orbitals + step * direction_proton
                )[0][:, :proton_number]
                trial_orbitals = assemble(trial_neutron, trial_proton)
                trial_energy = hamiltonian.energy(
                    HFBState.from_slater(trial_orbitals)
                )
                if trial_energy <= energy + 1e-4 * step * directional_derivative:
                    previous_gradient = (
                        tangent_neutron.copy(),
                        tangent_proton.copy(),
                    )
                    previous_direction = (
                        direction_neutron.copy(),
                        direction_proton.copy(),
                    )
                    previous_norm_squared = residual ** 2
                    neutron_orbitals, proton_orbitals = (
                        trial_neutron,
                        trial_proton,
                    )
                    next_step = min(1.0, 1.5 * step)
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                message = "Line search failed"
                break

        orbitals = assemble(neutron_orbitals, proton_orbitals)
        energy, gradient, fock, state = hartree_fock_energy_gradient(
            orbitals, hamiltonian
        )
        gradient_neutron = gradient[neutron, :neutron_number]
        gradient_proton = gradient[proton, neutron_number:]
        tangent_neutron = gradient_neutron - neutron_orbitals @ (
            neutron_orbitals.conj().T @ gradient_neutron
        )
        tangent_proton = gradient_proton - proton_orbitals @ (
            proton_orbitals.conj().T @ gradient_proton
        )
        residual = float(np.sqrt(
            np.linalg.norm(tangent_neutron) ** 2
            + np.linalg.norm(tangent_proton) ** 2
        ))
        converged = bool(residual <= gradient_tolerance)
        attempts.append({
            "converged": converged,
            "energy": energy,
            "number_error": 0.0,
            "gradient_norm": residual,
            "iterations": iteration + 1,
            "message": message,
        })
        candidates.append((converged, energy, residual, orbitals, state, fock))

    converged, energy, residual, orbitals, state, fock = min(
        candidates, key=lambda item: (not item[0], item[1], item[2])
    )
    neutron_fock = orbitals[:, :neutron_number].conj().T @ fock @ orbitals[:, :neutron_number]
    proton_fock = orbitals[:, neutron_number:].conj().T @ fock @ orbitals[:, neutron_number:]
    chemical_potentials = np.array([
        float(np.trace(neutron_fock).real / neutron_number)
        if neutron_number else np.nan,
        float(np.trace(proton_fock).real / proton_number)
        if proton_number else np.nan,
    ])
    return HFBResult(
        state=state,
        energy=energy,
        numbers=integer_targets.astype(float),
        converged=converged,
        message=next(
            item["message"]
            for item in attempts
            if item["energy"] == energy
        ),
        parameters=orbitals,
        attempts=attempts,
        chemical_potentials=chemical_potentials,
        stationarity_error=residual,
    )


def _solve_hfb_slsqp_legacy(
    hamiltonian,
    neutron_modes,
    targets,
    *,
    starts=3,
    seed=0,
    maxiter=300,
    tolerance=1e-8,
    initial_parameters=None,
    stationarity_diagnostics=True,
    shared_finite_difference_jacobian=False,
    analytic_jacobian=False,
    real_bogoliubov=False,
    number_penalty_max=1000.0,
    method="hfb",
    initial_orbitals=None,
):
    """Minimize E subject to <N>=targets[0], <Z>=targets[1].

    With ``analytic_jacobian=True``, an analytic-gradient quadratic-penalty
    continuation first brings a trial point onto the physically relevant
    number surface, followed by an exactly constrained SLSQP refinement.  The
    continuation avoids rank-deficient or poor stationary branches that SLSQP
    can encounter when started from a generic Bogoliubov vacuum.  This is the
    standard quadratic-penalty strategy of J. Nocedal and S. J. Wright,
    *Numerical Optimization*, 2nd ed., Springer (2006), Sec. 17.1.

    Set ``real_bogoliubov=True`` to restrict the antisymmetric generator and
    hence U, V, rho and kappa to real values.  This reduces the number of real
    optimization coordinates from ``modes*(modes-1)`` to
    ``modes*(modes-1)/2``.  The restriction is appropriate for real,
    time-reversal-compatible solutions but can exclude lower complex minima.

    The final equality-constrained refinement implements the particle-number
    Lagrange-multiplier problem without retaining a finite penalty.  Dense
    matrix exponentials still make this a reference solver rather than a
    large-scale production nuclear solver.
    Multiple paired starts reduce trapping; global optimality is not guaranteed.
    """
    if method not in ("hfb", "hf"):
        raise ValueError("method must be 'hfb' or 'hf'")
    if method == "hf":
        if real_bogoliubov:
            raise ValueError(
                "real_bogoliubov applies to method='hfb'; HF has its own "
                "orbital parameterization"
            )
        if initial_parameters is not None:
            raise ValueError("HF mode uses initial_orbitals, not initial_parameters")
        return solve_hartree_fock(
            hamiltonian,
            neutron_modes,
            targets,
            starts=starts,
            seed=seed,
            maxiter=maxiter,
            tolerance=tolerance,
            initial_orbitals=initial_orbitals,
        )
    if initial_orbitals is not None:
        raise ValueError("initial_orbitals is only valid with method='hf'")

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
    if not np.isfinite(number_penalty_max) or number_penalty_max < 1:
        raise ValueError("number_penalty_max must be finite and at least one")
    if analytic_jacobian and shared_finite_difference_jacobian:
        raise ValueError("Select either analytic or shared finite-difference Jacobian")

    def numbers(state):
        """Compute <N> and <Z> by summing diagonal occupations."""
        occupation = state.rho.diagonal().real
        return np.array([occupation[mask].sum(), occupation[~mask].sum()])

    analytic_cache = {"x": None, "values": None}

    def analytic_values(x):
        x = np.asarray(x, float)
        if (
            analytic_cache["x"] is None
            or not np.array_equal(x, analytic_cache["x"])
        ):
            analytic_cache["x"] = x.copy()
            analytic_cache["values"] = hfb_energy_number_jacobian(
                x,
                hamiltonian,
                indices,
                real_parameters=real_bogoliubov,
            )
        return analytic_cache["values"]

    def constraint(x):
        """Return equality-constraint residuals [<N>-N0, <Z>-Z0]."""
        if analytic_jacobian:
            return analytic_values(x)[1] - targets
        return numbers(
            state_from_parameters(
                x, m, real_parameters=real_bogoliubov
            )
        ) - targets

    def objective(x):
        """Map optimizer coordinates to the physical HFB energy."""
        if analytic_jacobian:
            return analytic_values(x)[0]
        return hamiltonian.energy(
            state_from_parameters(
                x, m, real_parameters=real_bogoliubov
            )
        )

    derivative_cache = {"x": None, "energy": None, "constraints": None}

    def joint_derivatives(x):
        """Finite-difference E, N and Z together at identical trial points."""
        x = np.asarray(x, float)
        if (
            derivative_cache["x"] is not None
            and np.array_equal(x, derivative_cache["x"])
        ):
            return derivative_cache["energy"], derivative_cache["constraints"]
        base_state = state_from_parameters(
            x, m, real_parameters=real_bogoliubov
        )
        base_energy = hamiltonian.energy(base_state)
        base_numbers = numbers(base_state)
        energy_gradient = np.empty(len(x), float)
        constraint_jacobian = np.empty((2, len(x)), float)
        steps = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x))
        for index, step_size in enumerate(steps):
            shifted = x.copy()
            shifted[index] += step_size
            shifted_state = state_from_parameters(
                shifted, m, real_parameters=real_bogoliubov
            )
            energy_gradient[index] = (
                hamiltonian.energy(shifted_state) - base_energy
            ) / step_size
            constraint_jacobian[:, index] = (
                numbers(shifted_state) - base_numbers
            ) / step_size
        derivative_cache.update({
            "x": x.copy(),
            "energy": energy_gradient,
            "constraints": constraint_jacobian,
        })
        return energy_gradient, constraint_jacobian

    def finite_difference_objective_jacobian(x):
        return joint_derivatives(x)[0]

    def finite_difference_number_jacobian(x):
        return joint_derivatives(x)[1]

    def analytic_objective_jacobian(x):
        return analytic_values(x)[2]

    def analytic_number_jacobian(x):
        return analytic_values(x)[3]

    # Run several randomized starts because the constrained HFB landscape is
    # non-convex.  A caller-supplied initial point is used for the first start.
    rng = np.random.default_rng(seed)
    attempts, candidates = [], []
    parameter_count = (
        m * (m - 1) // 2 if real_bogoliubov else m * (m - 1)
    )
    for attempt in range(starts):
        x = (
            np.array(initial_parameters, float)
            if attempt == 0 and initial_parameters is not None
            else rng.normal(
                scale=0.5 / np.sqrt(m), size=parameter_count
            )
        )
        preconditioner_evaluations = 0
        if analytic_jacobian:
            # Exact gradients make this continuation far cheaper than even a
            # single finite-difference SLSQP iteration in a large model space.
            # Its role is only to select a good constrained basin; SLSQP below
            # still enforces the particle numbers to the requested tolerance.
            penalty_iterations = max(5, min(100, maxiter // 4))
            penalty_schedule = []
            penalty = 1.0
            while penalty < number_penalty_max:
                penalty_schedule.append(penalty)
                penalty *= 10.0
            penalty_schedule.append(float(number_penalty_max))
            for penalty in penalty_schedule:
                def penalized_value_and_gradient(point):
                    energy, particle_numbers, gradient, jacobian, _ = (
                        analytic_values(point)
                    )
                    residual = particle_numbers - targets
                    return (
                        energy + penalty * residual @ residual,
                        gradient + 2.0 * penalty * jacobian.T @ residual,
                    )

                prefit = minimize(
                    penalized_value_and_gradient,
                    x,
                    method="L-BFGS-B",
                    jac=True,
                    options={
                        "maxiter": penalty_iterations,
                        "ftol": max(tolerance, 1e-12),
                    },
                )
                x = prefit.x
                preconditioner_evaluations += int(prefit.nfev)
        constraint_specification = {"type": "eq", "fun": constraint}
        if analytic_jacobian:
            constraint_specification["jac"] = analytic_number_jacobian
            scipy_jacobian = analytic_objective_jacobian
        elif shared_finite_difference_jacobian:
            constraint_specification["jac"] = finite_difference_number_jacobian
            scipy_jacobian = finite_difference_objective_jacobian
        else:
            scipy_jacobian = None
        fit = minimize(
            objective,
            x,
            method="SLSQP",
            jac=scipy_jacobian,
            constraints=constraint_specification,
            options={"maxiter": maxiter, "ftol": tolerance},
        )

        # Recompute the physical state and number residual from the returned
        # parameters rather than relying only on the optimizer success flag.
        state = state_from_parameters(
            fit.x, m, real_parameters=real_bogoliubov
        )
        residual = float(np.max(np.abs(constraint(fit.x))))
        ok = bool(fit.success and residual < max(1e-7, 10 * tolerance))
        attempts.append(
            {
                "converged": ok,
                "energy": float(fit.fun),
                "number_error": residual,
                "message": str(fit.message),
                "iterations": int(fit.nit),
                "function_evaluations": int(fit.nfev),
                "jacobian_evaluations": int(getattr(fit, "njev", 0)),
                "preconditioner_function_evaluations": preconditioner_evaluations,
                "real_bogoliubov": bool(real_bogoliubov),
                "parameter_count": int(parameter_count),
                "maximum_number_penalty": float(number_penalty_max),
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

    if not stationarity_diagnostics:
        # Large model spaces can spend more time on the 2*p central-difference
        # post-fit audit than on a bounded pilot optimization. Skipping this
        # audit does not change the SLSQP solution or its feasibility test.
        return HFBResult(
            state=state,
            energy=float(fit.fun),
            numbers=numbers(state),
            converged=ok,
            message=str(fit.message),
            parameters=fit.x,
            attempts=attempts,
            chemical_potentials=np.full(2, np.nan),
            stationarity_error=np.nan,
        )

    # Recover multipliers in grad(E) = lambda_n grad(N) + lambda_p grad(Z).
    # Central finite differences are adequate here because this is a compact
    # reference implementation and the optimizer itself is finite-difference
    # based.  Each row of ``jacobian`` is the gradient of one constraint.
    if analytic_jacobian:
        _, _, gradient, jacobian, _ = analytic_values(fit.x)
    else:
        step = 1e-5
        directions = np.eye(len(fit.x)) * step
        gradient = np.array(
            [
                (objective(fit.x + d) - objective(fit.x - d)) / (2 * step)
                for d in directions
            ]
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
    stationarity_diagnostics=True,
    shared_finite_difference_jacobian=False,
    analytic_jacobian=False,
    real_bogoliubov=False,
    number_penalty_max=None,
    method="hfb",
    initial_orbitals=None,
    step_size=0.05,
    momentum=0.3,
    constraint_iterations=12,
    gradient_tolerance=None,
):
    """Optimize an HF or HFB vacuum with structure-preserving gradients.

    HFB uses the TAURUS strategy: compute the local quasiparticle ``H20``
    gradient, determine the neutron/proton Lagrange multipliers from their
    two-by-two Gram system, take a heavy-ball Thouless step, and correct the
    number constraints locally.  It never builds the derivative of a global
    matrix exponential and does not use a quadratic particle-number penalty.

    ``method='hf'`` fixes the anomalous density to zero and uses the analogous
    analytic-gradient manifold solver for occupied orbitals.  Thus the public
    entry point, multistart behavior and result diagnostics are common to both
    approximations; only their physical manifolds differ.

    ``analytic_jacobian``, ``shared_finite_difference_jacobian`` and
    ``number_penalty_max`` are accepted as compatibility no-ops for older
    scripts.  The new HFB path always uses the local analytic ``H20`` gradient.

    References
    ----------
    B. Bally et al., Eur. Phys. J. A 57, 69 (2021),
    doi:10.1140/epja/s10050-021-00369-z.
    P. Ring and P. Schuck, *The Nuclear Many-Body Problem*, Springer (1980),
    Chs. 7-8.
    """
    if method not in ("hfb", "hf"):
        raise ValueError("method must be 'hfb' or 'hf'")
    if method == "hf":
        if initial_parameters is not None:
            raise ValueError("HF mode uses initial_orbitals, not initial_parameters")
        return solve_hartree_fock(
            hamiltonian,
            neutron_modes,
            targets,
            starts=starts,
            seed=seed,
            maxiter=maxiter,
            tolerance=tolerance,
            initial_orbitals=initial_orbitals,
        )
    if initial_orbitals is not None:
        raise ValueError("initial_orbitals is only valid with method='hf'")

    modes = len(hamiltonian.h)
    neutron = np.asarray(neutron_modes, dtype=int)
    if gradient_tolerance is not None and (
        not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0
    ):
        raise ValueError("gradient_tolerance must be finite and positive")
    if (
        neutron.ndim != 1
        or len(set(neutron)) != len(neutron)
        or np.any(neutron < 0)
        or np.any(neutron >= modes)
    ):
        raise ValueError("neutron_modes must contain distinct valid indices")
    neutron_mask = np.zeros(modes, bool)
    neutron_mask[neutron] = True
    proton = np.nonzero(~neutron_mask)[0]
    requested = np.asarray(targets, float)
    capacities = np.array([len(neutron), len(proton)])
    if (
        requested.shape != (2,)
        or not np.isfinite(requested).all()
        or np.any(requested < 0)
        or np.any(requested > capacities)
        or np.any(capacities == 0)
    ):
        raise ValueError("Invalid neutron/proton targets")
    if starts < 1 or maxiter < 1 or tolerance <= 0:
        raise ValueError("Positive starts, maxiter and tolerance required")
    if (
        not np.isfinite(step_size)
        or step_size <= 0
        or not np.isfinite(momentum)
        or momentum < 0
        or momentum >= 1
        or constraint_iterations < 1
    ):
        raise ValueError("Invalid heavy-ball solver controls")

    parameter_count = (
        modes * (modes - 1) // 2
        if real_bogoliubov
        else modes * (modes - 1)
    )
    number_tolerance = max(1e-8, 10 * tolerance)
    stationarity_tolerance = (
        1e-3 if gradient_tolerance is None else float(gradient_tolerance)
    )

    def paired_seed():
        """Construct a real BCS seed with the requested mean N and Z."""
        z = np.zeros((modes, modes), complex)
        for indices, target in ((neutron, requested[0]), (proton, requested[1])):
            if len(indices) % 2:
                return None
            angle = np.arcsin(np.sqrt(float(target) / len(indices)))
            for left, right in zip(indices[::2], indices[1::2]):
                z[left, right] = angle
                z[right, left] = -angle
        return apply_thouless_step(
            HFBState(np.eye(modes, dtype=complex), np.zeros((modes, modes), complex)),
            _antisymmetric_coordinates(z, real_parameters=real_bogoliubov),
            real_parameters=real_bogoliubov,
        )

    def hartree_fock_seed():
        """Use the lowest species orbitals as a deterministic HFB boundary."""
        rounded = np.rint(requested).astype(int)
        if not np.allclose(requested, rounded, atol=1e-12):
            return None
        occupied = []
        for indices, count in ((neutron, rounded[0]), (proton, rounded[1])):
            block = hamiltonian.h[np.ix_(indices, indices)]
            _, vectors = np.linalg.eigh(block)
            embedded = np.zeros((modes, count), complex)
            if count:
                embedded[indices, :] = vectors[:, :count]
            occupied.append(embedded)
        return HFBState.from_slater(np.hstack(occupied))

    def correct_numbers(state):
        """Apply minimum-norm local Newton corrections to N and Z."""
        corrections = 0
        for corrections in range(constraint_iterations):
            values, jacobian = _hfb_local_number_jacobian(
                state,
                neutron,
                real_parameters=real_bogoliubov,
            )
            residual = requested - values
            if np.max(np.abs(residual)) <= number_tolerance:
                return state, corrections, True
            correction = np.linalg.lstsq(jacobian, residual, rcond=1e-10)[0]
            correction_norm = np.linalg.norm(correction)
            if not np.isfinite(correction_norm) or correction_norm < 1e-14:
                break
            if correction_norm > 0.3:
                correction *= 0.3 / correction_norm
            state = apply_thouless_step(
                state, correction, real_parameters=real_bogoliubov
            )
        values, _ = _hfb_local_number_jacobian(
            state,
            neutron,
            real_parameters=real_bogoliubov,
        )
        return (
            state,
            corrections + 1,
            bool(np.max(np.abs(values - requested)) <= number_tolerance),
        )

    rng = np.random.default_rng(seed)
    seeds = []
    if initial_parameters is not None:
        supplied = np.asarray(initial_parameters, float)
        supplied = supplied[None, :] if supplied.ndim == 1 else supplied
        if supplied.ndim != 2 or supplied.shape[1] != parameter_count:
            raise ValueError("initial_parameters has the wrong shape")
        seeds.extend(
            state_from_parameters(
                coordinates, modes, real_parameters=real_bogoliubov
            )
            for coordinates in supplied
        )
    hf_base = hartree_fock_seed()
    if hf_base is not None:
        seeds.append(hf_base)
    paired_base = paired_seed()
    base = paired_base if paired_base is not None else hf_base
    if base is None:
        base = state_from_parameters(
            rng.normal(scale=0.15, size=parameter_count),
            modes,
            real_parameters=real_bogoliubov,
        )
        seeds.append(base)
    while len(seeds) < starts:
        perturbation = rng.normal(scale=0.08, size=parameter_count)
        seeds.append(apply_thouless_step(
            base, perturbation, real_parameters=real_bogoliubov
        ))

    attempts = []
    candidates = []
    for start_index, state in enumerate(seeds[:starts]):
        state, initial_corrections, _ = correct_numbers(state)
        velocity = np.zeros(parameter_count, float)
        accumulated = np.zeros(parameter_count, float)
        eta = float(step_size)
        message = "Iteration limit reached"
        total_constraint_corrections = initial_corrections

        for iteration in range(maxiter):
            energy, values, gradient, jacobian = hfb_local_gradient(
                state,
                hamiltonian,
                neutron,
                real_parameters=real_bogoliubov,
            )
            multipliers = np.linalg.lstsq(
                jacobian.T, gradient, rcond=1e-10
            )[0]
            constrained_gradient = gradient - jacobian.T @ multipliers
            stationarity = float(np.linalg.norm(constrained_gradient))
            number_error = float(np.max(np.abs(values - requested)))
            if (
                stationarity <= stationarity_tolerance
                and number_error <= number_tolerance
            ):
                message = "Constrained H20 gradient converged"
                break

            proposed_velocity = (
                momentum * velocity - eta * constrained_gradient
            )
            proposed_norm = np.linalg.norm(proposed_velocity)
            if proposed_norm > 0.25:
                proposed_velocity *= 0.25 / proposed_norm
            directional_derivative = float(
                constrained_gradient @ proposed_velocity
            )
            if directional_derivative >= 0:
                proposed_velocity = -eta * constrained_gradient
                proposed_norm = np.linalg.norm(proposed_velocity)
                if proposed_norm > 0.25:
                    proposed_velocity *= 0.25 / proposed_norm
                directional_derivative = float(
                    constrained_gradient @ proposed_velocity
                )

            accepted = False
            trial_step = proposed_velocity.copy()
            for _ in range(14):
                trial = apply_thouless_step(
                    state, trial_step, real_parameters=real_bogoliubov
                )
                trial, corrections, feasible = correct_numbers(trial)
                total_constraint_corrections += corrections
                trial_energy = hamiltonian.energy(trial)
                if feasible and trial_energy <= (
                    energy + 1e-4 * directional_derivative
                ):
                    state = trial
                    velocity = trial_step
                    accumulated += trial_step
                    eta = min(0.2, eta * 1.1)
                    accepted = True
                    break
                trial_step *= 0.5
                directional_derivative *= 0.5
            if not accepted:
                velocity.fill(0.0)
                eta *= 0.25
                if eta < 1e-10:
                    message = "Heavy-ball line search stalled"
                    break

        energy, values, gradient, jacobian = hfb_local_gradient(
            state,
            hamiltonian,
            neutron,
            real_parameters=real_bogoliubov,
        )
        multipliers = np.linalg.lstsq(
            jacobian.T, gradient, rcond=1e-10
        )[0]
        constrained_gradient = gradient - jacobian.T @ multipliers
        stationarity = float(np.linalg.norm(constrained_gradient))
        number_error = float(np.max(np.abs(values - requested)))
        converged = bool(
            stationarity <= stationarity_tolerance
            and number_error <= number_tolerance
        )
        if not stationarity_diagnostics:
            reported_stationarity = np.nan
            reported_multipliers = np.full(2, np.nan)
        else:
            reported_stationarity = stationarity
            reported_multipliers = multipliers
        attempt = {
            "converged": converged,
            "energy": float(energy),
            "number_error": number_error,
            "gradient_norm": stationarity,
            "gradient_tolerance": stationarity_tolerance,
            "iterations": int(iteration + 1),
            "message": message,
            "solver": "TAURUS-style constrained H20 heavy ball",
            "step_size": eta,
            "momentum": float(momentum),
            "constraint_corrections": int(total_constraint_corrections),
            "real_bogoliubov": bool(real_bogoliubov),
            "parameter_count": int(parameter_count),
        }
        attempts.append(attempt)
        candidates.append((
            converged,
            energy,
            number_error,
            stationarity,
            state,
            accumulated,
            reported_multipliers,
            reported_stationarity,
            message,
        ))

    # Return the lowest-energy number-feasible state even when a stricter
    # stationarity threshold was not reached at the iteration cap. This keeps
    # the best variational result while preserving ``converged=False`` and its
    # residual diagnostics, instead of preferring a higher stationary basin.
    feasible = [
        candidate for candidate in candidates
        if candidate[2] <= number_tolerance
    ]
    best = (
        min(feasible, key=lambda item: item[1])
        if feasible
        else min(candidates, key=lambda item: (item[2], item[3], item[1]))
    )
    (
        converged,
        energy,
        _,
        _,
        state,
        parameters,
        multipliers,
        stationarity,
        message,
    ) = best
    _, values, _, _ = hfb_local_gradient(
        state,
        hamiltonian,
        neutron,
        real_parameters=real_bogoliubov,
    )
    return HFBResult(
        state=state,
        energy=float(energy),
        numbers=values,
        converged=converged,
        message=message,
        parameters=parameters,
        attempts=attempts,
        chemical_potentials=multipliers,
        stationarity_error=stationarity,
    )
