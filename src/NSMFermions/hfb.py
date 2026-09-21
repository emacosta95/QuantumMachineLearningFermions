"""Complex, species-unrestricted HFB reference implementation (NumPy/SciPy).

H = h_ij c_i^dag c_j + v_ijkl c_i^dag c_j^dag c_l c_k / 4.
rho_ij = <c_j^dag c_i>, kappa_ij = <c_j c_i>.
No species, reality, or time-reversal blocks are imposed. The vacuum-based
solver currently covers even total number parity only, not blocked odd states.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


@dataclass
class HFBState:
    """Bogoliubov amplitudes defining an even-parity quasiparticle vacuum.

    The convention used throughout this module is

        beta = U^dagger c + V^dagger c^dagger.

    Both arrays have shape ``(modes, modes)``.  They are not separately
    unitary; together they form the canonical transformation in Nambu space.
    """

    U: np.ndarray
    V: np.ndarray

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
