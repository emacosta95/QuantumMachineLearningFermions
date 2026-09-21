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
    U: np.ndarray
    V: np.ndarray

    @property
    def rho(self):
        # from <c_j^dag c_i> = sum_k V_ik V_jk^* = (V V^dag)_ij
        return self.V.conj() @ self.V.T

    @property
    def kappa(self):
        # from <c_j c_i> = sum_k V_ik U_jk^* = (V U^dag)_ij
        return self.V.conj() @ self.U.T

    def canonical_error(self):
        m = len(self.U)
        return max(
            np.linalg.norm(
                self.U.conj().T @ self.U + self.V.conj().T @ self.V - np.eye(m)
            ),
            np.linalg.norm(self.U.T @ self.V + self.V.T @ self.U),
        )


def state_from_parameters(x, modes):
    """Exponentiate an arbitrary complex antisymmetric pairing generator.

    Unlike a vacuum Thouless inverse, this representation permits singular U.
    All mode pairs, including neutron-proton pairs, are parameterized.
    This is used for the unconstrained optimization of the HFB energy with respect to independent parameters.
    """
    # upper triangular indices of an m x m matrix, excluding the diagonal since k=1
    ij = np.triu_indices(modes, 1)
    p = len(ij[0])
    x = np.asarray(x, dtype=float)
    if x.shape != (2 * p,) or not np.isfinite(x).all():
        raise ValueError("Expected modes*(modes-1) finite real parameters")
    z = np.zeros((modes, modes), complex)
    z[ij] = x[:p] + 1j * x[p:]
    z -= z.T
    zero = np.zeros_like(z)
    # this w has the structure w=exp(A) with A generator of the Bogoliubov transformation, and is unitary
    w = expm(np.block([[zero, z.conj()], [z, zero]]))
    return HFBState(w[:modes, :modes], w[modes:, :modes])


class HFBHamiltonian:
    def __init__(self, h, interaction):
        self.h = np.array(h, dtype=complex, copy=True)
        if self.h.ndim != 2 or self.h.shape[0] != self.h.shape[1]:
            raise ValueError("h must be square")
        m = len(self.h)
        self.v = np.zeros((m,) * 4, complex)
        if isinstance(interaction, dict):
            for indices, value in interaction.items():
                if len(indices) != 4 or any(i < 0 or i >= m for i in indices):
                    raise ValueError("Interaction index out of range")
                self.v[indices] = value
        else:
            self.v = np.array(interaction, dtype=complex, copy=True)
        if self.v.shape != (m,) * 4:
            raise ValueError("Interaction must have shape (m,m,m,m)")
        if not np.isfinite(self.h).all() or not np.isfinite(self.v).all():
            raise ValueError("Hamiltonian must be finite")
        for a, b in [
            (self.h, self.h.conj().T),
            (self.v, -self.v.swapaxes(0, 1)),
            (self.v, -self.v.swapaxes(2, 3)),
            (self.v, self.v.transpose(2, 3, 0, 1).conj()),
        ]:
            if not np.allclose(a, b, atol=1e-10, rtol=1e-10):
                raise ValueError("Hamiltonian violates Hermiticity or antisymmetry")

    def energy(self, state):
        r, k = state.rho, state.kappa
        e = (
            np.einsum("ij,ji->", self.h, r)
            + 0.5 * np.einsum("ijkl,ki,lj->", self.v, r, r)
            + 0.25 * np.einsum("ijkl,ij,kl->", self.v, k.conj(), k)
        )
        if abs(e.imag) > 1e-8:
            raise ValueError("Non-real HFB energy")
        return float(e.real)


@dataclass
class HFBResult:
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
        occupation = state.rho.diagonal().real
        return np.array([occupation[mask].sum(), occupation[~mask].sum()])

    def constraint(x):
        return numbers(state_from_parameters(x, m)) - targets

    def objective(x):
        return hamiltonian.energy(state_from_parameters(x, m))

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
    feasible = [c for c in candidates if c[0]]
    best = (
        min(feasible, key=lambda c: c[2].fun)
        if feasible
        else min(candidates, key=lambda c: c[1])
    )
    ok, _, fit, state = best
    # Recover multipliers in grad(E) = lambda_n grad(N) + lambda_p grad(Z).
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
    stationarity = float(np.linalg.norm(gradient - jacobian.T @ multipliers))
    # At rank-deficient constraints (e.g. an HF limit), chemical potentials
    # are not uniquely determined by these first derivatives.
    if np.linalg.matrix_rank(jacobian, tol=1e-7) < 2:
        multipliers[:] = np.nan
    ok = bool(ok and stationarity < max(1e-5, 10 * np.sqrt(tolerance)))
    return HFBResult(
        state,
        float(fit.fun),
        numbers(state),
        ok,
        str(fit.message),
        fit.x,
        attempts,
        multipliers,
        stationarity,
    )
