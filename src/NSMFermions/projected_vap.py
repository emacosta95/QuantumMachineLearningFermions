"""Variation after particle-number and J=0 projection.

The optimizer varies a finite Thouless matrix and constructs the configured
``BogoliubovVacuumSeries`` at every objective evaluation.  Consequently the
same gauge/Euler transformations and weights used during variation are
retained for final energy and fidelity analysis.

Two energy backends are intentionally distinct:

``kernel``
    Uses generalized-Wick transition kernels and is polynomial in the
    one-body dimension.  A guaranteed-exact number grid is required by
    default because an undersized Fourier sum leaves other particle sectors
    in the VAP norm.

``fixed_sector_basis``
    Expands the same series directly in an existing fixed-(N,Z) determinant
    basis.  This is suitable for small CKI diagnostics and makes a 1x1 number
    grid harmless: selecting the determinant basis itself performs the exact
    number-sector restriction.  Its cost is combinatorial and it is not the
    scalable production backend.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize

if __package__:
    from .angular_momentum import ParticleNumberJ0ProjectedEnergy, HFBState
    from .number_projection import projected_series_observables
else:
    # Import the state class held by the projector module itself.  This avoids
    # class-identity mismatches in environments that load ``hfb.py`` through an
    # explicit module specification for isolated tests.
    from angular_momentum import ParticleNumberJ0ProjectedEnergy, HFBState
    from number_projection import projected_series_observables


@dataclass
class ProjectedVAPResult:
    """Best projected-HFB stationary point found by a multi-start search."""

    state: object
    projected_series: object
    projected_energy: float
    parameters: np.ndarray
    initial_parameters: np.ndarray
    converged: bool
    gradient_norm: float
    attempts: list
    energy_backend: str
    kappa_norm: float
    number_grid_guaranteed_exact: bool
    euler_grid_guaranteed_exact: bool


class ProjectedVAPObjective:
    """Projected energy of a finite-Thouless Bogoliubov vacuum."""

    _BACKENDS = {"kernel", "fixed_sector_basis"}

    def __init__(
        self,
        projector,
        *,
        energy_backend="kernel",
        fermionic_hamiltonian=None,
        allow_aliased_number_kernel=False,
    ):
        if not isinstance(projector, ParticleNumberJ0ProjectedEnergy):
            raise TypeError("projector must be ParticleNumberJ0ProjectedEnergy")
        if energy_backend not in self._BACKENDS:
            raise ValueError(
                "energy_backend must be 'kernel' or 'fixed_sector_basis'"
            )
        if energy_backend == "fixed_sector_basis" and fermionic_hamiltonian is None:
            raise ValueError(
                "fixed_sector_basis requires fermionic_hamiltonian"
            )
        if (
            energy_backend == "kernel"
            and not projector.number_grid_guaranteed_exact
            and not allow_aliased_number_kernel
        ):
            raise ValueError(
                "Kernel VAP requires an exact number grid because an undersized "
                "grid leaves aliased particle sectors in the projected norm. "
                "Use fixed_sector_basis for a small-space fixed-(N,Z) study, "
                "or set allow_aliased_number_kernel=True to accept that the "
                "objective is not a pure fixed-(N,Z) energy."
            )
        if energy_backend == "kernel" and not projector.number_grid_guaranteed_exact:
            warnings.warn(
                "Kernel VAP is using an aliased particle-number sum; the result "
                "is not guaranteed to be a fixed-(N,Z) variational energy.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.projector = projector
        self.energy_backend = energy_backend
        self.fermionic_hamiltonian = fermionic_hamiltonian
        self.modes = len(projector.ham.h)
        self.ij = np.triu_indices(self.modes, 1)

    def unpack(self, parameters):
        return self.projector.unpack(parameters)

    def pack(self, state):
        if not isinstance(state, HFBState):
            raise TypeError("initial_state must be an HFBState")
        z = state.thouless_matrix
        entries = z[self.ij]
        return np.concatenate((entries.real, entries.imag))

    def state(self, parameters):
        return HFBState.from_thouless(self.unpack(parameters))

    def series(self, parameters):
        return self.projector.projected_series(self.state(parameters))

    def series_energy(self, series):
        if self.energy_backend == "kernel":
            return self.projector.series_energy(series)
        return projected_series_observables(
            series, self.fermionic_hamiltonian
        ).energy

    def energy(self, parameters):
        return self.series_energy(self.series(parameters))


def number_projected_slater_seed(orbitals, *, pairing_scale=1.0):
    """Embed an even Slater determinant in a finite Thouless vacuum.

    Consecutive occupied orbitals are paired in the auxiliary BCS vacuum.  Its
    component with particle number equal to the number of supplied orbitals is
    exactly their Slater determinant (up to an irrelevant scalar and phase).
    Consequently, exact particle-number projection gives precisely the PAV
    trial state while keeping the VAP optimizer inside the finite Thouless
    chart.  Callers should order columns so that consecutive pairs do not mix
    separately projected species.
    """
    orbitals = np.asarray(orbitals, dtype=complex)
    if orbitals.ndim != 2:
        raise ValueError("orbitals must be a two-dimensional matrix")
    modes, particles = orbitals.shape
    if particles == 0 or particles % 2:
        raise ValueError("an even, nonzero number of occupied orbitals is required")
    if not np.isfinite(orbitals).all() or not np.isfinite(pairing_scale):
        raise ValueError("orbitals and pairing_scale must be finite")
    if pairing_scale <= 0:
        raise ValueError("pairing_scale must be positive")
    gram = orbitals.conj().T @ orbitals
    if not np.allclose(gram, np.eye(particles), atol=1e-9):
        raise ValueError("occupied orbitals must be orthonormal")
    pairing = np.zeros((particles, particles), dtype=complex)
    for column in range(0, particles, 2):
        pairing[column, column + 1] = pairing_scale
        pairing[column + 1, column] = -pairing_scale
    z = orbitals @ pairing @ orbitals.T
    if z.shape != (modes, modes):
        raise RuntimeError("internal Slater-seed dimension mismatch")
    return HFBState.from_thouless(z)


def solve_projected_hfb_vap(
    projector,
    *,
    energy_backend="kernel",
    fermionic_hamiltonian=None,
    allow_aliased_number_kernel=False,
    starts=4,
    seed=0,
    maxiter=300,
    tolerance=1e-10,
    gradient_tolerance=2e-5,
    parameter_bound=8.0,
    initial_state=None,
    initial_parameters=None,
    seed_scales=(0.15, 0.6, 1.5),
    optimizer="L-BFGS-B",
    spsa_learning_rate=0.08,
    spsa_perturbation=0.12,
):
    """Minimize the configured projected energy over Bogoliubov vacua.

    This is projected-energy VAP, not a fidelity optimization.  The finite
    Thouless chart covers paired vacua with nonzero particle-vacuum overlap;
    exact nonempty Slater determinants lie on its infinite-norm boundary.
    Multiple seed scales therefore probe both weakly and strongly paired
    intrinsic states without imposing an artificial pairing constraint.
    """
    if starts < 1 or maxiter < 1:
        raise ValueError("starts and maxiter must be positive")
    if tolerance <= 0 or gradient_tolerance <= 0 or parameter_bound <= 0:
        raise ValueError("optimizer tolerances and parameter bound must be positive")
    if optimizer not in {"L-BFGS-B", "SPSA"}:
        raise ValueError("optimizer must be 'L-BFGS-B' or 'SPSA'")
    if spsa_learning_rate <= 0 or spsa_perturbation <= 0:
        raise ValueError("SPSA scales must be positive")
    scales = tuple(float(scale) for scale in seed_scales)
    if not scales or any(not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("seed_scales must contain positive finite values")

    objective = ProjectedVAPObjective(
        projector,
        energy_backend=energy_backend,
        fermionic_hamiltonian=fermionic_hamiltonian,
        allow_aliased_number_kernel=allow_aliased_number_kernel,
    )
    dimension = objective.modes * (objective.modes - 1)
    supplied = []
    if initial_parameters is not None:
        parameters = np.asarray(initial_parameters, float)
        if parameters.shape != (dimension,) or not np.isfinite(parameters).all():
            raise ValueError("initial_parameters have the wrong shape or are nonfinite")
        supplied.append(parameters.copy())
    if initial_state is not None:
        supplied.append(objective.pack(initial_state))

    rng = np.random.default_rng(seed)
    seeds = supplied[:starts]
    while len(seeds) < starts:
        scale = scales[len(seeds) % len(scales)]
        seeds.append(rng.normal(scale=scale, size=dimension))

    attempts = []
    candidates = []
    bounds = [(-parameter_bound, parameter_bound)] * dimension
    for initial in seeds:
        failures = 0
        evaluations = 0

        def guarded_energy(parameters):
            nonlocal failures, evaluations
            evaluations += 1
            try:
                value = objective.energy(parameters)
                if np.isfinite(value):
                    return value
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                pass
            failures += 1
            return 1e12

        initial_energy = guarded_energy(initial)
        if optimizer == "L-BFGS-B":
            fit = minimize(
                guarded_energy,
                initial,
                method="L-BFGS-B",
                jac=False,
                bounds=bounds,
                options={
                    "maxiter": maxiter,
                    "ftol": tolerance,
                    "gtol": gradient_tolerance / 10,
                    "maxls": 40,
                },
            )
            returned_parameters = fit.x
            residual = (
                float(np.linalg.norm(fit.jac))
                if fit.jac is not None else np.inf
            )
            iterations = int(fit.nit)
            success = bool(fit.success)
            message = str(fit.message)
        else:
            # Simultaneous perturbation gives a dimension-independent two-point
            # gradient estimate, making 132-parameter CKI VAP scans practical.
            x = np.clip(np.asarray(initial, float), -parameter_bound,
                        parameter_bound)
            current = initial_energy
            best_energy = current
            best_parameters = x.copy()
            residual = np.inf
            recent = []
            success = False
            for iteration in range(maxiter):
                power = iteration + 1.0
                perturbation = spsa_perturbation / power ** 0.101
                learning_rate = spsa_learning_rate / power ** 0.602
                direction = rng.choice((-1.0, 1.0), size=dimension)
                plus = np.clip(
                    x + perturbation * direction,
                    -parameter_bound,
                    parameter_bound,
                )
                minus = np.clip(
                    x - perturbation * direction,
                    -parameter_bound,
                    parameter_bound,
                )
                plus_energy = guarded_energy(plus)
                minus_energy = guarded_energy(minus)
                gradient = (
                    (plus_energy - minus_energy)
                    / (2 * perturbation)
                    * direction
                )
                residual = float(np.linalg.norm(gradient))
                # Normalize unusually large stochastic gradients so a single
                # noisy direction cannot jump across the entire bounded chart.
                scale = max(1.0, residual / np.sqrt(dimension))
                trial = np.clip(
                    x - learning_rate * gradient / scale,
                    -parameter_bound,
                    parameter_bound,
                )
                trial_energy = guarded_energy(trial)
                x, current = trial, trial_energy
                if current < best_energy:
                    best_energy = current
                    best_parameters = x.copy()
                recent.append(best_energy)
                if len(recent) > 12:
                    recent.pop(0)
                    spread = max(recent) - min(recent)
                    if spread <= tolerance * max(1.0, abs(best_energy)):
                        success = True
                        break
            returned_parameters = best_parameters
            iterations = iteration + 1
            message = (
                "SPSA energy plateau reached" if success
                else "SPSA iteration limit reached"
            )
        try:
            energy = float(objective.energy(returned_parameters))
            valid = True
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            energy = float("inf")
            valid = False
        converged = bool(
            valid
            and optimizer == "L-BFGS-B"
            and success
            and residual <= gradient_tolerance
        )
        attempts.append({
            "optimizer": optimizer,
            "initial_projected_energy": float(initial_energy),
            "projected_energy": energy,
            "converged": converged,
            "gradient_norm": residual,
            "iterations": iterations,
            "evaluations": int(evaluations),
            "failed_evaluations": int(failures),
            "message": message,
        })
        if valid:
            candidates.append((
                energy,
                converged,
                residual,
                returned_parameters.copy(),
                np.asarray(initial, float).copy(),
            ))

    if not candidates:
        raise RuntimeError("Every projected-VAP optimization attempt failed")
    energy, converged, residual, parameters, selected_initial = min(
        candidates, key=lambda item: item[0]
    )
    state = objective.state(parameters)
    series = objective.series(parameters)
    # Re-evaluate through the retained final series so downstream fidelity and
    # the reported VAP energy consume precisely the same transformation list.
    energy = float(objective.series_energy(series))
    return ProjectedVAPResult(
        state=state,
        projected_series=series,
        projected_energy=energy,
        parameters=parameters,
        initial_parameters=selected_initial,
        converged=converged,
        gradient_norm=residual,
        attempts=attempts,
        energy_backend=energy_backend,
        kappa_norm=float(np.linalg.norm(state.kappa)),
        number_grid_guaranteed_exact=bool(
            series.number_grid_guaranteed_exact
        ),
        euler_grid_guaranteed_exact=bool(
            series.euler_grid_guaranteed_exact
        ),
    )


__all__ = [
    "ProjectedVAPObjective",
    "ProjectedVAPResult",
    "number_projected_slater_seed",
    "solve_projected_hfb_vap",
]
