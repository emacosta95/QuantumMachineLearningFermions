"""Variation after particle-number and J=0 projection.

The optimizer varies a finite Thouless matrix. Kernel and direct-basis modes
construct the configured ``BogoliubovVacuumSeries`` at every evaluation.  The
analytic fixed-sector mode differentiates the same Euler-transformed
Pfaffian amplitudes after performing the target-sector U(1) sum algebraically.
The final state is always retained as a complete gauge/Euler vacuum series for
independent energy and fidelity validation.

Three energy backends are intentionally distinct:

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

``analytic_fixed_sector``
    Uses the same small fixed sector but differentiates every projected
    Pfaffian amplitude exactly.  It is the deterministic CKI VAP backend; the
    final state can be cross-checked with the full transition-kernel series.
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

    _BACKENDS = {"kernel", "fixed_sector_basis", "analytic_fixed_sector"}

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
                "energy_backend must be 'kernel', 'fixed_sector_basis', or "
                "'analytic_fixed_sector'"
            )
        if (energy_backend in {"fixed_sector_basis", "analytic_fixed_sector"}
                and fermionic_hamiltonian is None):
            raise ValueError(
                f"{energy_backend} requires fermionic_hamiltonian"
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
        self.has_analytic_gradient = energy_backend == "analytic_fixed_sector"
        if self.has_analytic_gradient:
            self._prepare_analytic_sector()

    def _prepare_analytic_sector(self):
        occupations = tuple(
            tuple(int(mode) for mode in occupied)
            for occupied in self.fermionic_hamiltonian.occupations
        )
        if not occupations:
            raise ValueError("analytic fixed-sector VAP requires a nonempty basis")
        particle_counts = {len(occupied) for occupied in occupations}
        if len(particle_counts) != 1 or next(iter(particle_counts)) % 2:
            raise ValueError(
                "analytic fixed-sector VAP requires one even particle number"
            )
        neutron_modes = set(np.flatnonzero(self.projector.mask))
        targets = tuple(int(value) for value in self.projector.targets)
        for occupied in occupations:
            neutrons = sum(mode in neutron_modes for mode in occupied)
            if (neutrons, len(occupied) - neutrons) != targets:
                raise ValueError(
                    "fermionic basis does not match the projector N,Z targets"
                )
        matrix_source = self.fermionic_hamiltonian.matrix
        if hasattr(matrix_source, "toarray"):
            matrix_source = matrix_source.toarray()
        matrix = np.asarray(matrix_source, dtype=complex)
        if matrix.shape != (len(occupations), len(occupations)):
            raise ValueError("fermionic Hamiltonian matrix has the wrong shape")
        self._analytic_occupations = occupations
        self._analytic_occupation_array = np.asarray(occupations, dtype=int)
        self._analytic_hamiltonian = matrix
        self._analytic_rotations = np.asarray(
            [rotation for rotation, _ in self.projector.rotations], dtype=complex
        )
        self._analytic_rotation_weights = np.asarray(
            [weight for _, weight in self.projector.rotations], dtype=complex
        )
        particles = next(iter(particle_counts))
        sub_a, sub_b = np.triu_indices(particles, 1)
        self._analytic_sub_a = sub_a
        self._analytic_sub_b = sub_b
        sub_pair_index = {
            (int(row), int(column)): index
            for index, (row, column) in enumerate(zip(sub_a, sub_b))
        }
        self._analytic_matching_terms = tuple(
            (
                sign,
                tuple(sub_pair_index[pair] for pair in matching),
            )
            for sign, matching in self._perfect_matchings(tuple(range(particles)))
        )

    @classmethod
    def _perfect_matchings(cls, indices):
        """Return signed perfect matchings in Pfaffian expansion order."""
        if not indices:
            return ((1, ()),)
        first = indices[0]
        terms = []
        for position in range(1, len(indices)):
            second = indices[position]
            remaining = indices[1:position] + indices[position + 1:]
            local_sign = (-1) ** (position + 1)
            for nested_sign, nested_pairs in cls._perfect_matchings(remaining):
                terms.append((
                    local_sign * nested_sign,
                    ((first, second),) + nested_pairs,
                ))
        return tuple(terms)

    @staticmethod
    def _small_pfaffian(matrix):
        """Exact recursive Pfaffian for the small CKI determinant minors."""
        matrix = np.asarray(matrix, dtype=complex)
        size = len(matrix)
        if size == 0:
            return 1.0 + 0.0j
        if size == 2:
            return matrix[0, 1]
        value = 0.0j
        for column in range(1, size):
            keep = [index for index in range(size)
                    if index not in (0, column)]
            value += ((-1) ** (column + 1) * matrix[0, column]
                      * ProjectedVAPObjective._small_pfaffian(
                          matrix[np.ix_(keep, keep)]
                      ))
        return value

    @classmethod
    def _pfaffian_with_cofactor(cls, matrix):
        """Return Pf(A) and d Pf(A)/d A_rs for independent r<s entries."""
        matrix = np.asarray(matrix, dtype=complex)
        size = len(matrix)
        value = cls._small_pfaffian(matrix)
        rows, columns = np.triu_indices(size, 1)
        cofactor = np.empty(len(rows), dtype=complex)
        for index, (row, column) in enumerate(zip(rows, columns)):
            keep = [position for position in range(size)
                    if position not in (row, column)]
            cofactor[index] = (
                (-1) ** (row + column + 1)
                * cls._small_pfaffian(matrix[np.ix_(keep, keep)])
            )
        return value, rows, columns, cofactor

    @staticmethod
    def _batch_pfaffian_cofactors(pair_values, matching_terms):
        """Evaluate Pfaffians and all independent-entry derivatives in batch."""
        pair_values = np.asarray(pair_values, dtype=complex)
        values = np.zeros(pair_values.shape[:-1], dtype=complex)
        cofactors = np.zeros_like(pair_values)
        for sign, matching in matching_terms:
            factors = pair_values[..., matching]
            values += sign * np.prod(factors, axis=-1)
            for position, pair_index in enumerate(matching):
                other = tuple(
                    index for offset, index in enumerate(matching)
                    if offset != position
                )
                cofactors[..., pair_index] += sign * np.prod(
                    pair_values[..., other], axis=-1
                )
        return values, cofactors

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
        if self.has_analytic_gradient:
            return self.energy_and_gradient(parameters)[0]
        return self.series_energy(self.series(parameters))

    def energy_and_gradient(self, parameters):
        """Exact projected energy and real Thouless gradient in a fixed sector.

        The determinant basis already has the requested N,Z. At every gauge
        node, its phase cancels the Fourier character exactly, so the complete
        number-projection sum is one. The configured Euler transformations are
        retained explicitly and differentiated through their Pfaffian
        amplitudes. No J2 eigenprojector or finite difference is used here.
        """
        if not self.has_analytic_gradient:
            raise ValueError("this backend does not provide an analytic gradient")
        z = self.unpack(parameters)
        pair_a, pair_b = self.ij
        pairs = len(pair_a)
        amplitudes = np.zeros(len(self._analytic_occupations), dtype=complex)
        holomorphic = np.zeros((len(amplitudes), pairs), dtype=complex)

        sub_a = self._analytic_sub_a
        sub_b = self._analytic_sub_b
        # Batch a few Euler rotations along with every determinant.  A modest
        # batch keeps the (rotation,basis,subpair,Thouless-pair) wedge tensor
        # below about 30 MB for the 12-mode CKI space.
        rotation_batch = 8
        for start in range(0, len(self._analytic_rotations), rotation_batch):
            rotations = self._analytic_rotations[start:start + rotation_batch]
            weights = self._analytic_rotation_weights[
                start:start + rotation_batch
            ]
            # Shape: (rotations, basis, particles, modes).
            rows = rotations[:, self._analytic_occupation_array, :]
            rotated = np.einsum(
                "qbri,ij,qbsj->qbrs", rows, z, rows, optimize=True
            )
            pair_values = rotated[:, :, sub_a, sub_b]
            values, cofactors = self._batch_pfaffian_cofactors(
                pair_values, self._analytic_matching_terms
            )
            amplitudes += np.einsum("q,qb->b", weights, values)

            # d(TZT^T)_rs/dZ_ab for every determinant, submatrix pair, and
            # independent Thouless entry.  This batch contraction replaces the
            # previous 225 Python-level determinant loops.
            wedge = (
                rows[:, :, sub_a, :][:, :, :, pair_a]
                * rows[:, :, sub_b, :][:, :, :, pair_b]
                - rows[:, :, sub_a, :][:, :, :, pair_b]
                * rows[:, :, sub_b, :][:, :, :, pair_a]
            )
            holomorphic += np.einsum(
                "q,qbp,qbpk->bk", weights, cofactors, wedge, optimize=True
            )

        norm = float(np.vdot(amplitudes, amplitudes).real)
        if not np.isfinite(norm) or norm < 1e-20:
            raise ValueError("projected analytic state has vanishing norm")
        h_state = self._analytic_hamiltonian @ amplitudes
        energy = np.vdot(amplitudes, h_state) / norm
        if abs(energy.imag) > 1e-8:
            raise ValueError("analytic projected energy is not real")
        residual = h_state - energy.real * amplitudes
        real_gradient = 2 * np.real(holomorphic.conj().T @ residual) / norm
        imaginary_gradient = 2 * np.real(
            (1j * holomorphic).conj().T @ residual
        ) / norm
        gradient = np.concatenate((real_gradient, imaginary_gradient))
        if not np.isfinite(gradient).all():
            raise ValueError("analytic projected gradient is nonfinite")
        return float(energy.real), gradient


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
            if objective.has_analytic_gradient:
                def value_and_gradient(parameters):
                    nonlocal failures, evaluations
                    evaluations += 1
                    try:
                        value, gradient = objective.energy_and_gradient(parameters)
                        if np.isfinite(value) and np.isfinite(gradient).all():
                            return value, gradient
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                        pass
                    failures += 1
                    return 1e12, np.zeros_like(parameters)
                scipy_objective = value_and_gradient
                scipy_jacobian = True
            else:
                scipy_objective = guarded_energy
                scipy_jacobian = False
            fit = minimize(
                scipy_objective,
                initial,
                method="L-BFGS-B",
                jac=scipy_jacobian,
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
