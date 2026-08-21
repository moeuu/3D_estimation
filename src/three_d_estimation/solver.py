"""PF-independent Poisson reconstruction on an area-aware surface patch graph."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from .response_operator import ResponseOperator


@dataclass(frozen=True)
class SurfaceMapConfig:
    """Configure non-negative Poisson reconstruction with spatial regularization."""

    l1_weight: float = 0.0
    tv_weight: float = 0.0
    isotope_group_weight: float = 0.0
    nuisance_l1_weight: float = 0.0
    nuisance_l2_weight: float = 0.0
    nuisance_l2_weights: tuple[float, ...] = ()
    likelihood_family: str = "poisson"
    overdispersion_alpha: tuple[float, ...] = ()
    max_iterations: int = 4000
    poisson_em_warm_start_iterations: int = 0
    tolerance: float = 1.0e-6
    objective_tolerance: float = 1.0e-7
    kkt_tolerance: float | None = None
    check_interval: int = 20
    step_safety: float = 0.95
    over_relaxation: float = 1.0
    min_mean: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate solver parameters without changing their physical meaning."""
        non_negative = {
            "l1_weight": self.l1_weight,
            "tv_weight": self.tv_weight,
            "isotope_group_weight": self.isotope_group_weight,
            "nuisance_l1_weight": self.nuisance_l1_weight,
            "nuisance_l2_weight": self.nuisance_l2_weight,
            "tolerance": self.tolerance,
            "objective_tolerance": self.objective_tolerance,
        }
        if any(
            not np.isfinite(value) or value < 0.0 for value in non_negative.values()
        ):
            raise ValueError(
                "Regularization weights and tolerances must be finite and non-negative."
            )
        if any(
            not np.isfinite(value) or float(value) < 0.0
            for value in self.nuisance_l2_weights
        ):
            raise ValueError("nuisance_l2_weights must be finite and non-negative.")
        if self.likelihood_family not in {"poisson", "negative_binomial"}:
            raise ValueError("likelihood_family must be poisson or negative_binomial.")
        if any(
            not np.isfinite(value) or float(value) < 0.0
            for value in self.overdispersion_alpha
        ):
            raise ValueError("overdispersion_alpha must be finite and non-negative.")
        if self.likelihood_family == "poisson" and any(
            float(value) != 0.0 for value in self.overdispersion_alpha
        ):
            raise ValueError("Poisson likelihood cannot have positive overdispersion.")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be at least one.")
        if (
            isinstance(self.poisson_em_warm_start_iterations, (bool, np.bool_))
            or int(self.poisson_em_warm_start_iterations)
            != self.poisson_em_warm_start_iterations
            or int(self.poisson_em_warm_start_iterations) < 0
        ):
            raise ValueError(
                "poisson_em_warm_start_iterations must be a nonnegative integer."
            )
        if self.kkt_tolerance is not None and (
            not np.isfinite(self.kkt_tolerance)
            or float(self.kkt_tolerance) < 0.0
        ):
            raise ValueError("kkt_tolerance must be null or finite and nonnegative.")
        if int(self.check_interval) < 1:
            raise ValueError("check_interval must be at least one.")
        if not np.isfinite(self.step_safety) or not 0.0 < self.step_safety < 1.0:
            raise ValueError("step_safety must lie strictly between zero and one.")
        if (
            not np.isfinite(self.over_relaxation)
            or not 0.0 <= self.over_relaxation <= 1.0
        ):
            raise ValueError("over_relaxation must lie between zero and one.")
        if not np.isfinite(self.min_mean) or self.min_mean <= 0.0:
            raise ValueError("min_mean must be finite and positive.")


@dataclass(frozen=True)
class SurfaceMapObjective:
    """Store the terms of the regularized Poisson surface-map objective."""

    total: float
    poisson_nll: float
    l1_penalty: float
    tv_penalty: float
    group_penalty: float
    nuisance_penalty: float
    deviance: float


@dataclass(frozen=True)
class SurfaceMapResult:
    """Store a reconstructed surface intensity map and convergence diagnostics."""

    densities_cps_1m_m2: NDArray[np.float64]
    integrated_strengths_cps_1m: NDArray[np.float64]
    nuisance_coefficients: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    objective: float
    poisson_nll: float
    l1_penalty: float
    tv_penalty: float
    group_penalty: float
    nuisance_penalty: float
    deviance: float
    converged: bool
    iterations: int
    relative_change: float
    relative_objective_change: float
    kkt_residual: float
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class _PreparedSurfaceMapProblem:
    """Store validated, flattened arrays used by the batched solver."""

    observed: NDArray[np.float64]
    response_by_density: NDArray[np.float64]
    response_by_integrated_strength: NDArray[np.float64]
    background: NDArray[np.float64]
    nuisance_response: NDArray[np.float64]
    patch_areas: NDArray[np.float64]
    column_areas: NDArray[np.float64]
    incidence: csr_matrix
    edge_weights: NDArray[np.float64]
    observation_shape: tuple[int, ...]
    patch_count: int
    isotope_count: int


def _nuisance_l2_vector(
    config: SurfaceMapConfig,
    count: int,
) -> NDArray[np.float64]:
    """Return one L2 shrinkage weight per nuisance coefficient."""
    if not config.nuisance_l2_weights:
        return np.full(count, float(config.nuisance_l2_weight), dtype=np.float64)
    weights = np.asarray(config.nuisance_l2_weights, dtype=np.float64)
    if weights.shape != (count,):
        raise ValueError("nuisance_l2_weights must match nuisance_response columns.")
    return weights


def _as_non_negative_vector(
    values: ArrayLike,
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return a finite non-negative vector or raise for invalid physical inputs."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array < -1.0e-12):
        raise ValueError(f"{name} must be non-negative.")
    return np.maximum(array, 0.0)


def _broadcast_observation_vector(
    values: float | ArrayLike,
    observation_shape: tuple[int, ...],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return one finite non-negative value per flattened observation."""
    observation_count = int(np.prod(observation_shape, dtype=np.int64))
    array = np.asarray(values, dtype=float)
    if array.size == 1:
        vector = np.full(observation_count, float(array.reshape(-1)[0]), dtype=float)
    elif array.shape == observation_shape or array.size == observation_count:
        vector = array.reshape(-1).astype(float, copy=False)
    else:
        raise ValueError(
            f"{name} must be scalar, match observed_counts, or contain one value per observation."
        )
    return _as_non_negative_vector(vector, name=name)


def _flatten_response(
    observed_shape: tuple[int, ...],
    response: ArrayLike,
    patch_count: int,
) -> tuple[NDArray[np.float64], int]:
    """Flatten a shared-physics matrix or spectrum tensor to observations by columns."""
    response_array = np.asarray(response, dtype=float)
    prefix_rank = len(observed_shape)
    if response_array.ndim not in {prefix_rank + 1, prefix_rank + 2}:
        raise ValueError(
            "response must have shape observed_shape + (patches,) or "
            "observed_shape + (patches, isotopes)."
        )
    if tuple(response_array.shape[:prefix_rank]) != observed_shape:
        raise ValueError("response observation dimensions must match observed_counts.")
    if int(response_array.shape[prefix_rank]) != int(patch_count):
        raise ValueError("response patch dimension must match patch_areas_m2.")
    isotope_count = (
        int(response_array.shape[prefix_rank + 1])
        if response_array.ndim == prefix_rank + 2
        else 1
    )
    if isotope_count < 1:
        raise ValueError("response must contain at least one isotope channel.")
    if np.any(~np.isfinite(response_array)):
        raise ValueError("response must contain only finite values.")
    if np.any(response_array < -1.0e-12):
        raise ValueError("response must be non-negative.")
    observation_count = int(np.prod(observed_shape, dtype=np.int64))
    matrix = np.maximum(response_array, 0.0).reshape(
        observation_count,
        patch_count * isotope_count,
    )
    return matrix, isotope_count


def _flatten_nuisance_response(
    nuisance_response: ArrayLike | None,
    observation_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Return an observations-by-nuisance non-negative design matrix."""
    observation_count = int(np.prod(observation_shape, dtype=np.int64))
    if nuisance_response is None:
        return np.zeros((observation_count, 0), dtype=float)
    array = np.asarray(nuisance_response, dtype=float)
    if array.size == 0:
        return np.zeros((observation_count, 0), dtype=float)
    if array.ndim == 2 and array.shape[0] == observation_count:
        matrix = array
    elif (
        array.ndim == len(observation_shape) + 1
        and tuple(array.shape[: len(observation_shape)]) == observation_shape
    ):
        matrix = array.reshape(observation_count, -1)
    else:
        raise ValueError(
            "nuisance_response must be observations x nuisance or "
            "observed_shape + (nuisance,)."
        )
    if np.any(~np.isfinite(matrix)):
        raise ValueError("nuisance_response must contain only finite values.")
    if np.any(matrix < -1.0e-12):
        raise ValueError("nuisance_response must be non-negative.")
    return np.maximum(matrix, 0.0)


def _canonical_graph(
    adjacency_edges: ArrayLike | None,
    adjacency_weights: ArrayLike | None,
    patch_count: int,
) -> tuple[csr_matrix, NDArray[np.float64]]:
    """Return a deduplicated oriented incidence matrix and summed edge weights."""
    if adjacency_edges is None:
        return csr_matrix((0, patch_count), dtype=float), np.zeros(0, dtype=float)
    edges = np.asarray(adjacency_edges, dtype=np.int64)
    if edges.size == 0:
        return csr_matrix((0, patch_count), dtype=float), np.zeros(0, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("adjacency_edges must have shape edges x 2.")
    if np.any(edges < 0) or np.any(edges >= int(patch_count)):
        raise ValueError("adjacency edge indices must refer to existing patches.")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("adjacency edges must connect distinct patches.")
    canonical = np.sort(edges, axis=1)
    unique_edges, inverse = np.unique(canonical, axis=0, return_inverse=True)
    if adjacency_weights is None:
        raw_weights = np.ones(edges.shape[0], dtype=float)
    else:
        raw_weights = _as_non_negative_vector(
            adjacency_weights,
            name="adjacency_weights",
        )
        if raw_weights.size != edges.shape[0]:
            raise ValueError("adjacency_weights must contain one value per edge.")
    weights = np.bincount(
        inverse,
        weights=raw_weights,
        minlength=unique_edges.shape[0],
    ).astype(float, copy=False)
    row_indices = np.repeat(np.arange(unique_edges.shape[0], dtype=np.int64), 2)
    column_indices = unique_edges.reshape(-1)
    values = np.tile(np.asarray([-1.0, 1.0], dtype=float), unique_edges.shape[0])
    incidence = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(unique_edges.shape[0], patch_count),
        dtype=float,
    )
    return incidence, weights


def _prepare_surface_map_problem(
    observed_counts: ArrayLike,
    response: ArrayLike,
    patch_areas_m2: ArrayLike,
    adjacency_edges: ArrayLike | None,
    adjacency_weights: ArrayLike | None,
    *,
    background: float | ArrayLike,
    nuisance_response: ArrayLike | None,
) -> _PreparedSurfaceMapProblem:
    """Validate and flatten all inputs while preserving candidate-isotope ordering."""
    observed_array = np.asarray(observed_counts, dtype=float)
    if observed_array.ndim < 1 or observed_array.size == 0:
        raise ValueError("observed_counts must contain at least one observation.")
    observed_shape = tuple(int(value) for value in observed_array.shape)
    observed = _as_non_negative_vector(observed_array, name="observed_counts")
    patch_areas = _as_non_negative_vector(patch_areas_m2, name="patch_areas_m2")
    if patch_areas.size == 0 or np.any(patch_areas <= 0.0):
        raise ValueError("patch_areas_m2 must contain positive patch areas.")
    response_integrated, isotope_count = _flatten_response(
        observed_shape,
        response,
        int(patch_areas.size),
    )
    column_areas = np.repeat(patch_areas, isotope_count)
    response_density = response_integrated * column_areas[None, :]
    background_vector = _broadcast_observation_vector(
        background,
        observed_shape,
        name="background",
    )
    nuisance_matrix = _flatten_nuisance_response(
        nuisance_response,
        observed_shape,
    )
    incidence, edge_weights = _canonical_graph(
        adjacency_edges,
        adjacency_weights,
        int(patch_areas.size),
    )
    return _PreparedSurfaceMapProblem(
        observed=observed,
        response_by_density=response_density,
        response_by_integrated_strength=response_integrated,
        background=background_vector,
        nuisance_response=nuisance_matrix,
        patch_areas=patch_areas,
        column_areas=column_areas,
        incidence=incidence,
        edge_weights=edge_weights,
        observation_shape=observed_shape,
        patch_count=int(patch_areas.size),
        isotope_count=int(isotope_count),
    )


def _poisson_nll(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    *,
    min_mean: float,
) -> float:
    """Return Poisson negative log likelihood with model-independent constants omitted."""
    mean = np.maximum(np.asarray(expected, dtype=float).reshape(-1), float(min_mean))
    counts = np.asarray(observed, dtype=float).reshape(-1)
    return float(np.sum(mean - counts * np.log(mean)))


def _poisson_deviance(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    *,
    min_mean: float,
) -> float:
    """Return the Poisson deviance from the saturated count model."""
    counts = np.asarray(observed, dtype=float).reshape(-1)
    mean = np.maximum(np.asarray(expected, dtype=float).reshape(-1), float(min_mean))
    positive = counts > 0.0
    log_terms = np.zeros_like(counts, dtype=float)
    log_terms[positive] = counts[positive] * np.log(counts[positive] / mean[positive])
    return float(2.0 * np.sum(log_terms - counts + mean))


def _objective_from_prepared(
    problem: _PreparedSurfaceMapProblem,
    densities: NDArray[np.float64],
    nuisance_coefficients: NDArray[np.float64],
    config: SurfaceMapConfig,
) -> tuple[SurfaceMapObjective, NDArray[np.float64]]:
    """Evaluate all objective terms for validated density and nuisance arrays."""
    density_matrix = np.asarray(densities, dtype=float).reshape(
        problem.patch_count,
        problem.isotope_count,
    )
    nuisance = np.asarray(nuisance_coefficients, dtype=float).reshape(-1)
    signal = problem.response_by_density @ density_matrix.reshape(-1)
    if nuisance.size:
        signal = signal + problem.nuisance_response @ nuisance
    expected = np.maximum(problem.background + signal, float(config.min_mean))
    poisson_nll = _poisson_nll(
        problem.observed,
        expected,
        min_mean=float(config.min_mean),
    )
    l1_penalty = float(config.l1_weight) * float(
        np.sum(problem.patch_areas[:, None] * density_matrix)
    )
    if problem.incidence.shape[0] and float(config.tv_weight) > 0.0:
        differences = problem.incidence @ density_matrix
        tv_penalty = float(config.tv_weight) * float(
            np.sum(problem.edge_weights[:, None] * np.abs(differences))
        )
    else:
        tv_penalty = 0.0
    group_penalty = float(config.isotope_group_weight) * float(
        np.sum(problem.patch_areas * np.linalg.norm(density_matrix, axis=1))
    )
    nuisance_l2 = _nuisance_l2_vector(config, nuisance.size)
    nuisance_penalty = float(config.nuisance_l1_weight) * float(
        np.sum(nuisance)
    ) + 0.5 * float(np.dot(nuisance_l2, nuisance * nuisance))
    objective = SurfaceMapObjective(
        total=float(
            poisson_nll + l1_penalty + tv_penalty + group_penalty + nuisance_penalty
        ),
        poisson_nll=float(poisson_nll),
        l1_penalty=float(l1_penalty),
        tv_penalty=float(tv_penalty),
        group_penalty=float(group_penalty),
        nuisance_penalty=float(nuisance_penalty),
        deviance=_poisson_deviance(
            problem.observed,
            expected,
            min_mean=float(config.min_mean),
        ),
    )
    return objective, expected


def evaluate_surface_map_objective(
    observed_counts: ArrayLike,
    response: ArrayLike,
    patch_areas_m2: ArrayLike,
    densities_cps_1m_m2: ArrayLike,
    adjacency_edges: ArrayLike | None = None,
    adjacency_weights: ArrayLike | None = None,
    *,
    background: float | ArrayLike = 0.0,
    nuisance_response: ArrayLike | None = None,
    nuisance_coefficients: ArrayLike | None = None,
    config: SurfaceMapConfig | None = None,
) -> SurfaceMapObjective:
    """Evaluate the public regularized objective for a supplied surface map."""
    solver_config = SurfaceMapConfig() if config is None else config
    problem = _prepare_surface_map_problem(
        observed_counts,
        response,
        patch_areas_m2,
        adjacency_edges,
        adjacency_weights,
        background=background,
        nuisance_response=nuisance_response,
    )
    densities = _as_non_negative_vector(
        densities_cps_1m_m2,
        name="densities_cps_1m_m2",
    )
    expected_density_count = problem.patch_count * problem.isotope_count
    if densities.size != expected_density_count:
        raise ValueError(
            "densities_cps_1m_m2 must contain one value per patch and isotope."
        )
    nuisance_count = int(problem.nuisance_response.shape[1])
    nuisance = (
        np.zeros(nuisance_count, dtype=float)
        if nuisance_coefficients is None
        else _as_non_negative_vector(
            nuisance_coefficients,
            name="nuisance_coefficients",
        )
    )
    if nuisance.size != nuisance_count:
        raise ValueError("nuisance_coefficients must match nuisance_response columns.")
    objective, _expected = _objective_from_prepared(
        problem,
        densities.reshape(problem.patch_count, problem.isotope_count),
        nuisance,
        solver_config,
    )
    return objective


def _dual_poisson_prox(
    dual_trial: NDArray[np.float64],
    dual_steps: NDArray[np.float64],
    observed: NDArray[np.float64],
    background: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply the closed-form proximal operator of the shifted Poisson conjugate."""
    sigma = np.asarray(dual_steps, dtype=float)
    trial = np.asarray(dual_trial, dtype=float)
    gamma = 1.0 / sigma
    proximal_center = trial / sigma
    shifted = background + proximal_center - gamma
    mean = 0.5 * (
        shifted + np.sqrt(np.maximum(shifted * shifted + 4.0 * gamma * observed, 0.0))
    )
    primal_prox = mean - background
    return trial - sigma * primal_prox


def _preconditioned_steps(
    problem: _PreparedSurfaceMapProblem,
    *,
    tv_active: bool,
    safety: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return diagonal Chambolle-Pock steps from absolute operator row/column sums."""
    response = problem.response_by_density
    nuisance = problem.nuisance_response
    observation_row_sums = np.sum(response, axis=1)
    if nuisance.shape[1]:
        observation_row_sums = observation_row_sums + np.sum(nuisance, axis=1)
    observation_steps = float(safety) / np.maximum(observation_row_sums, 1.0e-12)
    density_column_sums = np.sum(response, axis=0).reshape(
        problem.patch_count,
        problem.isotope_count,
    )
    edge_steps = np.zeros(0, dtype=float)
    if tv_active:
        degrees = np.asarray(
            np.abs(problem.incidence).sum(axis=0), dtype=float
        ).reshape(-1)
        density_column_sums = density_column_sums + degrees[:, None]
        edge_steps = np.full(
            problem.incidence.shape[0], float(safety) / 2.0, dtype=float
        )
    # A shared step within each isotope group makes the non-negative group
    # shrinkage below an exact proximal map instead of a diagonal approximation.
    patch_column_bound = np.max(density_column_sums, axis=1, keepdims=True)
    density_steps = np.broadcast_to(
        float(safety) / np.maximum(patch_column_bound, 1.0e-12),
        density_column_sums.shape,
    ).copy()
    if nuisance.shape[1]:
        nuisance_column_sums = np.sum(nuisance, axis=0)
        nuisance_steps = float(safety) / np.maximum(nuisance_column_sums, 1.0e-12)
    else:
        nuisance_steps = np.zeros(0, dtype=float)
    return density_steps, nuisance_steps, observation_steps, edge_steps


def _kkt_residual(
    problem: _PreparedSurfaceMapProblem,
    densities: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    tv_dual: NDArray[np.float64],
    expected: NDArray[np.float64],
    config: SurfaceMapConfig,
) -> float:
    """Return a scale-normalized first-order residual for non-negative variables."""
    likelihood_gradient = 1.0 - problem.observed / np.maximum(
        expected,
        float(config.min_mean),
    )
    density_gradient = (problem.response_by_density.T @ likelihood_gradient).reshape(
        problem.patch_count, problem.isotope_count
    )
    density_gradient = (
        density_gradient + float(config.l1_weight) * problem.patch_areas[:, None]
    )
    if float(config.isotope_group_weight) > 0.0:
        density_norm = np.linalg.norm(densities, axis=1, keepdims=True)
        active = density_norm[:, 0] > 1.0e-12
        density_gradient[active] += (
            float(config.isotope_group_weight)
            * problem.patch_areas[active, None]
            * densities[active]
            / density_norm[active]
        )
    if tv_dual.size:
        density_gradient = density_gradient + problem.incidence.T @ tv_dual
    density_stationarity = np.where(
        densities > 1.0e-9,
        density_gradient,
        np.minimum(density_gradient, 0.0),
    )
    residual_parts = [density_stationarity.reshape(-1)]
    if nuisance.size:
        nuisance_gradient = problem.nuisance_response.T @ likelihood_gradient
        nuisance_gradient = (
            nuisance_gradient
            + float(config.nuisance_l1_weight)
            + _nuisance_l2_vector(config, nuisance.size) * nuisance
        )
        nuisance_stationarity = np.where(
            nuisance > 1.0e-9,
            nuisance_gradient,
            np.minimum(nuisance_gradient, 0.0),
        )
        residual_parts.append(nuisance_stationarity)
    residual = np.concatenate(residual_parts)
    scale = max(1.0, float(np.linalg.norm(likelihood_gradient)))
    return float(np.linalg.norm(residual) / scale)


def _poisson_em_warm_start_numpy(
    problem: _PreparedSurfaceMapProblem,
    densities: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    config: SurfaceMapConfig,
    *,
    progress_hook: Callable[[Mapping[str, object]], None] | None,
    progress_phase: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a data-scaled Poisson EM initialization for primal-dual fitting."""
    iteration_count = int(config.poisson_em_warm_start_iterations)
    if iteration_count == 0 or not np.any(problem.observed > 0.0):
        return densities, nuisance
    nuisance_l2 = _nuisance_l2_vector(config, nuisance.size)
    source_column_sums = np.sum(problem.response_by_density, axis=0).reshape(
        densities.shape
    )
    nuisance_column_sums = np.sum(problem.nuisance_response, axis=0)
    total_unit_counts = float(np.sum(source_column_sums)) + float(
        np.sum(nuisance_column_sums)
    )
    target_signal_counts = max(
        float(np.sum(problem.observed) - np.sum(problem.background)),
        0.0,
    )
    if total_unit_counts <= 0.0 or target_signal_counts <= 0.0:
        return densities, nuisance
    initial_scale = target_signal_counts / total_unit_counts
    positive_floor = max(initial_scale * 1.0e-12, np.finfo(np.float64).tiny)
    if np.any(densities > 0.0) or np.any(nuisance > 0.0):
        densities = np.maximum(densities, positive_floor)
        nuisance = np.maximum(nuisance, positive_floor)
    else:
        densities = np.full_like(densities, initial_scale)
        nuisance = np.full_like(nuisance, initial_scale)
    density_denominator = (
        source_column_sums
        + float(config.l1_weight) * problem.patch_areas[:, None]
    )
    nuisance_base_denominator = (
        nuisance_column_sums + float(config.nuisance_l1_weight)
    )
    started = perf_counter()
    for iteration in range(1, iteration_count + 1):
        expected = (
            problem.background
            + problem.response_by_density @ densities.reshape(-1)
        )
        if nuisance.size:
            expected = expected + problem.nuisance_response @ nuisance
        ratio = problem.observed / np.maximum(expected, float(config.min_mean))
        density_numerator = (
            problem.response_by_density.T @ ratio
        ).reshape(densities.shape)
        densities *= density_numerator / np.maximum(
            density_denominator,
            np.finfo(np.float64).tiny,
        )
        if nuisance.size:
            nuisance_numerator = problem.nuisance_response.T @ ratio
            nuisance_denominator = (
                nuisance_base_denominator + nuisance_l2 * nuisance
            )
            nuisance *= nuisance_numerator / np.maximum(
                nuisance_denominator,
                np.finfo(np.float64).tiny,
            )
        if progress_hook is not None and (
            iteration == iteration_count
            or iteration % max(1, min(20, iteration_count)) == 0
        ):
            elapsed = perf_counter() - started
            progress_hook(
                {
                    "phase": f"{progress_phase}:poisson_em_warm_start",
                    "completed": int(iteration),
                    "total": iteration_count,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed
                    * (iteration_count - iteration)
                    / iteration,
                }
            )
    return densities, nuisance


def fit_surface_map_poisson(
    observed_counts: ArrayLike,
    response: ArrayLike,
    patch_areas_m2: ArrayLike,
    adjacency_edges: ArrayLike | None = None,
    adjacency_weights: ArrayLike | None = None,
    *,
    background: float | ArrayLike = 0.0,
    nuisance_response: ArrayLike | None = None,
    initial_densities_cps_1m_m2: ArrayLike | None = None,
    initial_nuisance_coefficients: ArrayLike | None = None,
    config: SurfaceMapConfig | None = None,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    progress_phase: str = "mle_solver_iterations",
) -> SurfaceMapResult:
    """
    Fit a non-negative all-history Poisson surface intensity map.

    Response columns have unit integrated-strength semantics in cps at 1 m.
    The solver multiplies each column by its patch area, so optimized source
    variables are densities in cps at 1 m per square meter.  The L1 term is
    therefore total integrated strength, while graph TV compares neighboring
    densities and weights each difference by the supplied shared-edge measure.
    Matrix responses use shape ``observations x patches``.  Spectrum tensors
    use ``observed_shape + (patches,)`` or
    ``observed_shape + (patches, isotopes)`` and are flattened in one batch.
    """
    solver_config = SurfaceMapConfig() if config is None else config
    if progress_hook is not None and not callable(progress_hook):
        raise TypeError("progress_hook must be callable or None.")
    if solver_config.likelihood_family != "poisson":
        raise ValueError(
            "Calibrated overdispersion requires the matrix-free solver path."
        )
    problem = _prepare_surface_map_problem(
        observed_counts,
        response,
        patch_areas_m2,
        adjacency_edges,
        adjacency_weights,
        background=background,
        nuisance_response=nuisance_response,
    )
    density_shape = (problem.patch_count, problem.isotope_count)
    if initial_densities_cps_1m_m2 is None:
        densities = np.zeros(density_shape, dtype=float)
    else:
        density_vector = _as_non_negative_vector(
            initial_densities_cps_1m_m2,
            name="initial_densities_cps_1m_m2",
        )
        if density_vector.size != int(np.prod(density_shape, dtype=np.int64)):
            raise ValueError(
                "initial_densities_cps_1m_m2 must match patches by isotopes."
            )
        densities = density_vector.reshape(density_shape).copy()
    nuisance_count = int(problem.nuisance_response.shape[1])
    nuisance_l2 = _nuisance_l2_vector(solver_config, nuisance_count)
    if initial_nuisance_coefficients is None:
        nuisance = np.zeros(nuisance_count, dtype=float)
    else:
        nuisance = _as_non_negative_vector(
            initial_nuisance_coefficients,
            name="initial_nuisance_coefficients",
        ).copy()
        if nuisance.size != nuisance_count:
            raise ValueError(
                "initial_nuisance_coefficients must match nuisance_response columns."
            )
    densities, nuisance = _poisson_em_warm_start_numpy(
        problem,
        densities,
        nuisance,
        solver_config,
        progress_hook=progress_hook,
        progress_phase=progress_phase,
    )
    tv_active = bool(
        float(solver_config.tv_weight) > 0.0 and problem.incidence.shape[0] > 0
    )
    (
        density_steps,
        nuisance_steps,
        observation_dual_steps,
        edge_dual_steps,
    ) = _preconditioned_steps(
        problem,
        tv_active=tv_active,
        safety=float(solver_config.step_safety),
    )
    observation_dual = np.zeros(problem.observed.size, dtype=float)
    tv_dual = (
        np.zeros((problem.incidence.shape[0], problem.isotope_count), dtype=float)
        if tv_active
        else np.zeros((0, problem.isotope_count), dtype=float)
    )
    densities_bar = densities.copy()
    nuisance_bar = nuisance.copy()
    previous_check_density = densities.copy()
    previous_check_nuisance = nuisance.copy()
    previous_objective = float("inf")
    relative_change = float("inf")
    relative_objective_change = float("inf")
    kkt_residual = float("inf")
    converged = False
    iterations = 0
    objective_history: list[float] = []
    solve_started = perf_counter()
    if progress_hook is not None:
        progress_hook(
            {
                "phase": str(progress_phase),
                "completed": 0,
                "total": int(solver_config.max_iterations),
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
            }
        )

    for iteration in range(1, int(solver_config.max_iterations) + 1):
        signal_bar = problem.response_by_density @ densities_bar.reshape(-1)
        if nuisance_bar.size:
            signal_bar = signal_bar + problem.nuisance_response @ nuisance_bar
        observation_trial = observation_dual + observation_dual_steps * signal_bar
        observation_dual = _dual_poisson_prox(
            observation_trial,
            observation_dual_steps,
            problem.observed,
            problem.background,
        )
        if tv_active:
            difference_bar = problem.incidence @ densities_bar
            tv_trial = tv_dual + edge_dual_steps[:, None] * difference_bar
            tv_bound = float(solver_config.tv_weight) * problem.edge_weights[:, None]
            tv_dual = np.clip(tv_trial, -tv_bound, tv_bound)

        density_previous = densities
        nuisance_previous = nuisance
        density_gradient = (problem.response_by_density.T @ observation_dual).reshape(
            density_shape
        )
        if tv_active:
            density_gradient = density_gradient + problem.incidence.T @ tv_dual
        densities = np.maximum(
            densities
            - density_steps
            * (
                density_gradient
                + float(solver_config.l1_weight) * problem.patch_areas[:, None]
            ),
            0.0,
        )
        if float(solver_config.isotope_group_weight) > 0.0:
            group_norm = np.linalg.norm(densities, axis=1, keepdims=True)
            group_threshold = (
                density_steps[:, :1]
                * float(solver_config.isotope_group_weight)
                * problem.patch_areas[:, None]
            )
            group_scale = np.maximum(
                1.0 - group_threshold / np.maximum(group_norm, 1.0e-30),
                0.0,
            )
            densities = densities * group_scale
        if nuisance.size:
            nuisance_gradient = problem.nuisance_response.T @ observation_dual
            nuisance = np.maximum(
                nuisance
                - nuisance_steps
                * (nuisance_gradient + float(solver_config.nuisance_l1_weight)),
                0.0,
            ) / (1.0 + nuisance_steps * nuisance_l2)
        relaxation = float(solver_config.over_relaxation)
        densities_bar = densities + relaxation * (densities - density_previous)
        nuisance_bar = nuisance + relaxation * (nuisance - nuisance_previous)
        iterations = int(iteration)

        should_check = iteration % int(
            solver_config.check_interval
        ) == 0 or iteration == int(solver_config.max_iterations)
        if not should_check:
            continue
        density_delta = np.linalg.norm(densities - previous_check_density)
        nuisance_delta = np.linalg.norm(nuisance - previous_check_nuisance)
        state_delta = float(np.hypot(density_delta, nuisance_delta))
        state_norm = float(
            np.hypot(np.linalg.norm(densities), np.linalg.norm(nuisance))
        )
        relative_change = state_delta / max(state_norm, 1.0)
        objective_terms, _expected = _objective_from_prepared(
            problem,
            densities,
            nuisance,
            solver_config,
        )
        if solver_config.kkt_tolerance is not None:
            kkt_residual = _kkt_residual(
                problem,
                densities,
                nuisance,
                tv_dual,
                _expected,
                solver_config,
            )
        objective_history.append(float(objective_terms.total))
        if np.isfinite(previous_objective):
            relative_objective_change = abs(
                float(objective_terms.total) - previous_objective
            ) / max(abs(previous_objective), 1.0)
        previous_check_density = densities.copy()
        previous_check_nuisance = nuisance.copy()
        previous_objective = float(objective_terms.total)
        if progress_hook is not None:
            elapsed = perf_counter() - solve_started
            progress_hook(
                {
                    "phase": str(progress_phase),
                    "completed": int(iteration),
                    "total": int(solver_config.max_iterations),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed
                    * (int(solver_config.max_iterations) - int(iteration))
                    / int(iteration),
                    "relative_change": relative_change,
                    "relative_objective_change": relative_objective_change,
                    "kkt_residual": kkt_residual,
                }
            )
        if relative_change <= float(
            solver_config.tolerance
        ) and relative_objective_change <= float(
            solver_config.objective_tolerance
        ) and (
            solver_config.kkt_tolerance is None
            or kkt_residual <= float(solver_config.kkt_tolerance)
        ):
            converged = True
            break

    objective_terms, expected = _objective_from_prepared(
        problem,
        densities,
        nuisance,
        solver_config,
    )
    kkt_residual = _kkt_residual(
        problem,
        densities,
        nuisance,
        tv_dual,
        expected,
        solver_config,
    )
    integrated = densities * problem.patch_areas[:, None]
    return SurfaceMapResult(
        densities_cps_1m_m2=np.asarray(densities, dtype=float),
        integrated_strengths_cps_1m=np.asarray(integrated, dtype=float),
        nuisance_coefficients=np.asarray(nuisance, dtype=float),
        expected_counts=np.asarray(expected, dtype=float).reshape(
            problem.observation_shape
        ),
        objective=float(objective_terms.total),
        poisson_nll=float(objective_terms.poisson_nll),
        l1_penalty=float(objective_terms.l1_penalty),
        tv_penalty=float(objective_terms.tv_penalty),
        group_penalty=float(objective_terms.group_penalty),
        nuisance_penalty=float(objective_terms.nuisance_penalty),
        deviance=float(objective_terms.deviance),
        converged=bool(converged),
        iterations=int(iterations),
        relative_change=float(relative_change),
        relative_objective_change=float(relative_objective_change),
        kkt_residual=float(kkt_residual),
        objective_history=tuple(objective_history),
    )


@dataclass(frozen=True, slots=True)
class _IndexedCudaResponse:
    """Reuse immutable CUDA rows through an index view without copying them."""

    matrix: object
    row_indices: object


def _torch_response_product(
    operator: ResponseOperator,
    values: object,
    *,
    transpose: bool,
    torch_module: object,
    dense_response: object | None = None,
) -> object:
    """Apply a streamed response operator while retaining state on one device."""
    torch = torch_module
    vector = values
    if isinstance(dense_response, _IndexedCudaResponse):
        matrix = dense_response.matrix
        row_indices = dense_response.row_indices
        if transpose:
            original_rows = torch.zeros(
                int(matrix.shape[0]),
                dtype=vector.dtype,
                device=vector.device,
            )
            original_rows.index_add_(0, row_indices, vector)
            return matrix.T @ original_rows
        return (matrix @ vector).index_select(0, row_indices)
    if dense_response is not None:
        return dense_response.T @ vector if transpose else dense_response @ vector
    if transpose:
        result = torch.zeros(
            operator.source_count,
            dtype=vector.dtype,
            device=vector.device,
        )
        for block in operator.iter_blocks():
            rows = torch.as_tensor(
                np.array(block.observation_indices, dtype=np.int64, copy=True),
                dtype=torch.long,
                device=vector.device,
            )
            columns = torch.as_tensor(
                np.array(block.source_indices, dtype=np.int64, copy=True),
                dtype=torch.long,
                device=vector.device,
            )
            matrix = torch.as_tensor(
                np.array(block.values, dtype=np.float64, copy=True),
                dtype=vector.dtype,
                device=vector.device,
            )
            result.index_add_(0, columns, matrix.T @ vector[rows])
        return result
    result = torch.zeros(
        operator.observation_count,
        dtype=vector.dtype,
        device=vector.device,
    )
    for block in operator.iter_blocks():
        rows = torch.as_tensor(
            np.array(block.observation_indices, dtype=np.int64, copy=True),
            dtype=torch.long,
            device=vector.device,
        )
        columns = torch.as_tensor(
            np.array(block.source_indices, dtype=np.int64, copy=True),
            dtype=torch.long,
            device=vector.device,
        )
        matrix = torch.as_tensor(
            np.array(block.values, dtype=np.float64, copy=True),
            dtype=vector.dtype,
            device=vector.device,
        )
        result.index_add_(0, rows, matrix @ vector[columns])
    return result


def _prune_persistent_cuda_responses(
    persistent_cache: dict[str, object],
    active_identity: tuple[str, str, str],
    *,
    torch_module: object,
    maximum_entries: int = 2,
) -> int:
    """Evict stale dense CUDA responses before another layout is cached.

    Online coarse-to-fine fitting reuses one stable base layout while refined
    patch layouts can change at every station.  Keeping every refined matrix
    resident exhausts VRAM and forces later response kernels into very small
    chunks.  Two LRU entries retain the active base/refinement pair without
    allowing station-by-station growth.
    """
    entries = persistent_cache.get("entries")
    if not isinstance(entries, dict):
        return 0
    if maximum_entries < 1:
        raise ValueError("maximum_entries must be positive.")
    if active_identity in entries:
        active = entries.pop(active_identity)
        entries[active_identity] = active
        retained_before_prepare = maximum_entries
    else:
        retained_before_prepare = maximum_entries - 1
    evicted = 0
    while len(entries) > retained_before_prepare:
        oldest_identity = next(iter(entries))
        del entries[oldest_identity]
        evicted += 1
    if evicted:
        torch_module.cuda.empty_cache()
    return evicted


def _prepare_dense_torch_response(
    operator: ResponseOperator,
    *,
    device: object,
    dtype: object,
    cache_fraction: float,
    torch_module: object,
    persistent_cache: dict[str, object] | None = None,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    progress_phase: str = "mle_cuda_response_cache",
) -> tuple[
    object | None,
    dict[str, object],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Materialize an exact response matrix on CUDA when it safely fits.

    The response remains matrix-free on disk and during construction.  This
    opportunistic device cache only replaces repeated host-to-device block
    transfers inside the iterative solver.  Falling back to streamed blocks is
    deterministic when the required matrix does not fit the configured share
    of currently free device memory.
    """
    torch = torch_module
    required_bytes = (
        int(operator.observation_count)
        * int(operator.source_count)
        * int(torch.empty((), dtype=dtype).element_size())
    )
    diagnostics: dict[str, object] = {
        "mode": "streamed_host_blocks",
        "required_bytes": required_bytes,
        "cached_bytes": 0,
        "preparation_seconds": 0.0,
        "host_to_device_bytes": 0,
    }
    if getattr(device, "type", None) != "cuda" or float(cache_fraction) <= 0.0:
        return None, diagnostics, None, None
    operator_diagnostics = getattr(operator, "diagnostics", {})
    cache_key = operator_diagnostics.get("device_cache_key")
    row_keys_raw = operator_diagnostics.get("measurement_row_keys")
    selected_measurements = operator_diagnostics.get("selected_measurements")
    row_keys = (
        tuple(str(value) for value in row_keys_raw)
        if isinstance(row_keys_raw, list)
        else ()
    )
    complete_selection = selected_measurements is None or selected_measurements == list(
        range(len(row_keys))
    )
    persistent_eligible = bool(
        persistent_cache is not None
        and isinstance(cache_key, str)
        and cache_key
        and row_keys
        and complete_selection
        and not operator_diagnostics.get("source_masked", False)
    )
    persistent_identity = (str(device), str(dtype), str(cache_key))
    previous: dict[str, object] | None = None
    reusable: dict[str, object] | None = None
    if persistent_eligible:
        diagnostics["persistent_cache_entry_limit"] = 2
        diagnostics["persistent_cache_evicted_entries"] = (
            _prune_persistent_cuda_responses(
                persistent_cache,
                persistent_identity,
                torch_module=torch,
            )
        )
        entries = persistent_cache.get("entries")
        candidate = (
            entries.get(persistent_identity) if isinstance(entries, dict) else None
        )
        if (
            isinstance(candidate, dict)
            and candidate.get("identity") == (persistent_identity)
            and int(candidate.get("source_count", -1)) == operator.source_count
        ):
            reusable = candidate
            previous_keys = candidate.get("row_keys")
            if (
                isinstance(previous_keys, tuple)
                and row_keys[: len(previous_keys)] == previous_keys
            ):
                previous = candidate
        if reusable is None and isinstance(entries, dict):
            alternate = next(
                (
                    entry
                    for identity, entry in reversed(tuple(entries.items()))
                    if isinstance(identity, tuple)
                    and len(identity) == 3
                    and identity[0] == str(device)
                    and identity[2] == str(cache_key)
                    and identity != persistent_identity
                    and isinstance(entry, dict)
                    and int(entry.get("source_count", -1))
                    == operator.source_count
                ),
                None,
            )
            if alternate is not None:
                alternate_matrix = alternate.get("matrix")
                alternate_keys = alternate.get("row_keys")
                if (
                    alternate_matrix is not None
                    and isinstance(alternate_keys, tuple)
                    and all(key in alternate_keys for key in row_keys)
                ):
                    torch.cuda.empty_cache()
                    free_bytes, _ = torch.cuda.mem_get_info(device)
                    converted_bytes = int(alternate_matrix.numel()) * int(
                        torch.empty((), dtype=dtype).element_size()
                    )
                    if converted_bytes <= int(float(cache_fraction) * free_bytes):
                        converted_started = perf_counter()
                        try:
                            converted_matrix = alternate_matrix.to(dtype=dtype)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                        else:
                            converted = {
                                **alternate,
                                "identity": persistent_identity,
                                "matrix": converted_matrix,
                            }
                            entries[persistent_identity] = converted
                            reusable = converted
                            if row_keys[: len(alternate_keys)] == alternate_keys:
                                previous = converted
                            diagnostics.update(
                                {
                                    "cross_dtype_cache_reused": True,
                                    "cross_dtype_source": str(
                                        alternate_matrix.dtype
                                    ),
                                    "cross_dtype_conversion_bytes": (
                                        converted_bytes
                                    ),
                                    "cross_dtype_conversion_seconds": (
                                        perf_counter() - converted_started
                                    ),
                                }
                            )
    if previous is not None and previous.get("row_keys") == row_keys:
        cached_matrix = previous.get("matrix")
        cached_rows = previous.get("row_sums")
        cached_columns = previous.get("column_sums")
        if (
            isinstance(cached_rows, np.ndarray)
            and isinstance(cached_columns, np.ndarray)
            and cached_matrix is not None
        ):
            torch.cuda.empty_cache()
            diagnostics.update(
                {
                    "mode": "persistent_cuda_cache_hit",
                    "cached_bytes": required_bytes,
                    "persistent_prefix_measurements": len(row_keys),
                    "allocator_cache_cleared_before_reuse": True,
                }
            )
            return cached_matrix, diagnostics, cached_rows, cached_columns

    if reusable is not None:
        reusable_keys = reusable.get("row_keys")
        reusable_matrix = reusable.get("matrix")
        reusable_rows = reusable.get("row_sums_by_measurement")
        reusable_columns = reusable.get("column_sums_by_measurement")
        if (
            isinstance(reusable_keys, tuple)
            and reusable_matrix is not None
            and isinstance(reusable_rows, np.ndarray)
            and isinstance(reusable_columns, np.ndarray)
            and all(key in reusable_keys for key in row_keys)
        ):
            torch.cuda.empty_cache()
            key_to_index = {key: index for index, key in enumerate(reusable_keys)}
            selected = np.asarray(
                [key_to_index[key] for key in row_keys],
                dtype=np.int64,
            )
            selected_t = torch.as_tensor(
                selected,
                dtype=torch.long,
                device=device,
            )
            trailing = int(np.prod(operator.observation_shape[1:], dtype=np.int64))
            row_indices = (
                selected_t[:, None] * trailing
                + torch.arange(trailing, dtype=torch.long, device=device)[None, :]
            ).reshape(-1)
            indexed = _IndexedCudaResponse(
                matrix=reusable_matrix,
                row_indices=row_indices,
            )
            row_sums = np.asarray(reusable_rows[selected], dtype=np.float64).reshape(-1)
            column_sums = np.sum(
                reusable_columns[selected],
                axis=0,
                dtype=np.float64,
            )
            row_sums.setflags(write=False)
            column_sums.setflags(write=False)
            diagnostics.update(
                {
                    "mode": "persistent_cuda_row_gather",
                    "cached_bytes": required_bytes,
                    "persistent_reused_measurements": len(row_keys),
                    "materialized_row_gather_bytes": 0,
                    "allocator_cache_cleared_before_reuse": True,
                }
            )
            return indexed, diagnostics, row_sums, column_sums

    # Candidate planning can leave large, unreferenced blocks in PyTorch's
    # CUDA caching allocator.  The driver reports those reserved blocks as
    # unavailable, even though PyTorch can release them.  Flush only unused
    # allocator blocks before deciding whether the exact response fits;
    # live tensors, including the persistent response prefix, are preserved.
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    budget_bytes = int(float(cache_fraction) * int(free_bytes))
    diagnostics["free_device_bytes_at_prepare"] = int(free_bytes)
    diagnostics["budget_bytes"] = budget_bytes
    if required_bytes > budget_bytes:
        diagnostics["fallback_reason"] = "response_exceeds_device_cache_budget"
        return None, diagnostics, None, None

    started = perf_counter()
    last_progress_at = started
    matrix = None
    row_sums = np.zeros(operator.observation_count, dtype=np.float64)
    column_sums = np.zeros(operator.source_count, dtype=np.float64)
    trailing = int(np.prod(operator.observation_shape[1:], dtype=np.int64))
    measurement_count = int(operator.observation_shape[0])
    row_sums_by_measurement = np.zeros(
        (measurement_count, trailing),
        dtype=np.float64,
    )
    column_sums_by_measurement = np.zeros(
        (measurement_count, operator.source_count),
        dtype=np.float64,
    )
    old_observation_count = 0
    try:
        matrix = torch.zeros(
            (operator.observation_count, operator.source_count),
            dtype=dtype,
            device=device,
        )
        if previous is not None:
            previous_matrix = previous.get("matrix")
            previous_rows = previous.get("row_sums")
            previous_columns = previous.get("column_sums")
            previous_rows_by_measurement = previous.get("row_sums_by_measurement")
            previous_columns_by_measurement = previous.get("column_sums_by_measurement")
            previous_keys = previous.get("row_keys")
            if (
                previous_matrix is not None
                and isinstance(previous_rows, np.ndarray)
                and isinstance(previous_columns, np.ndarray)
                and isinstance(previous_keys, tuple)
            ):
                old_observation_count = int(previous_rows.size)
                matrix[:old_observation_count].copy_(
                    previous_matrix[:old_observation_count]
                )
                row_sums[:old_observation_count] = previous_rows
                column_sums[:] = previous_columns
                old_measurement_count = len(previous_keys)
                if isinstance(previous_rows_by_measurement, np.ndarray):
                    row_sums_by_measurement[:old_measurement_count] = (
                        previous_rows_by_measurement
                    )
                if isinstance(previous_columns_by_measurement, np.ndarray):
                    column_sums_by_measurement[:old_measurement_count] = (
                        previous_columns_by_measurement
                    )
        element_size = int(torch.empty((), dtype=dtype).element_size())
        completed_bytes = old_observation_count * operator.source_count * element_size
        if progress_hook is not None:
            progress_hook(
                {
                    "phase": str(progress_phase),
                    "completed": int(completed_bytes),
                    "total": int(required_bytes),
                    "elapsed_seconds": 0.0,
                    "eta_seconds": None,
                }
            )
        for block in operator.iter_blocks():
            rows = np.asarray(block.observation_indices, dtype=np.int64)
            columns = np.asarray(block.source_indices, dtype=np.int64)
            if rows.size and int(rows[-1]) < old_observation_count:
                continue
            block_values = torch.tensor(
                block.values,
                dtype=dtype,
                device=device,
            )
            block_row_sums = np.sum(block.values, axis=1, dtype=np.float64)
            block_column_sums = np.sum(block.values, axis=0, dtype=np.float64)
            row_sums[rows] += block_row_sums
            column_sums[columns] += block_column_sums
            measurement_indices = rows // trailing
            local_rows = rows % trailing
            row_sums_by_measurement[
                measurement_indices,
                local_rows,
            ] += block_row_sums
            if measurement_indices.size and np.all(
                measurement_indices == measurement_indices[0]
            ):
                column_sums_by_measurement[
                    int(measurement_indices[0]),
                    columns,
                ] += block_column_sums
            else:
                for measurement_index in np.unique(measurement_indices):
                    selected_rows = measurement_indices == measurement_index
                    column_sums_by_measurement[
                        measurement_index,
                        columns,
                    ] += np.sum(
                        block.values[selected_rows],
                        axis=0,
                        dtype=np.float64,
                    )
            diagnostics["host_to_device_bytes"] = int(
                diagnostics["host_to_device_bytes"]
            ) + int(block.values.nbytes)
            completed_bytes += int(block.values.size) * element_size
            contiguous_rows = bool(
                rows.size
                and np.array_equal(rows, np.arange(rows[0], rows[0] + rows.size))
            )
            contiguous_columns = bool(
                columns.size
                and np.array_equal(
                    columns,
                    np.arange(columns[0], columns[0] + columns.size),
                )
            )
            if contiguous_rows and contiguous_columns:
                matrix[
                    int(rows[0]) : int(rows[-1]) + 1,
                    int(columns[0]) : int(columns[-1]) + 1,
                ].copy_(block_values)
            else:
                row_indices = torch.as_tensor(rows, dtype=torch.long, device=device)
                column_indices = torch.as_tensor(
                    columns,
                    dtype=torch.long,
                    device=device,
                )
                matrix[row_indices[:, None], column_indices[None, :]] = block_values
            now = perf_counter()
            if progress_hook is not None and (
                completed_bytes >= required_bytes or now - last_progress_at >= 1.0
            ):
                elapsed = now - started
                work_done = max(
                    completed_bytes
                    - old_observation_count * operator.source_count * element_size,
                    1,
                )
                work_total = max(
                    required_bytes
                    - old_observation_count * operator.source_count * element_size,
                    0,
                )
                progress_hook(
                    {
                        "phase": str(progress_phase),
                        "completed": min(int(completed_bytes), int(required_bytes)),
                        "total": int(required_bytes),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": (
                            elapsed * max(work_total - work_done, 0) / work_done
                        ),
                    }
                )
                last_progress_at = now
    except torch.OutOfMemoryError:
        del matrix
        torch.cuda.empty_cache()
        diagnostics.update(
            {
                "fallback_reason": "cuda_out_of_memory_during_cache_prepare",
                "preparation_seconds": perf_counter() - started,
            }
        )
        return None, diagnostics, None, None
    torch.cuda.synchronize(device)
    if progress_hook is not None and completed_bytes < required_bytes:
        elapsed = perf_counter() - started
        progress_hook(
            {
                "phase": str(progress_phase),
                "completed": int(required_bytes),
                "total": int(required_bytes),
                "elapsed_seconds": elapsed,
                "eta_seconds": 0.0,
            }
        )
    diagnostics.update(
        {
            "mode": (
                "persistent_cuda_prefix_append"
                if old_observation_count
                else "dense_cuda_cache"
            ),
            "cached_bytes": required_bytes,
            "preparation_seconds": perf_counter() - started,
            "persistent_prefix_measurements": (
                0
                if old_observation_count == 0
                else old_observation_count // int(operator.observation_shape[1])
            ),
        }
    )
    row_sums.setflags(write=False)
    column_sums.setflags(write=False)
    row_sums_by_measurement.setflags(write=False)
    column_sums_by_measurement.setflags(write=False)
    if persistent_eligible and persistent_cache is not None:
        entries = persistent_cache.setdefault("entries", {})
        if not isinstance(entries, dict):
            raise TypeError("Persistent response cache entries must be a dictionary.")
        entries[persistent_identity] = {
            "identity": persistent_identity,
            "row_keys": row_keys,
            "source_count": operator.source_count,
            "matrix": matrix,
            "row_sums": row_sums,
            "column_sums": column_sums,
            "row_sums_by_measurement": row_sums_by_measurement,
            "column_sums_by_measurement": column_sums_by_measurement,
        }
    return matrix, diagnostics, row_sums, column_sums


def _enforce_cuda_response_cache_requirement(
    dense_response: object | None,
    diagnostics: Mapping[str, object],
    *,
    required: bool,
) -> None:
    """Fail before iteration when an explicitly required CUDA cache is absent."""
    if not required or dense_response is not None:
        return
    reason = diagnostics.get("fallback_reason", "unavailable")
    required_bytes = int(diagnostics.get("required_bytes", 0))
    free_bytes = int(diagnostics.get("free_device_bytes_at_prepare", 0))
    budget_bytes = int(diagnostics.get("budget_bytes", 0))
    raise RuntimeError(
        "Exact CUDA response cache is required but unavailable "
        f"(reason={reason}, required_bytes={required_bytes}, "
        f"free_device_bytes={free_bytes}, budget_bytes={budget_bytes}). "
        "Free GPU memory or increase response_device_cache_fraction; "
        "streamed host blocks were not started."
    )


def fit_surface_map_poisson_operator(
    observed_counts: ArrayLike,
    response_operator: ResponseOperator,
    patch_areas_m2: ArrayLike,
    adjacency_edges: ArrayLike | None = None,
    adjacency_weights: ArrayLike | None = None,
    *,
    background: float | ArrayLike = 0.0,
    nuisance_response: ArrayLike | None = None,
    initial_densities_cps_1m_m2: ArrayLike | None = None,
    initial_nuisance_coefficients: ArrayLike | None = None,
    config: SurfaceMapConfig | None = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
    gpu_dtype: str = "float64",
    response_device_cache_fraction: float = 0.6,
    require_gpu_response_cache: bool = False,
    persistent_response_cache: dict[str, object] | None = None,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    progress_phase: str = "mle_solver_iterations",
) -> SurfaceMapResult:
    """Fit Poisson surface density using streamed response products.

    The supplied operator maps density variables directly to expected counts;
    unlike :func:`fit_surface_map_poisson`, patch areas are therefore already
    included in each response block.  Areas remain explicit for L1, group
    regularization, and integrated-strength reporting.  CPU and CUDA execute
    the same Torch primal-dual updates. CUDA opportunistically caches the exact
    response matrix within a bounded share of free VRAM and otherwise streams
    the same response blocks unless ``require_gpu_response_cache`` is enabled.
    """
    if not isinstance(response_operator, ResponseOperator):
        raise TypeError("response_operator must implement ResponseOperator.")
    import torch

    solver_config = SurfaceMapConfig() if config is None else config
    if progress_hook is not None and not callable(progress_hook):
        raise TypeError("progress_hook must be callable or None.")
    observed_array = np.asarray(observed_counts, dtype=np.float64)
    if observed_array.shape != response_operator.observation_shape:
        raise ValueError("observed_counts must match response_operator shape.")
    observed_vector = _as_non_negative_vector(
        observed_array,
        name="observed_counts",
    )
    if solver_config.overdispersion_alpha:
        raw_alpha = np.asarray(
            solver_config.overdispersion_alpha,
            dtype=np.float64,
        )
        if raw_alpha.shape == (response_operator.observation_count,):
            alpha_vector = raw_alpha
        elif len(response_operator.observation_shape) == 2 and raw_alpha.shape == (
            response_operator.observation_shape[1],
        ):
            alpha_vector = np.tile(raw_alpha, response_operator.observation_shape[0])
        else:
            raise ValueError(
                "overdispersion_alpha must match all observations or energy bins."
            )
    else:
        alpha_vector = np.zeros(response_operator.observation_count, dtype=np.float64)
    if solver_config.likelihood_family == "negative_binomial" and not np.any(
        alpha_vector > 0.0
    ):
        raise ValueError("Negative-binomial likelihood requires positive calibration.")
    areas = _as_non_negative_vector(patch_areas_m2, name="patch_areas_m2")
    if areas.shape != (response_operator.patch_count,) or np.any(areas <= 0.0):
        raise ValueError("patch_areas_m2 must contain one positive area per patch.")
    nuisance_matrix = _flatten_nuisance_response(
        nuisance_response,
        response_operator.observation_shape,
    )
    background_vector = _broadcast_observation_vector(
        background,
        response_operator.observation_shape,
        name="background",
    )
    incidence, edge_weights = _canonical_graph(
        adjacency_edges,
        adjacency_weights,
        response_operator.patch_count,
    )
    density_shape = (
        response_operator.patch_count,
        response_operator.isotope_count,
    )
    if initial_densities_cps_1m_m2 is None:
        initial_density = np.zeros(density_shape, dtype=np.float64)
    else:
        initial_density = _as_non_negative_vector(
            initial_densities_cps_1m_m2,
            name="initial_densities_cps_1m_m2",
        )
        if initial_density.size != response_operator.source_count:
            raise ValueError("Initial densities must match patches by isotopes.")
        initial_density = initial_density.reshape(density_shape)
    if solver_config.likelihood_family == "negative_binomial" and not np.any(
        initial_density > 0.0
    ):
        unit_total = float(np.sum(response_operator.row_sums()))
        initial_scale = float(np.sum(observed_vector)) / max(unit_total, 1.0e-12)
        initial_density = np.full(density_shape, initial_scale, dtype=np.float64)
    nuisance_count = nuisance_matrix.shape[1]
    nuisance_l2_values = _nuisance_l2_vector(solver_config, nuisance_count)
    if initial_nuisance_coefficients is None:
        initial_nuisance = np.zeros(nuisance_count, dtype=np.float64)
    else:
        initial_nuisance = _as_non_negative_vector(
            initial_nuisance_coefficients,
            name="initial_nuisance_coefficients",
        )
        if initial_nuisance.size != nuisance_count:
            raise ValueError("Initial nuisance values must match nuisance columns.")
    if gpu_dtype not in {"float32", "float64"}:
        raise ValueError("gpu_dtype must be float32 or float64.")
    if (
        not np.isfinite(response_device_cache_fraction)
        or not 0.0 <= float(response_device_cache_fraction) < 1.0
    ):
        raise ValueError("response_device_cache_fraction must lie in [0, 1).")
    device = torch.device(gpu_device if use_gpu else "cpu")
    if use_gpu and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA matrix-free solve requested but CUDA is unavailable.")
    dtype = torch.float32 if gpu_dtype == "float32" else torch.float64
    solve_started = perf_counter()
    if progress_hook is not None:
        progress_hook(
            {
                "phase": str(progress_phase),
                "completed": 0,
                "total": int(solver_config.max_iterations),
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
            }
        )

    def tensor(values: ArrayLike, *, integer: bool = False) -> object:
        """Copy one bounded array to the selected solver device."""
        return torch.as_tensor(
            np.asarray(values),
            dtype=torch.long if integer else dtype,
            device=device,
        )

    (
        dense_response,
        response_cache_diagnostics,
        prepared_row_sums,
        prepared_column_sums,
    ) = _prepare_dense_torch_response(
        response_operator,
        device=device,
        dtype=dtype,
        cache_fraction=float(response_device_cache_fraction),
        torch_module=torch,
        persistent_cache=persistent_response_cache,
        progress_hook=progress_hook,
        progress_phase=f"{progress_phase}:cuda_response_cache",
    )
    _enforce_cuda_response_cache_requirement(
        dense_response,
        response_cache_diagnostics,
        required=(
            bool(require_gpu_response_cache) and use_gpu and device.type == "cuda"
        ),
    )
    observed = tensor(observed_vector)
    alpha_t = tensor(alpha_vector)
    background_t = tensor(background_vector)
    nuisance_design = tensor(nuisance_matrix)
    areas_t = tensor(areas)
    edge_weights_t = tensor(edge_weights)
    densities = tensor(initial_density).clone()
    nuisance = tensor(initial_nuisance).clone()
    tv_active = bool(float(solver_config.tv_weight) > 0.0 and incidence.shape[0] > 0)
    coo = incidence.tocoo()
    incidence_t = torch.sparse_coo_tensor(
        tensor(np.vstack((coo.row, coo.col)), integer=True),
        tensor(coo.data),
        size=incidence.shape,
        dtype=dtype,
        device=device,
        check_invariants=False,
    ).coalesce()

    if prepared_row_sums is None or prepared_column_sums is None:
        row_sums, flat_column_sums = response_operator.response_sums()
    else:
        row_sums = prepared_row_sums
        flat_column_sums = prepared_column_sums
    if nuisance_count:
        row_sums = row_sums + np.sum(nuisance_matrix, axis=1)
    observation_steps = tensor(
        float(solver_config.step_safety) / np.maximum(row_sums, 1.0e-12)
    )
    column_sums = flat_column_sums.reshape(density_shape)
    if tv_active:
        degrees = np.asarray(np.abs(incidence).sum(axis=0)).reshape(-1)
        column_sums = column_sums + degrees[:, None]
    group_bound = np.max(column_sums, axis=1, keepdims=True)
    density_steps = tensor(
        np.broadcast_to(
            float(solver_config.step_safety) / np.maximum(group_bound, 1.0e-12),
            density_shape,
        ).copy()
    )
    nuisance_steps = tensor(
        (
            float(solver_config.step_safety)
            / np.maximum(np.sum(nuisance_matrix, axis=0), 1.0e-12)
        )
        if nuisance_count
        else np.zeros(0, dtype=np.float64)
    )
    nuisance_l2_t = tensor(nuisance_l2_values)
    edge_steps = (
        torch.full(
            (incidence.shape[0],),
            float(solver_config.step_safety) / 2.0,
            dtype=dtype,
            device=device,
        )
        if tv_active
        else torch.zeros(0, dtype=dtype, device=device)
    )
    observation_dual = torch.zeros(
        response_operator.observation_count,
        dtype=dtype,
        device=device,
    )
    tv_dual = torch.zeros(
        (incidence.shape[0] if tv_active else 0, response_operator.isotope_count),
        dtype=dtype,
        device=device,
    )
    densities_bar = densities.clone()
    nuisance_bar = nuisance.clone()
    previous_density = densities.clone()
    previous_nuisance = nuisance.clone()
    previous_objective = float("inf")
    relative_change = float("inf")
    relative_objective_change = float("inf")
    kkt_residual = float("inf")
    converged = False
    iterations = 0
    objective_history: list[float] = []
    response_product_calls = 0

    def forward(density_values: object) -> object:
        """Apply the streamed density operator on the solver device."""
        nonlocal response_product_calls
        response_product_calls += 1
        return _torch_response_product(
            response_operator,
            density_values.reshape(-1),
            transpose=False,
            torch_module=torch,
            dense_response=dense_response,
        )

    def transpose(observation_values: object) -> object:
        """Apply the streamed transpose on the solver device."""
        nonlocal response_product_calls
        response_product_calls += 1
        return _torch_response_product(
            response_operator,
            observation_values,
            transpose=True,
            torch_module=torch,
            dense_response=dense_response,
        ).reshape(density_shape)

    def expected_counts(density_values: object, nuisance_values: object) -> object:
        """Return the current positive expected-count vector."""
        signal = forward(density_values)
        if nuisance_count:
            signal = signal + nuisance_design @ nuisance_values
        return torch.clamp(
            background_t + signal,
            min=float(solver_config.min_mean),
        )

    def objective_values(
        density_values: object,
        nuisance_values: object,
    ) -> tuple[tuple[float, ...], object]:
        """Return objective components and expected counts on the device."""
        expected = expected_counts(density_values, nuisance_values)
        if solver_config.likelihood_family == "negative_binomial":
            positive_alpha = alpha_t > 0.0
            concentration = torch.where(
                positive_alpha,
                1.0 / torch.clamp(alpha_t, min=1.0e-30),
                torch.ones_like(alpha_t),
            )
            nb_nll = (
                (observed + concentration) * torch.log(concentration + expected)
                - observed * torch.log(expected)
                - concentration * torch.log(concentration)
            )
            poisson_terms = expected - observed * torch.log(expected)
            likelihood = torch.sum(torch.where(positive_alpha, nb_nll, poisson_terms))
        else:
            likelihood = torch.sum(expected - observed * torch.log(expected))
        l1 = float(solver_config.l1_weight) * torch.sum(
            areas_t[:, None] * density_values
        )
        if tv_active:
            differences = torch.sparse.mm(incidence_t, density_values)
            tv = float(solver_config.tv_weight) * torch.sum(
                edge_weights_t[:, None] * torch.abs(differences)
            )
        else:
            tv = torch.zeros((), dtype=dtype, device=device)
        group = float(solver_config.isotope_group_weight) * torch.sum(
            areas_t * torch.linalg.vector_norm(density_values, dim=1)
        )
        nuisance_penalty = float(solver_config.nuisance_l1_weight) * torch.sum(
            nuisance_values
        ) + 0.5 * torch.dot(nuisance_l2_t * nuisance_values, nuisance_values)
        if solver_config.likelihood_family == "negative_binomial":
            saturated_mean = torch.clamp(observed, min=float(solver_config.min_mean))
            saturated_nb = (
                (observed + concentration) * torch.log(concentration + saturated_mean)
                - observed * torch.log(saturated_mean)
                - concentration * torch.log(concentration)
            )
            saturated_poisson = saturated_mean - observed * torch.log(saturated_mean)
            saturated = torch.where(
                positive_alpha,
                saturated_nb,
                saturated_poisson,
            )
            deviance = 2.0 * (likelihood - torch.sum(saturated))
        else:
            positive = observed > 0.0
            log_term = torch.where(
                positive,
                observed * torch.log(torch.clamp(observed, min=1.0e-30) / expected),
                torch.zeros_like(observed),
            )
            deviance = 2.0 * torch.sum(log_term - observed + expected)
        total = likelihood + l1 + tv + group + nuisance_penalty
        return (
            float(total.item()),
            float(likelihood.item()),
            float(l1.item()),
            float(tv.item()),
            float(group.item()),
            float(nuisance_penalty.item()),
            float(deviance.item()),
        ), expected

    def projected_kkt(
        density_values: object,
        nuisance_values: object,
        expected: object,
    ) -> float:
        """Return the projected KKT residual for the current device state."""
        if solver_config.likelihood_family == "negative_binomial":
            likelihood_gradient = (expected - observed) / torch.clamp(
                expected + alpha_t * expected * expected,
                min=float(solver_config.min_mean),
            )
        else:
            likelihood_gradient = 1.0 - observed / expected
        density_gradient = transpose(likelihood_gradient)
        density_gradient = (
            density_gradient + float(solver_config.l1_weight) * areas_t[:, None]
        )
        if float(solver_config.isotope_group_weight) > 0.0:
            norms = torch.linalg.vector_norm(density_values, dim=1, keepdim=True)
            active = norms[:, 0] > 1.0e-12
            density_gradient[active] += (
                float(solver_config.isotope_group_weight)
                * areas_t[active][:, None]
                * density_values[active]
                / norms[active]
            )
        if tv_active:
            density_gradient = density_gradient + torch.sparse.mm(
                incidence_t.transpose(0, 1), tv_dual
            )
        stationarity = torch.where(
            density_values > 1.0e-9,
            density_gradient,
            torch.minimum(density_gradient, torch.zeros_like(density_gradient)),
        ).reshape(-1)
        residual_parts = [stationarity]
        if nuisance_count:
            nuisance_gradient = (
                nuisance_design.T @ likelihood_gradient
                + float(solver_config.nuisance_l1_weight)
                + nuisance_l2_t * nuisance_values
            )
            residual_parts.append(
                torch.where(
                    nuisance_values > 1.0e-9,
                    nuisance_gradient,
                    torch.minimum(
                        nuisance_gradient,
                        torch.zeros_like(nuisance_gradient),
                    ),
                )
            )
        return float(
            (
                torch.linalg.vector_norm(torch.cat(residual_parts))
                / torch.clamp(
                    torch.linalg.vector_norm(likelihood_gradient),
                    min=1.0,
                )
            ).item()
        )

    em_warm_start_elapsed = 0.0
    em_warm_start_iterations_completed = 0
    em_warm_start_iterations = int(
        solver_config.poisson_em_warm_start_iterations
    )
    if (
        solver_config.likelihood_family == "poisson"
        and em_warm_start_iterations > 0
        and np.any(observed_vector > 0.0)
    ):
        total_unit_counts = float(np.sum(flat_column_sums)) + float(
            np.sum(nuisance_matrix)
        )
        target_signal_counts = max(
            float(np.sum(observed_vector) - np.sum(background_vector)),
            0.0,
        )
        if total_unit_counts > 0.0 and target_signal_counts > 0.0:
            initial_scale = target_signal_counts / total_unit_counts
            if np.any(initial_density > 0.0) or np.any(initial_nuisance > 0.0):
                positive_floor = max(
                    initial_scale * 1.0e-12,
                    np.finfo(np.float64).tiny,
                )
                densities = torch.clamp(densities, min=positive_floor)
                nuisance = torch.clamp(nuisance, min=positive_floor)
            else:
                densities.fill_(initial_scale)
                nuisance.fill_(initial_scale)
            density_em_denominator = tensor(
                flat_column_sums.reshape(density_shape)
                + float(solver_config.l1_weight) * areas[:, None]
            )
            nuisance_em_base_denominator = tensor(
                np.sum(nuisance_matrix, axis=0)
                + float(solver_config.nuisance_l1_weight)
            )
            em_started = perf_counter()
            for em_iteration in range(1, em_warm_start_iterations + 1):
                em_expected = expected_counts(densities, nuisance)
                em_ratio = observed / torch.clamp(
                    em_expected,
                    min=float(solver_config.min_mean),
                )
                density_em_numerator = transpose(em_ratio)
                densities *= density_em_numerator / torch.clamp(
                    density_em_denominator,
                    min=torch.finfo(dtype).tiny,
                )
                if nuisance_count:
                    nuisance_em_numerator = nuisance_design.T @ em_ratio
                    nuisance_em_denominator = (
                        nuisance_em_base_denominator + nuisance_l2_t * nuisance
                    )
                    nuisance *= nuisance_em_numerator / torch.clamp(
                        nuisance_em_denominator,
                        min=torch.finfo(dtype).tiny,
                    )
                em_warm_start_iterations_completed = em_iteration
                if progress_hook is not None and (
                    em_iteration == em_warm_start_iterations
                    or em_iteration
                    % max(1, min(20, em_warm_start_iterations))
                    == 0
                ):
                    elapsed = perf_counter() - em_started
                    progress_hook(
                        {
                            "phase": (
                                f"{progress_phase}:poisson_em_warm_start"
                            ),
                            "completed": int(em_iteration),
                            "total": em_warm_start_iterations,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": elapsed
                            * (em_warm_start_iterations - em_iteration)
                            / em_iteration,
                        }
                    )
            em_warm_start_elapsed = perf_counter() - em_started
            densities_bar = densities.clone()
            nuisance_bar = nuisance.clone()
            previous_density = densities.clone()
            previous_nuisance = nuisance.clone()

    for iteration in range(1, int(solver_config.max_iterations) + 1):
        if solver_config.likelihood_family == "negative_binomial":
            expected_current = expected_counts(densities, nuisance)
            likelihood_gradient = (expected_current - observed) / torch.clamp(
                expected_current + alpha_t * expected_current * expected_current,
                min=float(solver_config.min_mean),
            )
            density_old = densities
            nuisance_old = nuisance
            gradient = transpose(likelihood_gradient)
            gradient = gradient + float(solver_config.l1_weight) * areas_t[:, None]
            if float(solver_config.isotope_group_weight) > 0.0:
                norms = torch.linalg.vector_norm(densities, dim=1, keepdim=True)
                active = norms[:, 0] > 1.0e-12
                gradient[active] += (
                    float(solver_config.isotope_group_weight)
                    * areas_t[active][:, None]
                    * densities[active]
                    / norms[active]
                )
            if tv_active:
                differences = torch.sparse.mm(incidence_t, densities)
                tv_dual = (
                    float(solver_config.tv_weight)
                    * edge_weights_t[:, None]
                    * torch.sign(differences)
                )
                gradient = gradient + torch.sparse.mm(
                    incidence_t.transpose(0, 1),
                    tv_dual,
                )
            # Diagonal response-sum preconditioning is damped for the
            # non-quadratic calibrated likelihood.  The diminishing factor is
            # deterministic and prevents large early nuisance corrections.
            damping = min(0.25, 2.5 / math.sqrt(float(iteration)))
            densities = torch.clamp(
                densities - damping * density_steps * gradient,
                min=0.0,
            )
            if nuisance_count:
                nuisance_gradient = (
                    nuisance_design.T @ likelihood_gradient
                    + float(solver_config.nuisance_l1_weight)
                    + nuisance_l2_t * nuisance
                )
                nuisance = torch.clamp(
                    nuisance - damping * nuisance_steps * nuisance_gradient,
                    min=0.0,
                )
            densities_bar = densities
            nuisance_bar = nuisance
            iterations = iteration
            if iteration % int(solver_config.check_interval) != 0 and iteration != int(
                solver_config.max_iterations
            ):
                continue
            state_delta = torch.sqrt(
                torch.sum((densities - previous_density) ** 2)
                + torch.sum((nuisance - previous_nuisance) ** 2)
            )
            state_norm = torch.sqrt(torch.sum(densities**2) + torch.sum(nuisance**2))
            relative_change = float(
                (state_delta / torch.clamp(state_norm, min=1.0)).item()
            )
            terms, expected_check = objective_values(densities, nuisance)
            if solver_config.kkt_tolerance is not None:
                kkt_residual = projected_kkt(
                    densities,
                    nuisance,
                    expected_check,
                )
            objective_history.append(terms[0])
            if np.isfinite(previous_objective):
                relative_objective_change = abs(terms[0] - previous_objective) / max(
                    abs(previous_objective), 1.0
                )
            previous_density = densities.clone()
            previous_nuisance = nuisance.clone()
            previous_objective = terms[0]
            if progress_hook is not None:
                elapsed = perf_counter() - solve_started
                progress_hook(
                    {
                        "phase": str(progress_phase),
                        "completed": int(iteration),
                        "total": int(solver_config.max_iterations),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed
                        * (int(solver_config.max_iterations) - int(iteration))
                        / int(iteration),
                        "relative_change": relative_change,
                        "relative_objective_change": relative_objective_change,
                        "kkt_residual": kkt_residual,
                    }
                )
            if relative_change <= float(
                solver_config.tolerance
            ) and relative_objective_change <= float(
                solver_config.objective_tolerance
            ) and (
                solver_config.kkt_tolerance is None
                or kkt_residual <= float(solver_config.kkt_tolerance)
            ):
                converged = True
                break
            continue
        signal_bar = forward(densities_bar)
        if nuisance_count:
            signal_bar = signal_bar + nuisance_design @ nuisance_bar
        trial = observation_dual + observation_steps * signal_bar
        gamma = 1.0 / observation_steps
        center = trial / observation_steps
        shifted = background_t + center - gamma
        mean = 0.5 * (
            shifted
            + torch.sqrt(
                torch.clamp(shifted * shifted + 4.0 * gamma * observed, min=0.0)
            )
        )
        observation_dual = trial - observation_steps * (mean - background_t)
        if tv_active:
            difference_bar = torch.sparse.mm(incidence_t, densities_bar)
            tv_trial = tv_dual + edge_steps[:, None] * difference_bar
            tv_bound = float(solver_config.tv_weight) * edge_weights_t[:, None]
            tv_dual = torch.maximum(torch.minimum(tv_trial, tv_bound), -tv_bound)

        density_old = densities
        nuisance_old = nuisance
        gradient = transpose(observation_dual)
        if tv_active:
            gradient = gradient + torch.sparse.mm(incidence_t.transpose(0, 1), tv_dual)
        densities = torch.clamp(
            densities
            - density_steps
            * (gradient + float(solver_config.l1_weight) * areas_t[:, None]),
            min=0.0,
        )
        if float(solver_config.isotope_group_weight) > 0.0:
            group_norm = torch.linalg.vector_norm(densities, dim=1, keepdim=True)
            threshold = (
                density_steps[:, :1]
                * float(solver_config.isotope_group_weight)
                * areas_t[:, None]
            )
            densities = densities * torch.clamp(
                1.0 - threshold / torch.clamp(group_norm, min=1.0e-30),
                min=0.0,
            )
        if nuisance_count:
            nuisance_gradient = nuisance_design.T @ observation_dual
            nuisance = torch.clamp(
                nuisance
                - nuisance_steps
                * (nuisance_gradient + float(solver_config.nuisance_l1_weight)),
                min=0.0,
            ) / (1.0 + nuisance_steps * nuisance_l2_t)
        relaxation = float(solver_config.over_relaxation)
        densities_bar = densities + relaxation * (densities - density_old)
        nuisance_bar = nuisance + relaxation * (nuisance - nuisance_old)
        iterations = iteration
        if iteration % int(solver_config.check_interval) != 0 and iteration != int(
            solver_config.max_iterations
        ):
            continue
        state_delta = torch.sqrt(
            torch.sum((densities - previous_density) ** 2)
            + torch.sum((nuisance - previous_nuisance) ** 2)
        )
        state_norm = torch.sqrt(torch.sum(densities**2) + torch.sum(nuisance**2))
        relative_change = float((state_delta / torch.clamp(state_norm, min=1.0)).item())
        terms, expected_check = objective_values(densities, nuisance)
        if solver_config.kkt_tolerance is not None:
            kkt_residual = projected_kkt(
                densities,
                nuisance,
                expected_check,
            )
        objective_history.append(terms[0])
        if np.isfinite(previous_objective):
            relative_objective_change = abs(terms[0] - previous_objective) / max(
                abs(previous_objective), 1.0
            )
        previous_density = densities.clone()
        previous_nuisance = nuisance.clone()
        previous_objective = terms[0]
        if progress_hook is not None:
            elapsed = perf_counter() - solve_started
            progress_hook(
                {
                    "phase": str(progress_phase),
                    "completed": int(iteration),
                    "total": int(solver_config.max_iterations),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed
                    * (int(solver_config.max_iterations) - int(iteration))
                    / int(iteration),
                    "relative_change": relative_change,
                    "relative_objective_change": relative_objective_change,
                    "kkt_residual": kkt_residual,
                }
            )
        if relative_change <= float(
            solver_config.tolerance
        ) and relative_objective_change <= float(
            solver_config.objective_tolerance
        ) and (
            solver_config.kkt_tolerance is None
            or kkt_residual <= float(solver_config.kkt_tolerance)
        ):
            converged = True
            break

    terms, expected = objective_values(densities, nuisance)
    kkt = projected_kkt(densities, nuisance, expected)
    densities_numpy = densities.detach().cpu().numpy().astype(np.float64, copy=False)
    nuisance_numpy = nuisance.detach().cpu().numpy().astype(np.float64, copy=False)
    expected_numpy = expected.detach().cpu().numpy().astype(np.float64, copy=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    performance = getattr(response_operator, "diagnostics", {}).setdefault(
        "performance",
        {},
    )
    if isinstance(performance, dict):
        response_cache_diagnostics[
            "estimated_iterative_host_to_device_bytes_upper_bound"
        ] = (
            0
            if dense_response is not None or device.type != "cuda"
            else int(response_cache_diagnostics["required_bytes"])
            * int(response_product_calls)
        )
        solver_diagnostics = {
            "device": str(device),
            "dtype": str(gpu_dtype),
            "elapsed_seconds": perf_counter() - solve_started,
            "response_product_calls": int(response_product_calls),
            "poisson_em_warm_start_iterations": int(
                em_warm_start_iterations_completed
            ),
            "poisson_em_warm_start_elapsed_seconds": float(
                em_warm_start_elapsed
            ),
            "response_cache": response_cache_diagnostics,
        }
        performance["solver"] = solver_diagnostics
        solver_calls = performance.setdefault("solver_calls", [])
        if isinstance(solver_calls, list):
            solver_calls.append(solver_diagnostics)
    return SurfaceMapResult(
        densities_cps_1m_m2=densities_numpy,
        integrated_strengths_cps_1m=densities_numpy * areas[:, None],
        nuisance_coefficients=nuisance_numpy,
        expected_counts=expected_numpy.reshape(response_operator.observation_shape),
        objective=terms[0],
        poisson_nll=terms[1],
        l1_penalty=terms[2],
        tv_penalty=terms[3],
        group_penalty=terms[4],
        nuisance_penalty=terms[5],
        deviance=terms[6],
        converged=converged,
        iterations=iterations,
        relative_change=relative_change,
        relative_objective_change=relative_objective_change,
        kkt_residual=kkt,
        objective_history=tuple(objective_history),
    )
