"""All-history count and line-resolved spectral surface MLE estimators."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from runtime.discrepancy_calibration import load_discrepancy_calibration

from .count_covariance import fit_surface_map_count_covariance
from .config import MLEConfig
from .model_selection import (
    RegularizationCVResult,
    RegularizationCandidate,
    grouped_kfold_indices,
    select_regularization_one_standard_error,
)
from .postprocess import (
    cluster_surface_hotspots,
    poisson_deviance,
    response_identifiability_diagnostics,
)
from .provenance import estimator_provenance
from .response_builder import build_count_responses
from .response_operator import ResponseOperator
from .solver import (
    SurfaceMapConfig,
    SurfaceMapResult,
    fit_surface_map_poisson,
    fit_surface_map_poisson_operator,
)
from .spectral_response_builder import (
    SpectralResponseOperatorResult,
    SpectralResponseResult,
    build_spectral_response,
    build_spectral_response_operator,
)
from .surface_patches import build_surface_patches, refine_surface_patches
from .types import MLEEstimate, ObservationBatch, SurfacePatch, SurfacePatchSet
from .uncertainty import (
    active_support_laplace,
    augment_clusters_with_laplace,
    bootstrap_uncertainty_summary,
    station_bootstrap_batch,
)


@dataclass(frozen=True)
class _FitState:
    """Store one solver result and the response used to obtain it."""

    patches: SurfacePatchSet
    response: NDArray[np.float64] | ResponseOperator
    nuisance_response: NDArray[np.float64]
    nuisance_names: tuple[str, ...]
    nuisance_l2_weights: NDArray[np.float64]
    overdispersion_alpha_by_bin: NDArray[np.float64]
    result: SurfaceMapResult
    fit_indices: NDArray[np.int64]
    held_out_indices: NDArray[np.int64]
    spectral_details: SpectralResponseResult | SpectralResponseOperatorResult | None
    likelihood_diagnostics: dict[str, object]


def _surface_map_config(
    config: MLEConfig,
    *,
    regularized: bool = True,
    nuisance_l2_weights: NDArray[np.float64] | None = None,
    overdispersion_alpha_by_bin: NDArray[np.float64] | None = None,
) -> SurfaceMapConfig:
    """Translate public MLE settings into the numerical solver contract."""
    return SurfaceMapConfig(
        l1_weight=float(config.l1_weight) if regularized else 0.0,
        tv_weight=float(config.tv_weight) if regularized else 0.0,
        isotope_group_weight=(
            float(config.isotope_group_weight) if regularized else 0.0
        ),
        nuisance_l1_weight=float(config.nuisance_l1_weight),
        nuisance_l2_weight=float(config.nuisance_l2_weight),
        nuisance_l2_weights=(
            ()
            if nuisance_l2_weights is None
            else tuple(
                float(value) + float(config.nuisance_l2_weight)
                for value in np.asarray(nuisance_l2_weights, dtype=float)
            )
        ),
        likelihood_family=(
            "negative_binomial"
            if config.spectral_likelihood == "calibrated_overdispersed"
            else "poisson"
        ),
        overdispersion_alpha=(
            ()
            if overdispersion_alpha_by_bin is None
            else tuple(
                float(value)
                for value in np.asarray(overdispersion_alpha_by_bin, dtype=float)
            )
        ),
        max_iterations=int(config.max_iterations),
        tolerance=float(config.tolerance),
        objective_tolerance=float(config.objective_tolerance),
        check_interval=int(config.check_interval),
        step_safety=float(config.step_safety),
        over_relaxation=float(config.over_relaxation),
        min_mean=float(config.min_mean),
    )


def _union_group_labels(
    batch: ObservationBatch, grouping: str, tolerance: float
) -> tuple[int, ...]:
    """Return row component labels while keeping every station intact."""
    measurement_count = batch.measurement_count
    if grouping == "row":
        return tuple(range(measurement_count))
    parent = list(range(measurement_count))

    def find(index: int) -> int:
        """Return one disjoint-set root with path compression."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        """Merge two deterministic disjoint-set components."""
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        lower, upper = sorted((first_root, second_root))
        parent[upper] = lower

    station_first: dict[int, int] = {}
    for row, station_id in enumerate(batch.station_ids):
        station = int(station_id)
        if station in station_first:
            union(row, station_first[station])
        else:
            station_first[station] = row

    if grouping == "same_xy_height":
        xy = np.asarray(batch.detector_positions_xyz[:, :2], dtype=float)
        for first in range(measurement_count):
            distances = np.linalg.norm(xy[first + 1 :] - xy[first], axis=1)
            for offset in np.flatnonzero(distances <= float(tolerance)):
                union(first, first + 1 + int(offset))
    elif grouping == "shield_program_block":
        block_first: dict[str, int] = {}
        for row, block_id in enumerate(batch.shield_program_block_ids):
            if block_id in block_first:
                union(row, block_first[block_id])
            else:
                block_first[block_id] = row
    elif grouping != "station_id":
        raise ValueError(f"Unsupported held-out grouping: {grouping!r}.")

    canonical_roots: dict[int, int] = {}
    labels: list[int] = []
    for row in range(measurement_count):
        root = find(row)
        canonical_roots.setdefault(root, len(canonical_roots))
        labels.append(canonical_roots[root])
    return tuple(labels)


def _split_fit_indices(
    batch: ObservationBatch,
    fraction: float,
    seed: int,
    *,
    grouping: str,
    xy_tolerance_m: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], tuple[int, ...]]:
    """Return a deterministic whole-group fit/held-out split.

    A small subset-sum search chooses the group combination closest to the
    requested row fraction. The randomized group order is seed-controlled and
    only resolves equally good choices; rows within a related group never split.
    """
    measurement_count = batch.measurement_count
    all_indices = np.arange(measurement_count, dtype=np.int64)
    group_labels = _union_group_labels(batch, grouping, xy_tolerance_m)
    held_out_target = int(np.floor(float(fraction) * measurement_count))
    groups: dict[int, list[int]] = {}
    for row, label in enumerate(group_labels):
        groups.setdefault(label, []).append(row)
    if held_out_target <= 0 or len(groups) < 2:
        return all_indices, np.zeros(0, dtype=np.int64), group_labels

    ordered_labels = list(groups)
    rng = np.random.default_rng(int(seed))
    shuffled_labels = [
        ordered_labels[int(index)] for index in rng.permutation(len(ordered_labels))
    ]
    possibilities: dict[int, tuple[int, ...]] = {0: ()}
    for label in shuffled_labels:
        size = len(groups[label])
        additions: dict[int, tuple[int, ...]] = {}
        for row_count, selected in possibilities.items():
            candidate_count = row_count + size
            if candidate_count >= measurement_count:
                continue
            additions.setdefault(candidate_count, (*selected, label))
        for row_count, selected in additions.items():
            possibilities.setdefault(row_count, selected)
    feasible_counts = [
        count for count in possibilities if 0 < count < measurement_count
    ]
    if not feasible_counts:
        return all_indices, np.zeros(0, dtype=np.int64), group_labels
    selected_count = min(
        feasible_counts,
        key=lambda count: (
            abs(count - held_out_target),
            count > held_out_target,
            count,
        ),
    )
    selected_labels = set(possibilities[selected_count])
    held_out = np.asarray(
        [row for row, label in enumerate(group_labels) if label in selected_labels],
        dtype=np.int64,
    )
    fit = np.setdiff1d(all_indices, held_out, assume_unique=True)
    return fit, held_out, group_labels


def _count_solver_response(
    count_response: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Embed isotope-resolved count columns in a diagonal channel tensor."""
    isotope_count = int(count_response.shape[-1])
    identity = np.eye(isotope_count, dtype=float)
    return count_response[:, None, :, :] * identity[None, :, None, :]


def _count_nuisance_response(
    live_times_s: NDArray[np.float64],
    isotope_count: int,
    enabled: bool,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Return one isotope-specific background-rate basis per count channel."""
    measurement_count = int(np.asarray(live_times_s).size)
    if not enabled:
        return np.zeros((measurement_count, isotope_count, 0), dtype=float), ()
    identity = np.eye(isotope_count, dtype=float)
    basis = np.asarray(live_times_s, dtype=float)[:, None, None] * identity[None, :, :]
    names = tuple(f"background_rate_cps:{index}" for index in range(isotope_count))
    return basis, names


def _fit_problem(
    batch: ObservationBatch,
    patches: SurfacePatchSet,
    kernel: ContinuousKernel,
    config: MLEConfig,
    fit_indices: NDArray[np.int64],
    held_out_indices: NDArray[np.int64],
    *,
    initial_densities: NDArray[np.float64] | None = None,
    initial_nuisance: NDArray[np.float64] | None = None,
    persistent_response_cache: dict[str, object] | None = None,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    progress_label: str = "base",
) -> _FitState:
    """Build the configured forward model and solve one patch resolution."""
    response_started = perf_counter()
    if progress_hook is not None:
        progress_hook(
            {
                "phase": f"mle_response:{progress_label}",
                "completed": 0,
                "total": 1,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
            }
        )
    if config.mode == "count":
        if batch.isotope_counts is None:
            raise ValueError(
                "Count-domain MLE requires response_poisson isotope counts."
            )
        count_details = build_count_responses(
            batch,
            patches,
            config.isotope_names,
            kernel,
            kernel_chunk_size=int(config.response_chunk_size),
        )
        response = _count_solver_response(count_details.response_by_integrated_strength)
        nuisance_response, nuisance_names = _count_nuisance_response(
            batch.live_times_s,
            batch.isotope_count,
            bool(config.fit_background_nuisance),
        )
        nuisance_l2_weights = np.zeros(len(nuisance_names), dtype=np.float64)
        overdispersion_alpha = np.zeros(0, dtype=np.float64)
        likelihood_diagnostics = {"family": config.count_likelihood}
        observed = batch.isotope_counts
        spectral_details = None
    elif config.spectral_response_mode == "matrix_free":
        calibration = (
            None
            if config.discrepancy_calibration_path is None
            else load_discrepancy_calibration(config.discrepancy_calibration_path)
        )
        spectral_details = build_spectral_response_operator(
            batch,
            patches,
            config.isotope_names,
            kernel,
            chunk_size=int(config.response_chunk_size),
            measurement_chunk_size=int(config.response_measurement_chunk_size),
            energy_chunk_size=int(config.response_energy_chunk_size),
            patch_chunk_size=int(config.response_patch_chunk_size),
            worker_count=int(config.response_worker_count),
            cache_directory=config.response_cache_dir,
            continuum_to_peak=float(config.continuum_to_peak),
            backscatter_fraction=float(config.backscatter_fraction),
            require_line_resolved=True,
            include_background_nuisance=bool(config.fit_background_nuisance),
            include_scatter_nuisance=bool(config.fit_scatter_nuisance),
            discrepancy_calibration=calibration,
            include_shield_leakage_nuisance=bool(config.fit_shield_leakage_nuisance),
            include_station_rate_nuisance=bool(config.fit_station_rate_nuisance),
            include_low_rank_residual_nuisance=bool(
                config.fit_low_rank_residual_nuisance
            ),
            include_gain_resolution_drift=bool(config.fit_gain_resolution_drift),
        )
        response = spectral_details.operator
        nuisance_response = spectral_details.nuisance_response
        nuisance_names = spectral_details.nuisance_names
        nuisance_l2_weights = spectral_details.nuisance_l2_weights
        overdispersion_alpha = spectral_details.overdispersion_alpha_by_bin
        observed = batch.spectrum_counts
        likelihood_diagnostics = {
            "family": config.spectral_likelihood,
            "calibration_path": config.discrepancy_calibration_path,
        }
    else:
        calibration = (
            None
            if config.discrepancy_calibration_path is None
            else load_discrepancy_calibration(config.discrepancy_calibration_path)
        )
        spectral_details = build_spectral_response(
            batch,
            patches,
            config.isotope_names,
            kernel,
            chunk_size=int(config.response_chunk_size),
            continuum_to_peak=float(config.continuum_to_peak),
            backscatter_fraction=float(config.backscatter_fraction),
            require_line_resolved=True,
            include_background_nuisance=bool(config.fit_background_nuisance),
            include_scatter_nuisance=bool(config.fit_scatter_nuisance),
            discrepancy_calibration=calibration,
            include_shield_leakage_nuisance=bool(config.fit_shield_leakage_nuisance),
            include_station_rate_nuisance=bool(config.fit_station_rate_nuisance),
            include_low_rank_residual_nuisance=bool(
                config.fit_low_rank_residual_nuisance
            ),
            include_gain_resolution_drift=bool(config.fit_gain_resolution_drift),
        )
        response = spectral_details.response_per_integrated_strength
        nuisance_response = spectral_details.nuisance_response
        nuisance_names = spectral_details.nuisance_names
        nuisance_l2_weights = spectral_details.nuisance_l2_weights
        overdispersion_alpha = spectral_details.overdispersion_alpha_by_bin
        observed = batch.spectrum_counts
        likelihood_diagnostics = {
            "family": config.spectral_likelihood,
            "calibration_path": config.discrepancy_calibration_path,
        }

    if progress_hook is not None:
        response_elapsed = perf_counter() - response_started
        progress_hook(
            {
                "phase": f"mle_response:{progress_label}",
                "completed": 1,
                "total": 1,
                "elapsed_seconds": response_elapsed,
                "eta_seconds": 0.0,
            }
        )
    solver_started = perf_counter()
    if progress_hook is not None:
        progress_hook(
            {
                "phase": f"mle_solver:{progress_label}",
                "completed": 0,
                "total": 1,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
            }
        )

    if config.mode == "count" and config.count_likelihood != "poisson":
        if batch.isotope_covariances is None:
            raise ValueError(
                "Covariance-aware count likelihood requires isotope_covariances."
            )
        fitted, covariance_diagnostics = fit_surface_map_count_covariance(
            observed[fit_indices],
            count_details.response_by_integrated_strength[fit_indices],
            batch.isotope_covariances[fit_indices],
            patches.areas_m2,
            adjacency_edges=patches.adjacency_index_edges,
            adjacency_weights=patches.shared_edge_lengths_m,
            nuisance_response=nuisance_response[fit_indices],
            initial_densities_cps_1m_m2=initial_densities,
            initial_nuisance_coefficients=initial_nuisance,
            likelihood_family=config.count_likelihood,
            student_t_degrees_of_freedom=float(
                config.count_student_t_degrees_of_freedom
            ),
            covariance_regularization=float(config.count_covariance_regularization),
            maximum_condition_number=float(
                config.count_covariance_max_condition_number
            ),
            config=_surface_map_config(
                config,
                nuisance_l2_weights=nuisance_l2_weights,
            ),
        )
        likelihood_diagnostics.update(
            {
                "covariance_regularization": (
                    covariance_diagnostics.covariance_regularization
                ),
                "maximum_condition_number": (
                    covariance_diagnostics.maximum_condition_number
                ),
                "condition_numbers": list(covariance_diagnostics.condition_numbers),
            }
        )
    elif isinstance(response, ResponseOperator):
        fitted = fit_surface_map_poisson_operator(
            observed[fit_indices],
            response.select_measurements(fit_indices.tolist()),
            patches.areas_m2,
            adjacency_edges=patches.adjacency_index_edges,
            adjacency_weights=patches.shared_edge_lengths_m,
            background=0.0,
            nuisance_response=nuisance_response[fit_indices],
            initial_densities_cps_1m_m2=initial_densities,
            initial_nuisance_coefficients=initial_nuisance,
            config=_surface_map_config(
                config,
                nuisance_l2_weights=nuisance_l2_weights,
                overdispersion_alpha_by_bin=overdispersion_alpha,
            ),
            use_gpu=bool(config.use_gpu),
            gpu_device=str(config.gpu_device),
            gpu_dtype=str(config.gpu_dtype),
            response_device_cache_fraction=float(config.response_device_cache_fraction),
            persistent_response_cache=persistent_response_cache,
            progress_hook=progress_hook,
            progress_phase=f"mle_solver_iterations:{progress_label}",
        )
    else:
        fitted = fit_surface_map_poisson(
            observed[fit_indices],
            response[fit_indices],
            patches.areas_m2,
            adjacency_edges=patches.adjacency_index_edges,
            adjacency_weights=patches.shared_edge_lengths_m,
            background=0.0,
            nuisance_response=nuisance_response[fit_indices],
            initial_densities_cps_1m_m2=initial_densities,
            initial_nuisance_coefficients=initial_nuisance,
            config=_surface_map_config(
                config,
                nuisance_l2_weights=nuisance_l2_weights,
                overdispersion_alpha_by_bin=overdispersion_alpha,
            ),
            progress_hook=progress_hook,
            progress_phase=f"mle_solver_iterations:{progress_label}",
        )
    if progress_hook is not None:
        solver_elapsed = perf_counter() - solver_started
        progress_hook(
            {
                "phase": f"mle_solver:{progress_label}",
                "completed": 1,
                "total": 1,
                "elapsed_seconds": solver_elapsed,
                "eta_seconds": 0.0,
            }
        )
    return _FitState(
        patches=patches,
        response=response,
        nuisance_response=nuisance_response,
        nuisance_names=nuisance_names,
        nuisance_l2_weights=nuisance_l2_weights,
        overdispersion_alpha_by_bin=overdispersion_alpha,
        result=fitted,
        fit_indices=fit_indices,
        held_out_indices=held_out_indices,
        spectral_details=spectral_details,
        likelihood_diagnostics=likelihood_diagnostics,
    )


def _warm_start_refined(
    old_patches: SurfacePatchSet,
    new_patches: SurfacePatchSet,
    old_densities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transfer density, not total strength, to unchanged and child patches."""
    old_by_id = {
        patch.patch_id: np.asarray(old_densities[index], dtype=float)
        for index, patch in enumerate(old_patches.patches)
    }
    result = np.zeros((new_patches.patch_count, old_densities.shape[1]), dtype=float)
    for index, patch in enumerate(new_patches.patches):
        source_id = (
            patch.patch_id if patch.patch_id in old_by_id else patch.parent_patch_id
        )
        if source_id is not None and source_id in old_by_id:
            result[index] = old_by_id[source_id]
    return result


def _refinement_patch_ids(
    state: _FitState,
    fraction: float,
    maximum_patches: int,
) -> tuple[int, ...]:
    """Select bounded isotope-balanced support for local surface refinement."""
    strengths = np.asarray(
        state.result.integrated_strengths_cps_1m,
        dtype=np.float64,
    )
    if not np.any(strengths > 0.0) or float(fraction) <= 0.0:
        return ()
    normalized = strengths / np.maximum(
        np.max(strengths, axis=0, keepdims=True),
        1.0e-30,
    )
    selected: set[int] = set()
    for isotope_index in range(strengths.shape[1]):
        positive = np.flatnonzero(strengths[:, isotope_index] > 0.0)
        if not positive.size:
            continue
        selected_count = max(1, int(np.ceil(float(fraction) * positive.size)))
        order = positive[
            np.argsort(normalized[positive, isotope_index], kind="stable")[::-1][
                :selected_count
            ]
        ]
        selected.update(int(index) for index in order)
    ranked = sorted(
        selected,
        key=lambda index: (
            -float(np.max(normalized[index])),
            int(state.patches.patches[index].patch_id),
        ),
    )[: int(maximum_patches)]
    return tuple(state.patches.patches[index].patch_id for index in ranked)


def _bounded_debias_support(
    state: _FitState,
    config: MLEConfig,
) -> NDArray[np.bool_]:
    """Return a bounded isotope-balanced support mask for shrinkage removal."""
    strengths = np.asarray(
        state.result.integrated_strengths_cps_1m,
        dtype=np.float64,
    )
    maxima = np.max(strengths, axis=0, keepdims=True)
    relative = strengths / np.maximum(maxima, 1.0e-30)
    eligible = (
        (strengths > 0.0)
        & (relative >= float(config.support_threshold_fraction))
        & (maxima > 0.0)
    )
    flat_eligible = np.flatnonzero(eligible.reshape(-1))
    limit = int(config.debias_max_active_parameters)
    if flat_eligible.size <= limit:
        return eligible

    selected: set[int] = set()
    for isotope_index in range(strengths.shape[1]):
        candidates = np.flatnonzero(eligible[:, isotope_index])
        if candidates.size:
            strongest = int(candidates[np.argmax(relative[candidates, isotope_index])])
            selected.add(strongest * strengths.shape[1] + isotope_index)
    ranked = sorted(
        (int(index) for index in flat_eligible if int(index) not in selected),
        key=lambda index: (
            -float(relative.reshape(-1)[index]),
            index,
        ),
    )
    selected.update(ranked[: max(limit - len(selected), 0)])
    support = np.zeros(strengths.size, dtype=bool)
    support[list(selected)] = True
    return support.reshape(strengths.shape)


def _debias_state(
    state: _FitState,
    observed: NDArray[np.float64],
    config: MLEConfig,
    persistent_response_cache: dict[str, object] | None = None,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
) -> _FitState:
    """Refit selected support without L1, TV, or group shrinkage bias."""
    if state.likelihood_diagnostics.get("family") in {
        "covariance_gaussian",
        "multivariate_student_t",
    }:
        # The count-covariance path is diagnostic and must never be silently
        # replaced by a Poisson support refit.
        return state
    if bool(config.debias_requires_convergence) and not state.result.converged:
        return replace(
            state,
            likelihood_diagnostics={
                **state.likelihood_diagnostics,
                "debias_applied": False,
                "debias_skip_reason": "regularized_fit_not_converged",
            },
        )
    fitted_prediction = _full_prediction(state)[state.fit_indices]
    fitted_observed = np.asarray(observed[state.fit_indices], dtype=np.float64)
    pearson_dispersion = float(
        np.mean(
            np.square(fitted_observed - fitted_prediction)
            / np.maximum(fitted_prediction, 1.0)
        )
    )
    if pearson_dispersion > float(config.debias_max_pearson_dispersion):
        return replace(
            state,
            likelihood_diagnostics={
                **state.likelihood_diagnostics,
                "debias_applied": False,
                "debias_skip_reason": "pearson_dispersion_exceeds_limit",
                "pearson_dispersion": pearson_dispersion,
                "debias_max_pearson_dispersion": float(
                    config.debias_max_pearson_dispersion
                ),
            },
        )
    densities = state.result.densities_cps_1m_m2
    support = _bounded_debias_support(state, config)
    if not np.any(support):
        return state
    if isinstance(state.response, ResponseOperator):
        masked_response = state.response.masked_sources(support)
        result = fit_surface_map_poisson_operator(
            observed[state.fit_indices],
            masked_response.select_measurements(state.fit_indices.tolist()),
            state.patches.areas_m2,
            adjacency_edges=state.patches.adjacency_index_edges,
            adjacency_weights=state.patches.shared_edge_lengths_m,
            background=0.0,
            nuisance_response=state.nuisance_response[state.fit_indices],
            initial_densities_cps_1m_m2=np.where(support, densities, 0.0),
            initial_nuisance_coefficients=state.result.nuisance_coefficients,
            config=_surface_map_config(
                config,
                regularized=False,
                nuisance_l2_weights=state.nuisance_l2_weights,
                overdispersion_alpha_by_bin=state.overdispersion_alpha_by_bin,
            ),
            use_gpu=bool(config.use_gpu),
            gpu_device=str(config.gpu_device),
            gpu_dtype=str(config.gpu_dtype),
            response_device_cache_fraction=float(config.response_device_cache_fraction),
            persistent_response_cache=persistent_response_cache,
            progress_hook=progress_hook,
            progress_phase="mle_solver_iterations:debias",
        )
        if bool(config.debias_requires_convergence) and not result.converged:
            return replace(
                state,
                likelihood_diagnostics={
                    **state.likelihood_diagnostics,
                    "debias_applied": False,
                    "debias_skip_reason": "debias_fit_not_converged",
                    "debias_active_parameter_count": int(np.count_nonzero(support)),
                },
            )
        return replace(
            state,
            response=masked_response,
            result=result,
            likelihood_diagnostics={
                **state.likelihood_diagnostics,
                "debias_applied": True,
                "debias_active_parameter_count": int(np.count_nonzero(support)),
            },
        )
    if state.response.ndim == 4:
        masked_response = state.response * support[None, None, :, :]
    else:  # pragma: no cover - all production response tensors are rank four
        masked_response = state.response * support[None, :, :]
    result = fit_surface_map_poisson(
        observed[state.fit_indices],
        masked_response[state.fit_indices],
        state.patches.areas_m2,
        adjacency_edges=state.patches.adjacency_index_edges,
        adjacency_weights=state.patches.shared_edge_lengths_m,
        background=0.0,
        nuisance_response=state.nuisance_response[state.fit_indices],
        initial_densities_cps_1m_m2=np.where(support, densities, 0.0),
        initial_nuisance_coefficients=state.result.nuisance_coefficients,
        config=_surface_map_config(
            config,
            regularized=False,
            nuisance_l2_weights=state.nuisance_l2_weights,
            overdispersion_alpha_by_bin=state.overdispersion_alpha_by_bin,
        ),
        progress_hook=progress_hook,
        progress_phase="mle_solver_iterations:debias",
    )
    if bool(config.debias_requires_convergence) and not result.converged:
        return replace(
            state,
            likelihood_diagnostics={
                **state.likelihood_diagnostics,
                "debias_applied": False,
                "debias_skip_reason": "debias_fit_not_converged",
                "debias_active_parameter_count": int(np.count_nonzero(support)),
            },
        )
    return replace(
        state,
        response=masked_response,
        result=result,
        likelihood_diagnostics={
            **state.likelihood_diagnostics,
            "debias_applied": True,
            "debias_active_parameter_count": int(np.count_nonzero(support)),
        },
    )


def _full_prediction(state: _FitState) -> NDArray[np.float64]:
    """Evaluate the fitted density and nuisance model for every measurement."""
    if isinstance(state.response, ResponseOperator):
        expected = state.response.matvec(
            state.result.densities_cps_1m_m2.reshape(-1)
        ).reshape(state.response.observation_shape)
    else:
        strengths = state.result.integrated_strengths_cps_1m
        expected = np.einsum("...gi,gi->...", state.response, strengths, optimize=True)
    if state.result.nuisance_coefficients.size:
        expected = expected + np.einsum(
            "...n,n->...",
            state.nuisance_response,
            state.result.nuisance_coefficients,
            optimize=True,
        )
    return np.maximum(expected, 0.0)


def _operator_identifiability(
    operator: ResponseOperator,
    densities: NDArray[np.float64],
    fit_indices: NDArray[np.int64],
    threshold: float,
) -> dict[str, object]:
    """Return bounded active-support response-correlation diagnostics."""
    maximum = np.max(densities, axis=0, keepdims=True)
    active_mask = (densities > 0.0) & (densities >= maximum * 1.0e-3)
    active = np.flatnonzero(active_mask.reshape(-1))
    if active.size == 0:
        return {
            "matrix_free": True,
            "active_column_count": 0,
            "maximum_column_correlation": None,
            "high_correlation_pairs": [],
        }
    # Keep diagnostics bounded even when regularization leaves a diffuse tail.
    if active.size > 256:
        values = densities.reshape(-1)[active]
        active = active[np.argsort(values)[-256:]]
    selected = operator.select_measurements(fit_indices.tolist())
    gram = np.zeros((active.size, active.size), dtype=np.float64)
    active_lookup = np.full(operator.source_count, -1, dtype=np.int64)
    active_lookup[active] = np.arange(active.size, dtype=np.int64)
    for block in selected.iter_blocks():
        local = active_lookup[block.source_indices]
        keep = local >= 0
        if not np.any(keep):
            continue
        values = block.values[:, keep]
        indices = local[keep]
        gram[np.ix_(indices, indices)] += values.T @ values
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, 0.0)
    pairs = np.argwhere(np.triu(correlation >= float(threshold), k=1))
    return {
        "matrix_free": True,
        "active_column_count": int(active.size),
        "active_source_indices": active.astype(int).tolist(),
        "maximum_column_correlation": (
            float(np.max(correlation)) if active.size > 1 else 0.0
        ),
        "high_correlation_pairs": [
            [int(active[first]), int(active[second])] for first, second in pairs[:1024]
        ],
        "high_correlation_pair_count": int(pairs.shape[0]),
    }


def _json_floats(values: NDArray[np.float64]) -> list[float]:
    """Return a flat JSON-safe float list."""
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _initial_density_for_patches(
    initial_estimate: MLEEstimate | None,
    patches: SurfacePatchSet,
    isotope_names: Sequence[str],
) -> NDArray[np.float64] | None:
    """Map a prior estimate onto a newly rebuilt deterministic base patch set."""
    if initial_estimate is None:
        return None
    if tuple(initial_estimate.isotope_names) != tuple(isotope_names):
        raise ValueError("Warm-start isotope order must match MLEConfig.isotope_names.")
    old_density = np.asarray(initial_estimate.density_by_isotope, dtype=float).T
    old_by_id = {
        patch.patch_id: (patch, old_density[index])
        for index, patch in enumerate(initial_estimate.patches)
    }

    def is_descendant(candidate: SurfacePatch, ancestor: SurfacePatch) -> bool:
        """Return whether one active leaf belongs below a rebuilt base patch.

        Refined estimates retain only active leaves, so the immediate parent of
        a level-two-or-deeper leaf is no longer present.  Stable lineage handles
        direct children; exact rectangular containment recovers the remainder
        while checking the same face identity, orientation, and refinement
        direction.
        """
        if candidate.parent_patch_id == ancestor.patch_id:
            return True
        if (
            candidate.parent_patch_id is None
            or candidate.refinement_level <= ancestor.refinement_level
            or candidate.surface_kind != ancestor.surface_kind
            or candidate.object_id != ancestor.object_id
            or not np.allclose(
                candidate.normal_xyz,
                ancestor.normal_xyz,
                rtol=1.0e-9,
                atol=1.0e-9,
            )
        ):
            return False

        origin = ancestor.vertices_xyz[0]
        u_vector = ancestor.vertices_xyz[1] - origin
        v_vector = ancestor.vertices_xyz[3] - origin
        relative = candidate.vertices_xyz - origin
        scale = max(
            float(np.linalg.norm(u_vector)),
            float(np.linalg.norm(v_vector)),
            1.0,
        )
        plane_tolerance = 1.0e-9 * scale
        if np.any(np.abs(relative @ ancestor.normal_xyz) > plane_tolerance):
            return False
        coordinate_tolerance = 1.0e-9
        u_coordinate = relative @ u_vector / float(np.dot(u_vector, u_vector))
        v_coordinate = relative @ v_vector / float(np.dot(v_vector, v_vector))
        return bool(
            np.all(u_coordinate >= -coordinate_tolerance)
            and np.all(u_coordinate <= 1.0 + coordinate_tolerance)
            and np.all(v_coordinate >= -coordinate_tolerance)
            and np.all(v_coordinate <= 1.0 + coordinate_tolerance)
        )

    result = np.zeros((patches.patch_count, len(isotope_names)), dtype=float)
    for index, patch in enumerate(patches.patches):
        previous = old_by_id.get(patch.patch_id)
        if previous is not None:
            previous_patch, density = previous
            if not np.isclose(previous_patch.area_m2, patch.area_m2) or not np.allclose(
                previous_patch.vertices_xyz,
                patch.vertices_xyz,
            ):
                raise ValueError(
                    f"Warm-start patch {patch.patch_id} geometry does not match."
                )
            result[index] = density
            continue
        descendants = [
            (candidate, old_density[candidate_index])
            for candidate_index, candidate in enumerate(initial_estimate.patches)
            if is_descendant(candidate, patch)
        ]
        if descendants:
            integrated = np.sum(
                [descendant.area_m2 * density for descendant, density in descendants],
                axis=0,
            )
            result[index] = integrated / patch.area_m2
    return result


def _initial_nuisance_from_estimate(
    initial_estimate: MLEEstimate | None,
) -> NDArray[np.float64] | None:
    """Recombine persisted background and other nuisance coefficients."""
    if initial_estimate is None:
        return None
    return np.concatenate(
        [
            np.asarray(initial_estimate.background_parameters, dtype=float),
            np.asarray(initial_estimate.nuisance_parameters, dtype=float),
        ]
    )


def _held_out_likelihood_score(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    state: _FitState,
    config: MLEConfig,
) -> float:
    """Return mean validation loss under the configured observation family."""
    values = np.asarray(observed, dtype=np.float64)
    mean = np.maximum(np.asarray(predicted, dtype=np.float64), config.min_mean)
    if (
        config.mode == "spectral"
        and config.spectral_likelihood == "calibrated_overdispersed"
    ):
        alpha = np.asarray(state.overdispersion_alpha_by_bin, dtype=float)
        if alpha.shape != (values.shape[1],):
            raise ValueError("Calibrated validation alpha does not match energy bins.")
        alpha_grid = np.broadcast_to(alpha[None, :], values.shape)
        poisson = alpha_grid <= 1.0e-15
        loss = np.empty_like(values)
        loss[poisson] = (
            mean[poisson]
            - values[poisson] * np.log(mean[poisson])
            + gammaln(values[poisson] + 1.0)
        )
        if np.any(~poisson):
            size = 1.0 / alpha_grid[~poisson]
            selected_mean = mean[~poisson]
            selected_values = values[~poisson]
            loss[~poisson] = -(
                gammaln(selected_values + size)
                - gammaln(size)
                - gammaln(selected_values + 1.0)
                + size * np.log(size / (size + selected_mean))
                + selected_values * np.log(selected_mean / (size + selected_mean))
            )
        return float(np.mean(loss))
    return float(
        poisson_deviance(values, mean, min_mean=float(config.min_mean)) / values.size
    )


def _select_regularization(
    batch: ObservationBatch,
    environment: EnvironmentConfig,
    kernel: ContinuousKernel,
    config: MLEConfig,
    obstacle_grid: ObstacleGrid | None,
) -> RegularizationCVResult:
    """Run grouped CV on the base surface dictionary without truth access."""
    patches = build_surface_patches(
        environment,
        obstacle_grid,
        config.patch_spacing_m,
        obstacle_height_m=float(config.obstacle_height_m),
        quadrature_points_per_patch=int(config.quadrature_order),
    )
    labels = _union_group_labels(
        batch,
        config.cv_grouping,
        config.held_out_xy_tolerance_m,
    )
    folds = grouped_kfold_indices(
        labels,
        min(int(config.cv_fold_count), len(set(labels))),
        random_seed=int(config.random_seed),
    )
    candidates = tuple(
        RegularizationCandidate(l1_weight=l1, tv_weight=tv)
        for l1 in config.cv_l1_weights
        for tv in config.cv_tv_weights
    )
    observed = batch.isotope_counts if config.mode == "count" else batch.spectrum_counts
    if observed is None:
        raise ValueError("Configured observations are unavailable for grouped CV.")

    def score(
        candidate: RegularizationCandidate,
        fit_indices: NDArray[np.int64],
        validation_indices: NDArray[np.int64],
    ) -> float:
        """Fit one fold and score only its untouched related groups."""
        candidate_config = replace(
            config,
            regularization_selection="fixed",
            l1_weight=float(candidate.l1_weight),
            tv_weight=float(candidate.tv_weight),
            held_out_fraction=0.0,
            coarse_to_fine_levels=0,
            debias_refit=False,
        )
        state = _fit_problem(
            batch,
            patches,
            kernel,
            candidate_config,
            fit_indices,
            validation_indices,
        )
        prediction = _full_prediction(state)
        return _held_out_likelihood_score(
            observed[validation_indices],
            prediction[validation_indices],
            state,
            candidate_config,
        )

    return select_regularization_one_standard_error(
        candidates,
        folds,
        score,
        use_one_standard_error=bool(config.cv_one_standard_error),
        group_labels=labels,
    )


class SurfaceMLEEstimator:
    """Fit count-domain or line-resolved spectral surface intensity maps."""

    def __init__(
        self,
        config: MLEConfig,
        *,
        persistent_response_cache: dict[str, object] | None = None,
        progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        """Store configuration and an optional exact device response cache."""
        if not isinstance(config, MLEConfig):
            raise TypeError("config must be an MLEConfig.")
        if progress_hook is not None and not callable(progress_hook):
            raise TypeError("progress_hook must be callable or None.")
        self.config = config
        self._progress_hook = progress_hook
        self._persistent_response_cache = (
            {} if persistent_response_cache is None else persistent_response_cache
        )

    def _report_bootstrap_progress(
        self,
        completed: int,
        total: int,
        started: float,
    ) -> None:
        """Report completed exact bootstrap replicates and their running ETA."""
        if self._progress_hook is None:
            return
        elapsed = perf_counter() - float(started)
        self._progress_hook(
            {
                "phase": "mle_bootstrap",
                "completed": int(completed),
                "total": int(total),
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed * (int(total) - int(completed)) / completed,
            }
        )

    def fit(
        self,
        batch: ObservationBatch,
        environment: EnvironmentConfig,
        kernel: ContinuousKernel,
        *,
        obstacle_grid: ObstacleGrid | None = None,
        initial_estimate: MLEEstimate | None = None,
    ) -> MLEEstimate:
        """Fit all history, optionally warm-starting from a prior surface map."""
        if tuple(batch.isotope_names) != tuple(self.config.isotope_names):
            raise ValueError(
                "Observation isotope order must match MLEConfig.isotope_names."
            )
        if self.config.regularization_selection == "grouped_cv":
            selection = _select_regularization(
                batch,
                environment,
                kernel,
                self.config,
                obstacle_grid,
            )
            selected_config = replace(
                self.config,
                regularization_selection="fixed",
                l1_weight=float(selection.selected.l1_weight),
                tv_weight=float(selection.selected.tv_weight),
            )
            selected_estimate = SurfaceMLEEstimator(
                selected_config,
                progress_hook=self._progress_hook,
            ).fit(
                batch,
                environment,
                kernel,
                obstacle_grid=obstacle_grid,
                initial_estimate=initial_estimate,
            )
            return replace(
                selected_estimate,
                diagnostics={
                    **selected_estimate.diagnostics,
                    "regularization_selection": {
                        **selection.to_dict(),
                        "grouping": self.config.cv_grouping,
                        "tuning_environment_id": (self.config.tuning_environment_id),
                        "final_holdout_environment_id": (
                            self.config.final_holdout_environment_id
                        ),
                    },
                },
            )
        kernel.use_gpu = bool(self.config.use_gpu)
        kernel.gpu_device = str(self.config.gpu_device)
        kernel.gpu_dtype = str(self.config.gpu_dtype)
        patches = build_surface_patches(
            environment,
            obstacle_grid,
            self.config.patch_spacing_m,
            obstacle_height_m=float(self.config.obstacle_height_m),
            quadrature_points_per_patch=int(self.config.quadrature_order),
        )
        base_patch_ids = patches.patch_ids.astype(int).tolist()
        base_patch_count = patches.patch_count
        fit_indices, held_out_indices, held_out_group_labels = _split_fit_indices(
            batch,
            self.config.held_out_fraction,
            self.config.random_seed,
            grouping=self.config.held_out_grouping,
            xy_tolerance_m=self.config.held_out_xy_tolerance_m,
        )
        initial_densities = _initial_density_for_patches(
            initial_estimate,
            patches,
            self.config.isotope_names,
        )
        initial_nuisance = _initial_nuisance_from_estimate(initial_estimate)
        state = _fit_problem(
            batch,
            patches,
            kernel,
            self.config,
            fit_indices,
            held_out_indices,
            initial_densities=initial_densities,
            initial_nuisance=initial_nuisance,
            persistent_response_cache=self._persistent_response_cache,
            progress_hook=self._progress_hook,
            progress_label="base",
        )
        for level in range(int(self.config.coarse_to_fine_levels)):
            selected = _refinement_patch_ids(
                state,
                self.config.refinement_fraction,
                self.config.refinement_max_patches,
            )
            if not selected:
                break
            refined = refine_surface_patches(
                state.patches,
                selected,
                quadrature_points_per_patch=int(self.config.quadrature_order),
            )
            warm = _warm_start_refined(
                state.patches,
                refined,
                state.result.densities_cps_1m_m2,
            )
            state = _fit_problem(
                batch,
                refined,
                kernel,
                self.config,
                fit_indices,
                held_out_indices,
                initial_densities=warm,
                initial_nuisance=state.result.nuisance_coefficients,
                persistent_response_cache=self._persistent_response_cache,
                progress_hook=self._progress_hook,
                progress_label=f"refinement_{level + 1}",
            )

        observed = (
            batch.isotope_counts
            if self.config.mode == "count"
            else batch.spectrum_counts
        )
        if observed is None:  # guarded by _fit_problem; keeps static typing explicit
            raise ValueError("Configured observation domain is unavailable.")
        if self.config.debias_refit:
            state = _debias_state(
                state,
                observed,
                self.config,
                self._persistent_response_cache,
                self._progress_hook,
            )
        predicted = _full_prediction(state)
        held_out_deviance = (
            poisson_deviance(
                observed[held_out_indices],
                predicted[held_out_indices],
                min_mean=float(self.config.min_mean),
            )
            if held_out_indices.size
            else None
        )
        clusters = cluster_surface_hotspots(
            state.patches,
            state.result.densities_cps_1m_m2,
            self.config.isotope_names,
            threshold_fraction=float(self.config.cluster_threshold_fraction),
            min_strength_cps_1m=float(self.config.cluster_min_strength_cps_1m),
        )
        if isinstance(state.response, ResponseOperator):
            identifiability = _operator_identifiability(
                state.response,
                state.result.densities_cps_1m_m2,
                state.fit_indices,
                float(self.config.response_correlation_threshold),
            )
            response_shape = [
                *state.response.observation_shape,
                state.response.patch_count,
                state.response.isotope_count,
            ]
        else:
            identifiability = response_identifiability_diagnostics(
                state.response[state.fit_indices],
                correlation_threshold=float(self.config.response_correlation_threshold),
            )
            response_shape = [int(value) for value in state.response.shape]
        residual = observed - predicted
        pearson_dispersion = float(
            np.mean(
                np.square(residual[state.fit_indices])
                / np.maximum(predicted[state.fit_indices], 1.0)
            )
        )
        diagnostics: dict[str, object] = {
            "provenance": estimator_provenance(variant=self.config.mode),
            "estimator_family": "surface_mle",
            "estimator_variant": self.config.mode,
            "candidate_domain": "complete_surface_dictionary",
            "uses_pf_state": False,
            "uses_pf_candidates": False,
            "mode": self.config.mode,
            "warm_started": initial_estimate is not None,
            "density_unit": "detector_cps_1m_per_m2",
            "patch_strength_unit": "detector_cps_1m",
            "patch_count": state.patches.patch_count,
            "base_surface_dictionary_patch_count": base_patch_count,
            "base_surface_dictionary_patch_ids": base_patch_ids,
            "full_surface_dictionary_used": True,
            "response_shape": response_shape,
            "spectral_response_mode": self.config.spectral_response_mode,
            "observation_step_ids": batch.step_ids.astype(int).tolist(),
            "observation_action_ids": batch.action_ids.astype(int).tolist(),
            "observation_station_ids": batch.station_ids.astype(int).tolist(),
            "detector_positions_xyz": batch.detector_positions_xyz.tolist(),
            "detector_heights_m": batch.detector_positions_xyz[:, 2].tolist(),
            "live_times_s": batch.live_times_s.tolist(),
            "travel_times_s": batch.travel_times_s.tolist(),
            "shield_actuation_times_s": batch.shield_actuation_times_s.tolist(),
            "shield_program_block_ids": list(batch.shield_program_block_ids),
            "fit_measurement_indices": state.fit_indices.astype(int).tolist(),
            "held_out_measurement_indices": state.held_out_indices.astype(int).tolist(),
            "held_out_grouping": self.config.held_out_grouping,
            "held_out_group_labels": list(held_out_group_labels),
            "held_out_group_ids": sorted(
                {
                    int(held_out_group_labels[int(index)])
                    for index in state.held_out_indices
                }
            ),
            "held_out_poisson_deviance": held_out_deviance,
            "residual_l2": float(np.linalg.norm(residual)),
            "pearson_dispersion": pearson_dispersion,
            "deviance_per_fitted_channel": float(state.result.deviance)
            / max(int(observed[state.fit_indices].size), 1),
            "residual_by_observation": np.asarray(residual, dtype=float).tolist(),
            "objective_history": _json_floats(
                np.asarray(state.result.objective_history, dtype=float)
            ),
            "poisson_nll": float(state.result.poisson_nll),
            "l1_penalty": float(state.result.l1_penalty),
            "tv_penalty": float(state.result.tv_penalty),
            "group_penalty": float(state.result.group_penalty),
            "nuisance_penalty": float(state.result.nuisance_penalty),
            "relative_change": float(state.result.relative_change),
            "relative_objective_change": float(state.result.relative_objective_change),
            "kkt_residual": float(state.result.kkt_residual),
            "likelihood": dict(state.likelihood_diagnostics),
            "nuisance_names": list(state.nuisance_names),
            "isotope_covariance_preserved": batch.isotope_covariances is not None,
            "hotspot_clusters": [cluster.to_dict() for cluster in clusters],
            "identifiability": identifiability,
        }
        if state.spectral_details is not None:
            diagnostics["line_energies_keV_by_isotope"] = {
                key: list(values)
                for key, values in state.spectral_details.line_energies_keV_by_isotope.items()
            }
            diagnostics["line_weights_by_isotope"] = {
                key: list(values)
                for key, values in state.spectral_details.line_weights_by_isotope.items()
            }
            if isinstance(state.spectral_details, SpectralResponseOperatorResult):
                diagnostics["response_operator"] = dict(
                    state.spectral_details.operator.diagnostics
                )
                diagnostics["response_cache_directory"] = (
                    None
                    if state.spectral_details.cache_directory is None
                    else state.spectral_details.cache_directory.as_posix()
                )

        nuisance = state.result.nuisance_coefficients
        background_mask = np.asarray(
            [name.startswith("background") for name in state.nuisance_names],
            dtype=bool,
        )
        background_parameters = nuisance[background_mask]
        other_nuisance = nuisance[~background_mask]
        if self.config.mode == "count":
            predicted_spectra = None
            predicted_isotope_counts = predicted
        else:
            predicted_spectra = predicted
            if isinstance(state.response, ResponseOperator):
                density = state.result.densities_cps_1m_m2
                isotope_predictions: list[NDArray[np.float64]] = []
                for isotope_index in range(state.response.isotope_count):
                    selected = np.zeros_like(density)
                    selected[:, isotope_index] = density[:, isotope_index]
                    isotope_predictions.append(
                        np.sum(
                            state.response.matvec(selected.reshape(-1)).reshape(
                                state.response.observation_shape
                            ),
                            axis=1,
                        )
                    )
                predicted_isotope_counts = np.stack(
                    isotope_predictions,
                    axis=1,
                )
            else:
                predicted_isotope_counts = np.einsum(
                    "mbgi,gi->mi",
                    state.response,
                    state.result.integrated_strengths_cps_1m,
                    optimize=True,
                )
        estimate = MLEEstimate(
            isotope_names=self.config.isotope_names,
            patches=state.patches.patches,
            density_by_isotope=state.result.densities_cps_1m_m2.T,
            patch_strength_by_isotope=state.result.integrated_strengths_cps_1m.T,
            predicted_spectra=predicted_spectra,
            predicted_isotope_counts=predicted_isotope_counts,
            background_parameters=background_parameters,
            nuisance_parameters=other_nuisance,
            objective_value=float(state.result.objective),
            poisson_deviance=float(state.result.deviance),
            iterations=int(state.result.iterations),
            converged=bool(state.result.converged),
            diagnostics=diagnostics,
        )
        if not bool(self.config.uncertainty_enable):
            return estimate
        laplace = active_support_laplace(
            state.response,
            observed,
            predicted,
            state.result.densities_cps_1m_m2,
            state.patches.areas_m2,
            state.fit_indices,
            support_threshold_fraction=float(
                self.config.laplace_support_threshold_fraction
            ),
            maximum_active_parameters=int(self.config.laplace_max_active_parameters),
            ridge=float(self.config.laplace_ridge),
            overdispersion_alpha_by_bin=(
                state.overdispersion_alpha_by_bin
                if state.overdispersion_alpha_by_bin.size
                else None
            ),
        )
        estimate = replace(
            estimate,
            diagnostics={
                **estimate.diagnostics,
                "hotspot_clusters": augment_clusters_with_laplace(
                    estimate,
                    laplace,
                    confidence_level=float(self.config.bootstrap_confidence_level),
                ),
            },
        )
        bootstrap_estimates: list[MLEEstimate] = []
        replicate_count = int(self.config.station_bootstrap_replicates)
        bootstrap_seconds = 0.0
        if replicate_count:
            bootstrap_started = perf_counter()
            if self._progress_hook is not None:
                self._progress_hook(
                    {
                        "phase": "mle_bootstrap",
                        "completed": 0,
                        "total": replicate_count,
                        "elapsed_seconds": 0.0,
                        "eta_seconds": None,
                    }
                )
            rng = np.random.default_rng(int(self.config.bootstrap_seed))
            bootstrap_config = replace(
                self.config,
                uncertainty_enable=False,
                station_bootstrap_replicates=0,
                regularization_selection="fixed",
                held_out_fraction=0.0,
            )
            replicate_batches = tuple(
                station_bootstrap_batch(batch, rng)
                for _replicate in range(replicate_count)
            )
            batch_size = min(
                int(self.config.bootstrap_batch_size),
                replicate_count,
            )
            if batch_size == 1:
                bootstrap_estimator = SurfaceMLEEstimator(
                    bootstrap_config,
                    persistent_response_cache=self._persistent_response_cache,
                )
                for replicate_index, replicate_batch in enumerate(
                    replicate_batches,
                    start=1,
                ):
                    bootstrap_estimates.append(
                        bootstrap_estimator.fit(
                            replicate_batch,
                            environment,
                            kernel,
                            obstacle_grid=obstacle_grid,
                        )
                    )
                    self._report_bootstrap_progress(
                        replicate_index,
                        replicate_count,
                        bootstrap_started,
                    )
            else:
                cached_entries = self._persistent_response_cache.get("entries")
                response_entries = (
                    dict(cached_entries) if isinstance(cached_entries, dict) else {}
                )

                def fit_replicate(
                    replicate_batch: ObservationBatch,
                ) -> MLEEstimate:
                    """Fit one exact replicate with shared immutable GPU rows."""
                    local_cache: dict[str, object] = {"entries": dict(response_entries)}
                    replicate_estimator = SurfaceMLEEstimator(
                        bootstrap_config,
                        persistent_response_cache=local_cache,
                    )
                    if not bootstrap_config.use_gpu:
                        return replicate_estimator.fit(
                            replicate_batch,
                            environment,
                            kernel,
                            obstacle_grid=obstacle_grid,
                        )
                    import torch

                    device = torch.device(str(bootstrap_config.gpu_device))
                    stream = torch.cuda.Stream(device=device)
                    with torch.cuda.stream(stream):
                        result = replicate_estimator.fit(
                            replicate_batch,
                            environment,
                            kernel,
                            obstacle_grid=obstacle_grid,
                        )
                    stream.synchronize()
                    return result

                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    for replicate_index, result in enumerate(
                        executor.map(fit_replicate, replicate_batches),
                        start=1,
                    ):
                        bootstrap_estimates.append(result)
                        self._report_bootstrap_progress(
                            replicate_index,
                            replicate_count,
                            bootstrap_started,
                        )
            bootstrap_seconds = perf_counter() - bootstrap_started
        bootstrap, augmented_clusters = bootstrap_uncertainty_summary(
            estimate,
            bootstrap_estimates,
            confidence_level=float(self.config.bootstrap_confidence_level),
        )
        if replicate_count:
            bootstrap = {
                **bootstrap,
                "execution": {
                    "batch_size": min(
                        int(self.config.bootstrap_batch_size),
                        replicate_count,
                    ),
                    "shared_response_cache": True,
                    "elapsed_seconds": bootstrap_seconds,
                },
            }
        uncertainty = {
            "laplace": laplace.to_dict(
                patch_ids=state.patches.patch_ids.astype(int).tolist(),
                isotope_names=self.config.isotope_names,
            ),
            "station_bootstrap": bootstrap,
        }
        return replace(
            estimate,
            diagnostics={
                **estimate.diagnostics,
                "hotspot_clusters": augmented_clusters,
                "uncertainty": uncertainty,
            },
        )


def fit_surface_mle(
    batch: ObservationBatch,
    environment: EnvironmentConfig,
    kernel: ContinuousKernel,
    config: MLEConfig,
    *,
    obstacle_grid: ObstacleGrid | None = None,
    initial_estimate: MLEEstimate | None = None,
) -> MLEEstimate:
    """Convenience wrapper for one all-history surface MLE fit."""
    return SurfaceMLEEstimator(config).fit(
        batch,
        environment,
        kernel,
        obstacle_grid=obstacle_grid,
        initial_estimate=initial_estimate,
    )


__all__ = ["SurfaceMLEEstimator", "fit_surface_mle"]
