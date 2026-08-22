"""Laplace/Fisher optimal experimental design for online surface MLE."""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from runtime.discrepancy_calibration import load_discrepancy_calibration

from .config import MLEConfig
from .response_operator import LineFactorizedResponseOperator
from .spectral_response_builder import (
    _hash_canonical_value,
    build_spectral_response,
    build_spectral_response_operator,
)
from .types import MLEEstimate, ObservationBatch, SurfacePatch
from .uncertainty import _project_patch_strengths_to_base


PLANNING_METHOD = "two_stage_grouped_likelihood_fisher_d_s_station_block_v5"
_BEAM_PRECISION_WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024


def _beam_precision_chunk_size(parameter_count: int) -> int:
    """Bound temporary beam precision matrices to a fixed workspace."""
    count = int(parameter_count)
    if count < 1:
        raise ValueError("parameter_count must be positive.")
    bytes_per_expansion = 3 * count * count * np.dtype(np.float64).itemsize
    return max(1, _BEAM_PRECISION_WORKSPACE_LIMIT_BYTES // bytes_per_expansion)


@dataclass(frozen=True, slots=True)
class MLEPlanningConfig:
    """Configure local-Fisher measurement-pose and shield-program selection."""

    live_time_s: float = 10.0
    shield_program_length: int = 8
    max_active_source_parameters: int = 48
    max_total_source_parameters: int = 96
    active_strength_fraction: float = 1.0e-3
    source_strength_scale_floor_cps_1m: float = 1.0
    nuisance_scale_floor: float = 1.0
    laplace_prior_precision: float = 1.0
    minimum_expected_bin_count: float = 1.0e-3
    motion_cost_weight: float = 0.0
    rotation_cost_weight: float = 0.0
    candidate_pose_chunk_size: int = 8
    ranked_action_limit: int = 32
    shield_program_beam_width: int = 64
    future_station_rate_prior_precision: float = 1.0
    floor_ceiling_separation_weight: float = 1.0
    support_hypothesis_separation_weight: float = 0.5
    z_fisher_weight: float = 0.5
    response_correlation_reduction_weight: float = 0.25
    elevation_diversity_weight: float = 0.25
    geometry_exploration_weight: float = 0.5
    surface_coverage_weight: float = 1.0
    geometry_bootstrap_measurements: int = 6
    local_refinement_top_k: int = 8
    two_stage_screening: bool = True
    screening_energy_bin_count: int = 64
    screening_pair_limit: int = 16
    screening_pose_chunk_size: int = 64
    screening_source_parameter_limit: int = 24
    screening_points_per_mode: int = 2
    exact_candidate_min: int = 4
    exact_candidate_max: int = 8
    exact_score_margin_fraction: float = 0.05
    exact_diversity_weight: float = 0.15

    def __post_init__(self) -> None:
        """Validate all values that affect the planning objective."""
        integer_fields = {
            "shield_program_length": self.shield_program_length,
            "max_active_source_parameters": self.max_active_source_parameters,
            "max_total_source_parameters": self.max_total_source_parameters,
            "candidate_pose_chunk_size": self.candidate_pose_chunk_size,
            "ranked_action_limit": self.ranked_action_limit,
            "shield_program_beam_width": self.shield_program_beam_width,
            "geometry_bootstrap_measurements": self.geometry_bootstrap_measurements,
            "local_refinement_top_k": self.local_refinement_top_k,
            "screening_energy_bin_count": self.screening_energy_bin_count,
            "screening_pair_limit": self.screening_pair_limit,
            "screening_pose_chunk_size": self.screening_pose_chunk_size,
            "screening_source_parameter_limit": (self.screening_source_parameter_limit),
            "screening_points_per_mode": self.screening_points_per_mode,
            "exact_candidate_min": self.exact_candidate_min,
            "exact_candidate_max": self.exact_candidate_max,
        }
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if int(self.max_active_source_parameters) > int(
            self.max_total_source_parameters
        ):
            raise ValueError(
                "max_active_source_parameters cannot exceed "
                "max_total_source_parameters."
            )
        if int(self.exact_candidate_min) > int(self.exact_candidate_max):
            raise ValueError("exact_candidate_min cannot exceed exact_candidate_max.")
        if not isinstance(self.two_stage_screening, (bool, np.bool_)):
            raise TypeError("two_stage_screening must be boolean.")
        if self.two_stage_screening and int(self.ranked_action_limit) < int(
            self.exact_candidate_max
        ):
            raise ValueError("ranked_action_limit must cover exact_candidate_max.")
        if self.two_stage_screening and int(self.screening_pair_limit) < int(
            self.shield_program_length
        ):
            raise ValueError("screening_pair_limit must cover shield_program_length.")
        positive_fields = {
            "live_time_s": self.live_time_s,
            "source_strength_scale_floor_cps_1m": (
                self.source_strength_scale_floor_cps_1m
            ),
            "nuisance_scale_floor": self.nuisance_scale_floor,
            "laplace_prior_precision": self.laplace_prior_precision,
            "minimum_expected_bin_count": self.minimum_expected_bin_count,
            "future_station_rate_prior_precision": (
                self.future_station_rate_prior_precision
            ),
        }
        for name, value in positive_fields.items():
            parsed = float(value)
            if not np.isfinite(parsed) or parsed <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative_fields = {
            "active_strength_fraction": self.active_strength_fraction,
            "motion_cost_weight": self.motion_cost_weight,
            "rotation_cost_weight": self.rotation_cost_weight,
            "floor_ceiling_separation_weight": (self.floor_ceiling_separation_weight),
            "support_hypothesis_separation_weight": (
                self.support_hypothesis_separation_weight
            ),
            "z_fisher_weight": self.z_fisher_weight,
            "response_correlation_reduction_weight": (
                self.response_correlation_reduction_weight
            ),
            "elevation_diversity_weight": self.elevation_diversity_weight,
            "geometry_exploration_weight": self.geometry_exploration_weight,
            "surface_coverage_weight": self.surface_coverage_weight,
            "exact_score_margin_fraction": self.exact_score_margin_fraction,
            "exact_diversity_weight": self.exact_diversity_weight,
        }
        for name, value in nonnegative_fields.items():
            parsed = float(value)
            if not np.isfinite(parsed) or parsed < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if float(self.active_strength_fraction) > 1.0:
            raise ValueError("active_strength_fraction must not exceed one.")
        if float(self.exact_diversity_weight) > 1.0:
            raise ValueError("exact_diversity_weight must not exceed one.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe planner configuration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MLEPlanningConfig":
        """Construct planner settings from one JSON object."""
        if not isinstance(payload, Mapping):
            raise TypeError("MLE planning configuration must be a mapping.")
        return cls(**dict(payload))

    @classmethod
    def load(cls, path: str | Path) -> "MLEPlanningConfig":
        """Load one strict JSON planner configuration file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("MLE planning configuration root must be an object.")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class MLEPlanningAction:
    """Describe one candidate pose and its jointly optimized shield program."""

    candidate_index: int
    detector_pose_xyz: tuple[float, float, float]
    shield_pair_ids: tuple[int, ...]
    fe_orientation_indices: tuple[int, ...]
    pb_orientation_indices: tuple[int, ...]
    information_gain_nats: float
    travel_cost: float
    rotation_radians: float
    score: float
    live_time_s_by_view: tuple[float, ...]
    expected_total_counts_by_view: tuple[float, ...]
    floor_ceiling_separation: float = 0.0
    support_hypothesis_separation: float = 0.0
    z_fisher_information: float = 0.0
    response_correlation_reduction: float = 0.0
    elevation_diversity: float = 0.0
    geometry_exploration: float = 0.0
    surface_coverage: float = 0.0

    def __post_init__(self) -> None:
        """Validate that the recommendation is a complete executable program."""
        count = len(self.shield_pair_ids)
        if count == 0 or any(
            len(values) != count
            for values in (
                self.fe_orientation_indices,
                self.pb_orientation_indices,
                self.live_time_s_by_view,
                self.expected_total_counts_by_view,
            )
        ):
            raise ValueError("Planning action view fields must be nonempty and align.")
        if int(self.candidate_index) < 0:
            raise ValueError("candidate_index must be nonnegative.")
        pose = np.asarray(self.detector_pose_xyz, dtype=np.float64)
        if pose.shape != (3,) or np.any(~np.isfinite(pose)):
            raise ValueError("detector_pose_xyz must contain three finite values.")
        if any(int(value) < 0 for value in self.shield_pair_ids):
            raise ValueError("shield_pair_ids must be nonnegative.")
        if any(int(value) < 0 for value in self.fe_orientation_indices) or any(
            int(value) < 0 for value in self.pb_orientation_indices
        ):
            raise ValueError("Shield orientation indices must be nonnegative.")
        finite_values = (
            self.information_gain_nats,
            self.travel_cost,
            self.rotation_radians,
            self.score,
            *self.live_time_s_by_view,
            *self.expected_total_counts_by_view,
            self.floor_ceiling_separation,
            self.support_hypothesis_separation,
            self.z_fisher_information,
            self.response_correlation_reduction,
            self.elevation_diversity,
            self.geometry_exploration,
            self.surface_coverage,
        )
        if any(not np.isfinite(float(value)) for value in finite_values):
            raise ValueError("Planning action numerical values must be finite.")
        if float(self.information_gain_nats) < -1.0e-10:
            raise ValueError("information_gain_nats must be nonnegative.")
        if float(self.travel_cost) < 0.0 or float(self.rotation_radians) < 0.0:
            raise ValueError("Travel and rotation costs must be nonnegative.")
        if any(float(value) <= 0.0 for value in self.live_time_s_by_view):
            raise ValueError("Every planned live time must be positive.")
        if any(float(value) < 0.0 for value in self.expected_total_counts_by_view):
            raise ValueError("Expected counts must be nonnegative.")

    def to_dict(self) -> dict[str, object]:
        """Return the runtime-neutral action recommendation as JSON data."""
        program = [
            {
                "sequence_index": index,
                "shield_pair_id": int(pair_id),
                "fe_orientation_index": int(self.fe_orientation_indices[index]),
                "pb_orientation_index": int(self.pb_orientation_indices[index]),
                "live_time_s": float(self.live_time_s_by_view[index]),
                "station_complete": index == len(self.shield_pair_ids) - 1,
            }
            for index, pair_id in enumerate(self.shield_pair_ids)
        ]
        return {
            "action_schema_version": 1,
            "candidate_index": int(self.candidate_index),
            "detector_pose_xyz": [float(value) for value in self.detector_pose_xyz],
            "shield_pair_ids": [int(value) for value in self.shield_pair_ids],
            "fe_orientation_indices": [
                int(value) for value in self.fe_orientation_indices
            ],
            "pb_orientation_indices": [
                int(value) for value in self.pb_orientation_indices
            ],
            "information_gain_nats": float(self.information_gain_nats),
            "travel_cost": float(self.travel_cost),
            "rotation_radians": float(self.rotation_radians),
            "score": float(self.score),
            "live_time_s_by_view": [float(value) for value in self.live_time_s_by_view],
            "expected_total_counts_by_view": [
                float(value) for value in self.expected_total_counts_by_view
            ],
            "floor_ceiling_separation": float(self.floor_ceiling_separation),
            "support_hypothesis_separation": float(self.support_hypothesis_separation),
            "z_fisher_information": float(self.z_fisher_information),
            "response_correlation_reduction": float(
                self.response_correlation_reduction
            ),
            "elevation_diversity": float(self.elevation_diversity),
            "geometry_exploration": float(self.geometry_exploration),
            "surface_coverage": float(self.surface_coverage),
            "measurement_program": program,
        }


@dataclass(frozen=True, slots=True)
class MLEPlanningResult:
    """Return the selected action and bounded deterministic candidate ranking."""

    selected_action: MLEPlanningAction
    ranked_actions: tuple[MLEPlanningAction, ...]
    diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON planning artifact."""
        return {
            "schema_version": 1,
            "planning_method": PLANNING_METHOD,
            "selected_action": self.selected_action.to_dict(),
            "ranked_actions": [action.to_dict() for action in self.ranked_actions],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class _PlanningGeometry:
    """Expose only geometry fields consumed by the spectral response builder."""

    detector_positions_xyz: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    energy_bin_edges_keV: NDArray[np.float64]
    station_ids: NDArray[np.int64] | None = None


def _historical_row_keys(
    observations: ObservationBatch,
) -> tuple[tuple[object, ...], ...]:
    """Return causal cache keys for every response-relevant history row."""
    return tuple(
        (
            int(step_id),
            np.asarray(position, dtype=np.float64).tobytes(),
            np.asarray(quaternion, dtype=np.float64).tobytes(),
            int(fe_index),
            int(pb_index),
            float(live_time),
            int(station_id),
        )
        for (
            step_id,
            position,
            quaternion,
            fe_index,
            pb_index,
            live_time,
            station_id,
        ) in zip(
            observations.step_ids,
            observations.detector_positions_xyz,
            observations.detector_quaternions_wxyz,
            observations.fe_indices,
            observations.pb_indices,
            observations.live_times_s,
            observations.station_ids,
            strict=True,
        )
    )


def _calibration_identity(path: str | None) -> tuple[str, str | None] | None:
    """Return a content-sensitive identity for one calibration artifact."""
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    digest = sha256(resolved.read_bytes()).hexdigest() if resolved.is_file() else None
    return resolved.as_posix(), digest


def _kernel_physical_identity(kernel: object) -> object:
    """Return a value-sensitive identity for shared runtime kernel settings."""
    if not is_dataclass(kernel) or isinstance(kernel, type):
        return (type(kernel).__module__, type(kernel).__qualname__, id(kernel))
    digest = sha256()
    digest.update(b"planner-continuous-kernel-v1\0")
    for kernel_field in dataclass_fields(kernel):
        if kernel_field.init and kernel_field.name != "gpu_device":
            _hash_canonical_value(digest, kernel_field.name)
            _hash_canonical_value(digest, getattr(kernel, kernel_field.name))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _PatchView:
    """Expose estimated patches with aggregate area access."""

    patches: tuple[SurfacePatch, ...]

    @property
    def areas_m2(self) -> NDArray[np.float64]:
        """Return one physical area per estimated patch."""
        return np.asarray(
            [patch.area_m2 for patch in self.patches],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class _LineSpectralDesign:
    """Store compact integrated-strength line factors for planning."""

    spatial_factors: NDArray[np.float64]
    pulse_shapes: NDArray[np.float64]
    line_isotope_indices: NDArray[np.int64]
    nuisance_response: NDArray[np.float64]
    nuisance_names: tuple[str, ...]
    energy_chunk_size: int
    nuisance_l2_weights: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    overdispersion_alpha_by_bin: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )

    @property
    def observation_shape(self) -> tuple[int, int]:
        """Return action and energy-bin dimensions."""
        return (
            int(self.spatial_factors.shape[0]),
            int(self.pulse_shapes.shape[1]),
        )

    @property
    def patch_count(self) -> int:
        """Return the shared surface patch count."""
        return int(self.spatial_factors.shape[1])

    @property
    def isotope_count(self) -> int:
        """Return the isotope dimension encoded by line membership."""
        return int(np.max(self.line_isotope_indices)) + 1


@dataclass(frozen=True, slots=True)
class _SpectralHypothesisMoments:
    """Store grouped source counts and complete observation variance."""

    source_counts: NDArray[np.float64]
    total_variance: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _GroupedNB2LineDesign:
    """Project grouped NB2 hypotheses without a dense fine-bin response."""

    grouped_response: NDArray[np.float64]
    grouped_background: NDArray[np.float64]
    spatial_factors: NDArray[np.float64]
    pulse_shapes: NDArray[np.float64]
    line_isotope_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    background_rate_by_bin: NDArray[np.float64]
    overdispersion_alpha_by_bin: NDArray[np.float64]
    fine_group_indices: NDArray[np.int64]
    group_count: int
    energy_chunk_size: int

    def project_hypothesis(
        self,
        strengths: NDArray[np.float64],
    ) -> _SpectralHypothesisMoments:
        """Return exact grouped source counts and NB2 variance for one hypothesis."""
        values = np.asarray(strengths, dtype=np.float64)
        isotope_count = int(np.max(self.line_isotope_indices)) + 1
        expected_shape = (self.spatial_factors.shape[1], isotope_count)
        if values.shape != expected_shape:
            raise ValueError(
                f"Grouped screening strengths must have shape {expected_shape}."
            )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "Grouped screening strengths must be finite and non-negative."
            )
        amplitudes = np.einsum(
            "mgl,gl->ml",
            self.spatial_factors,
            values[:, self.line_isotope_indices],
            optimize=True,
        )
        measurement_count = int(self.spatial_factors.shape[0])
        grouped_source = np.zeros(
            (measurement_count, int(self.group_count)),
            dtype=np.float64,
        )
        grouped_variance = np.zeros_like(grouped_source)
        bin_count = int(self.pulse_shapes.shape[1])
        for start in range(0, bin_count, int(self.energy_chunk_size)):
            stop = min(start + int(self.energy_chunk_size), bin_count)
            source_chunk = amplitudes @ self.pulse_shapes[:, start:stop]
            common_chunk = (
                self.live_times_s[:, None]
                * self.background_rate_by_bin[None, start:stop]
            )
            mean_chunk = source_chunk + common_chunk
            variance_chunk = mean_chunk + (
                self.overdispersion_alpha_by_bin[None, start:stop] * mean_chunk**2
            )
            group_indices = self.fine_group_indices[start:stop]
            grouped_source += _group_last_axis(
                source_chunk,
                group_indices,
                int(self.group_count),
            )
            grouped_variance += _group_last_axis(
                variance_chunk,
                group_indices,
                int(self.group_count),
            )
        return _SpectralHypothesisMoments(
            source_counts=grouped_source,
            total_variance=grouped_variance,
        )


@dataclass(frozen=True, slots=True)
class _ScreeningPatchView:
    """Expose compact representative points as unit-area pseudo patches."""

    quadrature_points_xyz: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]
    areas_m2: NDArray[np.float64]
    strength_projection: NDArray[np.float64]


def _validated_candidate_poses(
    candidate_poses_xyz: object,
) -> NDArray[np.float64]:
    """Return a nonempty finite C x 3 candidate-pose array."""
    poses = np.asarray(candidate_poses_xyz, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1:] != (3,) or poses.shape[0] == 0:
        raise ValueError("candidate_poses_xyz must have nonempty shape (C, 3).")
    if np.any(~np.isfinite(poses)):
        raise ValueError("candidate_poses_xyz must contain only finite values.")
    return np.ascontiguousarray(poses)


def _validated_travel_costs(
    travel_costs: object | None,
    candidate_count: int,
) -> NDArray[np.float64]:
    """Return one finite nonnegative externally supplied cost per pose."""
    if travel_costs is None:
        return np.zeros(candidate_count, dtype=np.float64)
    costs = np.asarray(travel_costs, dtype=np.float64)
    if costs.shape != (candidate_count,):
        raise ValueError(f"travel_costs must have shape ({candidate_count},).")
    if np.any(~np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("travel_costs must contain finite nonnegative values.")
    return np.ascontiguousarray(costs)


def _validated_pair_ids(
    allowed_pair_ids: Sequence[int] | None,
    orientation_count: int,
) -> NDArray[np.int64]:
    """Return unique valid pair IDs under the shared runtime pair convention."""
    pair_count = int(orientation_count) ** 2
    if allowed_pair_ids is None:
        return np.arange(pair_count, dtype=np.int64)
    raw = np.asarray(tuple(allowed_pair_ids))
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("allowed_pair_ids must contain at least one pair ID.")
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError("allowed_pair_ids must contain only integers.")
    pairs = np.asarray(raw, dtype=np.int64)
    if np.any(pairs < 0) or np.any(pairs >= pair_count):
        raise ValueError(f"allowed_pair_ids must lie in [0, {pair_count - 1}].")
    if np.unique(pairs).size != pairs.size:
        raise ValueError("allowed_pair_ids must not contain duplicates.")
    return np.ascontiguousarray(pairs)


def _screening_source_basis(
    estimate: MLEEstimate,
    config: MLEPlanningConfig,
) -> tuple[NDArray[np.float64], tuple[dict[str, object], ...]]:
    """Build a compact isotope-by-surface basis for approximate screening."""
    patch_count = len(estimate.patches)
    isotope_count = len(estimate.isotope_names)
    strengths = np.asarray(estimate.patch_strength_by_isotope, dtype=np.float64).T
    groups: dict[tuple[int, str], list[int]] = {}
    for isotope_index in range(isotope_count):
        for patch_index, patch in enumerate(estimate.patches):
            groups.setdefault((isotope_index, patch.surface_kind), []).append(
                patch_index
            )
    ordered = sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1]),
    )
    limit = int(config.screening_source_parameter_limit)
    if len(ordered) > limit:
        collapsed: dict[tuple[int, str], list[int]] = {}
        for (isotope_index, _surface), indices in ordered:
            collapsed.setdefault((isotope_index, "all_surfaces"), []).extend(indices)
        ordered = sorted(collapsed.items(), key=lambda item: item[0][0])
    if len(ordered) > limit:
        raise ValueError(
            "screening_source_parameter_limit must cover every isotope mode."
        )
    basis = np.zeros(
        (patch_count, isotope_count, len(ordered)),
        dtype=np.float64,
    )
    labels: list[dict[str, object]] = []
    floor = float(config.source_strength_scale_floor_cps_1m)
    for column, ((isotope_index, surface_kind), indices) in enumerate(ordered):
        fitted = strengths[indices, isotope_index]
        total = float(np.sum(fitted))
        if total > 0.0:
            weights = fitted / total
        else:
            areas = np.asarray(
                [estimate.patches[index].area_m2 for index in indices],
                dtype=np.float64,
            )
            weights = areas / float(np.sum(areas))
        scale = max(total, floor)
        basis[indices, isotope_index, column] = scale * weights
        labels.append(
            {
                "kind": "screening_surface_mode",
                "isotope": estimate.isotope_names[isotope_index],
                "surface_kind": surface_kind,
                "patch_ids": [estimate.patches[index].patch_id for index in indices],
                "scale_cps_1m": scale,
            }
        )
    if not ordered or np.any(np.sum(basis, axis=(0, 1)) <= 0.0):
        raise RuntimeError("Every screening source mode must be nonzero.")
    return basis, tuple(labels)


def _screening_pseudo_model(
    estimate: MLEEstimate,
    config: MLEPlanningConfig,
) -> tuple[
    _ScreeningPatchView,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[dict[str, object], ...],
]:
    """Compress each screening mode to a few weighted representative points."""
    full_basis, labels = _screening_source_basis(estimate, config)
    full_strengths = np.asarray(
        estimate.patch_strength_by_isotope,
        dtype=np.float64,
    ).T
    patch_points = np.asarray(
        [patch.centroid_xyz for patch in estimate.patches],
        dtype=np.float64,
    )
    patch_ids = np.asarray(
        [patch.patch_id for patch in estimate.patches],
        dtype=np.int64,
    )
    patch_areas = np.asarray(
        [patch.area_m2 for patch in estimate.patches],
        dtype=np.float64,
    )
    patch_index_by_id = {
        int(patch_id): patch_index for patch_index, patch_id in enumerate(patch_ids)
    }
    pseudo_points: list[NDArray[np.float64]] = []
    pseudo_strength_rows: list[NDArray[np.float64]] = []
    pseudo_basis_rows: list[NDArray[np.float64]] = []
    pseudo_strength_projection: list[NDArray[np.float64]] = []
    point_limit = int(config.screening_points_per_mode)
    isotope_count = len(estimate.isotope_names)
    parameter_count = full_basis.shape[2]
    for parameter_index in range(parameter_count):
        coordinates = np.argwhere(full_basis[:, :, parameter_index] > 0.0)
        if not coordinates.size:
            raise RuntimeError("Screening mode unexpectedly has no coordinates.")
        isotope_index = int(coordinates[0, 1])
        mode_patch_ids = labels[parameter_index]["patch_ids"]
        if not isinstance(mode_patch_ids, list):
            raise RuntimeError("Screening mode patch IDs must be stored as a list.")
        indices = np.asarray(
            [patch_index_by_id[int(patch_id)] for patch_id in mode_patch_ids],
            dtype=np.int64,
        )
        weights = full_basis[indices, isotope_index, parameter_index]
        local_points = patch_points[indices]
        representative_count = min(point_limit, int(indices.size))
        first_candidates = np.flatnonzero(weights == float(np.max(weights)))
        first = int(first_candidates[np.argmin(patch_ids[indices[first_candidates]])])
        representatives = [first]
        minimum_distance = np.linalg.norm(local_points - local_points[first], axis=1)
        while len(representatives) < representative_count:
            scores = minimum_distance.copy()
            scores[np.asarray(representatives, dtype=np.int64)] = -np.inf
            next_candidates = np.flatnonzero(scores == float(np.max(scores)))
            next_index = int(
                next_candidates[np.argmin(patch_ids[indices[next_candidates]])]
            )
            representatives.append(next_index)
            minimum_distance = np.minimum(
                minimum_distance,
                np.linalg.norm(local_points - local_points[next_index], axis=1),
            )
        representative_points = local_points[
            np.asarray(representatives, dtype=np.int64)
        ]
        distances = np.linalg.norm(
            local_points[:, None, :] - representative_points[None, :, :],
            axis=2,
        )
        assignments = np.argmin(distances, axis=1)
        for cluster_index, representative_index in enumerate(representatives):
            assignments[representative_index] = cluster_index
        for cluster_index in range(representative_count):
            members = assignments == cluster_index
            if not np.any(members):
                raise RuntimeError("Screening representative cluster is empty.")
            cluster_weights = weights[members]
            weight_sum = float(np.sum(cluster_weights))
            centroid_weights = (
                cluster_weights if weight_sum > 0.0 else patch_areas[indices[members]]
            )
            centroid = np.average(
                local_points[members],
                axis=0,
                weights=centroid_weights,
            )
            strength_row = np.zeros(isotope_count, dtype=np.float64)
            strength_row[isotope_index] = float(
                np.sum(full_strengths[indices[members], isotope_index])
            )
            basis_row = np.zeros(
                (isotope_count, parameter_count),
                dtype=np.float64,
            )
            basis_row[isotope_index, parameter_index] = weight_sum
            projection_row = np.zeros(
                (len(estimate.patches), isotope_count),
                dtype=np.float64,
            )
            projection_row[
                indices[members],
                isotope_index,
            ] = 1.0
            pseudo_points.append(np.asarray(centroid, dtype=np.float64))
            pseudo_strength_rows.append(strength_row)
            pseudo_basis_rows.append(basis_row)
            pseudo_strength_projection.append(projection_row)
    points = np.asarray(pseudo_points, dtype=np.float64)
    patch_view = _ScreeningPatchView(
        quadrature_points_xyz=points[:, None, :],
        quadrature_weights=np.ones((points.shape[0], 1), dtype=np.float64),
        areas_m2=np.ones(points.shape[0], dtype=np.float64),
        strength_projection=np.asarray(
            pseudo_strength_projection,
            dtype=np.float64,
        ),
    )
    pseudo_strengths = np.asarray(pseudo_strength_rows, dtype=np.float64)
    pseudo_basis = np.asarray(pseudo_basis_rows, dtype=np.float64)
    if not np.allclose(
        np.sum(pseudo_basis, axis=(0, 1)),
        np.sum(full_basis, axis=(0, 1)),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("Pseudo screening model did not preserve mode scales.")
    return patch_view, pseudo_strengths, pseudo_basis, full_basis, labels


def _representative_pair_ids(
    pair_ids: NDArray[np.int64],
    orientations: NDArray[np.float64],
    limit: int,
    current_pair_id: int | None,
) -> NDArray[np.int64]:
    """Select a deterministic farthest-first covering set of shield pairs."""
    pairs = np.asarray(pair_ids, dtype=np.int64)
    if pairs.size <= int(limit):
        return np.ascontiguousarray(pairs)
    rotation_costs, _ = _pair_rotation_cost_cache(
        pairs,
        orientations,
        current_pair_id,
    )
    matches = (
        np.flatnonzero(pairs == int(current_pair_id))
        if current_pair_id is not None
        else np.zeros(0, dtype=np.int64)
    )
    first = int(matches[0]) if matches.size else int(np.argmin(pairs))
    selected = np.zeros(int(limit), dtype=np.int64)
    selected[0] = first
    available = np.ones(pairs.size, dtype=bool)
    available[first] = False
    minimum_distance = rotation_costs[first].copy()
    for offset in range(1, int(limit)):
        scores = np.where(available, minimum_distance, -np.inf)
        maximum = float(np.max(scores))
        tied = np.flatnonzero(scores == maximum)
        next_index = int(tied[np.argmin(pairs[tied])])
        selected[offset] = next_index
        available[next_index] = False
        minimum_distance = np.minimum(
            minimum_distance,
            rotation_costs[next_index],
        )
    return np.ascontiguousarray(pairs[selected])


def _source_marginal_precision(
    precision: NDArray[np.float64],
    nuisance_count: int,
) -> NDArray[np.float64]:
    """Return the source Schur complement of one joint precision matrix."""
    matrix = np.asarray(precision, dtype=np.float64)
    source_count = int(matrix.shape[0]) - int(nuisance_count)
    source = matrix[:source_count, :source_count]
    if int(nuisance_count) == 0:
        return source.copy()
    cross = matrix[:source_count, source_count:]
    nuisance = matrix[source_count:, source_count:]
    try:
        solved = np.linalg.solve(nuisance, cross.T)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(nuisance, rcond=1.0e-12) @ cross.T
    marginal = source - cross @ solved
    return 0.5 * (marginal + marginal.T)


def _planning_prior_precision(
    source_count: int,
    nuisance_scales: NDArray[np.float64],
    nuisance_l2_weights: NDArray[np.float64],
    *,
    laplace_prior_precision: float,
) -> NDArray[np.float64]:
    """Return source ridge plus correctly scaled fitted nuisance precision."""
    scales = np.asarray(nuisance_scales, dtype=np.float64)
    weights = np.asarray(nuisance_l2_weights, dtype=np.float64)
    if scales.shape != weights.shape or np.any(weights < 0.0):
        raise ValueError("Planner nuisance scales and L2 weights must align.")
    parameter_count = int(source_count) + int(scales.size)
    prior = float(laplace_prior_precision) * np.eye(
        parameter_count,
        dtype=np.float64,
    )
    if scales.size:
        nuisance_indices = np.arange(int(source_count), parameter_count)
        prior[nuisance_indices, nuisance_indices] += weights * scales**2
    return prior


def _screening_background_rate(
    observations: ObservationBatch,
    edges: NDArray[np.float64],
    predicted_source_counts: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Aggregate the non-source historical rate into coarse Poisson groups."""
    full_centers = 0.5 * (
        observations.energy_bin_edges_keV[:-1] + observations.energy_bin_edges_keV[1:]
    )
    indices = np.searchsorted(edges, full_centers, side="right") - 1
    indices = np.clip(indices, 0, edges.size - 2)
    observed = np.asarray(observations.spectrum_counts, dtype=np.float64)
    if predicted_source_counts is None:
        residual = observed
    else:
        predicted = np.asarray(predicted_source_counts, dtype=np.float64)
        if predicted.shape != observed.shape or np.any(~np.isfinite(predicted)):
            raise ValueError(
                "Predicted screening source counts must align with history."
            )
        residual = np.maximum(observed - predicted, 0.0)
    total_counts = np.sum(residual, axis=0, dtype=np.float64)
    grouped = np.bincount(
        indices,
        weights=total_counts,
        minlength=edges.size - 1,
    ).astype(np.float64)
    total_live_time = max(float(np.sum(observations.live_times_s)), 1.0e-12)
    return grouped / total_live_time


def _group_last_axis(
    values: NDArray[np.float64],
    group_indices: NDArray[np.int64],
    group_count: int,
) -> NDArray[np.float64]:
    """Sum the final array axis into deterministic non-overlapping groups."""
    array = np.asarray(values, dtype=np.float64)
    indices = np.asarray(group_indices, dtype=np.int64)
    count = int(group_count)
    if array.ndim < 1 or indices.shape != (array.shape[-1],):
        raise ValueError("Group indices must align with the final array axis.")
    if count < 1 or np.any(indices < 0) or np.any(indices >= count):
        raise ValueError("Group indices must lie within the output group count.")
    flat = array.reshape(-1, array.shape[-1])
    grouped = np.zeros((flat.shape[0], count), dtype=np.float64)
    for group_index in range(count):
        selected = indices == group_index
        if np.any(selected):
            grouped[:, group_index] = np.sum(flat[:, selected], axis=1)
    return grouped.reshape(*array.shape[:-1], count)


def _grouped_overdispersed_variance(
    expected_counts_by_bin: NDArray[np.float64],
    alpha_by_bin: NDArray[np.float64],
    group_indices: NDArray[np.int64],
    group_count: int,
) -> NDArray[np.float64]:
    """Aggregate independent NB2 variances without losing fine-bin alpha."""
    expected = np.asarray(expected_counts_by_bin, dtype=np.float64)
    alpha = np.asarray(alpha_by_bin, dtype=np.float64)
    if expected.ndim != 2 or alpha.shape != (expected.shape[1],):
        raise ValueError("Fine expected counts and overdispersion must align.")
    if (
        np.any(~np.isfinite(expected))
        or np.any(expected < 0.0)
        or np.any(~np.isfinite(alpha))
        or np.any(alpha < 0.0)
    ):
        raise ValueError("Expected counts and overdispersion must be non-negative.")
    fine_variance = expected + alpha[None, :] * expected**2
    return _group_last_axis(fine_variance, group_indices, group_count)


def _screening_fisher_information(
    source_response: NDArray[np.float64],
    source_basis: NDArray[np.float64],
    source_strengths: NDArray[np.float64],
    background_counts: NDArray[np.float64],
    *,
    minimum_expected_count: float,
    observation_variance: NDArray[np.float64] | None = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return source-only grouped likelihood Fisher matrices for screening."""
    response = np.asarray(source_response, dtype=np.float64)
    background = np.ascontiguousarray(background_counts, dtype=np.float64)
    if response.ndim != 4 or background.shape != response.shape[:2]:
        raise ValueError("Screening response and background counts must align.")
    supplied_variance = (
        None
        if observation_variance is None
        else np.asarray(observation_variance, dtype=np.float64)
    )
    if supplied_variance is not None and (
        supplied_variance.shape != response.shape[:2]
        or np.any(~np.isfinite(supplied_variance))
        or np.any(supplied_variance < 0.0)
    ):
        raise ValueError("Screening observation variance must align and be finite.")
    if use_gpu:
        import torch

        device = torch.device(gpu_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA screening requested but CUDA is unavailable.")
        response_t = torch.as_tensor(response, dtype=torch.float64, device=device)
        basis_t = torch.as_tensor(source_basis, dtype=torch.float64, device=device)
        strength_t = torch.as_tensor(
            source_strengths,
            dtype=torch.float64,
            device=device,
        )
        background_t = torch.as_tensor(
            background,
            dtype=torch.float64,
            device=device,
        )
        jacobian_t = torch.einsum("abgi,gik->abk", response_t, basis_t)
        expected_t = torch.einsum("abgi,gi->ab", response_t, strength_t)
        expected_t = torch.clamp(
            expected_t + background_t,
            min=float(minimum_expected_count),
        )
        variance_t = (
            expected_t
            if supplied_variance is None
            else torch.clamp(
                torch.as_tensor(
                    supplied_variance,
                    dtype=torch.float64,
                    device=device,
                ),
                min=float(minimum_expected_count),
            )
        )
        information_t = torch.einsum(
            "abp,abq,ab->apq",
            jacobian_t,
            jacobian_t,
            1.0 / variance_t,
        )
        information = information_t.detach().cpu().numpy()
        expected = expected_t.detach().cpu().numpy()
    else:
        jacobian = np.einsum(
            "abgi,gik->abk",
            response,
            source_basis,
            optimize=True,
        )
        expected = np.einsum(
            "abgi,gi->ab",
            response,
            source_strengths,
            optimize=True,
        )
        expected = np.maximum(
            expected + background,
            float(minimum_expected_count),
        )
        variance = (
            expected
            if supplied_variance is None
            else np.maximum(
                supplied_variance,
                float(minimum_expected_count),
            )
        )
        information = np.einsum(
            "abp,abq,ab->apq",
            jacobian,
            jacobian,
            1.0 / variance,
            optimize=True,
        )
    information = 0.5 * (information + np.swapaxes(information, 1, 2))
    return information, np.sum(expected, axis=1)


def _diverse_exact_candidate_indices(
    ranked: Sequence[MLEPlanningAction],
    poses: NDArray[np.float64],
    config: MLEPlanningConfig,
) -> NDArray[np.int64]:
    """Choose an adaptive exact shortlist with score and spatial diversity."""
    if not ranked:
        raise ValueError("ranked screening actions must be nonempty.")
    maximum = min(int(config.exact_candidate_max), len(ranked))
    minimum = min(int(config.exact_candidate_min), maximum)
    best_score = float(ranked[0].score)
    margin = max(abs(best_score), 1.0) * float(config.exact_score_margin_fraction)
    within_margin = sum(float(action.score) >= best_score - margin for action in ranked)
    target = min(maximum, max(minimum, within_margin))
    pool = tuple(ranked[: min(len(ranked), max(4 * target, target))])
    selected: list[int] = []
    forced = min(
        target,
        len(pool),
        max(minimum, int(np.ceil((1.0 - config.exact_diversity_weight) * target))),
    )
    selected.extend(int(pool[index].candidate_index) for index in range(forced))
    coordinates = np.asarray(
        [poses[int(action.candidate_index)] for action in pool],
        dtype=np.float64,
    )
    span = np.maximum(np.ptp(coordinates, axis=0), 1.0e-9)
    normalized = coordinates / span[None, :]
    score_values = np.asarray([action.score for action in pool], dtype=np.float64)
    score_span = max(float(np.ptp(score_values)), 1.0e-12)
    score_quality = (score_values - float(np.min(score_values))) / score_span
    weight = float(config.exact_diversity_weight)
    while len(selected) < target:
        selected_positions = np.asarray(
            [poses[index] / span for index in selected],
            dtype=np.float64,
        )
        distances = np.linalg.norm(
            normalized[:, None, :] - selected_positions[None, :, :],
            axis=2,
        )
        diversity = np.min(distances, axis=1)
        diversity /= max(float(np.max(diversity)), 1.0e-12)
        combined = (1.0 - weight) * score_quality + weight * diversity
        for selected_index in selected:
            combined[
                next(
                    offset
                    for offset, action in enumerate(pool)
                    if int(action.candidate_index) == selected_index
                )
            ] = -np.inf
        next_offset = int(np.argmax(combined))
        selected.append(int(pool[next_offset].candidate_index))
    return np.asarray(selected, dtype=np.int64)


def _source_basis(
    estimate: MLEEstimate,
    config: MLEPlanningConfig,
) -> tuple[NDArray[np.float64], tuple[dict[str, object], ...]]:
    """Build active patch modes plus residual exploration modes.

    The basis acts on patch-integrated strengths. Strong fitted patch/isotope
    entries receive individual dimensions. Every remaining entry is retained
    in an object-, surface-, or isotope-level aggregate, so a zero MLE region
    never disappears from the planning hypothesis space.
    """
    patch_count = len(estimate.patches)
    isotope_count = len(estimate.isotope_names)
    maximum_parameters = int(config.max_total_source_parameters)
    if maximum_parameters < isotope_count:
        raise ValueError(
            "max_total_source_parameters must be at least the isotope count."
        )
    strengths = np.asarray(
        estimate.patch_strength_by_isotope,
        dtype=np.float64,
    ).T
    candidates: list[tuple[float, int, int]] = []
    for isotope_index in range(isotope_count):
        isotope_values = strengths[:, isotope_index]
        maximum = float(np.max(isotope_values))
        threshold = maximum * float(config.active_strength_fraction)
        for patch_index in np.flatnonzero(
            (isotope_values > 0.0) & (isotope_values >= threshold)
        ):
            candidates.append(
                (
                    float(isotope_values[int(patch_index)]),
                    isotope_index,
                    int(patch_index),
                )
            )
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            estimate.patches[item[2]].patch_id,
        )
    )
    active_limit = min(
        int(config.max_active_source_parameters),
        maximum_parameters - isotope_count,
    )
    active = candidates[:active_limit]
    active_indices = {(item[2], item[1]) for item in active}

    def grouped(mode: str) -> list[tuple[tuple[object, ...], list[tuple[int, int]]]]:
        """Return deterministic groups of all non-active source coordinates."""
        rows: dict[tuple[object, ...], list[tuple[int, int]]] = {}
        for isotope_index, isotope in enumerate(estimate.isotope_names):
            for patch_index, patch in enumerate(estimate.patches):
                coordinate = (patch_index, isotope_index)
                if coordinate in active_indices:
                    continue
                if mode == "object":
                    key = (isotope, patch.object_id)
                elif mode == "surface":
                    key = (isotope, patch.surface_kind)
                else:
                    key = (isotope,)
                rows.setdefault(key, []).append(coordinate)
        return sorted(rows.items(), key=lambda item: tuple(map(str, item[0])))

    residual_groups = grouped("object")
    grouping = "object_id"
    if len(active) + len(residual_groups) > maximum_parameters:
        residual_groups = grouped("surface")
        grouping = "surface_kind"
    if len(active) + len(residual_groups) > maximum_parameters:
        residual_groups = grouped("isotope")
        grouping = "isotope"
    if len(active) + len(residual_groups) > maximum_parameters:
        raise RuntimeError("Source planning basis could not satisfy its size cap.")

    parameter_count = len(active) + len(residual_groups)
    basis = np.zeros(
        (patch_count, isotope_count, parameter_count),
        dtype=np.float64,
    )
    labels: list[dict[str, object]] = []
    floor = float(config.source_strength_scale_floor_cps_1m)
    for column, (strength, isotope_index, patch_index) in enumerate(active):
        scale = max(float(strength), floor)
        basis[patch_index, isotope_index, column] = scale
        patch = estimate.patches[patch_index]
        labels.append(
            {
                "kind": "active_patch",
                "isotope": estimate.isotope_names[isotope_index],
                "patch_ids": [int(patch.patch_id)],
                "scale_cps_1m": scale,
            }
        )
    for offset, (group_key, coordinates) in enumerate(residual_groups):
        column = len(active) + offset
        areas = np.asarray(
            [estimate.patches[index].area_m2 for index, _ in coordinates],
            dtype=np.float64,
        )
        weights = areas / float(np.sum(areas))
        for weight, (patch_index, isotope_index) in zip(
            weights,
            coordinates,
            strict=True,
        ):
            basis[patch_index, isotope_index, column] = floor * float(weight)
        isotope_index = coordinates[0][1]
        labels.append(
            {
                "kind": "residual_exploration",
                "grouping": grouping,
                "group_key": [str(value) for value in group_key],
                "isotope": estimate.isotope_names[isotope_index],
                "patch_ids": [
                    int(estimate.patches[index].patch_id) for index, _ in coordinates
                ],
                "scale_cps_1m": floor,
            }
        )
    if parameter_count == 0 or np.any(np.sum(basis, axis=(0, 1)) <= 0.0):
        raise RuntimeError("Every source planning basis column must be nonzero.")
    return basis, tuple(labels)


def _nuisance_coefficients(
    estimate: MLEEstimate,
    nuisance_names: Sequence[str],
) -> NDArray[np.float64]:
    """Restore fitted nuisance values in the response builder's exact order."""
    fitted_names = tuple(
        str(value) for value in estimate.diagnostics.get("nuisance_names", [])
    )
    fitted_values = np.concatenate(
        (
            np.asarray(estimate.background_parameters, dtype=float),
            np.asarray(estimate.nuisance_parameters, dtype=float),
        )
    )
    if len(fitted_names) == fitted_values.size and fitted_names:
        by_name = dict(zip(fitted_names, fitted_values, strict=True))
        missing = [name for name in nuisance_names if name not in by_name]
        if missing:
            raise ValueError(
                f"Estimate does not contain planner nuisance coefficients {missing}."
            )
        return np.asarray([by_name[name] for name in nuisance_names], dtype=np.float64)
    background = iter(np.asarray(estimate.background_parameters, dtype=float))
    other = iter(np.asarray(estimate.nuisance_parameters, dtype=float))
    values: list[float] = []
    for name in nuisance_names:
        selected = background if name.startswith("background") else other
        try:
            values.append(float(next(selected)))
        except StopIteration as exc:
            raise ValueError(
                "Estimate nuisance parameters do not match the response basis."
            ) from exc
    try:
        next(background)
    except StopIteration:
        pass
    else:
        raise ValueError("Estimate contains unused background parameters.")
    try:
        next(other)
    except StopIteration:
        pass
    else:
        raise ValueError("Estimate contains unused non-background nuisance parameters.")
    return np.asarray(values, dtype=np.float64)


def _spectral_design(
    observations: object,
    estimate: MLEEstimate,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[str, ...],
]:
    """Build integrated-strength and nuisance responses with shared physics."""
    calibration = (
        None
        if mle_config.discrepancy_calibration_path is None
        else load_discrepancy_calibration(mle_config.discrepancy_calibration_path)
    )
    details = build_spectral_response(
        observations,
        _PatchView(estimate.patches),
        estimate.isotope_names,
        kernel,
        chunk_size=int(mle_config.response_chunk_size),
        continuum_to_peak=float(mle_config.continuum_to_peak),
        backscatter_fraction=float(mle_config.backscatter_fraction),
        require_line_resolved=True,
        include_background_nuisance=bool(mle_config.fit_background_nuisance),
        include_scatter_nuisance=bool(mle_config.fit_scatter_nuisance),
        discrepancy_calibration=calibration,
        include_shield_leakage_nuisance=bool(mle_config.fit_shield_leakage_nuisance),
        # A future station coefficient has no fitted value.  Planner Fisher
        # information therefore marginalizes only calibrated run-global bases.
        include_station_rate_nuisance=False,
        include_low_rank_residual_nuisance=bool(
            mle_config.fit_low_rank_residual_nuisance
        ),
        include_gain_resolution_drift=bool(mle_config.fit_gain_resolution_drift),
    )
    return (
        details.response_per_integrated_strength,
        details.nuisance_response,
        details.nuisance_names,
    )


def _factorized_spectral_design(
    observations: object,
    estimate: MLEEstimate,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
) -> _LineSpectralDesign:
    """Build compact exact line factors in integrated-strength coordinates."""
    calibration = (
        None
        if mle_config.discrepancy_calibration_path is None
        else load_discrepancy_calibration(mle_config.discrepancy_calibration_path)
    )
    patch_view = _PatchView(estimate.patches)
    details = build_spectral_response_operator(
        observations,
        patch_view,
        estimate.isotope_names,
        kernel,
        chunk_size=int(mle_config.response_chunk_size),
        measurement_chunk_size=int(mle_config.response_measurement_chunk_size),
        energy_chunk_size=int(mle_config.response_energy_chunk_size),
        patch_chunk_size=int(mle_config.response_patch_chunk_size),
        worker_count=int(mle_config.response_worker_count),
        cache_directory=None,
        continuum_to_peak=float(mle_config.continuum_to_peak),
        backscatter_fraction=float(mle_config.backscatter_fraction),
        require_line_resolved=True,
        include_background_nuisance=bool(mle_config.fit_background_nuisance),
        include_scatter_nuisance=bool(mle_config.fit_scatter_nuisance),
        discrepancy_calibration=calibration,
        include_shield_leakage_nuisance=bool(mle_config.fit_shield_leakage_nuisance),
        include_station_rate_nuisance=False,
        include_low_rank_residual_nuisance=bool(
            mle_config.fit_low_rank_residual_nuisance
        ),
        include_gain_resolution_drift=bool(mle_config.fit_gain_resolution_drift),
    )
    areas = patch_view.areas_m2
    spatial = (
        np.asarray(details.operator.spatial_factors, dtype=np.float64)
        / areas[
            None,
            :,
            None,
        ]
    )
    spatial = np.ascontiguousarray(spatial, dtype=np.float64)
    spatial.setflags(write=False)
    overdispersion = (
        details.overdispersion_alpha_by_bin
        if mle_config.spectral_likelihood == "calibrated_overdispersed"
        else np.zeros_like(details.overdispersion_alpha_by_bin)
    )
    nuisance_l2_weights = np.ascontiguousarray(
        np.asarray(details.nuisance_l2_weights, dtype=np.float64)
        + float(mle_config.nuisance_l2_weight)
    )
    nuisance_l2_weights.setflags(write=False)
    return _LineSpectralDesign(
        spatial_factors=spatial,
        pulse_shapes=details.operator.pulse_shapes,
        line_isotope_indices=details.operator.line_isotope_indices,
        nuisance_response=details.nuisance_response,
        nuisance_names=details.nuisance_names,
        energy_chunk_size=int(mle_config.response_energy_chunk_size),
        nuisance_l2_weights=nuisance_l2_weights,
        overdispersion_alpha_by_bin=overdispersion,
    )


def _concatenate_factorized_designs(
    first: _LineSpectralDesign,
    second: _LineSpectralDesign,
) -> _LineSpectralDesign:
    """Append causal action rows after validating one common line model."""
    if (
        not np.array_equal(first.pulse_shapes, second.pulse_shapes)
        or not np.array_equal(
            first.line_isotope_indices,
            second.line_isotope_indices,
        )
        or first.nuisance_names != second.nuisance_names
        or first.patch_count != second.patch_count
        or not np.array_equal(
            first.nuisance_l2_weights,
            second.nuisance_l2_weights,
        )
        or not np.array_equal(
            first.overdispersion_alpha_by_bin,
            second.overdispersion_alpha_by_bin,
        )
    ):
        raise ValueError("Factorized planning designs do not share one model.")
    spatial = np.concatenate((first.spatial_factors, second.spatial_factors), axis=0)
    nuisance = np.concatenate(
        (first.nuisance_response, second.nuisance_response),
        axis=0,
    )
    spatial.setflags(write=False)
    nuisance.setflags(write=False)
    return _LineSpectralDesign(
        spatial_factors=spatial,
        pulse_shapes=first.pulse_shapes,
        line_isotope_indices=first.line_isotope_indices,
        nuisance_response=nuisance,
        nuisance_names=first.nuisance_names,
        energy_chunk_size=first.energy_chunk_size,
        nuisance_l2_weights=first.nuisance_l2_weights,
        overdispersion_alpha_by_bin=first.overdispersion_alpha_by_bin,
    )


def _screening_spectral_design(
    observations: object,
    patches: object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
) -> NDArray[np.float64]:
    """Build coarse source response without exact-stage nuisance expansion."""
    details = build_spectral_response(
        observations,
        patches,
        isotopes,
        kernel,
        chunk_size=int(mle_config.response_chunk_size),
        continuum_to_peak=float(mle_config.continuum_to_peak),
        backscatter_fraction=float(mle_config.backscatter_fraction),
        require_line_resolved=True,
        include_background_nuisance=False,
        include_scatter_nuisance=False,
        discrepancy_calibration=None,
        include_shield_leakage_nuisance=False,
        include_station_rate_nuisance=False,
        include_low_rank_residual_nuisance=False,
        include_gain_resolution_drift=False,
    )
    return details.response_per_integrated_strength


def _calibrated_screening_spectral_design(
    observations: _PlanningGeometry,
    patches: object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
    background_rate_by_bin: NDArray[np.float64],
    grouped_edges_keV: NDArray[np.float64],
) -> _GroupedNB2LineDesign:
    """Build grouped response and exact grouped NB2 variance from line factors."""
    if mle_config.discrepancy_calibration_path is None:
        raise ValueError("Calibrated screening requires a calibration artifact.")
    calibration = load_discrepancy_calibration(mle_config.discrepancy_calibration_path)
    details = build_spectral_response_operator(
        observations,
        patches,
        isotopes,
        kernel,
        chunk_size=int(mle_config.response_chunk_size),
        measurement_chunk_size=int(mle_config.response_measurement_chunk_size),
        energy_chunk_size=int(mle_config.response_energy_chunk_size),
        patch_chunk_size=int(mle_config.response_patch_chunk_size),
        worker_count=int(mle_config.response_worker_count),
        cache_directory=None,
        continuum_to_peak=float(mle_config.continuum_to_peak),
        backscatter_fraction=float(mle_config.backscatter_fraction),
        require_line_resolved=True,
        include_background_nuisance=False,
        include_scatter_nuisance=False,
        discrepancy_calibration=calibration,
        include_shield_leakage_nuisance=False,
        include_station_rate_nuisance=False,
        include_low_rank_residual_nuisance=False,
        include_gain_resolution_drift=False,
    )
    operator = details.operator
    if not isinstance(operator, LineFactorizedResponseOperator):
        raise TypeError("Calibrated screening requires line-factorized response.")
    full_edges = np.asarray(observations.energy_bin_edges_keV, dtype=np.float64)
    grouped_edges = np.asarray(grouped_edges_keV, dtype=np.float64)
    full_centers = 0.5 * (full_edges[:-1] + full_edges[1:])
    group_indices = np.searchsorted(grouped_edges, full_centers, side="right") - 1
    group_count = int(grouped_edges.size - 1)
    group_indices = np.clip(group_indices, 0, group_count - 1).astype(
        np.int64,
        copy=False,
    )
    grouped_pulses = _group_last_axis(
        operator.pulse_shapes,
        group_indices,
        group_count,
    )
    measurement_count = int(operator.observation_shape[0])
    grouped_response = np.zeros(
        (
            measurement_count,
            group_count,
            operator.patch_count,
            operator.isotope_count,
        ),
        dtype=np.float64,
    )
    for line_index, isotope_index in enumerate(operator.line_isotope_indices):
        grouped_response[:, :, :, int(isotope_index)] += (
            operator.spatial_factors[:, None, :, line_index]
            * grouped_pulses[line_index][None, :, None]
        )
    background_rate = np.asarray(background_rate_by_bin, dtype=np.float64)
    if background_rate.shape != (operator.observation_shape[1],):
        raise ValueError("Fine screening background rate must match energy bins.")
    live_times = np.asarray(observations.live_times_s, dtype=np.float64)
    grouped_background = np.zeros(
        (measurement_count, group_count),
        dtype=np.float64,
    )
    energy_step = int(mle_config.response_energy_chunk_size)
    for start in range(0, operator.observation_shape[1], energy_step):
        stop = min(start + energy_step, operator.observation_shape[1])
        grouped_background += _group_last_axis(
            live_times[:, None] * background_rate[None, start:stop],
            group_indices[start:stop],
            group_count,
        )
    return _GroupedNB2LineDesign(
        grouped_response=grouped_response,
        grouped_background=grouped_background,
        spatial_factors=operator.spatial_factors,
        pulse_shapes=operator.pulse_shapes,
        line_isotope_indices=operator.line_isotope_indices,
        live_times_s=live_times,
        background_rate_by_bin=background_rate,
        overdispersion_alpha_by_bin=details.overdispersion_alpha_by_bin,
        fine_group_indices=group_indices,
        group_count=group_count,
        energy_chunk_size=energy_step,
    )


def _historical_spectral_design(
    observations: ObservationBatch,
    estimate: MLEEstimate,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
    cache: dict[str, object] | None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    tuple[str, ...],
    dict[str, object],
]:
    """Build or append the exact historical design for a causal prefix."""
    step_ids = tuple(int(value) for value in observations.step_ids)
    row_keys = _historical_row_keys(observations)
    identity = (
        tuple(
            (
                int(patch.patch_id),
                np.asarray(patch.centroid_xyz, dtype=np.float64).tobytes(),
                float(patch.area_m2),
                np.asarray(
                    patch.quadrature_points_xyz,
                    dtype=np.float64,
                ).tobytes(),
                np.asarray(
                    patch.quadrature_weights,
                    dtype=np.float64,
                ).tobytes(),
            )
            for patch in estimate.patches
        ),
        tuple(estimate.isotope_names),
        _kernel_physical_identity(kernel),
        observations.energy_bin_edges_keV.tobytes(),
        float(mle_config.continuum_to_peak),
        float(mle_config.backscatter_fraction),
        bool(mle_config.fit_background_nuisance),
        bool(mle_config.fit_scatter_nuisance),
        bool(mle_config.fit_shield_leakage_nuisance),
        bool(mle_config.fit_low_rank_residual_nuisance),
        bool(mle_config.fit_gain_resolution_drift),
        _calibration_identity(mle_config.discrepancy_calibration_path),
    )
    entry = None if cache is None else cache.get("historical_design")
    previous_count = 0
    if isinstance(entry, dict) and entry.get("identity") == identity:
        previous_rows = entry.get("row_keys")
        if isinstance(previous_rows, tuple) and row_keys[: len(previous_rows)] == (
            previous_rows
        ):
            previous_count = len(previous_rows)
            if previous_count == len(step_ids):
                return (
                    np.asarray(entry["source"], dtype=np.float64),
                    np.asarray(entry["nuisance"], dtype=np.float64),
                    tuple(entry["nuisance_names"]),
                    {
                        "mode": "prefix_hit",
                        "reused_measurements": previous_count,
                        "computed_measurements": 0,
                    },
                )
    if previous_count and mle_config.fit_gain_resolution_drift:
        previous_count = 0
    if previous_count:
        selected = slice(previous_count, len(step_ids))
        suffix = _PlanningGeometry(
            detector_positions_xyz=observations.detector_positions_xyz[selected],
            fe_indices=observations.fe_indices[selected],
            pb_indices=observations.pb_indices[selected],
            live_times_s=observations.live_times_s[selected],
            energy_bin_edges_keV=observations.energy_bin_edges_keV,
            station_ids=observations.station_ids[selected],
        )
        suffix_source, suffix_nuisance, nuisance_names = _spectral_design(
            suffix,
            estimate,
            kernel,
            mle_config,
        )
        assert isinstance(entry, dict)
        if tuple(entry["nuisance_names"]) == tuple(nuisance_names):
            source = np.concatenate(
                (np.asarray(entry["source"], dtype=np.float64), suffix_source),
                axis=0,
            )
            nuisance = np.concatenate(
                (np.asarray(entry["nuisance"], dtype=np.float64), suffix_nuisance),
                axis=0,
            )
            mode = "prefix_append"
        else:
            previous_count = 0
    if previous_count == 0:
        source, nuisance, nuisance_names = _spectral_design(
            observations,
            estimate,
            kernel,
            mle_config,
        )
        mode = "full_rebuild"
    if cache is not None:
        cache["historical_design"] = {
            "identity": identity,
            "step_ids": step_ids,
            "row_keys": row_keys,
            "source": source,
            "nuisance": nuisance,
            "nuisance_names": tuple(nuisance_names),
        }
    return (
        source,
        nuisance,
        tuple(nuisance_names),
        {
            "mode": mode,
            "reused_measurements": previous_count,
            "computed_measurements": len(step_ids) - previous_count,
        },
    )


def _historical_factorized_spectral_design(
    observations: ObservationBatch,
    estimate: MLEEstimate,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
    cache: dict[str, object] | None,
) -> tuple[_LineSpectralDesign, dict[str, object]]:
    """Build or append compact historical factors for one causal prefix."""
    step_ids = tuple(int(value) for value in observations.step_ids)
    row_keys = _historical_row_keys(observations)
    identity = (
        tuple(
            (
                int(patch.patch_id),
                np.asarray(patch.centroid_xyz, dtype=np.float64).tobytes(),
                float(patch.area_m2),
                np.asarray(
                    patch.quadrature_points_xyz,
                    dtype=np.float64,
                ).tobytes(),
                np.asarray(
                    patch.quadrature_weights,
                    dtype=np.float64,
                ).tobytes(),
            )
            for patch in estimate.patches
        ),
        tuple(estimate.isotope_names),
        _kernel_physical_identity(kernel),
        observations.energy_bin_edges_keV.tobytes(),
        float(mle_config.continuum_to_peak),
        float(mle_config.backscatter_fraction),
        bool(mle_config.fit_background_nuisance),
        bool(mle_config.fit_scatter_nuisance),
        bool(mle_config.fit_shield_leakage_nuisance),
        bool(mle_config.fit_low_rank_residual_nuisance),
        bool(mle_config.fit_gain_resolution_drift),
        str(mle_config.spectral_likelihood),
        float(mle_config.nuisance_l2_weight),
        int(mle_config.response_energy_chunk_size),
        _calibration_identity(mle_config.discrepancy_calibration_path),
    )
    entry = None if cache is None else cache.get("historical_factorized_design")
    previous_count = 0
    if isinstance(entry, dict) and entry.get("identity") == identity:
        previous_rows = entry.get("row_keys")
        previous_design = entry.get("design")
        if (
            isinstance(previous_rows, tuple)
            and isinstance(previous_design, _LineSpectralDesign)
            and row_keys[: len(previous_rows)] == previous_rows
        ):
            previous_count = len(previous_rows)
            if previous_count == len(step_ids):
                return previous_design, {
                    "mode": "prefix_hit",
                    "reused_measurements": previous_count,
                    "computed_measurements": 0,
                }
    if previous_count and mle_config.fit_gain_resolution_drift:
        previous_count = 0
    if previous_count:
        selected = slice(previous_count, len(step_ids))
        suffix = _PlanningGeometry(
            detector_positions_xyz=observations.detector_positions_xyz[selected],
            fe_indices=observations.fe_indices[selected],
            pb_indices=observations.pb_indices[selected],
            live_times_s=observations.live_times_s[selected],
            energy_bin_edges_keV=observations.energy_bin_edges_keV,
            station_ids=observations.station_ids[selected],
        )
        suffix_design = _factorized_spectral_design(
            suffix,
            estimate,
            kernel,
            mle_config,
        )
        assert isinstance(entry, dict)
        previous_design = entry.get("design")
        if isinstance(previous_design, _LineSpectralDesign):
            design = _concatenate_factorized_designs(
                previous_design,
                suffix_design,
            )
            mode = "prefix_append"
        else:
            previous_count = 0
    if previous_count == 0:
        design = _factorized_spectral_design(
            observations,
            estimate,
            kernel,
            mle_config,
        )
        mode = "full_rebuild"
    if cache is not None:
        cache["historical_factorized_design"] = {
            "identity": identity,
            "step_ids": step_ids,
            "row_keys": row_keys,
            "design": design,
        }
    return design, {
        "mode": mode,
        "reused_measurements": previous_count,
        "computed_measurements": len(step_ids) - previous_count,
    }


def _cached_model_identity(
    cache: dict[str, object] | None,
    entry_name: str,
) -> object | None:
    """Return the physical model identity stored with one response cache."""
    entry = None if cache is None else cache.get(entry_name)
    return entry.get("identity") if isinstance(entry, dict) else None


def _fisher_information(
    source_response: NDArray[np.float64],
    nuisance_response: NDArray[np.float64],
    source_basis: NDArray[np.float64],
    source_strengths: NDArray[np.float64],
    nuisance_coefficients: NDArray[np.float64],
    nuisance_scales: NDArray[np.float64],
    *,
    minimum_expected_count: float,
    overdispersion_alpha_by_bin: NDArray[np.float64] | None = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return Fisher terms including a shared future station-rate nuisance."""
    response = np.asarray(source_response, dtype=np.float64)
    nuisance = np.asarray(nuisance_response, dtype=np.float64)
    if response.ndim != 4:
        raise ValueError("Spectral source_response must have shape (A, B, G, I).")
    action_count, bin_count, patch_count, isotope_count = response.shape
    if source_basis.shape[:2] != (patch_count, isotope_count):
        raise ValueError("source_basis does not match response patch/isotope axes.")
    if source_strengths.shape != (patch_count, isotope_count):
        raise ValueError("source_strengths do not match the response axes.")
    if nuisance.shape[:2] != (action_count, bin_count):
        raise ValueError("nuisance_response does not match action/bin axes.")
    if nuisance.shape[2] != nuisance_coefficients.size or (
        nuisance_coefficients.shape != nuisance_scales.shape
    ):
        raise ValueError("Nuisance response, coefficients, and scales must align.")
    alpha = (
        np.zeros(bin_count, dtype=np.float64)
        if overdispersion_alpha_by_bin is None
        else np.asarray(overdispersion_alpha_by_bin, dtype=np.float64)
    )
    if alpha.shape != (bin_count,) or np.any(~np.isfinite(alpha)) or np.any(alpha < 0):
        raise ValueError("Fisher overdispersion alpha must match energy bins.")
    if use_gpu:
        import torch

        device = torch.device(gpu_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA Fisher planning requested but CUDA is unavailable."
            )

        def tensor(values: object) -> object:
            """Copy one validated planner array to float64 on the GPU."""
            return torch.as_tensor(values, dtype=torch.float64, device=device)

        response_t = tensor(response)
        nuisance_t = tensor(nuisance)
        source_jacobian_t = torch.einsum(
            "abgi,gik->abk",
            response_t,
            tensor(source_basis),
        )
        nuisance_jacobian_t = nuisance_t * tensor(nuisance_scales)[None, None, :]
        jacobian_t = torch.cat((source_jacobian_t, nuisance_jacobian_t), dim=2)
        expected_raw_t = torch.einsum(
            "abgi,gi->ab",
            response_t,
            tensor(source_strengths),
        )
        if nuisance_coefficients.size:
            expected_raw_t = expected_raw_t + torch.einsum(
                "abn,n->ab",
                nuisance_t,
                tensor(nuisance_coefficients),
            )
        expected_t = torch.clamp(
            expected_raw_t,
            min=float(minimum_expected_count),
        )
        variance_t = expected_t + tensor(alpha)[None, :] * expected_t.square()
        weighted_jacobian_t = jacobian_t * torch.rsqrt(variance_t).unsqueeze(-1)
        information_t = torch.bmm(
            weighted_jacobian_t.transpose(1, 2),
            weighted_jacobian_t,
        )
        information_t = 0.5 * information_t.add(information_t.transpose(1, 2))
        station_derivative_t = torch.clamp(expected_raw_t, min=0.0)
        station_weight_t = station_derivative_t / variance_t
        station_cross_t = torch.sum(
            jacobian_t * station_weight_t.unsqueeze(-1),
            dim=1,
        )
        station_information_t = torch.sum(
            station_derivative_t * station_weight_t,
            dim=1,
        )
        expected_totals_t = torch.sum(station_derivative_t, dim=1)
        information = information_t.detach().cpu().numpy()
        expected_totals = expected_totals_t.detach().cpu().numpy()
        station_cross = station_cross_t.detach().cpu().numpy()
        station_information = station_information_t.detach().cpu().numpy()
    else:
        source_jacobian = np.einsum(
            "abgi,gik->abk",
            response,
            source_basis,
            optimize=True,
        )
        nuisance_jacobian = nuisance * nuisance_scales[None, None, :]
        jacobian = np.concatenate((source_jacobian, nuisance_jacobian), axis=2)
        expected_raw = np.einsum(
            "abgi,gi->ab",
            response,
            source_strengths,
            optimize=True,
        )
        if nuisance_coefficients.size:
            expected_raw = expected_raw + np.einsum(
                "abn,n->ab",
                nuisance,
                nuisance_coefficients,
                optimize=True,
            )
        expected = np.maximum(expected_raw, float(minimum_expected_count))
        variance = expected + alpha[None, :] * expected**2
        weighted_jacobian = jacobian / np.sqrt(variance)[..., None]
        information = np.matmul(
            np.swapaxes(weighted_jacobian, 1, 2),
            weighted_jacobian,
        )
        information = 0.5 * (information + np.swapaxes(information, 1, 2))
        station_derivative = np.maximum(expected_raw, 0.0)
        station_weight = station_derivative / variance
        station_cross = np.sum(jacobian * station_weight[..., None], axis=1)
        station_information = np.sum(station_derivative * station_weight, axis=1)
        expected_totals = np.sum(station_derivative, axis=1)
    return information, expected_totals, station_cross, station_information


def _factorized_source_spectrum(
    design: _LineSpectralDesign,
    source_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project compact line factors onto one patch-isotope source map."""
    weights = np.asarray(source_weights, dtype=np.float64)
    if weights.shape != (design.patch_count, design.isotope_count):
        raise ValueError("Source weights do not match factorized planning axes.")
    weights_by_line = weights[:, design.line_isotope_indices]
    amplitudes = np.einsum(
        "mgl,gl->ml",
        design.spatial_factors,
        weights_by_line,
        optimize=True,
    )
    return amplitudes @ design.pulse_shapes


def _factorized_fisher_information(
    design: _LineSpectralDesign,
    source_basis: NDArray[np.float64],
    source_strengths: NDArray[np.float64],
    nuisance_coefficients: NDArray[np.float64],
    nuisance_scales: NDArray[np.float64],
    *,
    minimum_expected_count: float,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Reduce exact Fisher terms from line factors in bounded energy chunks."""
    basis = np.asarray(source_basis, dtype=np.float64)
    strengths = np.asarray(source_strengths, dtype=np.float64)
    nuisance_coefficients = np.asarray(nuisance_coefficients, dtype=np.float64)
    nuisance_scales = np.asarray(nuisance_scales, dtype=np.float64)
    if basis.shape[:2] != (design.patch_count, design.isotope_count):
        raise ValueError("source_basis does not match factorized response axes.")
    if strengths.shape != (design.patch_count, design.isotope_count):
        raise ValueError("source_strengths do not match factorized response axes.")
    nuisance_count = int(nuisance_coefficients.size)
    if (
        design.nuisance_response.shape != (*design.observation_shape, nuisance_count)
        or nuisance_scales.shape != nuisance_coefficients.shape
    ):
        raise ValueError("Factorized nuisance response and coefficients must align.")
    basis_by_line = basis[:, design.line_isotope_indices, :]
    strengths_by_line = strengths[:, design.line_isotope_indices]
    action_count, bin_count = design.observation_shape
    alpha = np.asarray(design.overdispersion_alpha_by_bin, dtype=np.float64)
    if alpha.size == 0:
        alpha = np.zeros(bin_count, dtype=np.float64)
    if alpha.shape != (bin_count,) or np.any(~np.isfinite(alpha)) or np.any(alpha < 0):
        raise ValueError("Factorized Fisher overdispersion must match energy bins.")
    parameter_count = int(basis.shape[2] + nuisance_count)
    energy_step = max(1, int(design.energy_chunk_size))
    if use_gpu:
        import torch

        device = torch.device(gpu_device)
        if device.type != "cuda":
            raise ValueError("GPU Fisher planning requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA Fisher planning requested but CUDA is unavailable."
            )

        def tensor(values: object) -> object:
            """Copy one factorized planner array to CUDA float64."""
            return torch.as_tensor(values, dtype=torch.float64, device=device)

        spatial_t = tensor(design.spatial_factors)
        pulses_t = tensor(design.pulse_shapes)
        nuisance_t = tensor(design.nuisance_response)
        amplitude_basis_t = torch.einsum(
            "mgl,glk->mlk",
            spatial_t,
            tensor(basis_by_line),
        )
        amplitude_strength_t = torch.einsum(
            "mgl,gl->ml",
            spatial_t,
            tensor(strengths_by_line),
        )
        nuisance_coefficients_t = tensor(nuisance_coefficients)
        nuisance_scales_t = tensor(nuisance_scales)
        alpha_t = tensor(alpha)
        information_t = torch.zeros(
            (action_count, parameter_count, parameter_count),
            dtype=torch.float64,
            device=device,
        )
        station_cross_t = torch.zeros(
            (action_count, parameter_count),
            dtype=torch.float64,
            device=device,
        )
        station_information_t = torch.zeros(
            action_count,
            dtype=torch.float64,
            device=device,
        )
        expected_totals_t = torch.zeros(
            action_count,
            dtype=torch.float64,
            device=device,
        )
        for energy_start in range(0, bin_count, energy_step):
            energy_stop = min(energy_start + energy_step, bin_count)
            pulse_chunk = pulses_t[:, energy_start:energy_stop]
            source_jacobian = torch.einsum(
                "mlk,le->mek",
                amplitude_basis_t,
                pulse_chunk,
            )
            nuisance_chunk = nuisance_t[:, energy_start:energy_stop]
            nuisance_jacobian = nuisance_chunk * nuisance_scales_t[None, None, :]
            jacobian = torch.cat((source_jacobian, nuisance_jacobian), dim=2)
            expected_raw = torch.einsum(
                "ml,le->me",
                amplitude_strength_t,
                pulse_chunk,
            )
            if nuisance_count:
                expected_raw = expected_raw + torch.einsum(
                    "men,n->me",
                    nuisance_chunk,
                    nuisance_coefficients_t,
                )
            expected = torch.clamp(
                expected_raw,
                min=float(minimum_expected_count),
            )
            alpha_chunk = alpha_t[energy_start:energy_stop]
            variance = expected + alpha_chunk[None, :] * expected.square()
            weighted_jacobian = jacobian * torch.rsqrt(variance).unsqueeze(-1)
            information_t += torch.bmm(
                weighted_jacobian.transpose(1, 2),
                weighted_jacobian,
            )
            station_derivative = torch.clamp(expected_raw, min=0.0)
            station_weight = station_derivative / variance
            station_cross_t += torch.sum(
                jacobian * station_weight.unsqueeze(-1),
                dim=1,
            )
            station_information_t += torch.sum(
                station_derivative * station_weight,
                dim=1,
            )
            expected_totals_t += torch.sum(station_derivative, dim=1)
        information_t = 0.5 * information_t.add(information_t.transpose(1, 2))
        information = information_t.cpu().numpy()
        expected_totals = expected_totals_t.cpu().numpy()
        station_cross = station_cross_t.cpu().numpy()
        station_information = station_information_t.cpu().numpy()
    else:
        amplitude_basis = np.einsum(
            "mgl,glk->mlk",
            design.spatial_factors,
            basis_by_line,
            optimize=True,
        )
        amplitude_strength = np.einsum(
            "mgl,gl->ml",
            design.spatial_factors,
            strengths_by_line,
            optimize=True,
        )
        information = np.zeros(
            (action_count, parameter_count, parameter_count),
            dtype=np.float64,
        )
        station_cross = np.zeros(
            (action_count, parameter_count),
            dtype=np.float64,
        )
        station_information = np.zeros(action_count, dtype=np.float64)
        expected_totals = np.zeros(action_count, dtype=np.float64)
        for energy_start in range(0, bin_count, energy_step):
            energy_stop = min(energy_start + energy_step, bin_count)
            pulse_chunk = design.pulse_shapes[:, energy_start:energy_stop]
            source_jacobian = np.einsum(
                "mlk,le->mek",
                amplitude_basis,
                pulse_chunk,
                optimize=True,
            )
            nuisance_chunk = design.nuisance_response[
                :,
                energy_start:energy_stop,
            ]
            nuisance_jacobian = nuisance_chunk * nuisance_scales[None, None, :]
            jacobian = np.concatenate(
                (source_jacobian, nuisance_jacobian),
                axis=2,
            )
            expected_raw = np.einsum(
                "ml,le->me",
                amplitude_strength,
                pulse_chunk,
                optimize=True,
            )
            if nuisance_count:
                expected_raw += np.einsum(
                    "men,n->me",
                    nuisance_chunk,
                    nuisance_coefficients,
                    optimize=True,
                )
            expected = np.maximum(expected_raw, float(minimum_expected_count))
            alpha_chunk = alpha[energy_start:energy_stop]
            variance = expected + alpha_chunk[None, :] * expected**2
            weighted_jacobian = jacobian / np.sqrt(variance)[..., None]
            information += np.matmul(
                np.swapaxes(weighted_jacobian, 1, 2),
                weighted_jacobian,
            )
            station_derivative = np.maximum(expected_raw, 0.0)
            station_weight = station_derivative / variance
            station_cross += np.sum(jacobian * station_weight[..., None], axis=1)
            station_information += np.sum(
                station_derivative * station_weight,
                axis=1,
            )
            expected_totals += np.sum(station_derivative, axis=1)
        information = 0.5 * (information + np.swapaxes(information, 1, 2))
    return information, expected_totals, station_cross, station_information


def _historical_fisher_precision(
    source_response: NDArray[np.float64],
    nuisance_response: NDArray[np.float64],
    source_basis: NDArray[np.float64],
    source_strengths: NDArray[np.float64],
    nuisance_coefficients: NDArray[np.float64],
    nuisance_scales: NDArray[np.float64],
    history_keys: Sequence[object],
    *,
    model_identity: object | None = None,
    minimum_expected_count: float,
    cache: dict[str, object] | None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Reuse historical Fisher terms only while their fitted state is exact."""
    steps = tuple(history_keys)
    parameter_identity = (
        model_identity,
        np.asarray(source_basis, dtype=np.float64).tobytes(),
        np.asarray(source_strengths, dtype=np.float64).tobytes(),
        np.asarray(nuisance_coefficients, dtype=np.float64).tobytes(),
        np.asarray(nuisance_scales, dtype=np.float64).tobytes(),
        float(minimum_expected_count),
    )
    entry = None if cache is None else cache.get("historical_fisher")
    previous_count = 0
    if isinstance(entry, dict) and entry.get("identity") == parameter_identity:
        previous_steps = entry.get("step_ids")
        if isinstance(previous_steps, tuple) and steps[: len(previous_steps)] == (
            previous_steps
        ):
            previous_count = len(previous_steps)
            if previous_count == len(steps):
                return np.asarray(entry["precision"], dtype=np.float64), {
                    "mode": "prefix_hit",
                    "reused_measurements": previous_count,
                    "computed_measurements": 0,
                }
    if previous_count:
        information, _, _, _ = _fisher_information(
            source_response[previous_count:],
            nuisance_response[previous_count:],
            source_basis,
            source_strengths,
            nuisance_coefficients,
            nuisance_scales,
            minimum_expected_count=minimum_expected_count,
        )
        assert isinstance(entry, dict)
        precision = np.asarray(entry["precision"], dtype=np.float64) + np.sum(
            information,
            axis=0,
        )
        mode = "prefix_append"
    else:
        information, _, _, _ = _fisher_information(
            source_response,
            nuisance_response,
            source_basis,
            source_strengths,
            nuisance_coefficients,
            nuisance_scales,
            minimum_expected_count=minimum_expected_count,
        )
        precision = np.sum(information, axis=0)
        mode = "full_rebuild"
    precision = np.asarray(precision, dtype=np.float64)
    precision.setflags(write=False)
    if cache is not None:
        cache["historical_fisher"] = {
            "identity": parameter_identity,
            "step_ids": steps,
            "precision": precision,
        }
    return precision, {
        "mode": mode,
        "reused_measurements": previous_count,
        "computed_measurements": len(steps) - previous_count,
    }


def _historical_factorized_fisher_precision(
    design: _LineSpectralDesign,
    source_basis: NDArray[np.float64],
    source_strengths: NDArray[np.float64],
    nuisance_coefficients: NDArray[np.float64],
    nuisance_scales: NDArray[np.float64],
    history_keys: Sequence[object],
    *,
    model_identity: object | None = None,
    minimum_expected_count: float,
    cache: dict[str, object] | None,
    use_gpu: bool,
    gpu_device: str,
    cache_entry_name: str = "historical_factorized_fisher",
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Reuse compact-design historical Fisher terms for a causal prefix."""
    steps = tuple(history_keys)
    parameter_identity = (
        model_identity,
        np.asarray(design.overdispersion_alpha_by_bin, dtype=np.float64).tobytes(),
        np.asarray(source_basis, dtype=np.float64).tobytes(),
        np.asarray(source_strengths, dtype=np.float64).tobytes(),
        np.asarray(nuisance_coefficients, dtype=np.float64).tobytes(),
        np.asarray(nuisance_scales, dtype=np.float64).tobytes(),
        float(minimum_expected_count),
    )
    entry = None if cache is None else cache.get(cache_entry_name)
    previous_count = 0
    if isinstance(entry, dict) and entry.get("identity") == parameter_identity:
        previous_steps = entry.get("step_ids")
        if isinstance(previous_steps, tuple) and steps[: len(previous_steps)] == (
            previous_steps
        ):
            previous_count = len(previous_steps)
            if previous_count == len(steps):
                return np.asarray(entry["precision"], dtype=np.float64), {
                    "mode": "prefix_hit",
                    "reused_measurements": previous_count,
                    "computed_measurements": 0,
                }
    selected_design = (
        _LineSpectralDesign(
            spatial_factors=design.spatial_factors[previous_count:],
            pulse_shapes=design.pulse_shapes,
            line_isotope_indices=design.line_isotope_indices,
            nuisance_response=design.nuisance_response[previous_count:],
            nuisance_names=design.nuisance_names,
            energy_chunk_size=design.energy_chunk_size,
            nuisance_l2_weights=design.nuisance_l2_weights,
            overdispersion_alpha_by_bin=design.overdispersion_alpha_by_bin,
        )
        if previous_count
        else design
    )
    information, _, _, _ = _factorized_fisher_information(
        selected_design,
        source_basis,
        source_strengths,
        nuisance_coefficients,
        nuisance_scales,
        minimum_expected_count=minimum_expected_count,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    precision = np.sum(information, axis=0)
    if previous_count:
        assert isinstance(entry, dict)
        precision += np.asarray(entry["precision"], dtype=np.float64)
        mode = "prefix_append"
    else:
        mode = "full_rebuild"
    precision = np.asarray(precision, dtype=np.float64)
    precision.setflags(write=False)
    if cache is not None:
        cache[cache_entry_name] = {
            "identity": parameter_identity,
            "step_ids": steps,
            "precision": precision,
        }
    return precision, {
        "mode": mode,
        "reused_measurements": previous_count,
        "computed_measurements": len(steps) - previous_count,
    }


def _symmetric_spectral_separation(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    overdispersion_alpha_by_bin: NDArray[np.float64] | None = None,
    common_counts: NDArray[np.float64] | None = None,
    *,
    first_variance: NDArray[np.float64] | None = None,
    second_variance: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return bounded separation using complete shared Poisson/NB2 means."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if first_values.shape != second_values.shape or first_values.ndim < 1:
        raise ValueError("Spectral hypotheses must have matching array shapes.")
    if (
        np.any(~np.isfinite(first_values))
        or np.any(first_values < 0.0)
        or np.any(~np.isfinite(second_values))
        or np.any(second_values < 0.0)
    ):
        raise ValueError("Spectral hypotheses must contain finite non-negative counts.")
    if (first_variance is None) != (second_variance is None):
        raise ValueError(
            "Both spectral hypothesis variances must be supplied together."
        )
    if first_variance is not None and second_variance is not None:
        first_variance_values = np.asarray(first_variance, dtype=np.float64)
        second_variance_values = np.asarray(second_variance, dtype=np.float64)
        if (
            first_variance_values.shape != first_values.shape
            or second_variance_values.shape != second_values.shape
            or np.any(~np.isfinite(first_variance_values))
            or np.any(first_variance_values < 0.0)
            or np.any(~np.isfinite(second_variance_values))
            or np.any(second_variance_values < 0.0)
        ):
            raise ValueError(
                "Spectral hypothesis variances must match finite non-negative counts."
            )
        denominator = first_variance_values + second_variance_values
    else:
        alpha = (
            np.zeros(first_values.shape[-1], dtype=np.float64)
            if overdispersion_alpha_by_bin is None
            else np.asarray(overdispersion_alpha_by_bin, dtype=np.float64)
        )
        if alpha.size == 0:
            alpha = np.zeros(first_values.shape[-1], dtype=np.float64)
        if (
            alpha.shape not in {(first_values.shape[-1],), first_values.shape}
            or np.any(~np.isfinite(alpha))
            or np.any(alpha < 0.0)
        ):
            raise ValueError("Spectral separation overdispersion must match bins.")
        common = (
            np.zeros_like(first_values)
            if common_counts is None
            else np.asarray(common_counts, dtype=np.float64)
        )
        try:
            common = np.broadcast_to(common, first_values.shape)
        except ValueError as exc:
            raise ValueError(
                "Common spectral counts must broadcast to both hypotheses."
            ) from exc
        if np.any(~np.isfinite(common)):
            raise ValueError("Common spectral counts must be finite.")
        first_mean = np.maximum(first_values + common, 0.0)
        second_mean = np.maximum(second_values + common, 0.0)
        denominator = (
            first_mean + second_mean + alpha * (first_mean**2 + second_mean**2)
        )
    distance = 0.5 * np.sum(
        np.divide(
            (first_values - second_values) ** 2,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        ),
        axis=-1,
    )
    return -np.expm1(-np.maximum(distance, 0.0))


def _aligned_support_strengths(
    base_strengths: NDArray[np.float64],
    alternative_strengths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Match isotope totals so only within-isotope support can differ."""
    base = np.asarray(base_strengths, dtype=np.float64)
    alternative = np.asarray(alternative_strengths, dtype=np.float64)
    if base.shape != alternative.shape or base.ndim != 2:
        raise ValueError("Support strength maps must have matching G x I shapes.")
    if (
        np.any(~np.isfinite(base))
        or np.any(base < 0.0)
        or np.any(~np.isfinite(alternative))
        or np.any(alternative < 0.0)
    ):
        raise ValueError("Support strength maps must be finite and non-negative.")
    aligned = base.copy()
    base_totals = np.sum(base, axis=0)
    alternative_totals = np.sum(alternative, axis=0)
    shared = (base_totals > 0.0) & (alternative_totals > 0.0)
    if np.any(shared):
        aligned[:, shared] = (
            alternative[:, shared]
            * (base_totals[shared] / alternative_totals[shared])[None, :]
        )
    return aligned


def _vertical_basis_scale(
    patch_z: NDArray[np.float64],
    source_basis: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return one centered vertical coordinate for each source basis mode."""
    z_values = np.asarray(patch_z, dtype=np.float64)
    basis = np.asarray(source_basis, dtype=np.float64)
    if basis.ndim != 3 or z_values.shape != (basis.shape[0],):
        raise ValueError("Patch heights and source basis must align.")
    basis_mass = np.sum(np.abs(basis), axis=(0, 1))
    basis_z = np.einsum(
        "g,gik->k",
        z_values,
        np.abs(basis),
        optimize=True,
    ) / np.maximum(basis_mass, 1.0e-30)
    z_span = max(float(np.ptp(z_values)), 1.0e-12)
    return (basis_z - float(np.mean(basis_z))) / z_span


def _response_ambiguity_metrics(
    project: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    information: NDArray[np.float64],
    floor_weights: NDArray[np.float64],
    ceiling_weights: NDArray[np.float64],
    base_strengths: NDArray[np.float64],
    alternative_strengths: Sequence[NDArray[np.float64]],
    z_scale: NDArray[np.float64],
    overdispersion_alpha_by_bin: NDArray[np.float64],
    common_counts: NDArray[np.float64] | None,
    project_moments: (
        Callable[[NDArray[np.float64]], _SpectralHypothesisMoments] | None
    ) = None,
) -> dict[str, NDArray[np.float64]]:
    """Return response-dependent ambiguity metrics for one action batch."""
    fisher = np.asarray(information, dtype=np.float64)

    def hypothesis_moments(
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Project source counts and an optional exact grouped variance."""
        if project_moments is None:
            return np.asarray(project(weights), dtype=np.float64), None
        moments = project_moments(weights)
        if not isinstance(moments, _SpectralHypothesisMoments):
            raise TypeError("Hypothesis moment projection returned an invalid value.")
        return (
            np.asarray(moments.source_counts, dtype=np.float64),
            np.asarray(moments.total_variance, dtype=np.float64),
        )

    def separation(
        first_spectrum: NDArray[np.float64],
        second_spectrum: NDArray[np.float64],
        first_variance: NDArray[np.float64] | None,
        second_variance: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return exact available separation for two projected hypotheses."""
        if first_variance is not None and second_variance is not None:
            return _symmetric_spectral_separation(
                first_spectrum,
                second_spectrum,
                first_variance=first_variance,
                second_variance=second_variance,
            )
        return _symmetric_spectral_separation(
            first_spectrum,
            second_spectrum,
            overdispersion_alpha_by_bin,
            common_counts,
        )

    floor_spectrum, floor_variance = hypothesis_moments(floor_weights)
    ceiling_spectrum, ceiling_variance = hypothesis_moments(ceiling_weights)
    if (
        fisher.ndim != 3
        or floor_spectrum.shape != ceiling_spectrum.shape
        or floor_spectrum.ndim != 2
        or fisher.shape[0] != floor_spectrum.shape[0]
    ):
        raise ValueError("Ambiguity spectra and Fisher actions must align.")
    floor_ceiling = separation(
        floor_spectrum,
        ceiling_spectrum,
        floor_variance,
        ceiling_variance,
    )
    floor_centered = floor_spectrum - np.mean(
        floor_spectrum,
        axis=1,
        keepdims=True,
    )
    ceiling_centered = ceiling_spectrum - np.mean(
        ceiling_spectrum,
        axis=1,
        keepdims=True,
    )
    denominator = np.linalg.norm(floor_centered, axis=1) * np.linalg.norm(
        ceiling_centered,
        axis=1,
    )
    correlation = np.divide(
        np.sum(floor_centered * ceiling_centered, axis=1),
        denominator,
        out=np.ones(floor_spectrum.shape[0], dtype=np.float64),
        where=denominator > 0.0,
    )
    correlation_reduction = 1.0 - np.clip(np.abs(correlation), 0.0, 1.0)

    vertical = np.asarray(z_scale, dtype=np.float64)
    source_count = int(vertical.size)
    nuisance_count = int(fisher.shape[1]) - source_count
    if fisher.shape[1] != fisher.shape[2] or nuisance_count < 0:
        raise ValueError("Ambiguity Fisher dimensions do not match source modes.")
    marginal_information = np.stack(
        tuple(
            _source_marginal_precision(action_information, nuisance_count)
            for action_information in fisher
        ),
        axis=0,
    )
    z_fisher = np.einsum(
        "k,akl,l->a",
        vertical,
        marginal_information,
        vertical,
        optimize=True,
    )
    z_fisher = np.log1p(np.maximum(z_fisher, 0.0))

    base = np.asarray(base_strengths, dtype=np.float64)
    base_prediction, base_variance = hypothesis_moments(base)
    support_separation = np.zeros(floor_spectrum.shape[0], dtype=np.float64)
    for alternative in alternative_strengths:
        aligned = _aligned_support_strengths(base, alternative)
        alternative_prediction, alternative_variance = hypothesis_moments(aligned)
        support_separation = np.maximum(
            support_separation,
            separation(
                base_prediction,
                alternative_prediction,
                base_variance,
                alternative_variance,
            ),
        )
    return {
        "floor_ceiling": floor_ceiling,
        "support": support_separation,
        "z_fisher": z_fisher,
        "correlation": correlation_reduction,
    }


def _project_alternative_strengths(
    estimate: MLEEstimate,
    alternatives: Sequence[MLEEstimate],
) -> tuple[NDArray[np.float64], ...]:
    """Project compatible alternative estimates onto the current patch grid."""
    projected: list[NDArray[np.float64]] = []
    for alternative in alternatives:
        if tuple(alternative.isotope_names) != tuple(estimate.isotope_names):
            continue
        try:
            projection = _project_patch_strengths_to_base(
                estimate.patches,
                alternative.patches,
            )
        except ValueError:
            continue
        strength = np.asarray(
            alternative.patch_strength_by_isotope,
            dtype=np.float64,
        ).T
        projected.append(projection @ strength)
    return tuple(projected)


def _geometry_normalization_scale(
    candidate_context_xyz: NDArray[np.float64],
    historical: ObservationBatch,
) -> float:
    """Return one fixed geometry scale for a complete candidate search."""
    context = _validated_candidate_poses(candidate_context_xyz)
    return max(
        float(
            np.linalg.norm(
                np.ptp(
                    np.vstack((context, historical.detector_positions_xyz)),
                    axis=0,
                )
            )
        ),
        1.0e-12,
    )


def _geometric_ambiguity_metrics(
    poses: NDArray[np.float64],
    estimate: MLEEstimate,
    historical: ObservationBatch,
    geometry_scale: float,
) -> dict[str, NDArray[np.float64]]:
    """Return pose-only ambiguity metrics under one fixed search scale."""
    candidate_poses = _validated_candidate_poses(poses)
    scale = float(geometry_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("geometry_scale must be finite and positive.")
    base_strength = np.asarray(estimate.patch_strength_by_isotope, dtype=float).T
    strengths = np.sum(base_strength, axis=1)
    patch_points = np.vstack(
        [np.asarray(patch.centroid_xyz, dtype=np.float64) for patch in estimate.patches]
    )
    source_centroid = (
        np.average(
            patch_points,
            axis=0,
            weights=np.maximum(strengths, 0.0),
        )
        if np.any(strengths > 0.0)
        else np.mean(patch_points, axis=0)
    )

    def elevation(candidate: NDArray[np.float64]) -> float:
        """Return source-centroid elevation from one detector pose."""
        delta = source_centroid - candidate
        return float(np.arctan2(delta[2], max(np.linalg.norm(delta[:2]), 1.0e-12)))

    historical_elevations = np.asarray(
        [elevation(pose) for pose in historical.detector_positions_xyz],
        dtype=np.float64,
    )
    pose_count = candidate_poses.shape[0]
    elevation_diversity = np.zeros(pose_count, dtype=np.float64)
    geometry_exploration = np.zeros(pose_count, dtype=np.float64)
    for pose_index, pose in enumerate(candidate_poses):
        angle = elevation(pose)
        elevation_diversity[pose_index] = min(
            1.0,
            float(np.min(np.abs(angle - historical_elevations))) / (0.5 * np.pi),
        )
        geometry_exploration[pose_index] = min(
            1.0,
            float(
                np.min(
                    np.linalg.norm(
                        historical.detector_positions_xyz - pose[None, :],
                        axis=1,
                    )
                )
            )
            / scale,
        )
    patch_areas = np.asarray(
        [float(patch.area_m2) for patch in estimate.patches],
        dtype=np.float64,
    )
    surface_kinds = np.asarray(
        [str(patch.surface_kind) for patch in estimate.patches],
        dtype=object,
    )
    coverage_weights = np.zeros(patch_points.shape[0], dtype=np.float64)
    unique_kinds = tuple(sorted(set(surface_kinds.tolist())))
    for kind in unique_kinds:
        mask = surface_kinds == kind
        kind_areas = patch_areas[mask]
        coverage_weights[mask] = kind_areas / max(float(np.sum(kind_areas)), 1.0e-30)
    coverage_weights /= max(float(len(unique_kinds)), 1.0)
    historical_distances = np.min(
        np.linalg.norm(
            historical.detector_positions_xyz[:, None, :] - patch_points[None, :, :],
            axis=2,
        ),
        axis=0,
    )
    surface_scale = max(
        float(np.linalg.norm(np.ptp(patch_points, axis=0))),
        1.0,
    )
    surface_coverage = np.zeros(pose_count, dtype=np.float64)
    for pose_index, pose in enumerate(candidate_poses):
        candidate_distances = np.linalg.norm(patch_points - pose[None, :], axis=1)
        improvement = np.maximum(
            np.minimum(historical_distances, surface_scale)
            - np.minimum(candidate_distances, historical_distances),
            0.0,
        )
        surface_coverage[pose_index] = min(
            1.0,
            float(np.sum(coverage_weights * improvement)) / surface_scale,
        )
    return {
        "elevation": elevation_diversity,
        "geometry": geometry_exploration,
        "surface_coverage": surface_coverage,
    }


def _ambiguity_metrics(
    response: NDArray[np.float64] | _LineSpectralDesign,
    information: NDArray[np.float64],
    poses: NDArray[np.float64],
    estimate: MLEEstimate,
    historical: ObservationBatch,
    source_basis: NDArray[np.float64],
    alternatives: Sequence[MLEEstimate],
    *,
    nuisance_coefficients: NDArray[np.float64] | None = None,
    common_counts: NDArray[np.float64] | None = None,
    geometry_scale: float | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Return pose/pair metrics for vertical and support-hypothesis ambiguity."""
    if isinstance(response, _LineSpectralDesign):
        action_count, _bin_count = response.observation_shape
        patch_count = response.patch_count
        isotope_count = response.isotope_count
        separation_alpha = response.overdispersion_alpha_by_bin
        coefficients = (
            np.zeros(len(response.nuisance_names), dtype=np.float64)
            if nuisance_coefficients is None
            else np.asarray(nuisance_coefficients, dtype=np.float64)
        )
        if coefficients.shape != (len(response.nuisance_names),):
            raise ValueError("Ambiguity nuisance coefficients do not align.")
        shared_counts = (
            np.einsum(
                "abn,n->ab",
                response.nuisance_response,
                coefficients,
                optimize=True,
            )
            if common_counts is None
            else np.asarray(common_counts, dtype=np.float64)
        )

        def project(weights: NDArray[np.float64]) -> NDArray[np.float64]:
            """Project one source hypothesis through compact line factors."""
            return _factorized_source_spectrum(response, weights)

    else:
        action_count, _bin_count, patch_count, isotope_count = response.shape
        separation_alpha = np.zeros(_bin_count, dtype=np.float64)
        shared_counts = common_counts

        def project(weights: NDArray[np.float64]) -> NDArray[np.float64]:
            """Project one source hypothesis through a dense test response."""
            return np.einsum("abgi,gi->ab", response, weights, optimize=True)

    if patch_count != len(estimate.patches):
        raise ValueError("Planner response and estimate patches do not align.")
    floor = np.asarray(
        [patch.surface_kind == "floor" for patch in estimate.patches],
        dtype=bool,
    )
    ceiling = np.asarray(
        [patch.surface_kind == "ceiling" for patch in estimate.patches],
        dtype=bool,
    )

    def surface_weights(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        """Return equal-total-strength weights for one competing surface."""
        weights = np.zeros((patch_count, isotope_count), dtype=np.float64)
        if np.any(mask):
            weights[mask] = 1.0 / (float(np.count_nonzero(mask)) * isotope_count)
        return weights

    patch_z = np.asarray([patch.centroid_xyz[2] for patch in estimate.patches])
    base_strength = np.asarray(estimate.patch_strength_by_isotope, dtype=float).T
    response_metrics = _response_ambiguity_metrics(
        project,
        information,
        surface_weights(floor),
        surface_weights(ceiling),
        base_strength,
        _project_alternative_strengths(estimate, alternatives),
        _vertical_basis_scale(patch_z, source_basis),
        separation_alpha,
        shared_counts,
    )
    pose_count = poses.shape[0]
    if action_count % pose_count:
        raise ValueError("Ambiguity actions must contain equal pair counts per pose.")
    geometric_metrics = _geometric_ambiguity_metrics(
        poses,
        estimate,
        historical,
        (
            _geometry_normalization_scale(poses, historical)
            if geometry_scale is None
            else float(geometry_scale)
        ),
    )
    repeats = action_count // pose_count
    return {
        **response_metrics,
        **{
            name: np.repeat(values, repeats)
            for name, values in geometric_metrics.items()
        },
    }


def _source_log_precision(
    precision: NDArray[np.float64],
    nuisance_count: int,
) -> float:
    """Return log determinant of source precision after nuisance marginalization."""
    matrix = np.asarray(precision, dtype=np.float64)
    sign, full_logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0 or not np.isfinite(full_logdet):
        raise np.linalg.LinAlgError("Planning precision must be positive definite.")
    if nuisance_count == 0:
        return float(full_logdet)
    nuisance = matrix[-nuisance_count:, -nuisance_count:]
    nuisance_sign, nuisance_logdet = np.linalg.slogdet(nuisance)
    if nuisance_sign <= 0.0 or not np.isfinite(nuisance_logdet):
        raise np.linalg.LinAlgError(
            "Planning nuisance precision must be positive definite."
        )
    return float(full_logdet - nuisance_logdet)


def _pair_rotation_radians(
    first_pair_id: int | None,
    second_pair_id: int,
    orientations: NDArray[np.float64],
) -> float:
    """Return summed Fe/Pb angular motion between two runtime pair IDs."""
    if first_pair_id is None:
        return 0.0
    count = int(orientations.shape[0])
    first_fe, first_pb = divmod(int(first_pair_id), count)
    second_fe, second_pb = divmod(int(second_pair_id), count)

    def angle(first: int, second: int) -> float:
        """Return the stable angle between two orientation normals."""
        dot = float(np.dot(orientations[first], orientations[second]))
        return float(np.arccos(np.clip(dot, -1.0, 1.0)))

    return angle(first_fe, second_fe) + angle(first_pb, second_pb)


def _pair_rotation_cost_cache(
    pair_ids: NDArray[np.int64],
    orientations: NDArray[np.float64],
    current_pair_id: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precompute exact pair-to-pair and initial rotation costs once."""
    pair_count = int(pair_ids.size)
    matrix = np.empty((pair_count, pair_count), dtype=np.float64)
    initial = np.empty(pair_count, dtype=np.float64)
    for first_index, first_pair_id in enumerate(pair_ids):
        initial[first_index] = _pair_rotation_radians(
            current_pair_id,
            int(first_pair_id),
            orientations,
        )
        for second_index, second_pair_id in enumerate(pair_ids):
            matrix[first_index, second_index] = _pair_rotation_radians(
                int(first_pair_id),
                int(second_pair_id),
                orientations,
            )
    matrix.setflags(write=False)
    initial.setflags(write=False)
    return matrix, initial


def _cuda_source_log_precision(
    precision: object,
    nuisance_count: int,
    *,
    torch_module: object,
) -> NDArray[np.float64]:
    """Return batched source log precision from float64 CUDA matrices."""
    torch = torch_module
    sign, full_logdet = torch.linalg.slogdet(precision)
    valid = (sign > 0.0) & torch.isfinite(full_logdet)
    if not bool(torch.all(valid).item()):
        raise np.linalg.LinAlgError("Planning precision must be positive definite.")
    if nuisance_count:
        nuisance = precision[:, -nuisance_count:, -nuisance_count:]
        nuisance_sign, nuisance_logdet = torch.linalg.slogdet(nuisance)
        nuisance_valid = (nuisance_sign > 0.0) & torch.isfinite(nuisance_logdet)
        if not bool(torch.all(nuisance_valid).item()):
            raise np.linalg.LinAlgError(
                "Planning nuisance precision must be positive definite."
            )
        full_logdet = full_logdet - nuisance_logdet
    return full_logdet.detach().cpu().numpy().astype(np.float64, copy=False)


def _cuda_beam_search_pose_program(
    pair_ids: NDArray[np.int64],
    information: NDArray[np.float64],
    precision: NDArray[np.float64],
    effective_nuisance_count: int,
    bonuses: NDArray[np.float64],
    rotation_cost_matrix: NDArray[np.float64],
    initial_rotation_costs: NDArray[np.float64],
    config: MLEPlanningConfig,
    *,
    current_pair_id: int | None,
    gpu_device: str,
) -> tuple[tuple[int, ...], tuple[int, ...], float, float]:
    """Run the exact beam objective with batched float64 CUDA log determinants."""
    import torch

    device = torch.device(gpu_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA beam planning requested but CUDA is unavailable.")
    dtype = torch.float64
    information_t = torch.as_tensor(information, dtype=dtype, device=device)
    state_precisions = torch.as_tensor(
        precision,
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    base_value = _source_log_precision(precision, effective_nuisance_count)
    states: list[tuple[tuple[int, ...], tuple[int, ...], float, float, int | None]] = [
        ((), (), base_value, 0.0, current_pair_id)
    ]
    for _ in range(int(config.shield_program_length)):
        parents: list[int] = []
        pair_indices: list[int] = []
        metadata: list[tuple[tuple[int, ...], tuple[int, ...], float, int]] = []
        for state_index, (
            selected_indices,
            selected_pairs,
            _state_value,
            rotation,
            _previous,
        ) in enumerate(states):
            selected_set = set(selected_indices)
            for pair_index, raw_pair_id in enumerate(pair_ids):
                if pair_index in selected_set:
                    continue
                pair_id = int(raw_pair_id)
                parents.append(state_index)
                pair_indices.append(pair_index)
                metadata.append(
                    (
                        (*selected_indices, pair_index),
                        (*selected_pairs, pair_id),
                        rotation
                        + (
                            float(initial_rotation_costs[pair_index])
                            if not selected_indices
                            else float(
                                rotation_cost_matrix[
                                    selected_indices[-1],
                                    pair_index,
                                ]
                            )
                        ),
                        pair_id,
                    )
                )
        if not metadata:
            raise RuntimeError("No unselected shield pair remains.")
        parent_array = np.asarray(parents, dtype=np.int64)
        pair_array = np.asarray(pair_indices, dtype=np.int64)
        next_values = np.empty(parent_array.size, dtype=np.float64)
        expansion_step = _beam_precision_chunk_size(int(precision.shape[0]))
        for expansion_start in range(0, parent_array.size, expansion_step):
            expansion_stop = min(
                expansion_start + expansion_step,
                parent_array.size,
            )
            parent_t = torch.as_tensor(
                parent_array[expansion_start:expansion_stop],
                dtype=torch.long,
                device=device,
            )
            pair_t = torch.as_tensor(
                pair_array[expansion_start:expansion_stop],
                dtype=torch.long,
                device=device,
            )
            precision_chunk = state_precisions[parent_t] + information_t[pair_t]
            next_values[expansion_start:expansion_stop] = _cuda_source_log_precision(
                precision_chunk,
                effective_nuisance_count,
                torch_module=torch,
            )
        expansions: list[
            tuple[
                tuple[float, float, tuple[int, ...]],
                int,
                tuple[tuple[int, ...], tuple[int, ...], float, float, int],
            ]
        ] = []
        for expansion_index, (
            (next_indices, next_pairs, next_rotation, pair_id),
            next_value,
        ) in enumerate(zip(metadata, next_values, strict=True)):
            information_gain = 0.5 * (float(next_value) - base_value)
            utility_bonus = float(np.mean(bonuses[list(next_indices)]))
            partial_score = (
                information_gain
                + utility_bonus
                - float(config.rotation_cost_weight) * next_rotation
            )
            expansions.append(
                (
                    (-partial_score, -information_gain, next_pairs),
                    expansion_index,
                    (
                        next_indices,
                        next_pairs,
                        float(next_value),
                        next_rotation,
                        pair_id,
                    ),
                )
            )
        expansions.sort(key=lambda item: item[0])
        retained = expansions[: int(config.shield_program_beam_width)]
        retained_indices = np.asarray(
            [item[1] for item in retained],
            dtype=np.int64,
        )
        retained_parent_t = torch.as_tensor(
            parent_array[retained_indices],
            dtype=torch.long,
            device=device,
        )
        retained_pair_t = torch.as_tensor(
            pair_array[retained_indices],
            dtype=torch.long,
            device=device,
        )
        state_precisions = (
            state_precisions[retained_parent_t] + information_t[retained_pair_t]
        )
        states = [item[2] for item in retained]
    selected_indices, selected_pairs, current_value, rotation, _ = states[0]
    return selected_indices, selected_pairs, current_value, rotation


def _cuda_beam_search_pose_programs(
    pair_ids: NDArray[np.int64],
    information: NDArray[np.float64],
    precision: NDArray[np.float64],
    effective_nuisance_count: int,
    bonuses: NDArray[np.float64],
    rotation_cost_matrix: NDArray[np.float64],
    initial_rotation_costs: NDArray[np.float64],
    config: MLEPlanningConfig,
    *,
    gpu_device: str,
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...], float, float],
    ...,
]:
    """Run all candidate-pose beams in shared float64 CUDA batches."""
    import torch

    device = torch.device(gpu_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA beam planning requested but CUDA is unavailable.")
    candidate_count = int(information.shape[0])
    information_t = torch.as_tensor(
        information,
        dtype=torch.float64,
        device=device,
    )
    base_precision_t = torch.as_tensor(
        precision,
        dtype=torch.float64,
        device=device,
    )
    state_precisions = base_precision_t[None, None].expand(
        candidate_count,
        1,
        *precision.shape,
    )
    base_value = _source_log_precision(precision, effective_nuisance_count)
    states: list[list[tuple[tuple[int, ...], tuple[int, ...], float, float]]] = [
        [((), (), base_value, 0.0)] for _ in range(candidate_count)
    ]
    for _ in range(int(config.shield_program_length)):
        candidate_indices: list[int] = []
        parent_indices: list[int] = []
        pair_indices: list[int] = []
        metadata: list[tuple[int, tuple[int, ...], tuple[int, ...], float]] = []
        for candidate_index, candidate_states in enumerate(states):
            for state_index, (
                selected_indices,
                selected_pairs,
                _state_value,
                rotation,
            ) in enumerate(candidate_states):
                selected_set = set(selected_indices)
                for pair_index, raw_pair_id in enumerate(pair_ids):
                    if pair_index in selected_set:
                        continue
                    next_indices = (*selected_indices, pair_index)
                    next_pairs = (*selected_pairs, int(raw_pair_id))
                    next_rotation = rotation + (
                        float(initial_rotation_costs[pair_index])
                        if not selected_indices
                        else float(
                            rotation_cost_matrix[selected_indices[-1], pair_index]
                        )
                    )
                    candidate_indices.append(candidate_index)
                    parent_indices.append(state_index)
                    pair_indices.append(pair_index)
                    metadata.append(
                        (
                            candidate_index,
                            next_indices,
                            next_pairs,
                            next_rotation,
                        )
                    )
        candidate_array = np.asarray(candidate_indices, dtype=np.int64)
        parent_array = np.asarray(parent_indices, dtype=np.int64)
        pair_array = np.asarray(pair_indices, dtype=np.int64)
        next_values = np.empty(candidate_array.size, dtype=np.float64)
        expansion_step = _beam_precision_chunk_size(int(precision.shape[0]))
        for expansion_start in range(0, candidate_array.size, expansion_step):
            expansion_stop = min(
                expansion_start + expansion_step,
                candidate_array.size,
            )
            candidate_t = torch.as_tensor(
                candidate_array[expansion_start:expansion_stop],
                dtype=torch.long,
                device=device,
            )
            parent_t = torch.as_tensor(
                parent_array[expansion_start:expansion_stop],
                dtype=torch.long,
                device=device,
            )
            pair_t = torch.as_tensor(
                pair_array[expansion_start:expansion_stop],
                dtype=torch.long,
                device=device,
            )
            precision_chunk = (
                state_precisions[candidate_t, parent_t]
                + information_t[candidate_t, pair_t]
            )
            next_values[expansion_start:expansion_stop] = _cuda_source_log_precision(
                precision_chunk,
                effective_nuisance_count,
                torch_module=torch,
            )
        expansions: list[
            list[
                tuple[
                    tuple[float, float, tuple[int, ...]],
                    int,
                    tuple[tuple[int, ...], tuple[int, ...], float, float],
                ]
            ]
        ] = [[] for _ in range(candidate_count)]
        for expansion_index, (
            (candidate_index, next_indices, next_pairs, next_rotation),
            next_value,
        ) in enumerate(zip(metadata, next_values, strict=True)):
            information_gain = 0.5 * (float(next_value) - base_value)
            utility_bonus = float(np.mean(bonuses[candidate_index, list(next_indices)]))
            partial_score = (
                information_gain
                + utility_bonus
                - float(config.rotation_cost_weight) * next_rotation
            )
            expansions[candidate_index].append(
                (
                    (-partial_score, -information_gain, next_pairs),
                    expansion_index,
                    (
                        next_indices,
                        next_pairs,
                        float(next_value),
                        next_rotation,
                    ),
                )
            )
        retained_indices: list[int] = []
        next_states: list[
            list[tuple[tuple[int, ...], tuple[int, ...], float, float]]
        ] = []
        retained_count = None
        for candidate_expansions in expansions:
            candidate_expansions.sort(key=lambda item: item[0])
            retained = candidate_expansions[: int(config.shield_program_beam_width)]
            if retained_count is None:
                retained_count = len(retained)
            elif len(retained) != retained_count:
                raise RuntimeError("Candidate beams retained inconsistent widths.")
            retained_indices.extend(item[1] for item in retained)
            next_states.append([item[2] for item in retained])
        retained_array = np.asarray(retained_indices, dtype=np.int64)
        retained_candidate_t = torch.as_tensor(
            candidate_array[retained_array],
            dtype=torch.long,
            device=device,
        )
        retained_parent_t = torch.as_tensor(
            parent_array[retained_array],
            dtype=torch.long,
            device=device,
        )
        retained_pair_t = torch.as_tensor(
            pair_array[retained_array],
            dtype=torch.long,
            device=device,
        )
        assert retained_count is not None
        state_precisions = (
            state_precisions[retained_candidate_t, retained_parent_t]
            + information_t[retained_candidate_t, retained_pair_t]
        ).reshape(
            candidate_count,
            retained_count,
            *precision.shape,
        )
        states = next_states
    return tuple(
        (
            candidate_states[0][0],
            candidate_states[0][1],
            candidate_states[0][2],
            candidate_states[0][3],
        )
        for candidate_states in states
    )


def _select_pose_programs_cuda(
    candidate_indices: NDArray[np.int64],
    poses: NDArray[np.float64],
    pair_ids: NDArray[np.int64],
    pair_information: NDArray[np.float64],
    expected_counts: NDArray[np.float64],
    base_precision: NDArray[np.float64],
    nuisance_count: int,
    orientations: NDArray[np.float64],
    config: MLEPlanningConfig,
    *,
    travel_costs: NDArray[np.float64],
    station_rate_cross_information: NDArray[np.float64] | None,
    station_rate_information: NDArray[np.float64] | None,
    pair_utility_bonus: NDArray[np.float64],
    rotation_cost_matrix: NDArray[np.float64],
    initial_rotation_costs: NDArray[np.float64],
    gpu_device: str,
) -> tuple[MLEPlanningAction, ...]:
    """Select all pose programs through one CUDA beam sequence."""
    precision = np.asarray(base_precision, dtype=np.float64)
    information = np.asarray(pair_information, dtype=np.float64)
    effective_nuisance_count = int(nuisance_count)
    if station_rate_cross_information is not None:
        if station_rate_information is None:
            raise ValueError("Both future station-rate Fisher terms are required.")
        parameter_count = int(precision.shape[0])
        extended_precision = np.zeros(
            (parameter_count + 1, parameter_count + 1),
            dtype=np.float64,
        )
        extended_precision[:-1, :-1] = precision
        extended_precision[-1, -1] = float(config.future_station_rate_prior_precision)
        extended_information = np.zeros(
            (
                information.shape[0],
                information.shape[1],
                parameter_count + 1,
                parameter_count + 1,
            ),
            dtype=np.float64,
        )
        extended_information[:, :, :-1, :-1] = information
        extended_information[:, :, :-1, -1] = station_rate_cross_information
        extended_information[:, :, -1, :-1] = station_rate_cross_information
        extended_information[:, :, -1, -1] = station_rate_information
        precision = extended_precision
        information = extended_information
        effective_nuisance_count += 1
    selections = _cuda_beam_search_pose_programs(
        pair_ids,
        information,
        precision,
        effective_nuisance_count,
        pair_utility_bonus,
        rotation_cost_matrix,
        initial_rotation_costs,
        config,
        gpu_device=gpu_device,
    )
    base_value = _source_log_precision(precision, effective_nuisance_count)
    orientation_count = int(orientations.shape[0])
    actions: list[MLEPlanningAction] = []
    for local_index, (
        selected_indices,
        selected_pair_ids,
        current_value,
        total_rotation,
    ) in enumerate(selections):
        information_gain = 0.5 * (current_value - base_value)
        utility_bonus = float(
            np.mean(pair_utility_bonus[local_index, list(selected_indices)])
        )
        score = (
            information_gain
            + utility_bonus
            - float(config.motion_cost_weight) * float(travel_costs[local_index])
            - float(config.rotation_cost_weight) * total_rotation
        )
        actions.append(
            MLEPlanningAction(
                candidate_index=int(candidate_indices[local_index]),
                detector_pose_xyz=tuple(float(value) for value in poses[local_index]),
                shield_pair_ids=selected_pair_ids,
                fe_orientation_indices=tuple(
                    pair_id // orientation_count for pair_id in selected_pair_ids
                ),
                pb_orientation_indices=tuple(
                    pair_id % orientation_count for pair_id in selected_pair_ids
                ),
                information_gain_nats=float(information_gain),
                travel_cost=float(travel_costs[local_index]),
                rotation_radians=float(total_rotation),
                score=float(score),
                live_time_s_by_view=tuple(
                    float(config.live_time_s) for _ in selected_indices
                ),
                expected_total_counts_by_view=tuple(
                    float(expected_counts[local_index, pair_index])
                    for pair_index in selected_indices
                ),
            )
        )
    return tuple(actions)


def _select_pose_program(
    candidate_index: int,
    pose_xyz: NDArray[np.float64],
    pair_ids: NDArray[np.int64],
    pair_information: NDArray[np.float64],
    expected_counts: NDArray[np.float64],
    base_precision: NDArray[np.float64],
    nuisance_count: int,
    orientations: NDArray[np.float64],
    config: MLEPlanningConfig,
    *,
    travel_cost: float,
    current_pair_id: int | None,
    station_rate_cross_information: NDArray[np.float64] | None = None,
    station_rate_information: NDArray[np.float64] | None = None,
    pair_utility_bonus: NDArray[np.float64] | None = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
    rotation_cost_matrix: NDArray[np.float64] | None = None,
    initial_rotation_costs: NDArray[np.float64] | None = None,
) -> MLEPlanningAction:
    """Jointly optimize a complete station shield program with beam search."""
    if int(config.shield_program_length) > pair_ids.size:
        raise ValueError("shield_program_length cannot exceed the allowed pair count.")
    precision = np.asarray(base_precision, dtype=np.float64)
    information = np.asarray(pair_information, dtype=np.float64)
    if (station_rate_cross_information is None) != (station_rate_information is None):
        raise ValueError("Both future station-rate Fisher terms must be supplied.")
    effective_nuisance_count = int(nuisance_count)
    if station_rate_cross_information is not None:
        cross = np.asarray(station_rate_cross_information, dtype=np.float64)
        station = np.asarray(station_rate_information, dtype=np.float64)
        parameter_count = int(precision.shape[0])
        if cross.shape != (pair_ids.size, parameter_count):
            raise ValueError("station_rate_cross_information has invalid shape.")
        if station.shape != (pair_ids.size,):
            raise ValueError("station_rate_information has invalid shape.")
        extended_precision = np.zeros(
            (parameter_count + 1, parameter_count + 1),
            dtype=np.float64,
        )
        extended_precision[:-1, :-1] = precision
        extended_precision[-1, -1] = float(config.future_station_rate_prior_precision)
        extended_information = np.zeros(
            (pair_ids.size, parameter_count + 1, parameter_count + 1),
            dtype=np.float64,
        )
        extended_information[:, :-1, :-1] = information
        extended_information[:, :-1, -1] = cross
        extended_information[:, -1, :-1] = cross
        extended_information[:, -1, -1] = station
        precision = extended_precision
        information = extended_information
        effective_nuisance_count += 1
    bonuses = (
        np.zeros(pair_ids.size, dtype=np.float64)
        if pair_utility_bonus is None
        else np.asarray(pair_utility_bonus, dtype=np.float64)
    )
    if bonuses.shape != (pair_ids.size,) or np.any(~np.isfinite(bonuses)):
        raise ValueError("pair_utility_bonus must be one finite value per pair.")
    if rotation_cost_matrix is None or initial_rotation_costs is None:
        rotation_cost_matrix, initial_rotation_costs = _pair_rotation_cost_cache(
            pair_ids,
            orientations,
            current_pair_id,
        )
    rotation_costs = np.asarray(rotation_cost_matrix, dtype=np.float64)
    initial_costs = np.asarray(initial_rotation_costs, dtype=np.float64)
    if rotation_costs.shape != (pair_ids.size, pair_ids.size) or (
        initial_costs.shape != (pair_ids.size,)
    ):
        raise ValueError("Precomputed rotation costs do not align with pair IDs.")
    base_value = _source_log_precision(precision, effective_nuisance_count)
    if use_gpu and precision.shape[0] >= 24:
        (
            selected_indices,
            selected_pair_ids,
            current_value,
            total_rotation,
        ) = _cuda_beam_search_pose_program(
            pair_ids,
            information,
            precision,
            effective_nuisance_count,
            bonuses,
            rotation_costs,
            initial_costs,
            config,
            current_pair_id=current_pair_id,
            gpu_device=gpu_device,
        )
    else:
        states: list[
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                NDArray[np.float64],
                float,
                float,
                int | None,
            ]
        ] = [((), (), precision.copy(), base_value, 0.0, current_pair_id)]
        for _ in range(int(config.shield_program_length)):
            expansions: list[
                tuple[
                    tuple[float, float, tuple[int, ...]],
                    tuple[
                        tuple[int, ...],
                        tuple[int, ...],
                        int,
                        int,
                        float,
                        float,
                        int,
                    ],
                ]
            ] = []
            for parent_index, (
                state_indices,
                state_pairs,
                state_precision,
                _,
                rotation,
                _previous,
            ) in enumerate(states):
                selected_set = set(state_indices)
                for pair_index, raw_pair_id in enumerate(pair_ids):
                    if pair_index in selected_set:
                        continue
                    pair_id = int(raw_pair_id)
                    next_indices = (*state_indices, pair_index)
                    next_pairs = (*state_pairs, pair_id)
                    next_precision = state_precision + information[pair_index]
                    next_value = _source_log_precision(
                        next_precision,
                        effective_nuisance_count,
                    )
                    next_rotation = rotation + (
                        float(initial_costs[pair_index])
                        if not state_indices
                        else float(rotation_costs[state_indices[-1], pair_index])
                    )
                    information_gain = 0.5 * (next_value - base_value)
                    utility_bonus = float(np.mean(bonuses[list(next_indices)]))
                    partial_score = (
                        information_gain
                        + utility_bonus
                        - float(config.rotation_cost_weight) * next_rotation
                    )
                    expansions.append(
                        (
                            (-partial_score, -information_gain, next_pairs),
                            (
                                next_indices,
                                next_pairs,
                                parent_index,
                                pair_index,
                                next_value,
                                next_rotation,
                                pair_id,
                            ),
                        )
                    )
            if not expansions:
                raise RuntimeError("No unselected shield pair remains.")
            expansions.sort(key=lambda item: item[0])
            previous_states = states
            states = []
            for _, retained in expansions[: int(config.shield_program_beam_width)]:
                (
                    next_indices,
                    next_pairs,
                    parent_index,
                    pair_index,
                    next_value,
                    next_rotation,
                    pair_id,
                ) = retained
                next_precision = (
                    previous_states[parent_index][2] + information[pair_index]
                )
                states.append(
                    (
                        next_indices,
                        next_pairs,
                        next_precision,
                        next_value,
                        next_rotation,
                        pair_id,
                    )
                )
        (
            selected_indices,
            selected_pair_ids,
            _,
            current_value,
            total_rotation,
            _,
        ) = states[0]
    information_gain = 0.5 * (current_value - base_value)
    utility_bonus = float(np.mean(bonuses[list(selected_indices)]))
    score = (
        information_gain
        + utility_bonus
        - float(config.motion_cost_weight) * float(travel_cost)
        - float(config.rotation_cost_weight) * total_rotation
    )
    orientation_count = int(orientations.shape[0])
    return MLEPlanningAction(
        candidate_index=int(candidate_index),
        detector_pose_xyz=tuple(float(value) for value in pose_xyz),
        shield_pair_ids=tuple(selected_pair_ids),
        fe_orientation_indices=tuple(
            pair_id // orientation_count for pair_id in selected_pair_ids
        ),
        pb_orientation_indices=tuple(
            pair_id % orientation_count for pair_id in selected_pair_ids
        ),
        information_gain_nats=float(information_gain),
        travel_cost=float(travel_cost),
        rotation_radians=float(total_rotation),
        score=float(score),
        live_time_s_by_view=tuple(float(config.live_time_s) for _ in selected_indices),
        expected_total_counts_by_view=tuple(
            float(expected_counts[index]) for index in selected_indices
        ),
    )


def select_fisher_action(
    candidate_poses_xyz: object,
    pair_ids: Sequence[int],
    candidate_information: object,
    expected_total_counts: object,
    base_precision: object,
    orientations: object,
    *,
    nuisance_count: int,
    config: MLEPlanningConfig | None = None,
    travel_costs: object | None = None,
    current_pair_id: int | None = None,
    station_rate_cross_information: object | None = None,
    station_rate_information: object | None = None,
    pair_utility_bonus: object | None = None,
    use_gpu: bool = False,
    gpu_device: str = "cuda",
) -> tuple[MLEPlanningAction, tuple[MLEPlanningAction, ...]]:
    """Select a joint pose/program from precomputed expected Fisher matrices."""
    resolved = MLEPlanningConfig() if config is None else config
    poses = _validated_candidate_poses(candidate_poses_xyz)
    information = np.asarray(candidate_information, dtype=np.float64)
    totals = np.asarray(expected_total_counts, dtype=np.float64)
    precision = np.asarray(base_precision, dtype=np.float64)
    orientation_array = np.asarray(orientations, dtype=np.float64)
    candidate_count = int(poses.shape[0])
    parameter_count = int(precision.shape[0]) if precision.ndim == 2 else 0
    if orientation_array.ndim != 2 or orientation_array.shape[1:] != (3,):
        raise ValueError("orientations must have shape (R, 3).")
    pair_array = _validated_pair_ids(
        tuple(pair_ids),
        int(orientation_array.shape[0]),
    )
    pair_count = int(pair_array.size)
    if information.shape != (
        candidate_count,
        pair_count,
        parameter_count,
        parameter_count,
    ):
        raise ValueError("candidate_information has incompatible dimensions.")
    if totals.shape != (candidate_count, pair_count):
        raise ValueError("expected_total_counts must align with candidates and pairs.")
    if precision.shape != (parameter_count, parameter_count):
        raise ValueError("base_precision must be a square matrix.")
    if isinstance(nuisance_count, bool) or not isinstance(
        nuisance_count,
        (int, np.integer),
    ):
        raise TypeError("nuisance_count must be an integer.")
    if not 0 <= int(nuisance_count) < parameter_count:
        raise ValueError("nuisance_count must lie in [0, parameter_count).")
    if np.any(~np.isfinite(information)) or np.any(~np.isfinite(totals)):
        raise ValueError("Candidate information and expected counts must be finite.")
    if np.any(totals < 0.0):
        raise ValueError("expected_total_counts must be nonnegative.")
    if (station_rate_cross_information is None) != (station_rate_information is None):
        raise ValueError("Both future station-rate Fisher terms must be supplied.")
    station_cross = None
    station_information = None
    if station_rate_cross_information is not None:
        station_cross = np.asarray(
            station_rate_cross_information,
            dtype=np.float64,
        )
        station_information = np.asarray(station_rate_information, dtype=np.float64)
        if station_cross.shape != (candidate_count, pair_count, parameter_count):
            raise ValueError("station_rate_cross_information has invalid dimensions.")
        if station_information.shape != (candidate_count, pair_count):
            raise ValueError("station_rate_information has invalid dimensions.")
        if np.any(~np.isfinite(station_cross)) or np.any(
            ~np.isfinite(station_information)
        ):
            raise ValueError("Future station-rate Fisher terms must be finite.")
        if np.any(station_information < 0.0):
            raise ValueError("station_rate_information must be nonnegative.")
    bonuses = (
        np.zeros((candidate_count, pair_count), dtype=np.float64)
        if pair_utility_bonus is None
        else np.asarray(pair_utility_bonus, dtype=np.float64)
    )
    if bonuses.shape != (candidate_count, pair_count) or np.any(~np.isfinite(bonuses)):
        raise ValueError("pair_utility_bonus must align with candidates and pairs.")
    costs = _validated_travel_costs(travel_costs, candidate_count)
    rotation_costs, initial_rotation_costs = _pair_rotation_cost_cache(
        pair_array,
        orientation_array,
        current_pair_id,
    )
    if use_gpu and candidate_count > 1:
        actions = list(
            _select_pose_programs_cuda(
                np.arange(candidate_count, dtype=np.int64),
                poses,
                pair_array,
                information,
                totals,
                precision,
                nuisance_count,
                orientation_array,
                resolved,
                travel_costs=costs,
                station_rate_cross_information=station_cross,
                station_rate_information=station_information,
                pair_utility_bonus=bonuses,
                rotation_cost_matrix=rotation_costs,
                initial_rotation_costs=initial_rotation_costs,
                gpu_device=gpu_device,
            )
        )
    else:
        actions = [
            _select_pose_program(
                index,
                poses[index],
                pair_array,
                information[index],
                totals[index],
                precision,
                nuisance_count,
                orientation_array,
                resolved,
                travel_cost=float(costs[index]),
                current_pair_id=current_pair_id,
                station_rate_cross_information=(
                    None if station_cross is None else station_cross[index]
                ),
                station_rate_information=(
                    None if station_information is None else station_information[index]
                ),
                pair_utility_bonus=bonuses[index],
                use_gpu=use_gpu,
                gpu_device=gpu_device,
                rotation_cost_matrix=rotation_costs,
                initial_rotation_costs=initial_rotation_costs,
            )
            for index in range(candidate_count)
        ]
    ranked = tuple(
        sorted(
            actions,
            key=lambda action: (
                -action.score,
                -action.information_gain_nats,
                action.candidate_index,
                action.shield_pair_ids,
            ),
        )
    )
    return ranked[0], ranked[: int(resolved.ranked_action_limit)]


def _screen_candidate_measurements(
    estimate: MLEEstimate,
    historical_observations: ObservationBatch,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
    poses: NDArray[np.float64],
    pair_ids: NDArray[np.int64],
    costs: NDArray[np.float64],
    current_pair_id: int | None,
    config: MLEPlanningConfig,
    historical_response_cache: dict[str, object] | None,
    progress_hook: Callable[[Mapping[str, object]], None] | None,
    alternative_estimates: Sequence[MLEEstimate] = (),
) -> MLEPlanningResult:
    """Screen many poses with grouped spectra and a compact source basis."""
    started = perf_counter()
    orientations = np.asarray(kernel.orientations, dtype=np.float64)
    representative_pairs = _representative_pair_ids(
        pair_ids,
        orientations,
        int(config.screening_pair_limit),
        current_pair_id,
    )
    (
        screening_patches,
        source_strengths,
        basis,
        historical_basis,
        basis_labels,
    ) = _screening_pseudo_model(estimate, config)
    isotope_count = len(estimate.isotope_names)
    full_surface_kinds = np.asarray(
        [patch.surface_kind for patch in estimate.patches],
        dtype=object,
    )

    def pseudo_surface_weights(kind: str) -> NDArray[np.float64]:
        """Project one equal-total full-surface hypothesis to pseudo points."""
        selected = full_surface_kinds == kind
        full_weights = np.zeros(
            (len(estimate.patches), isotope_count),
            dtype=np.float64,
        )
        if np.any(selected):
            full_weights[selected] = 1.0 / (
                float(np.count_nonzero(selected)) * isotope_count
            )
        return np.einsum(
            "rgi,gi->ri",
            screening_patches.strength_projection,
            full_weights,
            optimize=True,
        )

    screening_floor_weights = pseudo_surface_weights("floor")
    screening_ceiling_weights = pseudo_surface_weights("ceiling")
    projected_alternatives = _project_alternative_strengths(
        estimate,
        alternative_estimates,
    )
    screening_alternatives = tuple(
        np.einsum(
            "rgi,gi->ri",
            screening_patches.strength_projection,
            alternative,
            optimize=True,
        )
        for alternative in projected_alternatives
    )
    screening_z_scale = _vertical_basis_scale(
        np.asarray([patch.centroid_xyz[2] for patch in estimate.patches]),
        historical_basis,
    )
    geometry_scale = _geometry_normalization_scale(
        poses,
        historical_observations,
    )
    geometric_metrics = _geometric_ambiguity_metrics(
        poses,
        estimate,
        historical_observations,
        geometry_scale,
    )
    bootstrap_multiplier = (
        1.0
        if historical_observations.measurement_count
        < int(config.geometry_bootstrap_measurements)
        else 0.25
    )

    def pose_utility(
        elevation: float,
        geometry: float,
        surface_coverage: float,
    ) -> float:
        """Return the pair-independent ambiguity utility for one pose."""
        return (
            float(config.elevation_diversity_weight) * float(elevation)
            + bootstrap_multiplier
            * float(config.geometry_exploration_weight)
            * float(geometry)
            + float(config.surface_coverage_weight) * float(surface_coverage)
        )

    historical_strengths = np.asarray(
        estimate.patch_strength_by_isotope,
        dtype=np.float64,
    ).T
    historical_design, _ = _historical_factorized_spectral_design(
        historical_observations,
        estimate,
        kernel,
        mle_config,
        historical_response_cache,
    )
    nuisance_names = historical_design.nuisance_names
    nuisance_coefficients = _nuisance_coefficients(estimate, nuisance_names)
    nuisance_scales = np.maximum(
        nuisance_coefficients,
        float(config.nuisance_scale_floor),
    )
    historical_precision, _ = _historical_factorized_fisher_precision(
        historical_design,
        historical_basis,
        historical_strengths,
        nuisance_coefficients,
        nuisance_scales,
        _historical_row_keys(historical_observations),
        model_identity=_cached_model_identity(
            historical_response_cache,
            "historical_factorized_design",
        ),
        minimum_expected_count=float(config.minimum_expected_bin_count),
        cache=historical_response_cache,
        use_gpu=bool(mle_config.use_gpu),
        gpu_device=str(mle_config.gpu_device),
        cache_entry_name="historical_factorized_fisher:screening",
    )
    nuisance_count = int(nuisance_coefficients.size)
    joint_precision = (
        _planning_prior_precision(
            int(historical_basis.shape[2]),
            nuisance_scales,
            historical_design.nuisance_l2_weights,
            laplace_prior_precision=float(config.laplace_prior_precision),
        )
        + historical_precision
    )
    base_precision = _source_marginal_precision(joint_precision, nuisance_count)
    bin_count = int(config.screening_energy_bin_count)
    full_edges = historical_observations.energy_bin_edges_keV
    screening_edges = np.linspace(
        float(full_edges[0]),
        float(full_edges[-1]),
        bin_count + 1,
        dtype=np.float64,
    )
    calibrated_screening = mle_config.spectral_likelihood == "calibrated_overdispersed"
    background_rate = _screening_background_rate(
        historical_observations,
        full_edges if calibrated_screening else screening_edges,
        _factorized_source_spectrum(
            historical_design,
            historical_strengths,
        ),
    )
    cache_identity = (
        tuple(
            (
                int(patch.patch_id),
                str(patch.surface_kind),
                str(patch.object_id),
                np.asarray(patch.vertices_xyz, dtype=np.float64).tobytes(),
            )
            for patch in estimate.patches
        ),
        np.asarray(
            estimate.patch_strength_by_isotope,
            dtype=np.float64,
        ).tobytes(),
        np.asarray(source_strengths, dtype=np.float64).tobytes(),
        np.asarray(basis, dtype=np.float64).tobytes(),
        np.asarray(background_rate, dtype=np.float64).tobytes(),
        tuple(nuisance_names),
        np.asarray(nuisance_coefficients, dtype=np.float64).tobytes(),
        np.asarray(nuisance_scales, dtype=np.float64).tobytes(),
        np.asarray(base_precision, dtype=np.float64).tobytes(),
        _cached_model_identity(
            historical_response_cache,
            "historical_factorized_design",
        ),
        _historical_row_keys(historical_observations),
        tuple(int(value) for value in representative_pairs),
        config.to_dict(),
        mle_config.to_dict(),
        current_pair_id,
        tuple(
            (
                tuple(alternative.isotope_names),
                np.asarray(
                    alternative.patch_strength_by_isotope,
                    dtype=np.float64,
                ).tobytes(),
                tuple(
                    (
                        int(patch.patch_id),
                        str(patch.surface_kind),
                        str(patch.object_id),
                        np.asarray(patch.vertices_xyz, dtype=np.float64).tobytes(),
                    )
                    for patch in alternative.patches
                ),
            )
            for alternative in alternative_estimates
        ),
    )
    cache_entry = (
        None
        if historical_response_cache is None
        else historical_response_cache.get("candidate_screening")
    )
    if (
        not isinstance(cache_entry, dict)
        or cache_entry.get("identity") != cache_identity
    ):
        cache_entry = {"identity": cache_identity, "actions_by_pose": {}}
        if historical_response_cache is not None:
            historical_response_cache["candidate_screening"] = cache_entry
    actions_by_pose = cache_entry["actions_by_pose"]
    if not isinstance(actions_by_pose, dict):
        raise TypeError("candidate screening cache is invalid.")
    missing = np.asarray(
        [
            index
            for index, pose in enumerate(poses)
            if np.asarray(pose, dtype=np.float64).tobytes() not in actions_by_pose
        ],
        dtype=np.int64,
    )
    screening_config = replace(
        config,
        shield_program_beam_width=1,
        ranked_action_limit=max(int(config.ranked_action_limit), int(poses.shape[0])),
    )
    response_seconds = 0.0
    fisher_seconds = 0.0
    beam_seconds = 0.0
    completed_missing = 0
    if progress_hook is not None:
        progress_hook(
            {
                "phase": "candidate_screening",
                "completed_candidates": int(poses.shape[0] - missing.size),
                "total_candidates": int(poses.shape[0]),
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
                "reused_candidates": int(poses.shape[0] - missing.size),
            }
        )
    chunk_size = int(config.screening_pose_chunk_size)
    orientation_count = int(orientations.shape[0])
    pair_fe = representative_pairs // orientation_count
    pair_pb = representative_pairs % orientation_count
    for missing_start in range(0, missing.size, chunk_size):
        selected_indices = missing[missing_start : missing_start + chunk_size]
        local_poses = poses[selected_indices]
        local_count = int(local_poses.shape[0])
        expanded = np.repeat(local_poses, representative_pairs.size, axis=0)
        geometry = _PlanningGeometry(
            detector_positions_xyz=expanded,
            fe_indices=np.tile(pair_fe, local_count),
            pb_indices=np.tile(pair_pb, local_count),
            live_times_s=np.full(
                expanded.shape[0],
                float(config.live_time_s),
                dtype=np.float64,
            ),
            energy_bin_edges_keV=(
                full_edges if calibrated_screening else screening_edges
            ),
            station_ids=np.zeros(expanded.shape[0], dtype=np.int64),
        )
        response_started = perf_counter()
        grouped_nb2_design: _GroupedNB2LineDesign | None = None
        if calibrated_screening:
            grouped_nb2_design = _calibrated_screening_spectral_design(
                geometry,
                screening_patches,
                estimate.isotope_names,
                kernel,
                mle_config,
                background_rate,
                screening_edges,
            )
            response = grouped_nb2_design.grouped_response
            background = grouped_nb2_design.grouped_background
            observation_variance = grouped_nb2_design.project_hypothesis(
                source_strengths
            ).total_variance
        else:
            response = _screening_spectral_design(
                geometry,
                screening_patches,
                estimate.isotope_names,
                kernel,
                mle_config,
            )
            background = np.broadcast_to(
                float(config.live_time_s) * background_rate[None, :],
                response.shape[:2],
            )
            observation_variance = None
        response_seconds += perf_counter() - response_started
        fisher_started = perf_counter()
        information, totals = _screening_fisher_information(
            response,
            basis,
            source_strengths,
            background,
            minimum_expected_count=float(config.minimum_expected_bin_count),
            observation_variance=observation_variance,
            use_gpu=bool(mle_config.use_gpu),
            gpu_device=str(mle_config.gpu_device),
        )
        fisher_seconds += perf_counter() - fisher_started

        def project_screening(
            weights: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Project one pseudo-source hypothesis through grouped response."""
            return np.einsum(
                "abgi,gi->ab",
                response,
                weights,
                optimize=True,
            )

        def project_screening_moments(
            weights: NDArray[np.float64],
        ) -> _SpectralHypothesisMoments:
            """Project exact fine-bin NB2 moments into screening groups."""
            if grouped_nb2_design is None:
                raise RuntimeError("Grouped NB2 screening design is unavailable.")
            return grouped_nb2_design.project_hypothesis(weights)

        response_ambiguity = _response_ambiguity_metrics(
            project_screening,
            information,
            screening_floor_weights,
            screening_ceiling_weights,
            source_strengths,
            screening_alternatives,
            screening_z_scale,
            np.zeros(response.shape[1], dtype=np.float64),
            background,
            (project_screening_moments if grouped_nb2_design is not None else None),
        )
        information = information.reshape(
            local_count,
            representative_pairs.size,
            base_precision.shape[0],
            base_precision.shape[1],
        )
        totals = totals.reshape(local_count, representative_pairs.size)
        ambiguity_by_pose = {
            name: values.reshape(local_count, representative_pairs.size)
            for name, values in response_ambiguity.items()
        }
        pair_utility_bonuses = (
            float(config.floor_ceiling_separation_weight)
            * ambiguity_by_pose["floor_ceiling"]
            + float(config.support_hypothesis_separation_weight)
            * ambiguity_by_pose["support"]
            + float(config.z_fisher_weight) * ambiguity_by_pose["z_fisher"]
            + float(config.response_correlation_reduction_weight)
            * ambiguity_by_pose["correlation"]
        )
        beam_started = perf_counter()
        _, local_ranked = select_fisher_action(
            local_poses,
            representative_pairs,
            information,
            totals,
            base_precision,
            orientations,
            nuisance_count=0,
            config=screening_config,
            travel_costs=costs[selected_indices],
            current_pair_id=current_pair_id,
            pair_utility_bonus=pair_utility_bonuses,
            use_gpu=bool(mle_config.use_gpu),
            gpu_device=str(mle_config.gpu_device),
        )
        beam_seconds += perf_counter() - beam_started
        by_local_index = {action.candidate_index: action for action in local_ranked}
        for local_index, global_index in enumerate(selected_indices):
            action = by_local_index[local_index]
            selected_pair_indices = np.asarray(
                [
                    int(np.flatnonzero(representative_pairs == pair_id)[0])
                    for pair_id in action.shield_pair_ids
                ],
                dtype=np.int64,
            )

            def selected_mean(name: str) -> float:
                """Return one selected screening-program ambiguity mean."""
                return float(
                    np.mean(
                        ambiguity_by_pose[name][
                            local_index,
                            selected_pair_indices,
                        ]
                    )
                )

            floor_ceiling = selected_mean("floor_ceiling")
            support = selected_mean("support")
            z_fisher = selected_mean("z_fisher")
            correlation = selected_mean("correlation")
            elevation = float(geometric_metrics["elevation"][global_index])
            geometry_value = float(geometric_metrics["geometry"][global_index])
            surface_coverage = float(
                geometric_metrics["surface_coverage"][global_index]
            )
            actions_by_pose[
                np.asarray(poses[global_index], dtype=np.float64).tobytes()
            ] = replace(
                action,
                candidate_index=int(global_index),
                detector_pose_xyz=tuple(float(value) for value in poses[global_index]),
                score=float(action.score)
                + pose_utility(elevation, geometry_value, surface_coverage),
                floor_ceiling_separation=floor_ceiling,
                support_hypothesis_separation=support,
                z_fisher_information=z_fisher,
                response_correlation_reduction=correlation,
                elevation_diversity=elevation,
                geometry_exploration=geometry_value,
                surface_coverage=surface_coverage,
            )
        completed_missing += local_count
        if progress_hook is not None:
            elapsed = perf_counter() - started
            completed = int(poses.shape[0] - missing.size + completed_missing)
            progress_hook(
                {
                    "phase": "candidate_screening",
                    "completed_candidates": completed,
                    "total_candidates": int(poses.shape[0]),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": (
                        elapsed * (int(poses.shape[0]) - completed) / completed
                        if completed
                        else None
                    ),
                    "reused_candidates": int(poses.shape[0] - missing.size),
                }
            )
    actions: list[MLEPlanningAction] = []
    for index, pose in enumerate(poses):
        cached = actions_by_pose[np.asarray(pose, dtype=np.float64).tobytes()]
        travel_delta = float(costs[index]) - float(cached.travel_cost)
        elevation = float(geometric_metrics["elevation"][index])
        geometry_value = float(geometric_metrics["geometry"][index])
        surface_coverage = float(geometric_metrics["surface_coverage"][index])
        cached_pose_utility = pose_utility(
            cached.elevation_diversity,
            cached.geometry_exploration,
            cached.surface_coverage,
        )
        current_pose_utility = pose_utility(
            elevation,
            geometry_value,
            surface_coverage,
        )
        actions.append(
            replace(
                cached,
                candidate_index=index,
                detector_pose_xyz=tuple(float(value) for value in pose),
                travel_cost=float(costs[index]),
                score=float(cached.score)
                - cached_pose_utility
                + current_pose_utility
                - float(config.motion_cost_weight) * travel_delta,
                elevation_diversity=elevation,
                geometry_exploration=geometry_value,
                surface_coverage=surface_coverage,
            )
        )
    ranked = tuple(
        sorted(
            actions,
            key=lambda action: (
                -action.score,
                -action.information_gain_nats,
                action.candidate_index,
                action.shield_pair_ids,
            ),
        )
    )
    diagnostics = {
        "criterion": (
            "grouped-NB2 source Fisher screening"
            if calibrated_screening
            else "grouped-Poisson source Fisher screening"
        ),
        "approximate": True,
        "stage": "screening",
        "ambiguity_aware": True,
        "candidate_count": int(poses.shape[0]),
        "screening_energy_bin_count": bin_count,
        "screening_pair_ids": representative_pairs.astype(int).tolist(),
        "screening_source_basis": list(basis_labels),
        "reused_candidates": int(poses.shape[0] - missing.size),
        "computed_candidates": int(missing.size),
        "performance": {
            "response_seconds": response_seconds,
            "fisher_seconds": fisher_seconds,
            "beam_search_seconds": beam_seconds,
            "elapsed_seconds": perf_counter() - started,
            "dtype": "float64",
            "device": str(mle_config.gpu_device) if mle_config.use_gpu else "cpu",
        },
        "config": config.to_dict(),
    }
    return MLEPlanningResult(
        selected_action=ranked[0],
        ranked_actions=ranked[: int(config.ranked_action_limit)],
        diagnostics=diagnostics,
    )


def _plan_next_measurement_exact(
    estimate: MLEEstimate,
    historical_observations: ObservationBatch,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
    candidate_poses_xyz: object,
    *,
    planning_config: MLEPlanningConfig | None = None,
    allowed_pair_ids: Sequence[int] | None = None,
    travel_costs: object | None = None,
    current_pair_id: int | None = None,
    alternative_estimates: Sequence[MLEEstimate] = (),
    historical_response_cache: dict[str, object] | None = None,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    ambiguity_geometry_scale: float | None = None,
) -> MLEPlanningResult:
    """Exactly plan a joint next station and Fe/Pb program from one fitted MLE.

    Candidate generation, obstacle traversability, and exact travel costs stay
    with the shared runtime. This function ranks only the truth-free candidates
    supplied by that controller.
    """
    if not isinstance(estimate, MLEEstimate):
        raise TypeError("estimate must be an MLEEstimate.")
    if not isinstance(historical_observations, ObservationBatch):
        raise TypeError("historical_observations must be an ObservationBatch.")
    if not isinstance(kernel, ContinuousKernel):
        raise TypeError("kernel must be the shared runtime ContinuousKernel.")
    if not isinstance(mle_config, MLEConfig):
        raise TypeError("mle_config must be an MLEConfig.")
    if not isinstance(alternative_estimates, Sequence):
        raise TypeError("alternative_estimates must be a sequence.")
    if any(
        not isinstance(alternative, MLEEstimate)
        for alternative in alternative_estimates
    ):
        raise TypeError("Every alternative estimate must be an MLEEstimate.")
    if progress_hook is not None and not callable(progress_hook):
        raise TypeError("progress_hook must be callable or None.")
    if mle_config.mode != "spectral":
        raise ValueError("Online MLE planning requires spectral mode.")
    if tuple(estimate.isotope_names) != tuple(mle_config.isotope_names) or (
        tuple(historical_observations.isotope_names) != tuple(estimate.isotope_names)
    ):
        raise ValueError("Planner isotope order must match estimate and history.")
    resolved = MLEPlanningConfig() if planning_config is None else planning_config
    poses = _validated_candidate_poses(candidate_poses_xyz)
    resolved_geometry_scale = (
        _geometry_normalization_scale(poses, historical_observations)
        if ambiguity_geometry_scale is None
        else float(ambiguity_geometry_scale)
    )
    if not np.isfinite(resolved_geometry_scale) or resolved_geometry_scale <= 0.0:
        raise ValueError("ambiguity_geometry_scale must be finite and positive.")
    costs = _validated_travel_costs(travel_costs, int(poses.shape[0]))
    orientations = np.asarray(kernel.orientations, dtype=np.float64)
    if orientations.ndim != 2 or orientations.shape[1:] != (3,):
        raise ValueError("Shared kernel orientations must have shape (R, 3).")
    pair_ids = _validated_pair_ids(allowed_pair_ids, int(orientations.shape[0]))
    if int(resolved.shield_program_length) > pair_ids.size:
        raise ValueError("shield_program_length cannot exceed the allowed pair count.")
    pair_limit = int(orientations.shape[0]) ** 2
    if current_pair_id is not None and (
        isinstance(current_pair_id, bool)
        or not isinstance(current_pair_id, (int, np.integer))
        or not 0 <= int(current_pair_id) < pair_limit
    ):
        raise ValueError(f"current_pair_id must lie in [0, {pair_limit - 1}].")

    planning_started = perf_counter()
    if progress_hook is not None:
        progress_hook(
            {
                "phase": "historical_setup",
                "completed_candidates": 0,
                "total_candidates": int(poses.shape[0]),
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
            }
        )
    response_seconds = 0.0
    fisher_seconds = 0.0
    beam_seconds = 0.0
    source_basis, basis_labels = _source_basis(estimate, resolved)
    source_strengths = np.asarray(
        estimate.patch_strength_by_isotope,
        dtype=np.float64,
    ).T
    response_started = perf_counter()
    historical_design, historical_cache_diagnostics = (
        _historical_factorized_spectral_design(
            historical_observations,
            estimate,
            kernel,
            mle_config,
            historical_response_cache,
        )
    )
    response_seconds += perf_counter() - response_started
    nuisance_names = historical_design.nuisance_names
    nuisance_coefficients = _nuisance_coefficients(estimate, nuisance_names)
    nuisance_scales = np.maximum(
        nuisance_coefficients,
        float(resolved.nuisance_scale_floor),
    )
    fisher_started = perf_counter()
    historical_precision, historical_fisher_diagnostics = (
        _historical_factorized_fisher_precision(
            historical_design,
            source_basis,
            source_strengths,
            nuisance_coefficients,
            nuisance_scales,
            _historical_row_keys(historical_observations),
            model_identity=_cached_model_identity(
                historical_response_cache,
                "historical_factorized_design",
            ),
            minimum_expected_count=float(resolved.minimum_expected_bin_count),
            cache=historical_response_cache,
            use_gpu=bool(mle_config.use_gpu),
            gpu_device=str(mle_config.gpu_device),
            cache_entry_name="historical_factorized_fisher:exact",
        )
    )
    fisher_seconds += perf_counter() - fisher_started
    parameter_count = int(source_basis.shape[2] + nuisance_coefficients.size)
    base_precision = (
        _planning_prior_precision(
            int(source_basis.shape[2]),
            nuisance_scales,
            historical_design.nuisance_l2_weights,
            laplace_prior_precision=float(resolved.laplace_prior_precision),
        )
        + historical_precision
    )
    nuisance_count = int(nuisance_coefficients.size)
    actions: list[MLEPlanningAction] = []
    pose_chunk = int(resolved.candidate_pose_chunk_size)
    candidate_count = int(poses.shape[0])
    candidate_started = perf_counter()
    orientation_count = int(orientations.shape[0])
    pair_fe = pair_ids // orientation_count
    pair_pb = pair_ids % orientation_count
    rotation_costs, initial_rotation_costs = _pair_rotation_cost_cache(
        pair_ids,
        orientations,
        current_pair_id,
    )
    if progress_hook is not None:
        progress_hook(
            {
                "phase": "candidate_search",
                "completed_candidates": 0,
                "total_candidates": candidate_count,
                "elapsed_seconds": 0.0,
                "eta_seconds": None,
            }
        )
    for start in range(0, candidate_count, pose_chunk):
        stop = min(start + pose_chunk, candidate_count)
        if progress_hook is not None:
            progress_hook(
                {
                    "phase": "candidate_chunk",
                    "completed_candidates": int(start),
                    "total_candidates": candidate_count,
                    "elapsed_seconds": perf_counter() - candidate_started,
                    "eta_seconds": None,
                }
            )
        local_poses = poses[start:stop]
        local_count = int(local_poses.shape[0])
        expanded = np.repeat(local_poses, pair_ids.size, axis=0)
        geometry = _PlanningGeometry(
            detector_positions_xyz=expanded,
            fe_indices=np.tile(pair_fe, local_count),
            pb_indices=np.tile(pair_pb, local_count),
            live_times_s=np.full(
                expanded.shape[0],
                float(resolved.live_time_s),
                dtype=np.float64,
            ),
            energy_bin_edges_keV=historical_observations.energy_bin_edges_keV,
        )
        response_started = perf_counter()
        candidate_design = _factorized_spectral_design(
            geometry,
            estimate,
            kernel,
            mle_config,
        )
        response_seconds += perf_counter() - response_started
        if tuple(candidate_design.nuisance_names) != tuple(nuisance_names):
            raise RuntimeError("Historical and candidate nuisance bases differ.")
        fisher_started = perf_counter()
        (
            information,
            expected_counts,
            station_rate_cross,
            station_rate_information,
        ) = _factorized_fisher_information(
            candidate_design,
            source_basis,
            source_strengths,
            nuisance_coefficients,
            nuisance_scales,
            minimum_expected_count=float(resolved.minimum_expected_bin_count),
            use_gpu=bool(mle_config.use_gpu),
            gpu_device=str(mle_config.gpu_device),
        )
        fisher_seconds += perf_counter() - fisher_started
        ambiguity = _ambiguity_metrics(
            candidate_design,
            information,
            local_poses,
            estimate,
            historical_observations,
            source_basis,
            alternative_estimates,
            nuisance_coefficients=nuisance_coefficients,
            geometry_scale=resolved_geometry_scale,
        )
        information = information.reshape(
            local_count,
            pair_ids.size,
            parameter_count,
            parameter_count,
        )
        expected_counts = expected_counts.reshape(local_count, pair_ids.size)
        station_rate_cross = station_rate_cross.reshape(
            local_count,
            pair_ids.size,
            parameter_count,
        )
        station_rate_information = station_rate_information.reshape(
            local_count,
            pair_ids.size,
        )
        ambiguity_by_pose = {
            name: values.reshape(local_count, pair_ids.size)
            for name, values in ambiguity.items()
        }
        bootstrap_multiplier = (
            1.0
            if historical_observations.measurement_count
            < int(resolved.geometry_bootstrap_measurements)
            else 0.25
        )
        pair_utility_bonuses = (
            float(resolved.floor_ceiling_separation_weight)
            * ambiguity_by_pose["floor_ceiling"]
            + float(resolved.support_hypothesis_separation_weight)
            * ambiguity_by_pose["support"]
            + float(resolved.z_fisher_weight) * ambiguity_by_pose["z_fisher"]
            + float(resolved.response_correlation_reduction_weight)
            * ambiguity_by_pose["correlation"]
            + float(resolved.elevation_diversity_weight)
            * ambiguity_by_pose["elevation"]
            + bootstrap_multiplier
            * float(resolved.geometry_exploration_weight)
            * ambiguity_by_pose["geometry"]
            + float(resolved.surface_coverage_weight)
            * ambiguity_by_pose["surface_coverage"]
        )
        beam_started = perf_counter()
        if mle_config.use_gpu and parameter_count >= 24 and local_count > 1:
            local_actions = _select_pose_programs_cuda(
                np.arange(start, stop, dtype=np.int64),
                local_poses,
                pair_ids,
                information,
                expected_counts,
                base_precision,
                nuisance_count,
                orientations,
                resolved,
                travel_costs=costs[start:stop],
                station_rate_cross_information=(
                    station_rate_cross if mle_config.fit_station_rate_nuisance else None
                ),
                station_rate_information=(
                    station_rate_information
                    if mle_config.fit_station_rate_nuisance
                    else None
                ),
                pair_utility_bonus=pair_utility_bonuses,
                rotation_cost_matrix=rotation_costs,
                initial_rotation_costs=initial_rotation_costs,
                gpu_device=str(mle_config.gpu_device),
            )
        else:
            local_actions = tuple(
                _select_pose_program(
                    start + local_index,
                    local_poses[local_index],
                    pair_ids,
                    information[local_index],
                    expected_counts[local_index],
                    base_precision,
                    nuisance_count,
                    orientations,
                    resolved,
                    travel_cost=float(costs[start + local_index]),
                    current_pair_id=current_pair_id,
                    station_rate_cross_information=(
                        station_rate_cross[local_index]
                        if mle_config.fit_station_rate_nuisance
                        else None
                    ),
                    station_rate_information=(
                        station_rate_information[local_index]
                        if mle_config.fit_station_rate_nuisance
                        else None
                    ),
                    pair_utility_bonus=pair_utility_bonuses[local_index],
                    use_gpu=bool(mle_config.use_gpu and parameter_count >= 24),
                    gpu_device=str(mle_config.gpu_device),
                    rotation_cost_matrix=rotation_costs,
                    initial_rotation_costs=initial_rotation_costs,
                )
                for local_index in range(local_count)
            )
        beam_seconds += perf_counter() - beam_started
        for local_index, action in enumerate(local_actions):
            selected_indices = np.asarray(
                [
                    int(np.flatnonzero(pair_ids == pair_id)[0])
                    for pair_id in action.shield_pair_ids
                ],
                dtype=np.int64,
            )

            def selected_mean(name: str) -> float:
                """Return the selected program mean of one ambiguity metric."""
                return float(
                    np.mean(ambiguity_by_pose[name][local_index, selected_indices])
                )

            floor_ceiling = selected_mean("floor_ceiling")
            support = selected_mean("support")
            z_fisher = selected_mean("z_fisher")
            correlation = selected_mean("correlation")
            elevation = selected_mean("elevation")
            geometry = selected_mean("geometry")
            surface_coverage = selected_mean("surface_coverage")
            actions.append(
                replace(
                    action,
                    floor_ceiling_separation=floor_ceiling,
                    support_hypothesis_separation=support,
                    z_fisher_information=z_fisher,
                    response_correlation_reduction=correlation,
                    elevation_diversity=elevation,
                    geometry_exploration=geometry,
                    surface_coverage=surface_coverage,
                )
            )
        if progress_hook is not None:
            candidate_elapsed = perf_counter() - candidate_started
            completed = int(stop)
            progress_hook(
                {
                    "phase": "candidate_search",
                    "completed_candidates": completed,
                    "total_candidates": candidate_count,
                    "elapsed_seconds": candidate_elapsed,
                    "eta_seconds": (
                        candidate_elapsed
                        * float(candidate_count - completed)
                        / float(completed)
                    ),
                    "response_seconds": response_seconds,
                    "fisher_seconds": fisher_seconds,
                    "beam_search_seconds": beam_seconds,
                }
            )
    ranked = tuple(
        sorted(
            actions,
            key=lambda action: (
                -action.score,
                -action.information_gain_nats,
                action.candidate_index,
                action.shield_pair_ids,
            ),
        )
    )
    selected = ranked[0]
    likelihood_label = (
        "NB2"
        if mle_config.spectral_likelihood == "calibrated_overdispersed"
        else "Poisson"
    )
    diagnostics: dict[str, object] = {
        "criterion": f"D_s-optimal expected {likelihood_label} Fisher information",
        "laplace_approximation": True,
        "nuisance_marginalization": "Schur determinant",
        "shield_program_selection": "joint_station_block_beam_search",
        "future_station_rate_marginalized": bool(mle_config.fit_station_rate_nuisance),
        "candidate_generation_owner": "shared_runtime_controller",
        "candidate_count": int(poses.shape[0]),
        "allowed_pair_count": int(pair_ids.size),
        "source_parameter_count": int(source_basis.shape[2]),
        "nuisance_parameter_count": nuisance_count,
        "source_basis": list(basis_labels),
        "historical_measurement_count": historical_observations.measurement_count,
        "historical_step_ids": historical_observations.step_ids.astype(int).tolist(),
        "alternative_support_count": len(alternative_estimates),
        "ambiguity_aware": True,
        "geometry_bootstrap_active": (
            historical_observations.measurement_count
            < int(resolved.geometry_bootstrap_measurements)
        ),
        "config": resolved.to_dict(),
        "performance": {
            "fisher_device": (
                str(mle_config.gpu_device) if mle_config.use_gpu else "cpu"
            ),
            "beam_device": (
                str(mle_config.gpu_device)
                if mle_config.use_gpu and parameter_count >= 24
                else "cpu"
            ),
            "dtype": "float64",
            "response_seconds": response_seconds,
            "fisher_seconds": fisher_seconds,
            "beam_search_seconds": beam_seconds,
            "elapsed_seconds": perf_counter() - planning_started,
            "historical_response_cache": historical_cache_diagnostics,
            "historical_fisher_cache": historical_fisher_diagnostics,
        },
    }
    return MLEPlanningResult(
        selected_action=selected,
        ranked_actions=ranked[: int(resolved.ranked_action_limit)],
        diagnostics=diagnostics,
    )


def plan_next_measurement(
    estimate: MLEEstimate,
    historical_observations: ObservationBatch,
    kernel: ContinuousKernel,
    mle_config: MLEConfig,
    candidate_poses_xyz: object,
    *,
    planning_config: MLEPlanningConfig | None = None,
    allowed_pair_ids: Sequence[int] | None = None,
    travel_costs: object | None = None,
    current_pair_id: int | None = None,
    alternative_estimates: Sequence[MLEEstimate] = (),
    historical_response_cache: dict[str, object] | None = None,
    progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    screening_only: bool = False,
) -> MLEPlanningResult:
    """Screen many poses and exactly re-evaluate only an adaptive shortlist."""
    if not isinstance(estimate, MLEEstimate):
        raise TypeError("estimate must be an MLEEstimate.")
    if not isinstance(historical_observations, ObservationBatch):
        raise TypeError("historical_observations must be an ObservationBatch.")
    if not isinstance(kernel, ContinuousKernel):
        raise TypeError("kernel must be the shared runtime ContinuousKernel.")
    if not isinstance(mle_config, MLEConfig):
        raise TypeError("mle_config must be an MLEConfig.")
    if not isinstance(alternative_estimates, Sequence) or any(
        not isinstance(alternative, MLEEstimate)
        for alternative in alternative_estimates
    ):
        raise TypeError("alternative_estimates must contain only MLEEstimate values.")
    resolved = MLEPlanningConfig() if planning_config is None else planning_config
    if not isinstance(screening_only, (bool, np.bool_)):
        raise TypeError("screening_only must be boolean.")
    poses = _validated_candidate_poses(candidate_poses_xyz)
    ambiguity_geometry_scale = _geometry_normalization_scale(
        poses,
        historical_observations,
    )
    costs = _validated_travel_costs(travel_costs, int(poses.shape[0]))
    orientations = np.asarray(kernel.orientations, dtype=np.float64)
    if orientations.ndim != 2 or orientations.shape[1:] != (3,):
        raise ValueError("Shared kernel orientations must have shape (R, 3).")
    pairs = _validated_pair_ids(allowed_pair_ids, int(orientations.shape[0]))
    if not resolved.two_stage_screening and not screening_only:
        return _plan_next_measurement_exact(
            estimate,
            historical_observations,
            kernel,
            mle_config,
            poses,
            planning_config=resolved,
            allowed_pair_ids=pairs,
            travel_costs=costs,
            current_pair_id=current_pair_id,
            alternative_estimates=alternative_estimates,
            historical_response_cache=historical_response_cache,
            progress_hook=progress_hook,
            ambiguity_geometry_scale=ambiguity_geometry_scale,
        )
    screening = _screen_candidate_measurements(
        estimate,
        historical_observations,
        kernel,
        mle_config,
        poses,
        pairs,
        costs,
        current_pair_id,
        resolved,
        historical_response_cache,
        progress_hook,
        alternative_estimates,
    )
    if screening_only:
        return screening
    if int(poses.shape[0]) <= int(resolved.exact_candidate_min):
        exact_indices = np.arange(poses.shape[0], dtype=np.int64)
    else:
        exact_indices = _diverse_exact_candidate_indices(
            screening.ranked_actions,
            poses,
            resolved,
        )
    exact = _plan_next_measurement_exact(
        estimate,
        historical_observations,
        kernel,
        mle_config,
        poses[exact_indices],
        planning_config=resolved,
        allowed_pair_ids=pairs,
        travel_costs=costs[exact_indices],
        current_pair_id=current_pair_id,
        alternative_estimates=alternative_estimates,
        historical_response_cache=historical_response_cache,
        progress_hook=progress_hook,
        ambiguity_geometry_scale=ambiguity_geometry_scale,
    )

    def restore_index(action: MLEPlanningAction) -> MLEPlanningAction:
        """Restore one exact-shortlist index to the runtime candidate index."""
        return replace(
            action,
            candidate_index=int(exact_indices[int(action.candidate_index)]),
        )

    restored_ranked = tuple(restore_index(action) for action in exact.ranked_actions)
    restored_selected = restore_index(exact.selected_action)
    diagnostics = {
        **exact.diagnostics,
        "criterion": (
            "two-stage grouped likelihood screening then exact D_s-optimal Fisher"
        ),
        "approximate_candidate_screening": True,
        "screening": screening.diagnostics,
        "total_candidate_count": int(poses.shape[0]),
        "exact_candidate_indices": exact_indices.astype(int).tolist(),
        "exact_candidate_count": int(exact_indices.size),
        "screening_selected_candidate_index": int(
            screening.selected_action.candidate_index
        ),
        "screening_selected_score": float(screening.selected_action.score),
        "exact_selected_screening_rank": next(
            (
                index
                for index, action in enumerate(screening.ranked_actions)
                if action.candidate_index == restored_selected.candidate_index
            ),
            None,
        ),
    }
    return MLEPlanningResult(
        selected_action=restored_selected,
        ranked_actions=restored_ranked,
        diagnostics=diagnostics,
    )


def save_mle_planning_result(
    result: MLEPlanningResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save one deterministic runtime-neutral planning artifact."""
    if not isinstance(result, MLEPlanningResult):
        raise TypeError("result must be an MLEPlanningResult.")
    target = Path(path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"MLE planning output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            result.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"MLE planning staging file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "MLEPlanningAction",
    "MLEPlanningConfig",
    "MLEPlanningResult",
    "PLANNING_METHOD",
    "plan_next_measurement",
    "save_mle_planning_result",
    "select_fisher_action",
]
