"""Runtime adapter exposing the standalone surface MLE as an estimator backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
import inspect
import json
import math
from pathlib import Path
import numpy as np

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from three_d_estimation.backend_contracts import (
    EstimatorResult,
    EstimatorSnapshot,
    SourceMode,
    SurfaceMapSnapshot,
)
from runtime import ResolvedForwardContext
from runtime.records import MeasurementRecord, RunContext

from .config import MLEConfig
from .estimator import SurfaceMLEEstimator
from .information_planner import (
    MLEPlanningConfig,
    MLEPlanningResult,
    plan_next_measurement,
)
from .observation_batch import observation_batch_from_records
from .types import MLEEstimate


_SOURCE_RATE_MODEL = "detector_cps_1m"


EstimatorFactory = Callable[[MLEConfig], SurfaceMLEEstimator]


def _json_safe(value: object) -> object:
    """Return strict JSON data, replacing non-finite diagnostics with null."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_safe(child) for child in value]
    return str(value)


def _strict_json_dict(value: Mapping[str, object]) -> dict[str, object]:
    """Normalize a diagnostics mapping and prove strict JSON serialization."""
    safe = _json_safe(value)
    if not isinstance(safe, dict):
        raise TypeError("Diagnostics must normalize to a JSON object.")
    json.dumps(safe, allow_nan=False, sort_keys=True)
    return safe


def _cluster_source_modes(
    estimate: MLEEstimate,
) -> dict[str, tuple[SourceMode, ...]]:
    """Convert MLE hotspot diagnostics into generic source modes."""
    grouped: dict[str, list[SourceMode]] = {
        isotope: [] for isotope in estimate.isotope_names
    }
    raw_clusters = estimate.diagnostics.get("hotspot_clusters", [])
    if raw_clusters is None:
        raw_clusters = []
    if not isinstance(raw_clusters, Sequence) or isinstance(raw_clusters, (str, bytes)):
        raise ValueError("MLE hotspot_clusters diagnostics must be a sequence.")
    for index, cluster in enumerate(raw_clusters):
        if not isinstance(cluster, Mapping):
            raise ValueError(f"hotspot_clusters[{index}] must be an object.")
        isotope = str(cluster.get("isotope", ""))
        if isotope not in grouped:
            raise ValueError(
                f"hotspot_clusters[{index}] has unknown isotope {isotope!r}."
            )
        centroid_raw = cluster.get("centroid_xyz")
        if isinstance(centroid_raw, (str, bytes)) or not isinstance(
            centroid_raw,
            Sequence,
        ):
            raise ValueError(f"hotspot_clusters[{index}] lacks centroid_xyz.")
        centroid = tuple(float(value) for value in centroid_raw)
        if len(centroid) != 3 or any(not math.isfinite(value) for value in centroid):
            raise ValueError(f"hotspot_clusters[{index}] centroid_xyz is invalid.")
        strength = float(cluster.get("integrated_strength_cps_1m", 0.0))
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError(f"hotspot_clusters[{index}] strength is invalid.")
        metadata = {
            str(key): _json_safe(value)
            for key, value in cluster.items()
            if key not in {"isotope", "centroid_xyz", "integrated_strength_cps_1m"}
        }
        covariance_raw = cluster.get("centroid_covariance_xyz_m2")
        covariance = (
            None
            if covariance_raw is None
            else np.asarray(covariance_raw, dtype=np.float64)
        )
        grouped[isotope].append(
            SourceMode(
                position_xyz=centroid,
                strength_cps_1m=strength,
                covariance_xyz_m2=covariance,
                metadata=metadata,
            )
        )
    return {
        isotope: tuple(
            sorted(
                modes,
                key=lambda mode: (-mode.strength_cps_1m, mode.position_xyz),
            )
        )
        for isotope, modes in grouped.items()
    }


def _surface_maps(
    estimate: MLEEstimate,
) -> dict[str, SurfaceMapSnapshot]:
    """Convert every isotope density row into a generic surface map."""
    patch_ids = tuple(patch.patch_id for patch in estimate.patches)
    centroids = np.vstack([patch.centroid_xyz for patch in estimate.patches])
    areas = [float(patch.area_m2) for patch in estimate.patches]
    kinds = [str(patch.surface_kind) for patch in estimate.patches]
    object_ids = [str(patch.object_id) for patch in estimate.patches]
    result: dict[str, SurfaceMapSnapshot] = {}
    for isotope_index, isotope in enumerate(estimate.isotope_names):
        result[isotope] = SurfaceMapSnapshot(
            patch_ids=patch_ids,
            patch_centroids_xyz=centroids,
            density_cps_1m_per_m2=estimate.density_by_isotope[isotope_index],
            metadata={
                "density_unit": "detector_cps_1m_per_m2",
                "patch_strength_unit": "detector_cps_1m",
                "areas_m2": areas,
                "surface_kinds": kinds,
                "object_ids": object_ids,
                "patch_strengths_cps_1m": estimate.patch_strength_by_isotope[
                    isotope_index
                ].tolist(),
            },
        )
    return result


def _snapshot_from_estimate(
    estimate: MLEEstimate,
    *,
    step_id: int,
    measurement_count: int,
    fit_kind: str,
) -> EstimatorSnapshot:
    """Convert an MLE estimate to the generic estimator snapshot contract."""
    predicted_spectrum = None
    if estimate.predicted_spectra is not None:
        if estimate.predicted_spectra.shape[0] != measurement_count:
            raise ValueError(
                "Predicted spectrum rows do not match buffered measurements."
            )
        predicted_spectrum = estimate.predicted_spectra[-1]
    diagnostics: dict[str, object] = {
        **estimate.diagnostics,
        "backend": "surface_mle",
        "fit_kind": fit_kind,
        "measurement_count": int(measurement_count),
        "objective_value": float(estimate.objective_value),
        "poisson_deviance": float(estimate.poisson_deviance),
        "iterations": int(estimate.iterations),
        "converged": bool(estimate.converged),
    }
    if estimate.predicted_isotope_counts is not None:
        if estimate.predicted_isotope_counts.shape[0] != measurement_count:
            raise ValueError("Predicted count rows do not match buffered measurements.")
        diagnostics["latest_predicted_isotope_counts"] = (
            estimate.predicted_isotope_counts[-1].tolist()
        )
    return EstimatorSnapshot(
        step_id=step_id,
        source_modes_by_isotope=_cluster_source_modes(estimate),
        surface_map_by_isotope=_surface_maps(estimate),
        predicted_spectrum=predicted_spectrum,
        diagnostics=_strict_json_dict(diagnostics),
    )


class SurfaceMLEBackend:
    """Buffer finalized records and run periodic/final all-history surface MLE."""

    def __init__(
        self,
        config: MLEConfig,
        *,
        estimator_factory: EstimatorFactory = SurfaceMLEEstimator,
        run_root: str | Path | None = None,
        progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        """Store configuration, runtime asset root, and estimator factory."""
        if not isinstance(config, MLEConfig):
            raise TypeError("config must be MLEConfig.")
        if not callable(estimator_factory):
            raise TypeError("estimator_factory must be callable.")
        if progress_hook is not None and not callable(progress_hook):
            raise TypeError("progress_hook must be callable or None.")
        self.config = config
        self._estimator_factory = estimator_factory
        self._run_root = None if run_root is None else Path(run_root).resolve()
        self._progress_hook = progress_hook
        self._context: RunContext | None = None
        self._forward_context: ResolvedForwardContext | None = None
        self._environment: EnvironmentConfig | None = None
        self._obstacle_grid: ObstacleGrid | None = None
        self._kernel: object | None = None
        self._estimator: SurfaceMLEEstimator | None = None
        self._online_estimator: SurfaceMLEEstimator | None = None
        self._planning_response_cache: dict[str, object] = {}
        self._records: list[MeasurementRecord] = []
        self._step_ids: set[int] = set()
        self._latest_estimate: MLEEstimate | None = None
        self._fit_estimates: list[MLEEstimate] = []
        self._latest_fit_kind: str | None = None
        self._latest_fit_measurement_count = 0
        self._latest_fit_step_id = -1
        self._last_station_fit_count = 0
        self._final_result: EstimatorResult | None = None

    @property
    def records(self) -> tuple[MeasurementRecord, ...]:
        """Return the buffered finalized history in acquisition order."""
        return tuple(self._records)

    @property
    def latest_estimate(self) -> MLEEstimate | None:
        """Return the latest station or final all-history MLE estimate."""
        return self._latest_estimate

    @property
    def forward_context(self) -> ResolvedForwardContext | None:
        """Return the shared runtime context resolved during initialization."""
        return self._forward_context

    def initialize(self, context: RunContext) -> None:
        """Build all local physical objects for one independent run."""
        if self._context is not None:
            raise RuntimeError("SurfaceMLEBackend is already initialized.")
        if not isinstance(context, RunContext):
            raise TypeError("context must be RunContext.")
        if str(context.source_rate_model).strip().lower() != _SOURCE_RATE_MODEL:
            raise ValueError("RunContext source_rate_model must be 'detector_cps_1m'.")
        if tuple(context.isotopes) != tuple(self.config.isotope_names):
            raise ValueError(
                "RunContext isotopes must exactly match MLEConfig.isotope_names."
            )
        if self.config.mode == "count" and (
            str(context.spectrum_count_method).strip().lower() != "response_poisson"
        ):
            raise ValueError(
                "Count SurfaceMLEBackend requires response_poisson counts."
            )

        if self._run_root is None:
            raise ValueError(
                "SurfaceMLEBackend requires an explicit shared runtime asset root."
            )
        forward_context = ResolvedForwardContext.from_run_context(
            context,
            run_root=self._run_root,
        )
        kernel = forward_context.build_continuous_kernel(
            use_gpu=bool(self.config.use_gpu),
            gpu_device=str(self.config.gpu_device),
            gpu_dtype=str(self.config.gpu_dtype),
        )
        estimator = self._new_estimator(self.config)
        if not callable(getattr(estimator, "fit", None)):
            raise TypeError("estimator_factory must return an object with fit().")

        self._context = context
        self._forward_context = forward_context
        self._environment = forward_context.environment
        self._obstacle_grid = forward_context.obstacle_grid
        self._kernel = kernel
        self._estimator = estimator

    def _new_estimator(self, config: MLEConfig) -> SurfaceMLEEstimator:
        """Create an estimator while preserving custom one-argument factories."""
        if self._estimator_factory is SurfaceMLEEstimator:
            return SurfaceMLEEstimator(config, progress_hook=self._progress_hook)
        return self._estimator_factory(config)

    def _ensure_active(self) -> None:
        """Raise when the backend is uninitialized or already finalized."""
        if self._context is None:
            raise RuntimeError("SurfaceMLEBackend is not initialized.")
        if self._final_result is not None:
            raise RuntimeError("SurfaceMLEBackend is already finalized.")

    def update(self, measurement: MeasurementRecord) -> None:
        """Buffer one record without fitting an incomplete measurement station."""
        self._ensure_active()
        if not isinstance(measurement, MeasurementRecord):
            raise TypeError("measurement must be MeasurementRecord.")
        if measurement.step_id in self._step_ids:
            raise ValueError(f"Duplicate finalized step_id {measurement.step_id}.")
        self._records.append(measurement)
        self._step_ids.add(measurement.step_id)

    def _fit_history(self, *, fit_kind: str) -> MLEEstimate:
        """Fit the complete buffered history and cache its estimate."""
        self._ensure_active()
        if not self._records:
            raise RuntimeError("SurfaceMLEBackend has no measurements to fit.")
        assert self._context is not None
        assert self._environment is not None
        assert self._kernel is not None
        assert self._estimator is not None
        batch = observation_batch_from_records(
            self._records,
            self._context.isotopes,
            context=self._context,
        )
        fit_kwargs: dict[str, object] = {"obstacle_grid": self._obstacle_grid}
        fit_signature = inspect.signature(self._estimator.fit)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in fit_signature.parameters.values()
        )
        if (
            self._latest_estimate is not None
            and ("initial_estimate" in fit_signature.parameters or accepts_kwargs)
        ):
            fit_kwargs["initial_estimate"] = self._latest_estimate
        active_estimator = self._estimator
        if fit_kind != "final" and (
            bool(self.config.uncertainty_enable)
            or self.config.online_patch_spacing_m is not None
            or int(self.config.online_coarse_to_fine_levels)
            != int(self.config.coarse_to_fine_levels)
        ):
            if self._online_estimator is None:
                self._online_estimator = self._new_estimator(
                    replace(
                        self.config,
                        uncertainty_enable=False,
                        station_bootstrap_replicates=0,
                        patch_spacing_m=(
                            self.config.patch_spacing_m
                            if self.config.online_patch_spacing_m is None
                            else self.config.online_patch_spacing_m
                        ),
                        coarse_to_fine_levels=int(
                            self.config.online_coarse_to_fine_levels
                        ),
                    )
                )
            active_estimator = self._online_estimator
        estimate = active_estimator.fit(
            batch,
            self._environment,
            self._kernel,
            **fit_kwargs,
        )
        if not isinstance(estimate, MLEEstimate):
            raise TypeError("SurfaceMLEEstimator.fit() must return MLEEstimate.")
        self._latest_estimate = estimate
        self._fit_estimates.append(estimate)
        self._latest_fit_kind = fit_kind
        self._latest_fit_measurement_count = len(self._records)
        self._latest_fit_step_id = self._records[-1].step_id
        return estimate

    def on_station_complete(
        self,
        station_id: int,
        measurements: tuple[MeasurementRecord, ...],
    ) -> None:
        """Run a station-complete all-history fit for the next warm snapshot."""
        self._ensure_active()
        station_records = tuple(measurements)
        if not station_records:
            raise ValueError("Station completion requires at least one measurement.")
        if any(not isinstance(record, MeasurementRecord) for record in station_records):
            raise TypeError(
                "Station measurements must contain MeasurementRecord objects."
            )
        if any(record.station_id != int(station_id) for record in station_records):
            raise ValueError("Station-complete measurements do not match station_id.")
        buffered_suffix = self._records[-len(station_records) :]
        if len(station_records) > len(self._records) or any(
            supplied is not buffered
            for supplied, buffered in zip(station_records, buffered_suffix, strict=True)
        ):
            raise ValueError(
                "Station-complete measurements must be the buffered history suffix."
            )
        if len(self._records) == self._last_station_fit_count:
            raise RuntimeError("No new measurements are available for a station fit.")
        self._fit_history(fit_kind="station_warm")
        self._last_station_fit_count = len(self._records)

    def snapshot(self) -> EstimatorSnapshot:
        """Return an empty pre-fit snapshot or the latest warm/final estimate."""
        if self._context is None:
            raise RuntimeError("SurfaceMLEBackend is not initialized.")
        step_id = self._records[-1].step_id if self._records else -1
        if self._latest_estimate is None:
            return EstimatorSnapshot(
                step_id=step_id,
                source_modes_by_isotope={
                    name: () for name in self.config.isotope_names
                },
                surface_map_by_isotope=None,
                predicted_spectrum=None,
                diagnostics={
                    "backend": "surface_mle",
                    "fit_kind": "not_fitted",
                    "measurement_count": len(self._records),
                },
            )
        return _snapshot_from_estimate(
            self._latest_estimate,
            step_id=self._latest_fit_step_id,
            measurement_count=self._latest_fit_measurement_count,
            fit_kind=str(self._latest_fit_kind),
        )

    def plan_next_action(
        self,
        candidate_poses_xyz: object,
        *,
        planning_config: MLEPlanningConfig | None = None,
        allowed_pair_ids: Sequence[int] | None = None,
        travel_costs: object | None = None,
        current_pair_id: int | None = None,
        progress_hook: Callable[[Mapping[str, object]], None] | None = None,
        screening_only: bool = False,
    ) -> MLEPlanningResult:
        """Rank runtime-supplied poses and Fe/Pb programs from the latest fit."""
        self._ensure_active()
        if self._latest_estimate is None:
            raise RuntimeError("Complete an MLE station fit before planning.")
        if self._latest_fit_measurement_count != len(self._records):
            raise RuntimeError(
                "Planning requires an MLE fit covering the current durable history."
            )
        if not isinstance(self._kernel, ContinuousKernel):
            raise TypeError("MLE planning requires the shared ContinuousKernel.")
        assert self._context is not None
        batch = observation_batch_from_records(
            self._records,
            self._context.isotopes,
            context=self._context,
        )
        resolved_current_pair = current_pair_id
        if resolved_current_pair is None:
            orientation_count = int(len(self._kernel.orientations))
            latest = self._records[-1]
            resolved_current_pair = int(
                latest.fe_orientation_index
            ) * orientation_count + int(latest.pb_orientation_index)
        return plan_next_measurement(
            self._latest_estimate,
            batch,
            self._kernel,
            self.config,
            candidate_poses_xyz,
            planning_config=planning_config,
            allowed_pair_ids=allowed_pair_ids,
            travel_costs=travel_costs,
            current_pair_id=resolved_current_pair,
            alternative_estimates=tuple(self._fit_estimates[-4:-1]),
            historical_response_cache=self._planning_response_cache,
            progress_hook=progress_hook,
            screening_only=screening_only,
        )

    def finalize(self) -> EstimatorResult:
        """Run and return one final all-history fit, caching the generic result."""
        if self._final_result is not None:
            return self._final_result
        estimate = self._fit_history(fit_kind="final")
        snapshot = _snapshot_from_estimate(
            estimate,
            step_id=self._records[-1].step_id,
            measurement_count=len(self._records),
            fit_kind="final",
        )
        result = EstimatorResult(
            final_snapshot=snapshot,
            diagnostics=_strict_json_dict(
                {
                    **snapshot.diagnostics,
                    "source_rate_model": _SOURCE_RATE_MODEL,
                    "isotopes": list(self.config.isotope_names),
                }
            ),
        )
        self._final_result = result
        return result


__all__ = ["SurfaceMLEBackend"]
