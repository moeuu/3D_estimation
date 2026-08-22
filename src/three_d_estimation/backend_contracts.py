"""MLE backend and estimator-facing snapshot contracts.

Nothing in this module imports a particle filter, planner implementation, or
simulator. The MLE consumes finalized shared MeasurementLog records only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from runtime.records import MeasurementRecord, RunContext


def _readonly_array(
    value: object,
    *,
    name: str,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    """Return a finite owned immutable array with structural constraints."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array.") from exc
    if array.ndim != ndim or (
        trailing_shape and array.shape[-len(trailing_shape) :] != trailing_shape
    ):
        suffix = f" ending in {trailing_shape}" if trailing_shape else ""
        raise ValueError(f"{name} must be {ndim}-dimensional{suffix}; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values.")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _metadata(value: object, *, name: str) -> dict[str, object]:
    """Copy a metadata dictionary without accepting implicit mappings."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary.")
    return dict(value)


@dataclass(frozen=True, eq=False)
class SourceMode:
    """A compact point-like source mode exposed by any estimator."""

    position_xyz: tuple[float, float, float]
    strength_cps_1m: float
    covariance_xyz_m2: NDArray[np.float64] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate physical source-mode values and optional covariance."""
        position = _readonly_array(
            self.position_xyz,
            name="position_xyz",
            ndim=1,
            trailing_shape=(3,),
        )
        strength = float(self.strength_cps_1m)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("strength_cps_1m must be finite and nonnegative.")
        covariance = None
        if self.covariance_xyz_m2 is not None:
            covariance = _readonly_array(
                self.covariance_xyz_m2,
                name="covariance_xyz_m2",
                ndim=2,
                trailing_shape=(3, 3),
            )
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
                raise ValueError("covariance_xyz_m2 must be symmetric.")
            if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
                raise ValueError("covariance_xyz_m2 must be positive semidefinite.")
        object.__setattr__(self, "position_xyz", tuple(float(v) for v in position))
        object.__setattr__(self, "strength_cps_1m", strength)
        object.__setattr__(self, "covariance_xyz_m2", covariance)
        object.__setattr__(self, "metadata", _metadata(self.metadata, name="metadata"))


@dataclass(frozen=True, eq=False)
class SurfaceMapSnapshot:
    """One isotope's surface-density values at a specific estimator step."""

    patch_ids: tuple[int, ...]
    patch_centroids_xyz: NDArray[np.float64]
    density_cps_1m_per_m2: NDArray[np.float64]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate aligned surface-map arrays and patch identifiers."""
        patch_ids = tuple(int(patch_id) for patch_id in self.patch_ids)
        if any(patch_id < 0 for patch_id in patch_ids):
            raise ValueError("patch_ids must be nonnegative.")
        if len(set(patch_ids)) != len(patch_ids):
            raise ValueError("patch_ids must be unique.")
        centroids = _readonly_array(
            self.patch_centroids_xyz,
            name="patch_centroids_xyz",
            ndim=2,
            trailing_shape=(3,),
        )
        density = _readonly_array(
            self.density_cps_1m_per_m2,
            name="density_cps_1m_per_m2",
            ndim=1,
            nonnegative=True,
        )
        if centroids.shape[0] != len(patch_ids) or density.shape != (len(patch_ids),):
            raise ValueError(
                "patch_ids, patch_centroids_xyz, and density_cps_1m_per_m2 "
                "must describe the same number of patches."
            )
        object.__setattr__(self, "patch_ids", patch_ids)
        object.__setattr__(self, "patch_centroids_xyz", centroids)
        object.__setattr__(self, "density_cps_1m_per_m2", density)
        object.__setattr__(self, "metadata", _metadata(self.metadata, name="metadata"))


@dataclass(frozen=True, eq=False)
class EstimatorSnapshot:
    """Estimator-neutral visualization/reporting state."""

    step_id: int
    source_modes_by_isotope: dict[str, tuple[SourceMode, ...]]
    surface_map_by_isotope: dict[str, SurfaceMapSnapshot] | None
    predicted_spectrum: NDArray[np.float64] | None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze estimator-neutral snapshot content."""
        if isinstance(self.step_id, bool) or not isinstance(self.step_id, (int, np.integer)):
            raise TypeError("step_id must be an integer.")
        if int(self.step_id) < -1:
            raise ValueError("step_id must be -1 (uninitialized) or nonnegative.")

        modes: dict[str, tuple[SourceMode, ...]] = {}
        for isotope, isotope_modes in sorted(self.source_modes_by_isotope.items()):
            if not isotope:
                raise ValueError("source_modes_by_isotope keys must be non-empty.")
            mode_tuple = tuple(isotope_modes)
            if not all(isinstance(mode, SourceMode) for mode in mode_tuple):
                raise TypeError("source_modes_by_isotope values must contain SourceMode objects.")
            modes[isotope] = mode_tuple

        surface_maps = None
        if self.surface_map_by_isotope is not None:
            surface_maps = {}
            for isotope, surface_map in sorted(self.surface_map_by_isotope.items()):
                if not isotope or not isinstance(surface_map, SurfaceMapSnapshot):
                    raise TypeError(
                        "surface_map_by_isotope must map isotope names to SurfaceMapSnapshot."
                    )
                surface_maps[isotope] = surface_map

        predicted_spectrum = None
        if self.predicted_spectrum is not None:
            predicted_spectrum = _readonly_array(
                self.predicted_spectrum,
                name="predicted_spectrum",
                ndim=1,
                nonnegative=True,
            )

        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(self, "source_modes_by_isotope", modes)
        object.__setattr__(self, "surface_map_by_isotope", surface_maps)
        object.__setattr__(self, "predicted_spectrum", predicted_spectrum)
        object.__setattr__(
            self,
            "diagnostics",
            _metadata(self.diagnostics, name="diagnostics"),
        )


@dataclass(frozen=True, eq=False)
class EstimatorResult:
    """Final estimator-neutral result plus backend-specific diagnostics/artifacts."""

    final_snapshot: EstimatorSnapshot
    diagnostics: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate final snapshot, diagnostics, and artifact references."""
        if not isinstance(self.final_snapshot, EstimatorSnapshot):
            raise TypeError("final_snapshot must be an EstimatorSnapshot.")
        artifacts = dict(self.artifacts)
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in artifacts.items()
        ):
            raise TypeError("artifacts must map string names to string paths.")
        object.__setattr__(
            self,
            "diagnostics",
            _metadata(self.diagnostics, name="diagnostics"),
        )
        object.__setattr__(self, "artifacts", artifacts)


@runtime_checkable
class EstimatorBackend(Protocol):
    """Minimum estimator contract used by live acquisition sessions."""

    def initialize(self, context: RunContext) -> None:
        """Initialize the estimator for one run."""

    def update(self, measurement: MeasurementRecord) -> None:
        """Consume one finalized and already-recorded measurement."""

    def snapshot(self) -> EstimatorSnapshot:
        """Return the latest estimator-neutral state."""

    def finalize(self) -> EstimatorResult:
        """Return the final all-history estimate."""


@runtime_checkable
class StationCompleteEstimatorBackend(Protocol):
    """Optional hook for a warm solve after all records at one station."""

    def on_station_complete(
        self,
        station_id: int,
        measurements: tuple[MeasurementRecord, ...],
    ) -> None:
        """Perform any backend-specific station-complete warm update."""


ActionT = TypeVar("ActionT")
PredictionT = TypeVar("PredictionT")
UncertaintyT = TypeVar("UncertaintyT")
ModelOrderT = TypeVar("ModelOrderT")


@runtime_checkable
class PlannerBeliefProvider(
    Protocol[ActionT, PredictionT, UncertaintyT, ModelOrderT]
):
    """Separate optional contract for planner-facing belief projections."""

    def source_modes(self) -> dict[str, tuple[SourceMode, ...]]:
        """Return compact source modes for planning."""

    def predict_candidate_counts(self, actions: Sequence[ActionT]) -> PredictionT:
        """Predict observations for opaque planner-owned candidate actions."""

    def uncertainty_summary(self) -> UncertaintyT:
        """Return an opaque planner-facing uncertainty summary."""

    def model_order_summary(self) -> ModelOrderT:
        """Return opaque source-cardinality evidence."""
