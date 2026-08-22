"""Immutable MLE-owned snapshots for live rendering integrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from runtime import DigestIdentity

from .types import MLEEstimate, SURFACE_DENSITY_UNIT


def _immutable_float_array(
    values: ArrayLike,
    *,
    name: str,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Return a finite copied array backed by immutable bytes."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


def _freeze_hotspot_value(value: object, *, path: str) -> object:
    """Deep-copy one hotspot value into immutable strict finite data."""
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings.")
            frozen[key] = _freeze_hotspot_value(nested, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        return _freeze_hotspot_value(value.tolist(), path=path)
    if isinstance(value, np.generic):
        return _freeze_hotspot_value(value.item(), path=path)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            _freeze_hotspot_value(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"{path} must contain only finite values.")
        return parsed
    raise TypeError(f"{path} contains unsupported data {type(value).__name__}.")


def _immutable_hotspots(
    values: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Return a deeply immutable copy of MLE hotspot diagnostics."""
    frozen: list[Mapping[str, object]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise TypeError(f"hotspot_clusters[{index}] must be an object.")
        copied = _freeze_hotspot_value(value, path=f"hotspot_clusters[{index}]")
        if not isinstance(copied, Mapping):
            raise TypeError(f"hotspot_clusters[{index}] must remain an object.")
        frozen.append(copied)
    return tuple(frozen)


@dataclass(frozen=True, slots=True)
class MLELiveSurfaceSnapshot:
    """Expose one causal MLE surface grid without PF-shaped state."""

    measurement_run_id: str
    record_count: int
    data_cutoff_step: int
    data_cutoff_station: int
    covered_records_digest: DigestIdentity
    isotope_names: tuple[str, ...]
    patch_ids: tuple[int, ...]
    patch_centroids_xyz: NDArray[np.float64]
    patch_surface_kinds: tuple[str, ...]
    patch_object_ids: tuple[str, ...]
    density_by_isotope: NDArray[np.float64]
    hotspot_clusters: tuple[Mapping[str, object], ...]
    latest_predicted_spectrum: NDArray[np.float64] | None
    representation: str = field(default="mle_surface_density_grid", init=False)
    density_unit: str = field(default=SURFACE_DENSITY_UNIT, init=False)

    def __post_init__(self) -> None:
        """Validate lineage and freeze all estimator-owned rendering values."""
        run_id = str(self.measurement_run_id).strip()
        if not run_id:
            raise ValueError("measurement_run_id must be non-empty.")
        record_count = int(self.record_count)
        cutoff_step = int(self.data_cutoff_step)
        cutoff_station = int(self.data_cutoff_station)
        if record_count < 1 or cutoff_step < 0 or cutoff_station < 0:
            raise ValueError("MLE live snapshot lineage values must be non-negative.")
        if not isinstance(self.covered_records_digest, DigestIdentity):
            raise TypeError("covered_records_digest must be a DigestIdentity.")
        isotopes = tuple(str(value).strip() for value in self.isotope_names)
        if not isotopes or any(not value for value in isotopes):
            raise ValueError("isotope_names must be non-empty.")
        if len(set(isotopes)) != len(isotopes):
            raise ValueError("isotope_names must be unique.")
        patch_ids = tuple(int(value) for value in self.patch_ids)
        if any(value < 0 for value in patch_ids) or len(set(patch_ids)) != len(
            patch_ids
        ):
            raise ValueError("patch_ids must be non-negative and unique.")
        patch_count = len(patch_ids)
        kinds = tuple(str(value).strip() for value in self.patch_surface_kinds)
        object_ids = tuple(str(value).strip() for value in self.patch_object_ids)
        if len(kinds) != patch_count or any(not value for value in kinds):
            raise ValueError("patch_surface_kinds must align with patch_ids.")
        if len(object_ids) != patch_count or any(not value for value in object_ids):
            raise ValueError("patch_object_ids must align with patch_ids.")
        centroids = _immutable_float_array(
            self.patch_centroids_xyz,
            name="patch_centroids_xyz",
            shape=(patch_count, 3),
        )
        density = _immutable_float_array(
            self.density_by_isotope,
            name="density_by_isotope",
            shape=(len(isotopes), patch_count),
        )
        if np.any(density < 0.0):
            raise ValueError("density_by_isotope must be non-negative.")
        prediction = None
        if self.latest_predicted_spectrum is not None:
            raw_prediction = np.asarray(
                self.latest_predicted_spectrum,
                dtype=np.float64,
            ).reshape(-1)
            if raw_prediction.size < 1:
                raise ValueError("latest_predicted_spectrum must not be empty.")
            prediction = _immutable_float_array(
                raw_prediction,
                name="latest_predicted_spectrum",
                shape=(int(raw_prediction.size),),
            )
            if np.any(prediction < 0.0):
                raise ValueError("latest_predicted_spectrum must be non-negative.")
        hotspots = _immutable_hotspots(self.hotspot_clusters)
        object.__setattr__(self, "measurement_run_id", run_id)
        object.__setattr__(self, "record_count", record_count)
        object.__setattr__(self, "data_cutoff_step", cutoff_step)
        object.__setattr__(self, "data_cutoff_station", cutoff_station)
        object.__setattr__(self, "isotope_names", isotopes)
        object.__setattr__(self, "patch_ids", patch_ids)
        object.__setattr__(self, "patch_centroids_xyz", centroids)
        object.__setattr__(self, "patch_surface_kinds", kinds)
        object.__setattr__(self, "patch_object_ids", object_ids)
        object.__setattr__(self, "density_by_isotope", density)
        object.__setattr__(self, "hotspot_clusters", hotspots)
        object.__setattr__(self, "latest_predicted_spectrum", prediction)

    @property
    def covered_records_sha256(self) -> str:
        """Return the SHA-256 compatibility alias for the typed digest."""
        return self.covered_records_digest.sha256

    @classmethod
    def from_estimate(
        cls,
        estimate: MLEEstimate,
        *,
        measurement_run_id: str,
        record_count: int,
        data_cutoff_step: int,
        data_cutoff_station: int,
        covered_records_digest: DigestIdentity,
        include_latest_prediction: bool,
    ) -> MLELiveSurfaceSnapshot:
        """Copy one existing MLE estimate into the live surface contract."""
        if not isinstance(estimate, MLEEstimate):
            raise TypeError("estimate must be an MLEEstimate.")
        raw_hotspots = estimate.diagnostics.get("hotspot_clusters", ())
        if raw_hotspots is None:
            raw_hotspots = ()
        if isinstance(raw_hotspots, (str, bytes)) or not isinstance(
            raw_hotspots,
            Sequence,
        ):
            raise TypeError("MLE hotspot_clusters diagnostics must be a sequence.")
        prediction = None
        if include_latest_prediction and estimate.predicted_spectra is not None:
            predicted = np.asarray(estimate.predicted_spectra, dtype=np.float64)
            if predicted.ndim != 2 or predicted.shape[0] != int(record_count):
                raise ValueError(
                    "Current MLE prediction rows must cover the complete live history."
                )
            prediction = predicted[-1]
        return cls(
            measurement_run_id=measurement_run_id,
            record_count=record_count,
            data_cutoff_step=data_cutoff_step,
            data_cutoff_station=data_cutoff_station,
            covered_records_digest=covered_records_digest,
            isotope_names=estimate.isotope_names,
            patch_ids=tuple(patch.patch_id for patch in estimate.patches),
            patch_centroids_xyz=np.asarray(
                [patch.centroid_xyz for patch in estimate.patches],
                dtype=np.float64,
            ),
            patch_surface_kinds=tuple(
                patch.surface_kind for patch in estimate.patches
            ),
            patch_object_ids=tuple(patch.object_id for patch in estimate.patches),
            density_by_isotope=estimate.density_by_isotope,
            hotspot_clusters=tuple(raw_hotspots),
            latest_predicted_spectrum=prediction,
        )


__all__ = ["MLELiveSurfaceSnapshot"]
