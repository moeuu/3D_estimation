"""Line-resolved spectral response tensors built from the local shared kernel."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
import json
from multiprocessing import get_context
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from runtime.discrepancy_calibration import DiscrepancyCalibration
from spectrum.additive_scatter import (
    AdditiveNoncollidedTransportResponse,
    PhysicsOnlyNoncollidedTransportResponse,
)
from spectrum.library import default_library
from spectrum.response_matrix import (
    BACKSCATTER_FRACTION,
    COMPTON_CONTINUUM_TO_PEAK,
    cebr3_efficiency,
    compton_continuum_shape,
    default_background_shape,
    default_resolution,
    detector_response_kernel_for_incident_gamma,
)

from .response_operator import (
    LineFactorizedResponseOperator,
    atomic_save_npy,
)


@dataclass(frozen=True, slots=True)
class _SpectralProcessKernel:
    """Store a pickle-safe recipe for one fresh runtime kernel."""

    kernel_without_additive_response: ContinuousKernel
    additive_response_kind: str | None
    additive_response_payload: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class _PreparedSpectralIsotope:
    """Store one isotope's jointly evaluated transport lines and pulses."""

    isotope_index: int
    isotope: str
    line_start: int
    weights: NDArray[np.float64]
    positive_line_indices: NDArray[np.int64]
    kernel: ContinuousKernel | _SpectralProcessKernel
    pulses: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _SpectralProcessContext:
    """Store immutable inputs shared by CPU response worker processes."""

    detector_positions: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times: NDArray[np.float64]
    areas: NDArray[np.float64]
    quadrature_points: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]
    line_count: int
    prepared_isotopes: tuple[_PreparedSpectralIsotope, ...]
    kernel_chunk_size: int


_SPECTRAL_PROCESS_CONTEXT: _SpectralProcessContext | None = None
_SPECTRAL_THREADPOOL_LIMITER: object | None = None


def _available_cpu_count() -> int:
    """Return the process-affinity CPU count with a portable fallback."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _fresh_spectral_process_context(
    context: _SpectralProcessContext,
) -> _SpectralProcessContext:
    """Return a spawn-safe context without parent caches or mapping proxies."""
    cloned_kernels: dict[int, _SpectralProcessKernel] = {}
    prepared_isotopes: list[_PreparedSpectralIsotope] = []
    for prepared in context.prepared_isotopes:
        if not isinstance(prepared.kernel, ContinuousKernel):
            raise TypeError("Spectral process preparation requires runtime kernels.")
        identity = id(prepared.kernel)
        fresh_kernel = cloned_kernels.get(identity)
        if fresh_kernel is None:
            response = prepared.kernel.additive_scatter_response
            if response is None:
                response_kind = None
                response_payload = None
            elif type(response) is AdditiveNoncollidedTransportResponse:
                response_kind = "additive_noncollided"
                response_payload = response.to_payload()
            elif type(response) is PhysicsOnlyNoncollidedTransportResponse:
                response_kind = "physics_only_noncollided"
                response_payload = response.to_payload()
            else:
                raise TypeError("Unsupported additive scatter response type.")
            fresh_kernel = _SpectralProcessKernel(
                kernel_without_additive_response=replace(
                    prepared.kernel,
                    additive_scatter_response=None,
                ),
                additive_response_kind=response_kind,
                additive_response_payload=response_payload,
            )
            cloned_kernels[identity] = fresh_kernel
        prepared_isotopes.append(replace(prepared, kernel=fresh_kernel))
    return replace(context, prepared_isotopes=tuple(prepared_isotopes))


def _materialize_spectral_process_context(
    context: _SpectralProcessContext,
) -> _SpectralProcessContext:
    """Rebuild authenticated runtime responses inside one spawned worker."""
    restored_kernels: dict[int, ContinuousKernel] = {}
    prepared_isotopes: list[_PreparedSpectralIsotope] = []
    for prepared in context.prepared_isotopes:
        recipe = prepared.kernel
        if isinstance(recipe, ContinuousKernel):
            restored_kernel = replace(recipe)
        else:
            identity = id(recipe)
            restored_kernel = restored_kernels.get(identity)
            if restored_kernel is None:
                payload = recipe.additive_response_payload
                if recipe.additive_response_kind is None:
                    if payload is not None:
                        raise ValueError(
                            "Kernel response payload lacks a response kind."
                        )
                    response = None
                elif recipe.additive_response_kind == "additive_noncollided":
                    if payload is None:
                        raise ValueError("Additive response payload is unavailable.")
                    response = AdditiveNoncollidedTransportResponse.from_payload(
                        payload
                    )
                elif recipe.additive_response_kind == "physics_only_noncollided":
                    if payload is None:
                        raise ValueError(
                            "Physics-only response payload is unavailable."
                        )
                    response = PhysicsOnlyNoncollidedTransportResponse.from_payload(
                        payload
                    )
                else:
                    raise ValueError("Kernel response recipe kind is invalid.")
                restored_kernel = replace(
                    recipe.kernel_without_additive_response,
                    additive_scatter_response=response,
                )
                restored_kernels[identity] = restored_kernel
        prepared_isotopes.append(replace(prepared, kernel=restored_kernel))
    return replace(context, prepared_isotopes=tuple(prepared_isotopes))


def _initialize_spectral_process(context: _SpectralProcessContext) -> None:
    """Initialize one CPU worker without nested Torch oversubscription."""
    global _SPECTRAL_PROCESS_CONTEXT, _SPECTRAL_THREADPOOL_LIMITER
    _SPECTRAL_PROCESS_CONTEXT = _materialize_spectral_process_context(context)
    _SPECTRAL_THREADPOOL_LIMITER = threadpool_limits(limits=1)
    try:
        import torch

        torch.set_num_threads(1)
    except (ImportError, RuntimeError):
        pass


def _calculate_spectral_context_task(
    context: _SpectralProcessContext,
    task: tuple[tuple[int, ...], int, int],
) -> NDArray[np.float64]:
    """Calculate one exact measurement-batched spatial line-factor chunk."""
    measurement_indices, patch_start, patch_stop = task
    selected_measurements = np.asarray(measurement_indices, dtype=np.int64)
    selected_points = context.quadrature_points[patch_start:patch_stop]
    selected_weights = context.quadrature_weights[patch_start:patch_stop]
    selected_areas = context.areas[patch_start:patch_stop]
    quadrature_count = int(selected_points.shape[1])
    source_points = selected_points.reshape(-1, 3)
    factors = np.zeros(
        (
            selected_measurements.size,
            patch_stop - patch_start,
            context.line_count,
        ),
        dtype=np.float64,
    )
    for prepared in context.prepared_isotopes:
        if not isinstance(prepared.kernel, ContinuousKernel):
            raise RuntimeError("Spectral response kernel was not initialized.")
        raw = _joint_line_kernel_values(
            prepared.kernel,
            prepared.isotope,
            detector_positions=context.detector_positions[selected_measurements],
            sources=source_points,
            fe_indices=context.fe_indices[selected_measurements],
            pb_indices=context.pb_indices[selected_measurements],
            positive_line_indices=prepared.positive_line_indices,
            chunk_size=context.kernel_chunk_size,
        )
        expected_shape = (
            selected_measurements.size,
            (patch_stop - patch_start) * quadrature_count,
            prepared.positive_line_indices.size,
        )
        if raw.shape != expected_shape:
            raise ValueError(
                "Selected-pair line kernel returned "
                f"{raw.shape}, expected {expected_shape}."
            )
        values = raw.reshape(
            selected_measurements.size,
            patch_stop - patch_start,
            quadrature_count,
            prepared.positive_line_indices.size,
        )
        spatial = context.live_times[selected_measurements, None, None] * np.einsum(
            "mgql,gq->mgl",
            values,
            selected_weights,
            optimize=True,
        )
        line_stop = prepared.line_start + prepared.weights.size
        factors[:, :, prepared.line_start : line_stop] = (
            spatial * prepared.weights[None, None, :] * selected_areas[None, :, None]
        )
    return factors


def _calculate_spectral_process_task(
    task: tuple[tuple[int, ...], int, int],
) -> NDArray[np.float64]:
    """Calculate one measurement-batched response chunk in a CPU worker."""
    context = _SPECTRAL_PROCESS_CONTEXT
    if context is None:
        raise RuntimeError("Spectral response worker context is unavailable.")
    return _calculate_spectral_context_task(context, task)


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without accepting booleans or lossy casts."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _isotope_names(isotopes: Sequence[str]) -> tuple[str, ...]:
    """Return unique, non-empty isotope names in their requested order."""
    names = tuple(isotopes)
    if not names:
        raise ValueError("At least one isotope is required.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Isotopes must contain only non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Isotopes must not contain duplicates.")
    return names


def _orientation_indices(
    values: object,
    *,
    name: str,
    measurement_count: int,
    orientation_count: int,
) -> NDArray[np.int64]:
    """Return one in-range integer shield index per measurement."""
    raw = np.asarray(values)
    if raw.shape != (measurement_count,):
        raise ValueError(
            f"{name} must have shape ({measurement_count},), got {raw.shape}."
        )
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError(f"{name} must contain integer indices.")
    indices = np.asarray(raw, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= orientation_count):
        raise ValueError(f"{name} entries must lie in [0, {orientation_count - 1}].")
    return np.ascontiguousarray(indices)


def _validated_observation_geometry(
    observations: object,
    kernel: ContinuousKernel,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return validated geometry, timing, shield pairs, and spectrum edges."""
    detector_positions = np.asarray(
        getattr(observations, "detector_positions_xyz"),
        dtype=np.float64,
    )
    if (
        detector_positions.ndim != 2
        or detector_positions.shape[1:] != (3,)
        or detector_positions.shape[0] == 0
    ):
        raise ValueError("detector_positions_xyz must have non-empty shape (M, 3).")
    if not np.all(np.isfinite(detector_positions)):
        raise ValueError("detector_positions_xyz must contain only finite values.")
    detector_positions = np.ascontiguousarray(detector_positions)
    measurement_count = int(detector_positions.shape[0])

    orientations = np.asarray(kernel.orientations, dtype=float)
    if (
        orientations.ndim != 2
        or orientations.shape[1:] != (3,)
        or orientations.shape[0] == 0
        or not np.all(np.isfinite(orientations))
    ):
        raise ValueError("kernel.orientations must have non-empty finite shape (R, 3).")
    orientation_count = int(orientations.shape[0])
    fe_indices = _orientation_indices(
        getattr(observations, "fe_indices"),
        name="fe_indices",
        measurement_count=measurement_count,
        orientation_count=orientation_count,
    )
    pb_indices = _orientation_indices(
        getattr(observations, "pb_indices"),
        name="pb_indices",
        measurement_count=measurement_count,
        orientation_count=orientation_count,
    )

    live_times = np.asarray(
        getattr(observations, "live_times_s"),
        dtype=np.float64,
    )
    if live_times.shape != (measurement_count,):
        raise ValueError("live_times_s must contain one entry per measurement.")
    if not np.all(np.isfinite(live_times)) or np.any(live_times <= 0.0):
        raise ValueError("live_times_s must contain finite positive values.")
    live_times = np.ascontiguousarray(live_times)

    edges = np.asarray(
        getattr(observations, "energy_bin_edges_keV"),
        dtype=np.float64,
    )
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            "energy_bin_edges_keV must be a one-dimensional bin edge array."
        )
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy_bin_edges_keV must be finite and strictly increasing.")
    edges = np.ascontiguousarray(edges)

    if hasattr(observations, "spectrum_counts"):
        spectrum = np.asarray(getattr(observations, "spectrum_counts"))
        expected_shape = (measurement_count, edges.size - 1)
        if spectrum.shape != expected_shape:
            raise ValueError(
                f"spectrum_counts must have shape {expected_shape}, got {spectrum.shape}."
            )
    return detector_positions, fe_indices, pb_indices, live_times, edges


@dataclass(frozen=True)
class SpectralResponseResult:
    """Store a line-resolved response tensor and its construction diagnostics."""

    response_per_integrated_strength: NDArray[np.float64]
    response_per_density: NDArray[np.float64]
    nuisance_response: NDArray[np.float64]
    nuisance_names: tuple[str, ...]
    nuisance_l2_weights: NDArray[np.float64]
    overdispersion_alpha_by_bin: NDArray[np.float64]
    line_energies_keV_by_isotope: dict[str, tuple[float, ...]]
    line_weights_by_isotope: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class SpectralResponseOperatorResult:
    """Store a matrix-free density operator and compact nuisance responses."""

    operator: LineFactorizedResponseOperator
    nuisance_response: NDArray[np.float64]
    nuisance_names: tuple[str, ...]
    nuisance_l2_weights: NDArray[np.float64]
    overdispersion_alpha_by_bin: NDArray[np.float64]
    line_energies_keV_by_isotope: dict[str, tuple[float, ...]]
    line_weights_by_isotope: dict[str, tuple[float, ...]]
    cache_directory: Path | None


def _validated_patch_quadrature(
    patches: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return validated patch areas and fixed-width quadrature arrays."""
    if hasattr(patches, "quadrature_points_xyz") and hasattr(
        patches, "quadrature_weights"
    ):
        points = np.asarray(getattr(patches, "quadrature_points_xyz"), dtype=float)
        weights = np.asarray(getattr(patches, "quadrature_weights"), dtype=float)
    else:
        if not hasattr(patches, "patches"):
            raise TypeError(
                "patches must provide aggregate quadrature arrays or a patches sequence."
            )
        patch_items = tuple(getattr(patches, "patches"))
        if not patch_items:
            raise ValueError("patches must contain at least one surface patch.")
        rows: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for patch_index, patch in enumerate(patch_items):
            patch_points = np.asarray(patch.quadrature_points_xyz, dtype=float)
            patch_weights = np.asarray(
                patch.quadrature_weights,
                dtype=float,
            ).reshape(-1)
            if (
                patch_points.ndim != 2
                or patch_points.shape[1:] != (3,)
                or patch_points.shape[0] == 0
            ):
                raise ValueError(
                    "Patch "
                    f"{patch_index} quadrature_points_xyz must have shape (Q, 3), Q >= 1."
                )
            if patch_weights.shape != (patch_points.shape[0],):
                raise ValueError(
                    f"Patch {patch_index} quadrature weights must match its points."
                )
            rows.append((patch_points, patch_weights))
        maximum_count = max(int(row[1].size) for row in rows)
        points = np.empty((len(patch_items), maximum_count, 3), dtype=float)
        weights = np.zeros((len(patch_items), maximum_count), dtype=float)
        for patch_index, (patch_points, patch_weights) in enumerate(rows):
            count = int(patch_weights.size)
            points[patch_index, :count] = patch_points
            points[patch_index, count:] = patch_points[-1]
            weights[patch_index, :count] = patch_weights
    areas = np.asarray(getattr(patches, "areas_m2"), dtype=float).reshape(-1)
    if (
        points.ndim != 3
        or points.shape[2:] != (3,)
        or points.shape[0] == 0
        or points.shape[1] == 0
    ):
        raise ValueError("quadrature_points_xyz must have non-empty shape (G, Q, 3).")
    if weights.shape != points.shape[:2]:
        raise ValueError("quadrature_weights must have shape (G, Q).")
    if (
        areas.shape != (points.shape[0],)
        or not np.all(np.isfinite(areas))
        or np.any(areas <= 0.0)
    ):
        raise ValueError("areas_m2 must contain one finite positive value per patch.")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
        raise ValueError("Patch quadrature must contain only finite values.")
    if np.any(weights < 0.0) or not np.allclose(
        np.sum(weights, axis=1),
        1.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Each patch's quadrature weights must be non-negative and sum to one."
        )
    return (
        np.ascontiguousarray(areas, dtype=np.float64),
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
    )


def _line_entries(
    kernel: ContinuousKernel,
    isotope: str,
    *,
    require_line_resolved: bool,
) -> tuple[dict[str, float], ...]:
    """Return normalized line energy/weight/mu entries for one isotope."""
    table = kernel.line_mu_by_isotope
    raw: object | None = None
    if isinstance(table, Mapping):
        raw = table.get(isotope)
        if raw is None:
            normalized = {
                "".join(ch for ch in str(key).upper() if ch.isalnum()): value
                for key, value in table.items()
            }
            key = "".join(ch for ch in str(isotope).upper() if ch.isalnum())
            raw = normalized.get(key)
    entries: list[dict[str, float]] = []
    if isinstance(raw, (tuple, list)):
        for raw_index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            energy = float(item.get("energy_keV", np.nan))
            weight = float(item.get("weight", 0.0))
            mu_fe = float(item.get("fe", item.get("mu_fe", np.nan)))
            mu_pb = float(item.get("pb", item.get("mu_pb", np.nan)))
            if (
                np.isfinite(energy)
                and energy > 0.0
                and np.isfinite(weight)
                and weight > 0.0
                and np.isfinite(mu_fe)
                and mu_fe >= 0.0
                and np.isfinite(mu_pb)
                and mu_pb >= 0.0
            ):
                entries.append(
                    {
                        "energy_keV": energy,
                        "weight": weight,
                        "fe": mu_fe,
                        "pb": mu_pb,
                        "transport_line_index": float(raw_index),
                    }
                )
    if not entries:
        if require_line_resolved:
            raise ValueError(
                f"No line-resolved attenuation table is available for {isotope}."
            )
        library = default_library()
        nuclide = library.get(isotope)
        if nuclide is None or not nuclide.lines:
            raise ValueError(f"No gamma-line library entry is available for {isotope}.")
        mu_fe, mu_pb = kernel._mu_values(isotope)  # explicit diagnostic fallback
        entries = []
        for raw_index, line in enumerate(nuclide.lines):
            if float(line.intensity) <= 0.0:
                continue
            entries.append(
                {
                    "energy_keV": float(line.energy_keV),
                    "weight": max(float(line.intensity), 0.0),
                    "fe": float(mu_fe),
                    "pb": float(mu_pb),
                    "transport_line_index": float(raw_index),
                }
            )
    total_weight = float(sum(entry["weight"] for entry in entries))
    if total_weight <= 0.0:
        raise ValueError(f"Gamma-line weights for {isotope} sum to zero.")
    return tuple(
        {**entry, "weight": float(entry["weight"] / total_weight)} for entry in entries
    )


def _obstacle_grid_for_line(
    grid: ObstacleGrid | None,
    isotope: str,
    line_index: int,
    *,
    require_line_resolved: bool,
) -> ObstacleGrid | None:
    """Return a grid whose line table contains only the requested gamma line."""
    if grid is None or not grid.has_transport_model:
        return grid
    rows = grid.transport_line_mu_values(isotope)
    if rows is None:
        if require_line_resolved:
            raise ValueError(
                "No line-resolved obstacle attenuation table is available for "
                f"{isotope}; aggregate obstacle attenuation is not valid for "
                "spectral fitting."
            )
        return grid
    if not 0 <= int(line_index) < len(rows):
        raise ValueError(
            "Obstacle line attenuation table for "
            f"{isotope} has {len(rows)} rows but line index {line_index} was requested."
        )
    compton_rows = grid.transport_line_compton_mu_values(isotope)
    if compton_rows is not None and not 0 <= int(line_index) < len(compton_rows):
        raise ValueError(
            "Obstacle line Compton table for "
            f"{isotope} has {len(compton_rows)} rows but line index "
            f"{line_index} was requested."
        )
    mu_by_isotope = dict(grid.transport_mu_by_isotope)
    return grid.with_transport_model(
        boxes_m=grid.transport_boxes_m,
        mu_by_isotope=mu_by_isotope,
        line_mu_by_isotope={str(isotope): (rows[int(line_index)],)},
        line_compton_mu_by_isotope=(
            None
            if compton_rows is None
            else {str(isotope): (compton_rows[int(line_index)],)}
        ),
    )


def _kernel_for_line(
    kernel: ContinuousKernel,
    isotope: str,
    line: Mapping[str, float],
    line_index: int,
    *,
    require_line_resolved: bool,
) -> ContinuousKernel:
    """Clone the shared kernel with one line-specific shield and obstacle row."""
    grid = _obstacle_grid_for_line(
        kernel.obstacle_grid,
        isotope,
        line_index,
        require_line_resolved=require_line_resolved,
    )
    return replace(
        kernel,
        line_mu_by_isotope={
            str(isotope): (
                {
                    "energy_keV": float(line["energy_keV"]),
                    "weight": 1.0,
                    "fe": float(line["fe"]),
                    "pb": float(line["pb"]),
                },
            )
        },
        obstacle_grid=grid,
    )


def _positive_line_indices(
    kernel: ContinuousKernel,
    isotope: str,
    lines: Sequence[Mapping[str, float]],
    *,
    require_line_resolved: bool,
) -> NDArray[np.int64]:
    """Return the shared runtime's exact positive-line indices after validation."""
    for line_index, line in enumerate(lines):
        _kernel_for_line(
            kernel,
            isotope,
            line,
            int(line.get("transport_line_index", float(line_index))),
            require_line_resolved=require_line_resolved,
        )
    indices = np.asarray(
        kernel.positive_line_indices(isotope),
        dtype=np.int64,
    )
    if indices.shape != (len(lines),):
        raise ValueError(
            f"Shared runtime positive-line count for {isotope} does not match "
            "the spectral response table."
        )
    shared_weights = np.asarray(
        kernel.line_branching_weights(isotope, indices),
        dtype=np.float64,
    )
    expected_weights = np.asarray(
        [float(line["weight"]) for line in lines],
        dtype=np.float64,
    )
    if not np.allclose(
        shared_weights,
        expected_weights,
        rtol=1.0e-13,
        atol=1.0e-15,
    ):
        raise ValueError(
            f"Shared runtime branching weights for {isotope} do not match "
            "the spectral response table."
        )
    return np.ascontiguousarray(indices)


def _joint_line_kernel_values(
    kernel: ContinuousKernel,
    isotope: str,
    *,
    detector_positions: NDArray[np.float64],
    sources: NDArray[np.float64],
    fe_indices: NDArray[np.int64],
    pb_indices: NDArray[np.int64],
    positive_line_indices: NDArray[np.int64],
    chunk_size: int,
) -> NDArray[np.float64]:
    """Evaluate all requested lines while sharing geometry and obstacle rays."""
    values = np.asarray(
        kernel.kernel_values_selected_pairs_for_detectors_by_line(
            isotope=isotope,
            detector_positions=detector_positions,
            sources=sources,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            positive_line_indices=positive_line_indices,
            chunk_size=chunk_size,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            "Batched selected-pair line kernel must return finite non-negative values."
        )
    return values


def build_spectral_nuisance_response(
    live_times_s: NDArray[np.float64],
    energy_bin_edges_keV: NDArray[np.float64],
    *,
    include_background: bool = True,
    include_scatter: bool = True,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Build non-negative background-rate and scatter-rate nuisance columns."""
    live_times = np.asarray(live_times_s, dtype=float)
    if live_times.ndim != 1:
        raise ValueError("live_times_s must be one-dimensional.")
    if not np.all(np.isfinite(live_times)) or np.any(live_times < 0.0):
        raise ValueError("live_times_s must contain finite non-negative values.")
    edges = np.asarray(energy_bin_edges_keV, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            "energy_bin_edges_keV must be a one-dimensional bin edge array."
        )
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy_bin_edges_keV must be finite and strictly increasing.")
    centers = 0.5 * (edges[:-1] + edges[1:])
    columns: list[NDArray[np.float64]] = []
    names: list[str] = []
    if include_background:
        shape = default_background_shape(centers)
        shape = shape / max(float(np.sum(shape)), 1.0e-30)
        columns.append(live_times[:, None] * shape[None, :])
        names.append("background_rate_cps")
    if include_scatter:
        incident_energy = max(float(edges[-1]) * 0.85, 200.0)
        shape = compton_continuum_shape(centers, incident_energy, shape="exponential")
        shape = shape / max(float(np.sum(shape)), 1.0e-30)
        columns.append(live_times[:, None] * shape[None, :])
        names.append("scatter_rate_cps")
    if not columns:
        return np.zeros((live_times.size, centers.size, 0), dtype=float), ()
    return np.stack(columns, axis=-1), tuple(names)


def build_structured_spectral_nuisance_response(
    live_times_s: NDArray[np.float64],
    energy_bin_edges_keV: NDArray[np.float64],
    fe_indices: NDArray[np.int64],
    pb_indices: NDArray[np.int64],
    station_ids: NDArray[np.int64],
    calibration: DiscrepancyCalibration,
    *,
    include_background: bool = True,
    include_scatter: bool = True,
    include_shield_leakage: bool = True,
    include_station_rate: bool = True,
    include_low_rank_residual: bool = True,
    include_gain_resolution_drift: bool = False,
) -> tuple[
    NDArray[np.float64],
    tuple[str, ...],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build calibrated shared nuisance bases with family-specific shrinkage."""
    if not isinstance(calibration, DiscrepancyCalibration):
        raise TypeError("calibration must be a DiscrepancyCalibration.")
    live_times = np.asarray(live_times_s, dtype=np.float64).reshape(-1)
    edges = np.asarray(energy_bin_edges_keV, dtype=np.float64)
    calibration.validate_energy_axis(edges)
    measurement_count = live_times.size
    fe = np.asarray(fe_indices, dtype=np.int64)
    pb = np.asarray(pb_indices, dtype=np.int64)
    stations = np.asarray(station_ids, dtype=np.int64)
    if fe.shape != (measurement_count,) or pb.shape != (measurement_count,):
        raise ValueError("Shield indices must contain one row per measurement.")
    if stations.shape != (measurement_count,):
        raise ValueError("station_ids must contain one row per measurement.")
    pair_ids = 8 * fe + pb
    if np.any(pair_ids < 0) or np.any(pair_ids >= 64):
        raise ValueError("Shield pair IDs must lie in [0, 63].")
    bin_count = edges.size - 1
    columns: list[NDArray[np.float64]] = []
    names: list[str] = []
    weights: list[float] = []

    def add_global_basis(
        basis: NDArray[np.float64],
        family: str,
        prefix: str,
    ) -> None:
        """Add live-time-scaled run-global spectral basis columns."""
        for index, shape in enumerate(np.asarray(basis, dtype=np.float64)):
            columns.append(live_times[:, None] * shape[None, :])
            names.append(f"{prefix}:{index}")
            weights.append(float(calibration.shrinkage_l2_by_family[family]))

    if include_background:
        add_global_basis(calibration.background_basis, "background", "background")
    if include_scatter:
        add_global_basis(calibration.scatter_basis, "scatter", "scatter")
    if include_shield_leakage:
        features = calibration.shield_pair_feature_basis[pair_ids]
        for feature_index in range(features.shape[1]):
            for spectrum_index, shape in enumerate(calibration.shield_leakage_basis):
                columns.append(
                    live_times[:, None]
                    * features[:, feature_index, None]
                    * shape[None, :]
                )
                names.append(f"shield_leakage:f{feature_index}:s{spectrum_index}")
                weights.append(
                    float(calibration.shrinkage_l2_by_family["shield_leakage"])
                )
    if include_station_rate:
        base_shapes = np.vstack(
            [
                calibration.background_basis,
                calibration.scatter_basis,
                calibration.low_rank_spectral_residual_basis[:1],
            ]
        )
        station_shape = (
            np.mean(base_shapes, axis=0)
            if base_shapes.size
            else np.full(bin_count, 1.0 / bin_count)
        )
        station_shape = station_shape / max(float(np.sum(station_shape)), 1.0e-30)
        for station in np.unique(stations):
            indicator = stations == station
            columns.append(
                live_times[:, None] * indicator[:, None] * station_shape[None, :]
            )
            names.append(f"station_rate:{int(station)}")
            weights.append(float(calibration.shrinkage_l2_by_family["station_rate"]))
    if include_low_rank_residual:
        add_global_basis(
            calibration.low_rank_spectral_residual_basis,
            "low_rank_residual",
            "low_rank_residual",
        )
    if include_gain_resolution_drift:
        for family, prefix, basis in (
            ("gain_drift", "gain_drift", calibration.gain_derivative_basis),
            (
                "resolution_drift",
                "resolution_drift",
                calibration.resolution_derivative_basis,
            ),
        ):
            for index, derivative in enumerate(basis):
                positive = np.maximum(derivative, 0.0)
                negative = np.maximum(-derivative, 0.0)
                for sign, shape in (("positive", positive), ("negative", negative)):
                    if not np.any(shape):
                        continue
                    columns.append(live_times[:, None] * shape[None, :])
                    names.append(f"{prefix}:{index}:{sign}")
                    weights.append(float(calibration.shrinkage_l2_by_family[family]))
    if not columns:
        nuisance = np.zeros((measurement_count, bin_count, 0), dtype=np.float64)
    else:
        nuisance = np.stack(columns, axis=-1)
    return (
        nuisance,
        tuple(names),
        np.asarray(weights, dtype=np.float64),
        calibration.overdispersion_alpha_by_bin,
    )


def build_spectral_response(
    observations: object,
    patches: object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
    *,
    chunk_size: int = 262144,
    continuum_to_peak: float = COMPTON_CONTINUUM_TO_PEAK,
    backscatter_fraction: float = BACKSCATTER_FRACTION,
    require_line_resolved: bool = True,
    include_background_nuisance: bool = True,
    include_scatter_nuisance: bool = True,
    discrepancy_calibration: DiscrepancyCalibration | None = None,
    include_shield_leakage_nuisance: bool = True,
    include_station_rate_nuisance: bool = True,
    include_low_rank_residual_nuisance: bool = True,
    include_gain_resolution_drift: bool = False,
) -> SpectralResponseResult:
    """Build ``M x B x G x I`` line-resolved count response tensors."""
    kernel_chunk_size = _positive_integer(chunk_size, name="chunk_size")
    for name, value in {
        "continuum_to_peak": continuum_to_peak,
        "backscatter_fraction": backscatter_fraction,
    }.items():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    (
        detector_positions,
        fe_indices,
        pb_indices,
        live_times,
        edges,
    ) = _validated_observation_geometry(observations, kernel)
    measurement_count = int(detector_positions.shape[0])
    names = _isotope_names(isotopes)
    areas, quadrature_points, quadrature_weights = _validated_patch_quadrature(patches)
    patch_count, quadrature_count = quadrature_points.shape[:2]
    sources = quadrature_points.reshape(-1, 3)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(np.median(np.diff(edges)))
    response = np.zeros(
        (measurement_count, centers.size, patch_count, len(names)), dtype=float
    )
    energies_by_isotope: dict[str, tuple[float, ...]] = {}
    weights_by_isotope: dict[str, tuple[float, ...]] = {}
    resolution = default_resolution()

    for isotope_index, isotope in enumerate(names):
        lines = _line_entries(
            kernel,
            isotope,
            require_line_resolved=require_line_resolved,
        )
        positive_line_indices = _positive_line_indices(
            kernel,
            isotope,
            lines,
            require_line_resolved=require_line_resolved,
        )
        energies_by_isotope[isotope] = tuple(
            float(line["energy_keV"]) for line in lines
        )
        weights_by_isotope[isotope] = tuple(float(line["weight"]) for line in lines)
        raw_values = _joint_line_kernel_values(
            kernel,
            isotope,
            detector_positions=detector_positions,
            sources=sources,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            positive_line_indices=positive_line_indices,
            chunk_size=kernel_chunk_size,
        )
        expected_shape = (
            measurement_count,
            patch_count * quadrature_count,
            len(lines),
        )
        if raw_values.shape != expected_shape:
            raise ValueError(
                "Batched selected-pair line kernel returned shape "
                f"{raw_values.shape}, expected {expected_shape}."
            )
        values = raw_values.reshape(
            measurement_count,
            patch_count,
            quadrature_count,
            len(lines),
        )
        spatial = live_times[:, None, None] * np.einsum(
            "mgql,gq->mgl",
            values,
            quadrature_weights,
            optimize=True,
        )
        for line_index, line in enumerate(lines):
            pulse = detector_response_kernel_for_incident_gamma(
                centers,
                float(line["energy_keV"]),
                resolution,
                cebr3_efficiency,
                bin_width,
                continuum_to_peak=float(continuum_to_peak),
                backscatter_fraction=float(backscatter_fraction),
            )
            if pulse.shape != centers.shape:
                raise ValueError(
                    "Detector response returned an incompatible bin shape."
                )
            if not np.all(np.isfinite(pulse)) or np.any(pulse < 0.0):
                raise ValueError(
                    "Detector response must contain finite non-negative values."
                )
            response[:, :, :, isotope_index] += (
                float(line["weight"])
                * spatial[:, None, :, line_index]
                * pulse[None, :, None]
            )

    if discrepancy_calibration is None:
        nuisance, nuisance_names = build_spectral_nuisance_response(
            live_times,
            edges,
            include_background=include_background_nuisance,
            include_scatter=include_scatter_nuisance,
        )
        nuisance_l2_weights = np.zeros(len(nuisance_names), dtype=np.float64)
        overdispersion_alpha = np.zeros(centers.size, dtype=np.float64)
    else:
        nuisance, nuisance_names, nuisance_l2_weights, overdispersion_alpha = (
            build_structured_spectral_nuisance_response(
                live_times,
                edges,
                fe_indices,
                pb_indices,
                np.asarray(getattr(observations, "station_ids"), dtype=np.int64),
                discrepancy_calibration,
                include_background=include_background_nuisance,
                include_scatter=include_scatter_nuisance,
                include_shield_leakage=include_shield_leakage_nuisance,
                include_station_rate=include_station_rate_nuisance,
                include_low_rank_residual=include_low_rank_residual_nuisance,
                include_gain_resolution_drift=include_gain_resolution_drift,
            )
        )
    return SpectralResponseResult(
        response_per_integrated_strength=response,
        response_per_density=response * areas[None, None, :, None],
        nuisance_response=nuisance,
        nuisance_names=nuisance_names,
        nuisance_l2_weights=nuisance_l2_weights,
        overdispersion_alpha_by_bin=overdispersion_alpha,
        line_energies_keV_by_isotope=energies_by_isotope,
        line_weights_by_isotope=weights_by_isotope,
    )


def _hash_array(digest: object, values: NDArray[np.generic]) -> None:
    """Update a SHA-256 digest with one canonical array description."""
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _hash_framed_bytes(digest: object, values: bytes) -> None:
    """Hash one length-delimited byte sequence without concatenation ambiguity."""
    digest.update(len(values).to_bytes(8, byteorder="big", signed=False))
    digest.update(values)


def _hash_canonical_value(digest: object, value: object) -> None:
    """Hash nested physical configuration by value without address-based reprs."""
    type_name = f"{type(value).__module__}.{type(value).__qualname__}".encode()
    _hash_framed_bytes(digest, type_name)
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        digest.update(b"1" if bool(value) else b"0")
        return
    if isinstance(value, (int, np.integer)):
        _hash_framed_bytes(digest, str(int(value)).encode("ascii"))
        return
    if isinstance(value, (float, np.floating)):
        _hash_framed_bytes(digest, float(value).hex().encode("ascii"))
        return
    if isinstance(value, str):
        _hash_framed_bytes(digest, value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _hash_framed_bytes(digest, value)
        return
    if isinstance(value, np.ndarray):
        _hash_array(digest, value)
        return
    if isinstance(value, Mapping):
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        keyed_digests: list[tuple[bytes, object]] = []
        for key in value:
            key_digest = sha256()
            _hash_canonical_value(key_digest, key)
            keyed_digests.append((key_digest.digest(), key))
        for _, key in sorted(keyed_digests, key=lambda item: item[0]):
            _hash_canonical_value(digest, key)
            _hash_canonical_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
        for item in value:
            _hash_canonical_value(digest, item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        selected_fields = tuple(field for field in fields(value) if field.init)
        digest.update(len(selected_fields).to_bytes(8, byteorder="big", signed=False))
        for field in selected_fields:
            _hash_canonical_value(digest, field.name)
            _hash_canonical_value(digest, getattr(value, field.name))
        return
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        public_attributes = {
            str(name): item
            for name, item in attributes.items()
            if not str(name).startswith("_")
        }
        _hash_canonical_value(digest, public_attributes)
        return
    raise TypeError(
        "Physical response cache configuration contains an unsupported "
        f"value of type {type(value).__qualname__}."
    )


def _spectral_spatial_cache_key(
    *,
    kernel: ContinuousKernel,
    areas: NDArray[np.float64],
    quadrature_points: NDArray[np.float64],
    quadrature_weights: NDArray[np.float64],
    isotope_lines: Mapping[str, Sequence[Mapping[str, float]]],
) -> str:
    """Return a stable physical-model and patch line-factor cache key."""
    digest = sha256()
    digest.update(b"spectral-response-spatial-line-factor-v6\0")
    try:
        runtime_distribution = distribution("rotating-shield-simulation-runtime")
        runtime_identity = (
            runtime_distribution.version,
            runtime_distribution.read_text("direct_url.json"),
        )
    except PackageNotFoundError:
        runtime_identity = ("uninstalled", None)
    _hash_canonical_value(digest, runtime_identity)
    # ContinuousKernel is owned by the shared runtime. Hash its configured
    # constructor fields while excluding only the CUDA device ordinal and every
    # mutable cache/counter. Backend and dtype remain part of the key because
    # float32 GPU factors must never poison a float64 CPU cache.
    for field in fields(kernel):
        if field.init and field.name != "gpu_device":
            _hash_canonical_value(digest, field.name)
            _hash_canonical_value(digest, getattr(kernel, field.name))
    # Spatial-factor line columns follow the requested isotope order. Preserve
    # that order in the namespace instead of canonicalizing it as a mapping.
    ordered_isotope_lines = tuple(
        (isotope, tuple(lines)) for isotope, lines in isotope_lines.items()
    )
    _hash_canonical_value(digest, ordered_isotope_lines)
    for array in (areas, quadrature_points, quadrature_weights):
        _hash_array(digest, array)
    return digest.hexdigest()


def _spectral_cache_root(
    cache_directory: str | Path | None,
    spatial_cache_key: str,
) -> Path | None:
    """Return the optional disk-cache root for one spatial-factor key."""
    if cache_directory is None:
        return None
    root = Path(cache_directory).expanduser().resolve() / spatial_cache_key
    root.mkdir(parents=True, exist_ok=True)
    return root


def _measurement_cache_key(
    detector_position: NDArray[np.float64],
    fe_index: int,
    pb_index: int,
    live_time_s: float,
) -> str:
    """Return a stable append-only key for one acquired response row."""
    digest = sha256()
    _hash_array(digest, np.asarray(detector_position, dtype=np.float64))
    digest.update(
        json.dumps(
            {
                "fe": int(fe_index),
                "pb": int(pb_index),
                "live_time_s": float(live_time_s),
            },
            allow_nan=False,
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def build_spectral_response_operator(
    observations: object,
    patches: object,
    isotopes: Sequence[str],
    kernel: ContinuousKernel,
    *,
    chunk_size: int = 262144,
    measurement_chunk_size: int = 8,
    energy_chunk_size: int = 128,
    patch_chunk_size: int = 128,
    worker_count: int = 0,
    cache_directory: str | Path | None = None,
    continuum_to_peak: float = COMPTON_CONTINUUM_TO_PEAK,
    backscatter_fraction: float = BACKSCATTER_FRACTION,
    require_line_resolved: bool = True,
    include_background_nuisance: bool = True,
    include_scatter_nuisance: bool = True,
    discrepancy_calibration: DiscrepancyCalibration | None = None,
    include_shield_leakage_nuisance: bool = True,
    include_station_rate_nuisance: bool = True,
    include_low_rank_residual_nuisance: bool = True,
    include_gain_resolution_drift: bool = False,
) -> SpectralResponseOperatorResult:
    """Build an exact disk-cacheable line-factorized spectral operator.

    The disk cache stores one compact patch-by-line factor matrix for each
    acquired measurement/patch pair. Cache keys are per acquired row, so
    extending a causal observation prefix writes only newly appended rows.
    Energy-bin expansion is deferred to bounded products or diagnostics.
    """
    kernel_chunk_size = _positive_integer(chunk_size, name="chunk_size")
    measurement_step = _positive_integer(
        measurement_chunk_size,
        name="measurement_chunk_size",
    )
    energy_step = _positive_integer(energy_chunk_size, name="energy_chunk_size")
    patch_step = _positive_integer(patch_chunk_size, name="patch_chunk_size")
    if isinstance(worker_count, (bool, np.bool_)) or not isinstance(
        worker_count,
        (int, np.integer),
    ):
        raise TypeError("worker_count must be an integer.")
    if int(worker_count) < 0:
        raise ValueError("worker_count must be nonnegative.")
    resolved_workers = 1 if int(worker_count) == 0 else int(worker_count)
    (
        detector_positions,
        fe_indices,
        pb_indices,
        live_times,
        edges,
    ) = _validated_observation_geometry(observations, kernel)
    names = _isotope_names(isotopes)
    areas, quadrature_points, quadrature_weights = _validated_patch_quadrature(patches)
    measurement_count = int(detector_positions.shape[0])
    patch_count, quadrature_count = quadrature_points.shape[:2]
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(np.median(np.diff(edges)))
    resolution = default_resolution()
    lines_by_isotope = {
        isotope: _line_entries(
            kernel,
            isotope,
            require_line_resolved=require_line_resolved,
        )
        for isotope in names
    }
    prepared_isotopes: list[_PreparedSpectralIsotope] = []
    line_count = 0
    for isotope_index, isotope in enumerate(names):
        lines = lines_by_isotope[isotope]
        positive_line_indices = _positive_line_indices(
            kernel,
            isotope,
            lines,
            require_line_resolved=require_line_resolved,
        )
        pulses = np.vstack(
            [
                detector_response_kernel_for_incident_gamma(
                    centers,
                    float(line["energy_keV"]),
                    resolution,
                    cebr3_efficiency,
                    bin_width,
                    continuum_to_peak=float(continuum_to_peak),
                    backscatter_fraction=float(backscatter_fraction),
                )
                for line in lines
            ]
        )
        prepared_isotopes.append(
            _PreparedSpectralIsotope(
                isotope_index=isotope_index,
                isotope=isotope,
                line_start=line_count,
                weights=np.asarray(
                    [float(line["weight"]) for line in lines],
                    dtype=np.float64,
                ),
                positive_line_indices=positive_line_indices,
                kernel=kernel,
                pulses=np.ascontiguousarray(pulses, dtype=np.float64),
            )
        )
        line_count += len(lines)
    work_items = (
        measurement_count
        * patch_count
        * quadrature_count
        * sum(len(lines) for lines in lines_by_isotope.values())
    )
    if bool(kernel.use_gpu):
        # Concurrent launches through one shared runtime kernel increase device
        # memory pressure and do not improve the already-batched CUDA path.
        resolved_workers = 1
    energies_by_isotope = {
        isotope: tuple(float(line["energy_keV"]) for line in lines)
        for isotope, lines in lines_by_isotope.items()
    }
    weights_by_isotope = {
        isotope: tuple(float(line["weight"]) for line in lines)
        for isotope, lines in lines_by_isotope.items()
    }
    spatial_cache_key = _spectral_spatial_cache_key(
        kernel=kernel,
        areas=areas,
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        isotope_lines=lines_by_isotope,
    )
    cache_root = _spectral_cache_root(cache_directory, spatial_cache_key)
    measurement_row_keys = tuple(
        _measurement_cache_key(
            detector_positions[index],
            int(fe_indices[index]),
            int(pb_indices[index]),
            float(live_times[index]),
        )
        for index in range(measurement_count)
    )
    cache_stats = {"hits": 0, "misses": 0, "files": 0, "blocks": 0}
    performance: dict[str, object] = {
        "response_construction": {
            "worker_count": resolved_workers,
            "requested_worker_count": int(worker_count),
            "measurement_chunk_size": measurement_step,
            "estimated_kernel_work_items": work_items,
            "missing_kernel_work_items": 0,
            "iterations": 0,
            "kernel_batch_calls": 0,
            "kernel_batched_measurements": 0,
            "elapsed_seconds": 0.0,
        }
    }

    def measurement_cache_path(measurement_index: int) -> Path | None:
        """Return one immutable full-patch line-factor path per unique row."""
        if cache_root is None:
            return None
        return cache_root / measurement_row_keys[measurement_index] / "factors.npy"

    def calculate_patch_batch(
        measurement_indices: tuple[int, ...],
        patch_start: int,
        patch_stop: int,
    ) -> NDArray[np.float64]:
        """Calculate one bounded measurement and patch line-factor batch."""
        return _calculate_spectral_context_task(
            process_context,
            (measurement_indices, patch_start, patch_stop),
        )

    process_context = _SpectralProcessContext(
        detector_positions=detector_positions,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_times=live_times,
        areas=areas,
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        line_count=line_count,
        prepared_isotopes=tuple(prepared_isotopes),
        kernel_chunk_size=kernel_chunk_size,
    )

    spatial_factors = np.empty(
        (measurement_count, patch_count, line_count),
        dtype=np.float64,
    )

    started = perf_counter()
    construction = performance["response_construction"]
    try:
        row_members: dict[str, list[int]] = {}
        for measurement_index, row_key in enumerate(measurement_row_keys):
            row_members.setdefault(row_key, []).append(measurement_index)
        representative_indices = tuple(members[0] for members in row_members.values())
        cached_representatives = tuple(
            measurement_index
            for measurement_index in representative_indices
            if (
                measurement_cache_path(measurement_index) is not None
                and measurement_cache_path(measurement_index).exists()
            )
        )
        cached_set = set(cached_representatives)
        missing_representatives = tuple(
            measurement_index
            for measurement_index in representative_indices
            if measurement_index not in cached_set
        )
        expected_cache_shape = (patch_count, line_count)
        for measurement_index in cached_representatives:
            path = measurement_cache_path(measurement_index)
            assert path is not None
            cached_values = np.load(path, allow_pickle=False, mmap_mode="r")
            if cached_values.shape != expected_cache_shape:
                raise ValueError(
                    "Cached spectral line factors have shape "
                    f"{cached_values.shape}, expected {expected_cache_shape}."
                )
            for member_index in row_members[measurement_row_keys[measurement_index]]:
                spatial_factors[member_index] = cached_values
        calculation_groups: list[tuple[tuple[int, ...], int, int]] = []
        for patch_start in range(0, patch_count, patch_step):
            patch_stop = min(patch_start + patch_step, patch_count)
            calculation_groups.extend(
                (
                    missing_representatives[start : start + measurement_step],
                    patch_start,
                    patch_stop,
                )
                for start in range(0, len(missing_representatives), measurement_step)
            )
        missing_work_items = (
            len(missing_representatives)
            * patch_count
            * quadrature_count
            * sum(len(lines) for lines in lines_by_isotope.values())
        )
        if int(worker_count) == 0:
            resolved_workers = (
                min(4, max(1, _available_cpu_count() // 2))
                if not bool(kernel.use_gpu) and missing_work_items >= 10_000_000
                else 1
            )
        resolved_workers = min(resolved_workers, _available_cpu_count())
        resolved_workers = min(
            resolved_workers,
            max(1, len(calculation_groups)),
        )
        if isinstance(construction, dict):
            construction["worker_count"] = resolved_workers
            construction["missing_kernel_work_items"] = missing_work_items
            construction["kernel_batch_calls"] = len(calculation_groups)
            construction["kernel_batched_measurements"] = sum(
                len(group[0]) for group in calculation_groups
            )

        def consume_group(
            group: tuple[tuple[int, ...], int, int],
            calculated: NDArray[np.float64],
        ) -> None:
            """Publish one measurement batch into ordered resident factors."""
            measurement_indices, patch_start, patch_stop = group
            expected_shape = (
                len(measurement_indices),
                patch_stop - patch_start,
                line_count,
            )
            if calculated.shape != expected_shape:
                raise ValueError(
                    "Calculated spectral line-factor batch has shape "
                    f"{calculated.shape}, expected {expected_shape}."
                )
            spatial_factors[
                np.asarray(measurement_indices, dtype=np.int64),
                patch_start:patch_stop,
            ] = calculated

        if resolved_workers == 1 or len(calculation_groups) <= 1:
            for group in calculation_groups:
                consume_group(group, calculate_patch_batch(*group))
        else:
            worker_context = _fresh_spectral_process_context(process_context)
            with ProcessPoolExecutor(
                max_workers=resolved_workers,
                initializer=_initialize_spectral_process,
                initargs=(worker_context,),
                mp_context=get_context("spawn"),
            ) as executor:
                remaining = iter(calculation_groups)
                pending: deque[
                    tuple[
                        tuple[tuple[int, ...], int, int],
                        Future[NDArray[np.float64]],
                    ]
                ] = deque()
                for _ in range(min(len(calculation_groups), 2 * resolved_workers)):
                    group = next(remaining, None)
                    if group is None:
                        break
                    pending.append(
                        (
                            group,
                            executor.submit(
                                _calculate_spectral_process_task,
                                group,
                            ),
                        )
                    )
                while pending:
                    group, future = pending.popleft()
                    consume_group(group, future.result())
                    next_group = next(remaining, None)
                    if next_group is not None:
                        pending.append(
                            (
                                next_group,
                                executor.submit(
                                    _calculate_spectral_process_task,
                                    next_group,
                                ),
                            )
                        )
        for measurement_index in missing_representatives:
            row_key = measurement_row_keys[measurement_index]
            for member_index in row_members[row_key][1:]:
                spatial_factors[member_index] = spatial_factors[measurement_index]
            path = measurement_cache_path(measurement_index)
            if path is not None:
                atomic_save_npy(path, spatial_factors[measurement_index])
        unique_row_count = len(representative_indices)
        cache_stats.update(
            {
                "hits": len(cached_representatives),
                "misses": len(missing_representatives),
                "files": unique_row_count if cache_root is not None else 0,
                "blocks": (
                    unique_row_count
                    * ((centers.size + energy_step - 1) // energy_step)
                    * ((patch_count + patch_step - 1) // patch_step)
                ),
            }
        )
    finally:
        if isinstance(construction, dict):
            construction["iterations"] = 1
            construction["elapsed_seconds"] = perf_counter() - started

    pulse_shapes = np.concatenate(
        tuple(prepared.pulses for prepared in prepared_isotopes),
        axis=0,
    )
    line_isotope_indices = np.concatenate(
        tuple(
            np.full(prepared.weights.size, prepared.isotope_index, dtype=np.int64)
            for prepared in prepared_isotopes
        )
    )
    device_digest = sha256()
    _hash_canonical_value(device_digest, spatial_cache_key)
    _hash_array(device_digest, edges)
    _hash_canonical_value(
        device_digest,
        (float(continuum_to_peak), float(backscatter_fraction)),
    )
    device_cache_key = device_digest.hexdigest()
    operator = LineFactorizedResponseOperator(
        spatial_factors,
        pulse_shapes,
        line_isotope_indices,
        len(names),
        energy_chunk_size=energy_step,
        patch_chunk_size=patch_step,
        _copy_factors=False,
        diagnostics={
            "response_mode": "line_factorized",
            "energy_chunk_size": energy_step,
            "patch_chunk_size": patch_step,
            "cache_enabled": cache_root is not None,
            "cache_file_layout": "measurement_full_patch_line_factors_v4",
            "cache_stats": cache_stats,
            "device_cache_key": device_cache_key,
            "measurement_row_keys": list(measurement_row_keys),
            "line_count": line_count,
            "factor_response_bytes": int(spatial_factors.nbytes + pulse_shapes.nbytes),
            "dense_response_bytes": int(
                measurement_count
                * centers.size
                * patch_count
                * len(names)
                * np.dtype(np.float64).itemsize
            ),
            "performance": performance,
        },
    )
    if discrepancy_calibration is None:
        nuisance, nuisance_names = build_spectral_nuisance_response(
            live_times,
            edges,
            include_background=include_background_nuisance,
            include_scatter=include_scatter_nuisance,
        )
        nuisance_l2_weights = np.zeros(len(nuisance_names), dtype=np.float64)
        overdispersion_alpha = np.zeros(centers.size, dtype=np.float64)
    else:
        nuisance, nuisance_names, nuisance_l2_weights, overdispersion_alpha = (
            build_structured_spectral_nuisance_response(
                live_times,
                edges,
                fe_indices,
                pb_indices,
                np.asarray(getattr(observations, "station_ids"), dtype=np.int64),
                discrepancy_calibration,
                include_background=include_background_nuisance,
                include_scatter=include_scatter_nuisance,
                include_shield_leakage=include_shield_leakage_nuisance,
                include_station_rate=include_station_rate_nuisance,
                include_low_rank_residual=include_low_rank_residual_nuisance,
                include_gain_resolution_drift=include_gain_resolution_drift,
            )
        )
    return SpectralResponseOperatorResult(
        operator=operator,
        nuisance_response=nuisance,
        nuisance_names=nuisance_names,
        nuisance_l2_weights=nuisance_l2_weights,
        overdispersion_alpha_by_bin=overdispersion_alpha,
        line_energies_keV_by_isotope=energies_by_isotope,
        line_weights_by_isotope=weights_by_isotope,
        cache_directory=cache_root,
    )


__all__ = [
    "SpectralResponseOperatorResult",
    "SpectralResponseResult",
    "build_spectral_nuisance_response",
    "build_spectral_response",
    "build_spectral_response_operator",
    "build_structured_spectral_nuisance_response",
]
