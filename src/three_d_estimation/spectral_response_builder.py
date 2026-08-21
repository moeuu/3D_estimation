"""Line-resolved spectral response tensors built from the local shared kernel."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.obstacles import ObstacleGrid
from runtime.discrepancy_calibration import DiscrepancyCalibration
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
    BlockResponseOperator,
    ResponseBlock,
    atomic_save_npy,
)


@dataclass(frozen=True, slots=True)
class _PreparedSpectralIsotope:
    """Store one isotope's jointly evaluated transport lines and pulses."""

    isotope_index: int
    isotope: str
    weights: NDArray[np.float64]
    positive_line_indices: NDArray[np.int64]
    kernel: ContinuousKernel
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
    isotope_count: int
    prepared_isotopes: tuple[_PreparedSpectralIsotope, ...]
    kernel_chunk_size: int


_SPECTRAL_PROCESS_CONTEXT: _SpectralProcessContext | None = None


def _initialize_spectral_process(context: _SpectralProcessContext) -> None:
    """Initialize one CPU worker without nested Torch oversubscription."""
    global _SPECTRAL_PROCESS_CONTEXT
    _SPECTRAL_PROCESS_CONTEXT = context
    try:
        import torch

        torch.set_num_threads(1)
    except (ImportError, RuntimeError):
        pass


def _calculate_spectral_context_task(
    context: _SpectralProcessContext,
    task: tuple[tuple[int, ...], int, int],
) -> NDArray[np.float64]:
    """Calculate one exact measurement-batched response chunk."""
    measurement_indices, patch_start, patch_stop = task
    selected_measurements = np.asarray(measurement_indices, dtype=np.int64)
    selected_points = context.quadrature_points[patch_start:patch_stop]
    selected_weights = context.quadrature_weights[patch_start:patch_stop]
    selected_areas = context.areas[patch_start:patch_stop]
    quadrature_count = int(selected_points.shape[1])
    source_points = selected_points.reshape(-1, 3)
    response = np.zeros(
        (
            selected_measurements.size,
            context.prepared_isotopes[0].pulses.shape[1],
            patch_stop - patch_start,
            context.isotope_count,
        ),
        dtype=np.float64,
    )
    for prepared in context.prepared_isotopes:
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
        for line_index, weight in enumerate(prepared.weights):
            response[:, :, :, prepared.isotope_index] += (
                float(weight)
                * prepared.pulses[line_index][None, :, None]
                * spatial[:, None, :, line_index]
                * selected_areas[None, None, :]
            )
    return response


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

    operator: BlockResponseOperator
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
            "Batched selected-pair line kernel must return finite non-negative "
            "values."
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


def _spectral_cache_root(
    cache_directory: str | Path | None,
    *,
    kernel: ContinuousKernel,
    areas: NDArray[np.float64],
    quadrature_points: NDArray[np.float64],
    quadrature_weights: NDArray[np.float64],
    edges: NDArray[np.float64],
    isotope_lines: Mapping[str, Sequence[Mapping[str, float]]],
    continuum_to_peak: float,
    backscatter_fraction: float,
) -> Path | None:
    """Return a physical-model and patch-specific disk-cache namespace."""
    if cache_directory is None:
        return None
    digest = sha256()
    digest.update(b"spectral-response-block-v1\0")
    # ContinuousKernel is owned by the shared runtime.  Hash its configured
    # constructor fields while excluding execution-device choices and every
    # mutable cache/counter.  This keeps a causal prefix in one cache namespace.
    physical_kernel_fields = {
        field.name: repr(getattr(kernel, field.name))
        for field in fields(kernel)
        if field.init and field.name not in {"use_gpu", "gpu_device", "gpu_dtype"}
    }
    digest.update(json.dumps(physical_kernel_fields, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(dict(isotope_lines), sort_keys=True).encode("utf-8"))
    digest.update(
        repr((float(continuum_to_peak), float(backscatter_fraction))).encode()
    )
    for array in (areas, quadrature_points, quadrature_weights, edges):
        _hash_array(digest, array)
    root = Path(cache_directory).expanduser().resolve() / digest.hexdigest()
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
    """Build a disk-cacheable streaming ``A @ q`` and ``A.T @ r`` operator.

    Emitted blocks are one measurement by an energy-bin chunk by a patch chunk.
    The disk cache stores the full energy axis for each measurement/patch pair
    in one contiguous file.  Cache keys are per acquired row, so extending a
    causal observation prefix writes only files belonging to newly appended
    measurements without creating one filesystem entry per energy chunk.
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
                weights=np.asarray(
                    [float(line["weight"]) for line in lines],
                    dtype=np.float64,
                ),
                positive_line_indices=positive_line_indices,
                kernel=kernel,
                pulses=np.ascontiguousarray(pulses, dtype=np.float64),
            )
        )
    work_items = (
        measurement_count
        * patch_count
        * quadrature_count
        * sum(len(lines) for lines in lines_by_isotope.values())
    )
    if int(worker_count) == 0 and work_items >= 250_000:
        resolved_workers = min(4, max(1, (os.cpu_count() or 1) // 2))
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
    cache_root = _spectral_cache_root(
        cache_directory,
        kernel=kernel,
        areas=areas,
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        edges=edges,
        isotope_lines=lines_by_isotope,
        continuum_to_peak=continuum_to_peak,
        backscatter_fraction=backscatter_fraction,
    )
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
            "measurement_chunk_size": measurement_step,
            "estimated_kernel_work_items": work_items,
            "iterations": 0,
            "kernel_batch_calls": 0,
            "kernel_batched_measurements": 0,
            "elapsed_seconds": 0.0,
        }
    }

    def task_cache_path(
        measurement_index: int,
        patch_start: int,
        patch_stop: int,
    ) -> Path | None:
        """Return one immutable full-energy cache path for a patch task."""
        if cache_root is None:
            return None
        return (
            cache_root
            / measurement_row_keys[measurement_index]
            / f"g{patch_start}-{patch_stop}_all.npy"
        )

    def calculate_patch_batch(
        measurement_indices: tuple[int, ...],
        patch_start: int,
        patch_stop: int,
    ) -> NDArray[np.float64]:
        """Calculate one bounded measurement and patch response batch."""
        return _calculate_spectral_context_task(
            process_context,
            (measurement_indices, patch_start, patch_stop),
        )

    tasks = tuple(
        (measurement_index, patch_start, min(patch_start + patch_step, patch_count))
        for measurement_index in range(measurement_count)
        for patch_start in range(0, patch_count, patch_step)
    )
    process_context = _SpectralProcessContext(
        detector_positions=detector_positions,
        fe_indices=fe_indices,
        pb_indices=pb_indices,
        live_times=live_times,
        areas=areas,
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        isotope_count=len(names),
        prepared_isotopes=tuple(prepared_isotopes),
        kernel_chunk_size=kernel_chunk_size,
    )

    def task_requires_calculation(task: tuple[int, int, int]) -> bool:
        """Return whether the contiguous cache file for a task is absent."""
        measurement_index, patch_start, patch_stop = task
        path = task_cache_path(
            measurement_index,
            patch_start,
            patch_stop,
        )
        return path is None or not path.exists()

    def build_task(
        task: tuple[int, int, int],
        calculated_response: NDArray[np.float64] | None = None,
    ) -> tuple[tuple[ResponseBlock, ...], int, int]:
        """Build or load one patch chunk and return deterministic energy blocks."""
        measurement_index, patch_start, patch_stop = task
        energy_ranges = tuple(
            (start, min(start + energy_step, centers.size))
            for start in range(0, centers.size, energy_step)
        )
        path = task_cache_path(
            measurement_index,
            patch_start,
            patch_stop,
        )
        missing = path is None or not path.exists()
        calculated = calculated_response
        if missing and calculated is None:
            calculated = calculate_patch_batch(
                (measurement_index,),
                patch_start,
                patch_stop,
            )[0]
        if missing:
            assert calculated is not None
            cached_values = calculated.reshape(centers.size, -1)
            if path is not None:
                atomic_save_npy(path, cached_values)
            hits = 0
            misses = 1
        else:
            assert path is not None
            cached_values = np.load(path, allow_pickle=False, mmap_mode="r")
            expected_cache_shape = (
                centers.size,
                (patch_stop - patch_start) * len(names),
            )
            if cached_values.shape != expected_cache_shape:
                raise ValueError(
                    "Cached spectral response has shape "
                    f"{cached_values.shape}, expected {expected_cache_shape}."
                )
            hits = 1
            misses = 0
        source_indices = np.arange(
            patch_start * len(names),
            patch_stop * len(names),
            dtype=np.int64,
        )
        blocks: list[ResponseBlock] = []
        for energy_start, energy_stop in energy_ranges:
            values = cached_values[energy_start:energy_stop]
            observation_indices = measurement_index * centers.size + np.arange(
                energy_start,
                energy_stop,
                dtype=np.int64,
            )
            blocks.append(
                ResponseBlock(
                    observation_indices=observation_indices,
                    source_indices=source_indices,
                    values=np.asarray(values, dtype=np.float64),
                )
            )
        return tuple(blocks), hits, misses

    def factory() -> object:
        """Yield bounded blocks with deterministic bounded CPU parallelism."""
        started = perf_counter()
        construction = performance["response_construction"]

        def emit(result: tuple[tuple[ResponseBlock, ...], int, int]) -> object:
            """Record cache counters and yield one task's ordered blocks."""
            blocks, hits, misses = result
            cache_stats["hits"] += hits
            cache_stats["misses"] += misses
            cache_stats["files"] += hits + misses
            cache_stats["blocks"] += len(blocks)
            yield from blocks

        try:
            task_states = tuple(
                (task, task_requires_calculation(task)) for task in tasks
            )
            missing_tasks = tuple(task for task, missing in task_states if missing)
            cached_tasks = tuple(task for task, missing in task_states if not missing)
            calculation_groups: list[tuple[tuple[int, ...], int, int]] = []
            for patch_start in range(0, patch_count, patch_step):
                patch_stop = min(patch_start + patch_step, patch_count)
                missing_measurements = tuple(
                    measurement_index
                    for measurement_index, task_patch_start, _ in missing_tasks
                    if task_patch_start == patch_start
                )
                calculation_groups.extend(
                    (
                        missing_measurements[start : start + measurement_step],
                        patch_start,
                        patch_stop,
                    )
                    for start in range(0, len(missing_measurements), measurement_step)
                )
            if isinstance(construction, dict):
                construction["kernel_batch_calls"] = int(
                    construction["kernel_batch_calls"]
                ) + len(calculation_groups)
                construction["kernel_batched_measurements"] = int(
                    construction["kernel_batched_measurements"]
                ) + sum(len(group[0]) for group in calculation_groups)

            def emit_group(
                group: tuple[tuple[int, ...], int, int],
                calculated: NDArray[np.float64],
            ) -> object:
                """Yield one measurement batch as immutable row blocks."""
                measurement_indices, patch_start, patch_stop = group
                for local_index, measurement_index in enumerate(measurement_indices):
                    yield from emit(
                        build_task(
                            (measurement_index, patch_start, patch_stop),
                            calculated[local_index],
                        )
                    )

            if resolved_workers == 1 or len(calculation_groups) <= 1:
                for group in calculation_groups:
                    calculated = calculate_patch_batch(*group)
                    yield from emit_group(group, calculated)
                for task in cached_tasks:
                    yield from emit(build_task(task))
                return
            with ProcessPoolExecutor(
                max_workers=resolved_workers,
                initializer=_initialize_spectral_process,
                initargs=(process_context,),
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
                            executor.submit(_calculate_spectral_process_task, group),
                        )
                    )
                while pending:
                    group, future = pending.popleft()
                    yield from emit_group(group, future.result())
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
                for task in cached_tasks:
                    yield from emit(build_task(task))
        finally:
            if isinstance(construction, dict):
                construction["iterations"] = int(construction["iterations"]) + 1
                construction["elapsed_seconds"] = float(
                    construction["elapsed_seconds"]
                ) + (perf_counter() - started)

    operator = BlockResponseOperator(
        (measurement_count, centers.size),
        patch_count,
        len(names),
        factory,
        diagnostics={
            "response_mode": "matrix_free",
            "energy_chunk_size": energy_step,
            "patch_chunk_size": patch_step,
            "cache_enabled": cache_root is not None,
            "cache_file_layout": "measurement_patch_full_energy_v2",
            "cache_stats": cache_stats,
            "device_cache_key": (None if cache_root is None else cache_root.as_posix()),
            "measurement_row_keys": list(measurement_row_keys),
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
