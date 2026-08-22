"""Deterministic, self-contained persistence for :class:`MLEEstimate`.

Reports deliberately avoid pickle.  Dense numerical state and variable-length
patch data are stored in a standard NPZ archive, while diagnostics and the
resolved MLE configuration are strict JSON.  NPZ members are written in sorted
order with a fixed ZIP timestamp so identical estimates produce identical
bytes across repeated writes on the same NumPy/Python implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
from numpy.typing import NDArray
from runtime import AtomicBundlePublisher

from .config import MLEConfig
from .types import MLEEstimate, SurfacePatch


REPORT_SCHEMA_VERSION = 1
ESTIMATE_FILENAME = "mle_estimate.npz"
DIAGNOSTICS_FILENAME = "mle_diagnostics.json"
HOTSPOT_CLUSTERS_FILENAME = "hotspot_clusters.json"
_KNOWN_REPORT_FILENAMES = (
    ESTIMATE_FILENAME,
    DIAGNOSTICS_FILENAME,
    HOTSPOT_CLUSTERS_FILENAME,
)


@dataclass(frozen=True, slots=True)
class MLEReportPaths:
    """Return the files produced for one estimate report."""

    output_dir: Path
    estimate_npz: Path
    diagnostics_json: Path
    hotspot_clusters_json: Path | None


def _json_safe(value: object, *, path: str = "$") -> object:
    """Convert supported values to strict JSON data and reject non-finite data."""
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path=path)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value), path=path)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"Non-finite JSON value at {path}: {result!r}.")
        return result
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} must be a string.")
            result[key] = _json_safe(child, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_safe(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise TypeError(f"Unsupported JSON value at {path}: {type(value).__name__}.")


def _json_bytes(payload: object) -> bytes:
    """Return deterministic, strict, newline-terminated UTF-8 JSON."""
    safe = _json_safe(payload)
    text = json.dumps(
        safe,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _config_payload(config: MLEConfig | Mapping[str, object] | None) -> object:
    """Return the resolved configuration as strict JSON-compatible data."""
    if config is None:
        return None
    if isinstance(config, MLEConfig):
        return _json_safe(config.to_dict(), path="$.config")
    if isinstance(config, Mapping):
        return _json_safe(dict(config), path="$.config")
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict(), path="$.config")
    raise TypeError(
        "config must be MLEConfig, a mapping, expose to_dict(), or be None."
    )


def _offsets(lengths: Sequence[int]) -> NDArray[np.int64]:
    """Return prefix offsets for a sequence of variable-length rows."""
    result = np.zeros(len(lengths) + 1, dtype=np.int64)
    if lengths:
        result[1:] = np.cumsum(np.asarray(lengths, dtype=np.int64))
    return result


def _canonical_adjacency(
    patches: Sequence[SurfacePatch],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Validate symmetric patch neighbors and return stable-ID graph edges."""
    by_id = {patch.patch_id: patch for patch in patches}
    if len(by_id) != len(patches):
        raise ValueError("Patch IDs must be unique before reporting.")
    edge_lengths: dict[tuple[int, int], float] = {}
    for patch in patches:
        for neighbor_id, length in zip(
            patch.neighbor_patch_ids,
            patch.neighbor_shared_edge_lengths_m,
        ):
            if neighbor_id not in by_id:
                raise ValueError(
                    f"Patch {patch.patch_id} references missing neighbor {neighbor_id}."
                )
            neighbor = by_id[neighbor_id]
            if patch.patch_id not in neighbor.neighbor_patch_ids:
                raise ValueError(
                    f"Patch adjacency {patch.patch_id}<->{neighbor_id} is not symmetric."
                )
            reverse_index = neighbor.neighbor_patch_ids.index(patch.patch_id)
            reverse_length = neighbor.neighbor_shared_edge_lengths_m[reverse_index]
            if not np.isclose(length, reverse_length, rtol=1.0e-9, atol=1.0e-12):
                raise ValueError(
                    f"Patch adjacency {patch.patch_id}<->{neighbor_id} has inconsistent lengths."
                )
            key = tuple(sorted((patch.patch_id, neighbor_id)))
            prior = edge_lengths.get(key)
            if prior is not None and not np.isclose(
                prior,
                length,
                rtol=1.0e-9,
                atol=1.0e-12,
            ):
                raise ValueError(
                    f"Patch adjacency {key} has duplicate inconsistent lengths."
                )
            edge_lengths[key] = float(length)
    if not edge_lengths:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=float)
    keys = sorted(edge_lengths)
    return (
        np.asarray(keys, dtype=np.int64).reshape(-1, 2),
        np.asarray([edge_lengths[key] for key in keys], dtype=float),
    )


def _estimate_arrays(
    estimate: MLEEstimate,
    *,
    diagnostics_sha256: str,
) -> dict[str, NDArray[Any]]:
    """Flatten an estimate into non-object NumPy arrays."""
    patches = estimate.patches
    patch_count = len(patches)
    quadrature_counts = [patch.quadrature_count for patch in patches]
    quadrature_offsets = _offsets(quadrature_counts)
    quadrature_points = np.vstack([patch.quadrature_points_xyz for patch in patches])
    quadrature_weights = np.concatenate([patch.quadrature_weights for patch in patches])
    neighbor_counts = [len(patch.neighbor_patch_ids) for patch in patches]
    neighbor_offsets = _offsets(neighbor_counts)
    if neighbor_offsets[-1] > 0:
        neighbor_ids = np.concatenate(
            [np.asarray(patch.neighbor_patch_ids, dtype=np.int64) for patch in patches]
        )
        neighbor_lengths = np.concatenate(
            [
                np.asarray(patch.neighbor_shared_edge_lengths_m, dtype=float)
                for patch in patches
            ]
        )
    else:
        neighbor_ids = np.zeros(0, dtype=np.int64)
        neighbor_lengths = np.zeros(0, dtype=float)
    adjacency_edges, adjacency_lengths = _canonical_adjacency(patches)
    predicted_spectra_present = estimate.predicted_spectra is not None
    predicted_counts_present = estimate.predicted_isotope_counts is not None
    predicted_spectra = (
        np.asarray(estimate.predicted_spectra, dtype=float)
        if predicted_spectra_present
        else np.zeros((0, 0), dtype=float)
    )
    predicted_counts = (
        np.asarray(estimate.predicted_isotope_counts, dtype=float)
        if predicted_counts_present
        else np.zeros((0, len(estimate.isotope_names)), dtype=float)
    )
    arrays: dict[str, NDArray[Any]] = {
        "schema_version": np.asarray(REPORT_SCHEMA_VERSION, dtype=np.int64),
        "diagnostics_sha256": np.asarray(diagnostics_sha256, dtype=np.str_),
        "isotope_names": np.asarray(estimate.isotope_names, dtype=np.str_),
        "patch_ids": np.asarray([patch.patch_id for patch in patches], dtype=np.int64),
        "patch_centroids_xyz": np.vstack([patch.centroid_xyz for patch in patches]),
        "patch_normals_xyz": np.vstack([patch.normal_xyz for patch in patches]),
        "patch_areas_m2": np.asarray([patch.area_m2 for patch in patches], dtype=float),
        "patch_surface_kinds": np.asarray(
            [patch.surface_kind for patch in patches], dtype=np.str_
        ),
        "patch_object_ids": np.asarray(
            [patch.object_id for patch in patches], dtype=np.str_
        ),
        "patch_vertices_xyz": np.stack([patch.vertices_xyz for patch in patches]),
        "patch_quadrature_offsets": quadrature_offsets,
        "patch_quadrature_points_xyz": quadrature_points,
        "patch_quadrature_weights": quadrature_weights,
        "patch_neighbor_offsets": neighbor_offsets,
        "patch_neighbor_ids": neighbor_ids,
        "patch_neighbor_shared_edge_lengths_m": neighbor_lengths,
        "patch_parent_ids": np.asarray(
            [
                -1 if patch.parent_patch_id is None else patch.parent_patch_id
                for patch in patches
            ],
            dtype=np.int64,
        ),
        "patch_refinement_levels": np.asarray(
            [patch.refinement_level for patch in patches], dtype=np.int64
        ),
        "adjacency_patch_id_edges": adjacency_edges,
        "adjacency_shared_edge_lengths_m": adjacency_lengths,
        "density_by_isotope": np.asarray(estimate.density_by_isotope, dtype=float),
        "patch_strength_by_isotope": np.asarray(
            estimate.patch_strength_by_isotope, dtype=float
        ),
        "predicted_spectra_present": np.asarray(
            int(predicted_spectra_present), dtype=np.uint8
        ),
        "predicted_spectra": predicted_spectra,
        "predicted_isotope_counts_present": np.asarray(
            int(predicted_counts_present), dtype=np.uint8
        ),
        "predicted_isotope_counts": predicted_counts,
        "background_parameters": np.asarray(
            estimate.background_parameters, dtype=float
        ),
        "nuisance_parameters": np.asarray(estimate.nuisance_parameters, dtype=float),
        "objective_value": np.asarray(estimate.objective_value, dtype=float),
        "poisson_deviance": np.asarray(estimate.poisson_deviance, dtype=float),
        "iterations": np.asarray(estimate.iterations, dtype=np.int64),
        "converged": np.asarray(int(estimate.converged), dtype=np.uint8),
        "patch_count": np.asarray(patch_count, dtype=np.int64),
    }
    for name, array in arrays.items():
        if np.asarray(array).dtype.hasobject:
            raise TypeError(f"NPZ member {name!r} cannot use object dtype.")
    return arrays


def _npy_bytes(array: NDArray[Any]) -> bytes:
    """Return one deterministic NPY member without pickle."""
    buffer = BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _deterministic_npz_bytes(arrays: Mapping[str, NDArray[Any]]) -> bytes:
    """Return a standard NPZ archive with stable members and metadata."""
    buffer = BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"Invalid NPZ member name: {name!r}.")
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return buffer.getvalue()


def _commit_report(
    output_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    overwrite: bool,
) -> None:
    """Publish one complete report generation through shared runtime mechanics."""
    target_exists = output_dir.exists() or output_dir.is_symlink()
    if target_exists and not output_dir.is_dir():
        raise NotADirectoryError(f"Report output path is not a directory: {output_dir}")
    if target_exists and not overwrite:
        raise FileExistsError(
            f"MLE report output already exists; pass overwrite=True: {output_dir}"
        )
    policy = "replace_known" if target_exists else "create"
    with AtomicBundlePublisher(
        output_dir,
        policy=policy,
        known_members=_KNOWN_REPORT_FILENAMES,
    ) as publisher:
        for name in sorted(payloads):
            publisher.write_bytes(name, payloads[name])
        if (
            policy == "replace_known"
            and HOTSPOT_CLUSTERS_FILENAME not in payloads
        ):
            publisher.remove(HOTSPOT_CLUSTERS_FILENAME)
        publisher.publish()


def save_mle_estimate(
    estimate_or_output_dir: MLEEstimate | str | Path,
    output_dir_or_estimate: str | Path | MLEEstimate,
    config: MLEConfig | Mapping[str, object] | None = None,
    *,
    overwrite: bool = False,
) -> MLEReportPaths:
    """Save a complete estimate report and return its paths.

    A new report is published as one complete directory generation. With
    ``overwrite=True``, known report members are atomically replaced while
    unrelated files already present in ``output_dir`` remain unchanged.

    The canonical call is ``save_mle_estimate(estimate, output_dir)``.  The
    reversed ``(output_dir, estimate)`` order is also accepted for CLI and
    estimator-backend hooks that use an output-first persistence convention.
    """
    if isinstance(estimate_or_output_dir, MLEEstimate):
        estimate = estimate_or_output_dir
        output_dir = output_dir_or_estimate
    elif isinstance(output_dir_or_estimate, MLEEstimate):
        estimate = output_dir_or_estimate
        output_dir = estimate_or_output_dir
    else:
        raise TypeError(
            "save_mle_estimate requires one MLEEstimate and one output directory."
        )
    if not isinstance(estimate, MLEEstimate):
        raise TypeError("estimate must be an MLEEstimate.")
    if not isinstance(output_dir, (str, os.PathLike)):
        raise TypeError("output_dir must be a filesystem path.")
    diagnostics = _json_safe(estimate.diagnostics, path="$.diagnostics")
    if not isinstance(diagnostics, dict):  # guarded by MLEEstimate, kept explicit
        raise TypeError("estimate.diagnostics must serialize as a JSON object.")
    resolved_config = _config_payload(config)
    diagnostics_payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "isotope_names": list(estimate.isotope_names),
        "patch_count": len(estimate.patches),
        "predicted_spectra_present": estimate.predicted_spectra is not None,
        "predicted_isotope_counts_present": (
            estimate.predicted_isotope_counts is not None
        ),
        "objective_value": float(estimate.objective_value),
        "poisson_deviance": float(estimate.poisson_deviance),
        "iterations": int(estimate.iterations),
        "converged": bool(estimate.converged),
        "diagnostics": diagnostics,
        "config": resolved_config,
    }
    diagnostics_bytes = _json_bytes(diagnostics_payload)
    arrays = _estimate_arrays(
        estimate,
        diagnostics_sha256=sha256(diagnostics_bytes).hexdigest(),
    )
    payloads: dict[str, bytes] = {
        DIAGNOSTICS_FILENAME: diagnostics_bytes,
        ESTIMATE_FILENAME: _deterministic_npz_bytes(arrays),
    }
    hotspot_clusters = diagnostics.get("hotspot_clusters")
    if hotspot_clusters is not None:
        payloads[HOTSPOT_CLUSTERS_FILENAME] = _json_bytes(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "hotspot_clusters": hotspot_clusters,
            }
        )
    target = Path(output_dir)
    _commit_report(target, payloads, overwrite=bool(overwrite))
    hotspot_path = (
        target / HOTSPOT_CLUSTERS_FILENAME
        if HOTSPOT_CLUSTERS_FILENAME in payloads
        else None
    )
    return MLEReportPaths(
        output_dir=target,
        estimate_npz=target / ESTIMATE_FILENAME,
        diagnostics_json=target / DIAGNOSTICS_FILENAME,
        hotspot_clusters_json=hotspot_path,
    )


def _read_strict_json(path: Path) -> dict[str, object]:
    """Read and recursively validate one strict JSON object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read valid JSON report file: {path}") from exc
    safe = _json_safe(payload)
    if not isinstance(safe, dict):
        raise ValueError(f"JSON report root must be an object: {path}")
    return safe


def _scalar(array: NDArray[Any], *, name: str) -> object:
    """Return one scalar NPZ member."""
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"NPZ member {name!r} must be scalar.")
    return value.item()


def _validated_offsets(
    values: NDArray[Any],
    *,
    name: str,
    row_count: int,
    flat_count: int,
) -> NDArray[np.int64]:
    """Return validated monotonic ragged-array offsets."""
    raw = np.asarray(values)
    if raw.shape != (row_count + 1,):
        raise ValueError(f"{name} must have shape ({row_count + 1},).")
    offsets = np.asarray(raw, dtype=np.int64)
    if not np.array_equal(raw, offsets):
        raise ValueError(f"{name} must contain integer offsets.")
    if offsets[0] != 0 or offsets[-1] != flat_count or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{name} contains invalid ragged-array bounds.")
    return offsets


def _required_npz_members(archive: np.lib.npyio.NpzFile) -> None:
    """Reject incomplete archives before reconstructing partial state."""
    required = {
        "schema_version",
        "diagnostics_sha256",
        "isotope_names",
        "patch_ids",
        "patch_centroids_xyz",
        "patch_normals_xyz",
        "patch_areas_m2",
        "patch_surface_kinds",
        "patch_object_ids",
        "patch_vertices_xyz",
        "patch_quadrature_offsets",
        "patch_quadrature_points_xyz",
        "patch_quadrature_weights",
        "patch_neighbor_offsets",
        "patch_neighbor_ids",
        "patch_neighbor_shared_edge_lengths_m",
        "patch_parent_ids",
        "patch_refinement_levels",
        "adjacency_patch_id_edges",
        "adjacency_shared_edge_lengths_m",
        "density_by_isotope",
        "patch_strength_by_isotope",
        "predicted_spectra_present",
        "predicted_spectra",
        "predicted_isotope_counts_present",
        "predicted_isotope_counts",
        "background_parameters",
        "nuisance_parameters",
        "objective_value",
        "poisson_deviance",
        "iterations",
        "converged",
        "patch_count",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"MLE estimate NPZ is missing members: {', '.join(missing)}")


def _patches_from_archive(
    archive: np.lib.npyio.NpzFile,
    *,
    patch_count: int,
) -> tuple[SurfacePatch, ...]:
    """Reconstruct every rectangular patch and its variable-length metadata."""
    ids = np.asarray(archive["patch_ids"], dtype=np.int64)
    centroids = np.asarray(archive["patch_centroids_xyz"], dtype=float)
    normals = np.asarray(archive["patch_normals_xyz"], dtype=float)
    areas = np.asarray(archive["patch_areas_m2"], dtype=float)
    kinds = np.asarray(archive["patch_surface_kinds"], dtype=np.str_)
    object_ids = np.asarray(archive["patch_object_ids"], dtype=np.str_)
    vertices = np.asarray(archive["patch_vertices_xyz"], dtype=float)
    parent_ids = np.asarray(archive["patch_parent_ids"], dtype=np.int64)
    levels = np.asarray(archive["patch_refinement_levels"], dtype=np.int64)
    expected_shapes = {
        "patch_ids": (patch_count,),
        "patch_centroids_xyz": (patch_count, 3),
        "patch_normals_xyz": (patch_count, 3),
        "patch_areas_m2": (patch_count,),
        "patch_surface_kinds": (patch_count,),
        "patch_object_ids": (patch_count,),
        "patch_vertices_xyz": (patch_count, 4, 3),
        "patch_parent_ids": (patch_count,),
        "patch_refinement_levels": (patch_count,),
    }
    actual_arrays = {
        "patch_ids": ids,
        "patch_centroids_xyz": centroids,
        "patch_normals_xyz": normals,
        "patch_areas_m2": areas,
        "patch_surface_kinds": kinds,
        "patch_object_ids": object_ids,
        "patch_vertices_xyz": vertices,
        "patch_parent_ids": parent_ids,
        "patch_refinement_levels": levels,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_arrays[name].shape != expected_shape:
            raise ValueError(
                f"NPZ member {name!r} must have shape {expected_shape}, "
                f"got {actual_arrays[name].shape}."
            )

    quadrature_points = np.asarray(archive["patch_quadrature_points_xyz"], dtype=float)
    quadrature_weights = np.asarray(archive["patch_quadrature_weights"], dtype=float)
    if quadrature_points.ndim != 2 or quadrature_points.shape[1:] != (3,):
        raise ValueError("patch_quadrature_points_xyz must have shape (Q, 3).")
    if quadrature_weights.shape != (quadrature_points.shape[0],):
        raise ValueError("patch_quadrature_weights must have shape (Q,).")
    quadrature_offsets = _validated_offsets(
        archive["patch_quadrature_offsets"],
        name="patch_quadrature_offsets",
        row_count=patch_count,
        flat_count=int(quadrature_points.shape[0]),
    )
    neighbor_ids = np.asarray(archive["patch_neighbor_ids"], dtype=np.int64)
    neighbor_lengths = np.asarray(
        archive["patch_neighbor_shared_edge_lengths_m"], dtype=float
    )
    if neighbor_ids.ndim != 1 or neighbor_lengths.shape != neighbor_ids.shape:
        raise ValueError("Flattened neighbor IDs and lengths must be aligned vectors.")
    neighbor_offsets = _validated_offsets(
        archive["patch_neighbor_offsets"],
        name="patch_neighbor_offsets",
        row_count=patch_count,
        flat_count=int(neighbor_ids.size),
    )

    patches: list[SurfacePatch] = []
    for index in range(patch_count):
        q_start, q_stop = quadrature_offsets[index : index + 2]
        n_start, n_stop = neighbor_offsets[index : index + 2]
        parent_id = int(parent_ids[index])
        patches.append(
            SurfacePatch(
                patch_id=int(ids[index]),
                centroid_xyz=centroids[index],
                normal_xyz=normals[index],
                area_m2=float(areas[index]),
                surface_kind=str(kinds[index]),
                object_id=str(object_ids[index]),
                vertices_xyz=vertices[index],
                quadrature_points_xyz=quadrature_points[q_start:q_stop],
                quadrature_weights=quadrature_weights[q_start:q_stop],
                neighbor_patch_ids=tuple(
                    int(value) for value in neighbor_ids[n_start:n_stop]
                ),
                neighbor_shared_edge_lengths_m=tuple(
                    float(value) for value in neighbor_lengths[n_start:n_stop]
                ),
                parent_patch_id=None if parent_id == -1 else parent_id,
                refinement_level=int(levels[index]),
            )
        )
    return tuple(patches)


def _validate_diagnostics_mirror(
    metadata: Mapping[str, object],
    *,
    estimate: MLEEstimate,
) -> None:
    """Verify JSON metadata mirrors the numerical archive exactly."""
    if int(metadata.get("schema_version", -1)) != REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported MLE diagnostics schema_version.")
    if tuple(metadata.get("isotope_names", ())) != estimate.isotope_names:
        raise ValueError("JSON and NPZ isotope_names do not match.")
    mirrors = {
        "patch_count": len(estimate.patches),
        "predicted_spectra_present": estimate.predicted_spectra is not None,
        "predicted_isotope_counts_present": (
            estimate.predicted_isotope_counts is not None
        ),
        "iterations": estimate.iterations,
        "converged": estimate.converged,
    }
    for name, expected in mirrors.items():
        if metadata.get(name) != expected:
            raise ValueError(f"JSON and NPZ {name} do not match.")
    for name, expected in (
        ("objective_value", estimate.objective_value),
        ("poisson_deviance", estimate.poisson_deviance),
    ):
        value = metadata.get(name)
        if not isinstance(value, (int, float)) or not np.isclose(
            float(value), float(expected), rtol=0.0, atol=0.0
        ):
            raise ValueError(f"JSON and NPZ {name} do not match.")


def _report_directory(path: str | Path) -> Path:
    """Resolve either a report directory or its canonical NPZ member."""
    candidate = Path(path)
    if candidate.name == ESTIMATE_FILENAME:
        return candidate.parent
    return candidate


def mle_report_sha256(output_dir: str | Path) -> str:
    """Hash the canonical files of one validated MLE report artifact."""
    directory = _report_directory(output_dir)
    estimate_path = directory / ESTIMATE_FILENAME
    diagnostics_path = directory / DIAGNOSTICS_FILENAME
    if not estimate_path.is_file() or not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"MLE report requires {ESTIMATE_FILENAME} and {DIAGNOSTICS_FILENAME}."
        )
    inventory = {
        ESTIMATE_FILENAME: sha256(estimate_path.read_bytes()).hexdigest(),
        DIAGNOSTICS_FILENAME: sha256(diagnostics_path.read_bytes()).hexdigest(),
    }
    hotspot_path = directory / HOTSPOT_CLUSTERS_FILENAME
    if hotspot_path.is_file():
        inventory[HOTSPOT_CLUSTERS_FILENAME] = sha256(
            hotspot_path.read_bytes()
        ).hexdigest()
    return sha256(_json_bytes(inventory)).hexdigest()


def load_mle_estimate(output_dir: str | Path) -> MLEEstimate:
    """Load and validate an exact semantic round-trip of a saved estimate."""
    directory = _report_directory(output_dir)
    estimate_path = directory / ESTIMATE_FILENAME
    diagnostics_path = directory / DIAGNOSTICS_FILENAME
    if not estimate_path.is_file() or not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"MLE report requires {ESTIMATE_FILENAME} and {DIAGNOSTICS_FILENAME}."
        )
    diagnostics_bytes = diagnostics_path.read_bytes()
    metadata = _read_strict_json(diagnostics_path)
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("mle_diagnostics.json diagnostics must be an object.")

    try:
        archive_context = np.load(estimate_path, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Unable to read valid MLE estimate NPZ: {estimate_path}"
        ) from exc
    with archive_context as archive:
        _required_npz_members(archive)
        schema_version = int(_scalar(archive["schema_version"], name="schema_version"))
        if schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported MLE estimate schema_version: {schema_version}"
            )
        expected_digest = str(
            _scalar(archive["diagnostics_sha256"], name="diagnostics_sha256")
        )
        actual_digest = sha256(diagnostics_bytes).hexdigest()
        if expected_digest != actual_digest:
            raise ValueError("mle_diagnostics.json does not match mle_estimate.npz.")
        patch_count = int(_scalar(archive["patch_count"], name="patch_count"))
        if patch_count < 1:
            raise ValueError("patch_count must be positive.")
        patches = _patches_from_archive(archive, patch_count=patch_count)
        isotope_names = tuple(
            str(value) for value in np.asarray(archive["isotope_names"], dtype=np.str_)
        )
        predicted_spectra_present = bool(
            int(
                _scalar(
                    archive["predicted_spectra_present"],
                    name="predicted_spectra_present",
                )
            )
        )
        predicted_counts_present = bool(
            int(
                _scalar(
                    archive["predicted_isotope_counts_present"],
                    name="predicted_isotope_counts_present",
                )
            )
        )
        estimate = MLEEstimate(
            isotope_names=isotope_names,
            patches=patches,
            density_by_isotope=np.asarray(archive["density_by_isotope"], dtype=float),
            patch_strength_by_isotope=np.asarray(
                archive["patch_strength_by_isotope"], dtype=float
            ),
            predicted_spectra=(
                np.asarray(archive["predicted_spectra"], dtype=float)
                if predicted_spectra_present
                else None
            ),
            predicted_isotope_counts=(
                np.asarray(archive["predicted_isotope_counts"], dtype=float)
                if predicted_counts_present
                else None
            ),
            background_parameters=np.asarray(
                archive["background_parameters"], dtype=float
            ),
            nuisance_parameters=np.asarray(archive["nuisance_parameters"], dtype=float),
            objective_value=float(
                _scalar(archive["objective_value"], name="objective_value")
            ),
            poisson_deviance=float(
                _scalar(archive["poisson_deviance"], name="poisson_deviance")
            ),
            iterations=int(_scalar(archive["iterations"], name="iterations")),
            converged=bool(int(_scalar(archive["converged"], name="converged"))),
            diagnostics=diagnostics,
        )
        stored_edges = np.asarray(archive["adjacency_patch_id_edges"], dtype=np.int64)
        stored_lengths = np.asarray(
            archive["adjacency_shared_edge_lengths_m"], dtype=float
        )
    if stored_edges.size == 0:
        stored_edges = np.zeros((0, 2), dtype=np.int64)
    if stored_edges.ndim != 2 or stored_edges.shape[1] != 2:
        raise ValueError("adjacency_patch_id_edges must have shape (E, 2).")
    if stored_lengths.shape != (stored_edges.shape[0],):
        raise ValueError("adjacency_shared_edge_lengths_m must have shape (E,).")
    derived_edges, derived_lengths = _canonical_adjacency(estimate.patches)
    if not np.array_equal(stored_edges, derived_edges) or not np.allclose(
        stored_lengths,
        derived_lengths,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "Stored global adjacency does not match patch neighbor metadata."
        )
    _validate_diagnostics_mirror(metadata, estimate=estimate)

    hotspot_path = directory / HOTSPOT_CLUSTERS_FILENAME
    clusters_available = (
        "hotspot_clusters" in diagnostics
        and diagnostics.get("hotspot_clusters") is not None
    )
    if clusters_available:
        if not hotspot_path.is_file():
            raise FileNotFoundError(
                "Diagnostics contain hotspot_clusters but hotspot_clusters.json is missing."
            )
        hotspot_payload = _read_strict_json(hotspot_path)
        if int(hotspot_payload.get("schema_version", -1)) != REPORT_SCHEMA_VERSION:
            raise ValueError("Unsupported hotspot cluster schema_version.")
        if hotspot_payload.get("hotspot_clusters") != diagnostics.get(
            "hotspot_clusters"
        ):
            raise ValueError("hotspot_clusters.json does not match MLE diagnostics.")
    elif hotspot_path.exists():
        raise ValueError(
            "Stale hotspot_clusters.json exists without cluster diagnostics."
        )
    return estimate


def load_mle_config_payload(output_dir: str | Path) -> dict[str, object] | None:
    """Load the resolved configuration stored beside an estimate."""
    metadata = _read_strict_json(_report_directory(output_dir) / DIAGNOSTICS_FILENAME)
    config = metadata.get("config")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("Stored MLE config must be a JSON object or null.")
    return dict(config)


__all__ = [
    "DIAGNOSTICS_FILENAME",
    "ESTIMATE_FILENAME",
    "HOTSPOT_CLUSTERS_FILENAME",
    "MLEReportPaths",
    "REPORT_SCHEMA_VERSION",
    "load_mle_config_payload",
    "load_mle_estimate",
    "mle_report_sha256",
    "save_mle_estimate",
]
