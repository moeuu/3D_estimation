"""Future-only predictive verification for frozen count-MLE snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from runtime.measurement_log import MeasurementLog
from runtime.prefix import measurement_records_digest
from runtime.records import (
    canonical_json_bytes,
    validate_truth_free_estimator_input,
)

from .config import MLEConfig
from .lineage import validate_covered_records_lineage
from .observation_batch import subset_observation_batch
from .estimator_context import (
    EstimatorContext,
    prepare_estimator_context,
    validate_warm_start_artifact,
)
from .reporting import mle_report_sha256
from .response_builder import build_count_responses
from .types import MLEEstimate, SurfacePatch


_SNAPSHOT_FIELDS = {
    "schema_version",
    "snapshot_id",
    "trigger_id",
    "estimator_family",
    "estimator_variant",
    "data_cutoff_step",
    "data_cutoff_station",
    "cutoff_station_complete",
    "covered_step_ids",
    "source_run_id",
    "prefix_measurement_log_sha256",
    "covered_records_digest",
    "covered_records_sha256",
    "covered_station_boundaries_sha256",
    "mle_result_sha256",
    "warm_start",
    "clusters",
    "predicted_observations",
    "fit_diagnostics",
    "safety",
    "provenance",
}
_CLUSTER_FIELDS = {
    "snapshot_candidate_id",
    "cluster_id",
    "isotope",
    "centroid_xyz",
    "integrated_strength_cps_1m",
    "surface_kinds",
    "patch_ids",
}
_PREDICTION_FIELDS = {"step_id", "isotope_counts"}
_WARM_START_FIELDS = {"used", "snapshot_id", "mle_result_sha256"}
_SNAPSHOT_SAFETY = {
    "direct_mle_objective_reweight": False,
    "hard_prune_authorized": False,
}
_PROVENANCE_FIELDS = (
    "estimator_family",
    "estimator_variant",
    "candidate_domain",
    "uses_pf_state",
    "uses_pf_candidates",
    "estimator_commit",
    "measurement_run_id",
    "measurement_log_schema_version",
    "measurement_log_sha256",
    "forward_model_manifest_sha256",
    "config_sha256",
    "resolved_config_sha256",
    "resolved_estimator_config_sha256",
)


@dataclass(frozen=True, slots=True)
class _FrozenPatchGeometry:
    """Expose only saved patch geometry needed by count-response construction."""

    areas_m2: NDArray[np.float64]
    quadrature_points_xyz: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]


def _json_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys while constructing one object."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Snapshot JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    """Reject NaN and infinity tokens in strict snapshot JSON."""
    raise ValueError(f"Snapshot JSON contains non-finite constant {value!r}.")


def _load_snapshot(path: str | Path) -> tuple[dict[str, object], bytes]:
    """Load one strict MLESnapshot v2 object and its exact raw bytes."""
    target = Path(path)
    try:
        raw = target.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load strict snapshot JSON: {target}.") from exc
    if not isinstance(payload, dict):
        raise ValueError("MLESnapshot JSON root must be an object.")
    validate_truth_free_estimator_input(payload, path="mle_snapshot")
    return payload, raw


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return one nonnegative integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    """Return one finite nonnegative scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _finite_float(value: object, *, name: str) -> float:
    """Return one finite signed numeric scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _sha256_string(value: object, *, name: str) -> str:
    """Return a validated lowercase SHA-256 string."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    """Return one normalized nonempty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _station_boundary_payload(
    log: MeasurementLog,
    *,
    cutoff_step: int,
) -> dict[str, object]:
    """Build the declared station-boundary payload through one cutoff."""
    selected = tuple(record for record in log.records if record.step_id <= cutoff_step)
    if not selected or selected[-1].step_id != cutoff_step:
        raise ValueError("Station-boundary cutoff is absent from the MeasurementLog.")
    entries: list[dict[str, int]] = []
    for index, record in enumerate(selected):
        if record.metadata.get("station_complete") is not True:
            continue
        if index + 1 < len(selected) and (
            selected[index + 1].station_id == record.station_id
        ):
            raise ValueError("A station_complete marker precedes another station row.")
        entries.append(
            {
                "station_id": record.station_id,
                "terminal_step_id": record.step_id,
            }
        )
    if not entries or entries[-1]["terminal_step_id"] != cutoff_step:
        raise ValueError(
            "The requested cutoff row is not marked station_complete=true."
        )
    present_stations = {record.station_id for record in selected}
    declared_stations = {entry["station_id"] for entry in entries}
    if declared_stations != present_stations:
        raise ValueError("Every station through the cutoff must declare its boundary.")
    if any(
        right["station_id"] <= left["station_id"]
        or right["terminal_step_id"] <= left["terminal_step_id"]
        for left, right in zip(entries, entries[1:])
    ):
        raise ValueError("Declared station boundaries must be strictly increasing.")
    return {
        "schema_version": 1,
        "source_run_id": log.context.run_id,
        "station_end_steps": entries,
    }


def covered_station_boundaries_sha256(
    log: MeasurementLog,
    *,
    cutoff_step: int,
) -> str:
    """Hash all declared station boundaries through an exact cutoff step."""
    return sha256(
        canonical_json_bytes(_station_boundary_payload(log, cutoff_step=cutoff_step))
    ).hexdigest()


def _validate_manifest_schedule_hash(log: MeasurementLog) -> None:
    """Validate an optional complete station-boundary schedule hash."""
    candidates: list[object] = []
    direct = log.context.metadata.get("station_boundary_schedule_sha256")
    if direct is not None:
        candidates.append(direct)
    prefix_metadata = log.context.metadata.get("measurement_log_prefix")
    if isinstance(prefix_metadata, Mapping):
        nested = prefix_metadata.get("station_boundary_schedule_sha256")
        if nested is not None:
            candidates.append(nested)
    if not candidates:
        return
    expected = covered_station_boundaries_sha256(
        log,
        cutoff_step=log.records[-1].step_id,
    )
    for value in candidates:
        if _sha256_string(value, name="station_boundary_schedule_sha256") != expected:
            raise ValueError(
                "MeasurementLog station-boundary schedule hash mismatches."
            )


def _frozen_patch_geometry(
    patches: Sequence[SurfacePatch],
) -> _FrozenPatchGeometry:
    """Pack saved patch areas and mixed quadrature into dense response arrays."""
    rows = tuple(patches)
    maximum = max(patch.quadrature_count for patch in rows)
    points = np.empty((len(rows), maximum, 3), dtype=float)
    weights = np.zeros((len(rows), maximum), dtype=float)
    for index, patch in enumerate(rows):
        count = patch.quadrature_count
        points[index, :count] = patch.quadrature_points_xyz
        points[index, count:] = patch.quadrature_points_xyz[-1]
        weights[index, :count] = patch.quadrature_weights
    return _FrozenPatchGeometry(
        areas_m2=np.asarray([patch.area_m2 for patch in rows], dtype=float),
        quadrature_points_xyz=points,
        quadrature_weights=weights,
    )


def _validate_warm_start_mapping(value: object) -> dict[str, object]:
    """Validate the MLESnapshot v2 warm-start lineage object."""
    if not isinstance(value, dict) or set(value) != _WARM_START_FIELDS:
        raise ValueError("snapshot.warm_start has an incompatible schema.")
    used = value.get("used")
    if not isinstance(used, bool):
        raise ValueError("snapshot.warm_start.used must be boolean.")
    snapshot_id = value.get("snapshot_id")
    result_digest = value.get("mle_result_sha256")
    if used:
        _nonempty_string(snapshot_id, name="snapshot.warm_start.snapshot_id")
        _sha256_string(
            result_digest,
            name="snapshot.warm_start.mle_result_sha256",
        )
    elif snapshot_id is not None or result_digest is not None:
        raise ValueError("Unused snapshot.warm_start references must be null.")
    return dict(value)


def _validate_snapshot_clusters(
    value: object,
    *,
    estimate: MLEEstimate,
) -> list[dict[str, object]]:
    """Validate snapshot candidates exactly mirror saved hotspot clusters."""
    if not isinstance(value, list):
        raise ValueError("snapshot.clusters must be an array.")
    report_clusters = estimate.diagnostics.get("hotspot_clusters")
    if not isinstance(report_clusters, list) or len(value) != len(report_clusters):
        raise ValueError("Snapshot clusters do not match report hotspot clusters.")
    report_by_id: dict[int, Mapping[str, object]] = {}
    for raw in report_clusters:
        if not isinstance(raw, Mapping):
            raise ValueError("Report hotspot cluster must be an object.")
        cluster_id = _nonnegative_integer(raw.get("cluster_id"), name="cluster_id")
        if cluster_id in report_by_id:
            raise ValueError("Report hotspot cluster IDs must be unique.")
        report_by_id[cluster_id] = raw
    patch_ids = {patch.patch_id for patch in estimate.patches}
    candidate_ids: set[str] = set()
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != _CLUSTER_FIELDS:
            raise ValueError(f"snapshot.clusters[{index}] has an incompatible schema.")
        candidate_id = _nonempty_string(
            raw["snapshot_candidate_id"],
            name=f"snapshot.clusters[{index}].snapshot_candidate_id",
        )
        if candidate_id in candidate_ids:
            raise ValueError("snapshot_candidate_id values must be unique.")
        candidate_ids.add(candidate_id)
        cluster_id = _nonnegative_integer(
            raw["cluster_id"], name=f"snapshot.clusters[{index}].cluster_id"
        )
        report = report_by_id.get(cluster_id)
        if report is None:
            raise ValueError("Snapshot cluster_id is absent from the MLE report.")
        isotope = _nonempty_string(
            raw["isotope"], name=f"snapshot.clusters[{index}].isotope"
        )
        if isotope not in estimate.isotope_names:
            raise ValueError("Snapshot cluster isotope is unknown.")
        raw_centroid = raw["centroid_xyz"]
        if not isinstance(raw_centroid, list) or len(raw_centroid) != 3:
            raise ValueError("Snapshot cluster centroid_xyz must contain three values.")
        centroid = [
            _finite_float(item, name="snapshot cluster centroid")
            for item in raw_centroid
        ]
        raw_strength = _finite_nonnegative(
            raw["integrated_strength_cps_1m"],
            name="snapshot cluster integrated strength",
        )
        kinds = raw["surface_kinds"]
        raw_patch_ids = raw["patch_ids"]
        if not isinstance(kinds, list) or not all(
            isinstance(item, str) and item for item in kinds
        ):
            raise ValueError("Snapshot cluster surface_kinds must be a string array.")
        if not isinstance(raw_patch_ids, list) or not raw_patch_ids:
            raise ValueError("Snapshot cluster patch_ids must be a nonempty array.")
        cluster_patch_ids = [
            _nonnegative_integer(item, name="snapshot cluster patch_id")
            for item in raw_patch_ids
        ]
        if len(set(cluster_patch_ids)) != len(cluster_patch_ids) or not set(
            cluster_patch_ids
        ).issubset(patch_ids):
            raise ValueError("Snapshot cluster patch_ids are invalid.")
        comparisons = {
            "isotope": isotope,
            "centroid_xyz": centroid,
            "integrated_strength_cps_1m": raw_strength,
            "surface_kinds": kinds,
            "patch_ids": cluster_patch_ids,
        }
        for name, actual in comparisons.items():
            expected = report.get(name)
            if isinstance(expected, tuple):
                expected = list(expected)
            if actual != expected:
                raise ValueError(
                    f"Snapshot cluster {name} differs from the MLE report."
                )
        normalized.append(dict(raw))
    return normalized


def _validate_snapshot_predictions(
    value: object,
    *,
    estimate: MLEEstimate,
    covered_step_ids: tuple[int, ...],
) -> None:
    """Validate snapshot predictions exactly mirror the bound count report."""
    if not isinstance(value, list) or len(value) != len(covered_step_ids):
        raise ValueError("snapshot.predicted_observations has an incompatible length.")
    expected_counts = estimate.predicted_isotope_counts
    if expected_counts is None or expected_counts.shape[0] != len(covered_step_ids):
        raise ValueError("Count snapshot report lacks aligned isotope predictions.")
    for row, (raw, step_id) in enumerate(zip(value, covered_step_ids)):
        if not isinstance(raw, dict) or set(raw) != _PREDICTION_FIELDS:
            raise ValueError("snapshot.predicted_observations row schema is invalid.")
        if _nonnegative_integer(raw["step_id"], name="prediction step_id") != step_id:
            raise ValueError("Snapshot predicted observation step order is invalid.")
        counts = raw["isotope_counts"]
        if not isinstance(counts, dict) or set(counts) != set(estimate.isotope_names):
            raise ValueError("Snapshot predicted isotope channels are incompatible.")
        actual = np.asarray(
            [
                _finite_nonnegative(counts[name], name="predicted isotope count")
                for name in estimate.isotope_names
            ],
            dtype=float,
        )
        if not np.array_equal(actual, expected_counts[row]):
            raise ValueError("Snapshot predicted counts differ from the MLE report.")


def _validate_snapshot(
    snapshot: dict[str, object],
    *,
    context: EstimatorContext,
    estimate: MLEEstimate,
    report_digest: str,
) -> tuple[int, int, list[dict[str, object]]]:
    """Validate MLESnapshot v2 identities against report and current prefix."""
    if set(snapshot) != _SNAPSHOT_FIELDS:
        raise ValueError("MLESnapshot v2 top-level fields are incompatible.")
    if snapshot.get("schema_version") != 2:
        raise ValueError("MLESnapshot schema_version must be 2.")
    _nonempty_string(snapshot["snapshot_id"], name="snapshot_id")
    _nonempty_string(snapshot["trigger_id"], name="trigger_id")
    if snapshot.get("estimator_family") != "surface_mle":
        raise ValueError("MLESnapshot estimator_family must be surface_mle.")
    if snapshot.get("estimator_variant") != "count":
        raise ValueError("Future scoring requires a count MLESnapshot.")
    if snapshot.get("cutoff_station_complete") is not True:
        raise ValueError("MLESnapshot cutoff_station_complete must be true.")
    if snapshot.get("safety") != _SNAPSHOT_SAFETY:
        raise ValueError("MLESnapshot safety flags are incompatible.")
    if not isinstance(snapshot.get("fit_diagnostics"), dict):
        raise ValueError("MLESnapshot fit_diagnostics must be an object.")
    _validate_warm_start_mapping(snapshot["warm_start"])

    log = context.log
    batch = context.batch
    cutoff_step = _nonnegative_integer(
        snapshot["data_cutoff_step"], name="data_cutoff_step"
    )
    cutoff_station = _nonnegative_integer(
        snapshot["data_cutoff_station"], name="data_cutoff_station"
    )
    raw_steps = snapshot["covered_step_ids"]
    if not isinstance(raw_steps, list):
        raise ValueError("MLESnapshot covered_step_ids must be an array.")
    covered_steps = tuple(
        _nonnegative_integer(value, name="covered_step_id") for value in raw_steps
    )
    if (
        not covered_steps
        or covered_steps[-1] != cutoff_step
        or any(right <= left for left, right in zip(covered_steps, covered_steps[1:]))
    ):
        raise ValueError("MLESnapshot covered_step_ids are invalid.")
    if len(covered_steps) >= batch.measurement_count:
        raise ValueError("MLESnapshot cutoff must be a strict prefix of current data.")
    current_steps = tuple(int(value) for value in batch.step_ids[: len(covered_steps)])
    if current_steps != covered_steps:
        raise ValueError("MLESnapshot steps are not a current-log prefix.")
    if int(batch.station_ids[len(covered_steps) - 1]) != cutoff_station:
        raise ValueError("MLESnapshot cutoff station is incompatible.")
    if int(batch.station_ids[len(covered_steps)]) == cutoff_station:
        raise ValueError("MLESnapshot cutoff is not station-complete.")
    if snapshot.get("source_run_id") != log.context.run_id:
        raise ValueError("MLESnapshot source_run_id is incompatible.")
    covered_records = log.records[: len(covered_steps)]
    records_digest = validate_covered_records_lineage(
        snapshot,
        covered_records,
        location="MLESnapshot",
    )
    boundary_digest = covered_station_boundaries_sha256(log, cutoff_step=cutoff_step)
    if snapshot.get("covered_station_boundaries_sha256") != boundary_digest:
        raise ValueError("MLESnapshot station-boundary lineage is incompatible.")
    if _sha256_string(snapshot["mle_result_sha256"], name="mle_result_sha256") != (
        report_digest
    ):
        raise ValueError("MLESnapshot does not bind the supplied MLE report.")

    report_provenance = estimate.diagnostics.get("provenance")
    snapshot_provenance = snapshot.get("provenance")
    if not isinstance(report_provenance, Mapping) or not isinstance(
        snapshot_provenance, dict
    ):
        raise ValueError("Snapshot and report provenance must be objects.")
    for name in _PROVENANCE_FIELDS:
        if snapshot_provenance.get(name) != report_provenance.get(name):
            raise ValueError(f"MLESnapshot provenance {name} mismatches the report.")
    expected_current = {
        "measurement_run_id": log.context.run_id,
        "measurement_log_schema_version": log.context.schema_version,
        "forward_model_manifest_sha256": sha256(
            (context.measurement_log_path / "forward_model_manifest.json").read_bytes()
        ).hexdigest(),
        "config_sha256": context.config_sha256,
        "resolved_config_sha256": log.context.runtime_config_sha256,
        "resolved_estimator_config_sha256": (context.resolved_estimator_config_sha256),
    }
    for name, expected in expected_current.items():
        if snapshot_provenance.get(name) != expected:
            raise ValueError(f"MLESnapshot provenance {name} is incompatible.")
    prefix_digest = _sha256_string(
        snapshot["prefix_measurement_log_sha256"],
        name="prefix_measurement_log_sha256",
    )
    if prefix_digest != report_provenance.get("measurement_log_sha256") or (
        prefix_digest != snapshot_provenance.get("measurement_log_sha256")
    ):
        raise ValueError("MLESnapshot prefix MeasurementLog binding is incompatible.")
    report_lineage = estimate.diagnostics.get("causal_lineage")
    if not isinstance(report_lineage, Mapping) or (
        report_lineage.get("schema_version") != 2
        or report_lineage.get("covered_step_ids") != list(covered_steps)
        or report_lineage.get("data_cutoff_step") != cutoff_step
        or report_lineage.get("data_cutoff_station") != cutoff_station
    ):
        raise ValueError("MLESnapshot cutoff differs from bound report lineage.")
    report_records_digest = validate_covered_records_lineage(
        report_lineage,
        covered_records,
        location="MLE report causal_lineage",
    )
    if report_records_digest != records_digest:
        raise ValueError("MLESnapshot digest differs from bound report lineage.")
    _validate_snapshot_predictions(
        snapshot["predicted_observations"],
        estimate=estimate,
        covered_step_ids=covered_steps,
    )
    clusters = _validate_snapshot_clusters(snapshot["clusters"], estimate=estimate)
    return cutoff_step, cutoff_station, clusters


def _poisson_log_predictive_ratio(
    observed: NDArray[np.float64],
    full_mean: NDArray[np.float64],
    reduced_mean: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return row-wise log p(y|full) minus log p(y|reduced)."""
    return np.sum(
        observed * (np.log(full_mean) - np.log(reduced_mean))
        - (full_mean - reduced_mean),
        axis=1,
    )


def score_future_count_candidates(
    run_dir: str | Path,
    *,
    config: MLEConfig | Mapping[str, Any] | str | Path,
    snapshot_estimate: str | Path,
    snapshot: str | Path,
) -> dict[str, object]:
    """Score frozen snapshot clusters using only observations after cutoff."""
    context = prepare_estimator_context(run_dir, config=config)
    if context.config.mode != "count":
        raise ValueError("Future candidate scoring requires count MLE configuration.")
    if context.batch.isotope_counts is None:
        raise ValueError(
            "Future candidate scoring requires isotope count observations."
        )
    if context.log.records[-1].metadata.get("station_complete") is not True:
        raise ValueError(
            "Current MeasurementLog prefix must end at station_complete=true."
        )
    _validate_manifest_schedule_hash(context.log)
    artifact = validate_warm_start_artifact(context, snapshot_estimate)
    estimate = artifact.estimate
    report_digest = mle_report_sha256(snapshot_estimate)
    snapshot_payload, snapshot_raw = _load_snapshot(snapshot)
    cutoff_step, cutoff_station, clusters = _validate_snapshot(
        snapshot_payload,
        context=context,
        estimate=estimate,
        report_digest=report_digest,
    )
    future_indices = np.flatnonzero(context.batch.step_ids > cutoff_step)
    if future_indices.size == 0:
        raise ValueError("MLESnapshot cutoff has no future verification observations.")
    future = subset_observation_batch(context.batch, future_indices)
    if future.isotope_counts is None:
        raise ValueError("Future verification observations lack isotope counts.")
    geometry = _frozen_patch_geometry(estimate.patches)
    responses = build_count_responses(
        future,
        geometry,
        estimate.isotope_names,
        context.kernel,
        kernel_chunk_size=int(context.config.response_chunk_size),
    ).response_by_integrated_strength
    strengths = np.asarray(estimate.patch_strength_by_isotope, dtype=float)
    signal = np.einsum("mgi,ig->mi", responses, strengths, optimize=True)
    background = np.asarray(estimate.background_parameters, dtype=float)
    if background.size not in {0, len(estimate.isotope_names)}:
        raise ValueError("Frozen count-MLE background parameters are incompatible.")
    if bool(context.config.fit_background_nuisance) != bool(background.size):
        raise ValueError("Frozen background parameters do not match MLEConfig.")
    if estimate.nuisance_parameters.size:
        raise ValueError("Count snapshot contains unsupported non-background nuisance.")
    if background.size:
        signal = signal + future.live_times_s[:, None] * background[None, :]
    min_mean = float(context.config.min_mean)
    full_mean = np.maximum(signal, min_mean)
    patch_index = {
        patch.patch_id: index for index, patch in enumerate(estimate.patches)
    }
    isotope_index = {name: index for index, name in enumerate(estimate.isotope_names)}
    candidate_scores: list[dict[str, object]] = []
    for cluster in clusters:
        isotope = str(cluster["isotope"])
        channel = isotope_index[isotope]
        indices = np.asarray(
            [patch_index[int(value)] for value in cluster["patch_ids"]],
            dtype=np.int64,
        )
        removed = np.sum(
            responses[:, indices, channel] * strengths[channel, indices][None, :],
            axis=1,
        )
        reduced_raw = np.array(signal, copy=True)
        reduced_raw[:, channel] = np.maximum(
            reduced_raw[:, channel] - removed,
            0.0,
        )
        reduced_mean = np.maximum(reduced_raw, min_mean)
        ratios = _poisson_log_predictive_ratio(
            future.isotope_counts,
            full_mean,
            reduced_mean,
        )
        per_step = [
            {
                "step_id": int(step_id),
                "station_id": int(station_id),
                "log_predictive_likelihood_ratio": float(ratio),
            }
            for step_id, station_id, ratio in zip(
                future.step_ids,
                future.station_ids,
                ratios,
            )
        ]
        candidate_scores.append(
            {
                "snapshot_candidate_id": cluster["snapshot_candidate_id"],
                "cluster_id": cluster["cluster_id"],
                "isotope": isotope,
                "patch_ids": list(cluster["patch_ids"]),
                "future_step_scores": per_step,
                "cumulative_log_predictive_likelihood_ratio": float(np.sum(ratios)),
            }
        )
    current_records_digest = measurement_records_digest(context.log.records)
    return {
        "schema_version": 1,
        "score_family": "frozen_count_snapshot_cluster_log_predictive_ratio",
        "source_run_id": context.log.context.run_id,
        "snapshot_id": snapshot_payload["snapshot_id"],
        "snapshot_data_cutoff_step": cutoff_step,
        "snapshot_data_cutoff_station": cutoff_station,
        "future_step_ids": future.step_ids.astype(int).tolist(),
        "future_station_ids": future.station_ids.astype(int).tolist(),
        "isotope_names": list(estimate.isotope_names),
        "candidates": candidate_scores,
        "hashes": {
            "snapshot_file_sha256": sha256(snapshot_raw).hexdigest(),
            "snapshot_canonical_sha256": sha256(
                canonical_json_bytes(snapshot_payload)
            ).hexdigest(),
            "snapshot_mle_report_sha256": report_digest,
            "snapshot_prefix_measurement_log_sha256": snapshot_payload[
                "prefix_measurement_log_sha256"
            ],
            "current_measurement_log_sha256": context.log.content_sha256,
            "snapshot_covered_records_digest": snapshot_payload[
                "covered_records_digest"
            ],
            "snapshot_covered_records_sha256": snapshot_payload[
                "covered_records_sha256"
            ],
            "current_covered_records_digest": current_records_digest.to_payload(),
            "current_covered_records_sha256": current_records_digest.sha256,
            "snapshot_covered_station_boundaries_sha256": snapshot_payload[
                "covered_station_boundaries_sha256"
            ],
        },
        "safety": {
            "future_only": True,
            "snapshot_parameters_frozen": True,
            "no_refit": True,
            "truth_used": False,
        },
    }


def save_future_candidate_scores(
    output: str | Path,
    payload: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> Path:
    """Persist one deterministic future-candidate score JSON artifact."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(dict(payload))
    if target.exists() and not overwrite:
        raise FileExistsError(f"Future score output already exists: {target}")
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "covered_station_boundaries_sha256",
    "save_future_candidate_scores",
    "score_future_count_candidates",
]
