"""Validated runtime context for MLE configuration and lineage checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig
from measurement.observation_model import RuntimeObservationModel
from measurement.obstacles import ObstacleGrid
from runtime import ResolvedForwardContext
from runtime.measurement_log import MeasurementLog, load_measurement_log
from runtime.records import canonical_json_bytes, canonical_json_sha256

from .config import MLEConfig
from .lineage import validate_covered_records_lineage
from .observation_batch import observation_batch_from_log
from .reporting import (
    DIAGNOSTICS_FILENAME,
    ESTIMATE_FILENAME,
    load_mle_config_payload,
    load_mle_estimate,
    mle_report_sha256,
)
from .types import MLEEstimate, ObservationBatch


@dataclass(frozen=True, slots=True)
class EstimatorContext:
    """Hold validated runtime objects for estimator configuration checks."""

    measurement_log_path: Path
    log: MeasurementLog
    batch: ObservationBatch
    config: MLEConfig
    forward_context: ResolvedForwardContext
    environment: EnvironmentConfig
    obstacle_grid: ObstacleGrid | None
    resolved_obstacle_path: Path | None
    observation_model: RuntimeObservationModel
    kernel: ContinuousKernel
    config_sha256: str
    resolved_estimator_config_sha256: str


@dataclass(frozen=True, slots=True)
class WarmStartArtifact:
    """Hold a validated prior estimate and its immutable causal lineage."""

    estimate: MLEEstimate
    report_sha256: str
    estimate_sha256: str
    diagnostics_sha256: str
    measurement_log_sha256: str
    causal_lineage: dict[str, object]


def _resolve_mle_config(
    config: MLEConfig | Mapping[str, Any] | str | Path | None,
    batch: ObservationBatch,
) -> MLEConfig:
    """Resolve an MLE config and enforce the measurement isotope ordering."""
    if config is None:
        mode = "count" if batch.isotope_counts is not None else "spectral"
        resolved = MLEConfig(mode=mode, isotope_names=batch.isotope_names)
    elif isinstance(config, MLEConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = MLEConfig.from_dict(config)
    elif isinstance(config, (str, Path)):
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"MLE configuration file does not exist: {path}")
        resolved = MLEConfig.load(path)
    else:
        raise TypeError("config must be MLEConfig, a mapping, a file path, or None.")
    if tuple(resolved.isotope_names) != tuple(batch.isotope_names):
        raise ValueError(
            "MLEConfig isotope_names must exactly match the measurement-log isotope order."
        )
    return resolved


def prepare_estimator_context(
    measurement_log_path: str | Path,
    *,
    config: MLEConfig | Mapping[str, Any] | str | Path | None = None,
    config_source_sha256: str | None = None,
) -> EstimatorContext:
    """Resolve runtime physics and MLE identities for one measurement log."""
    resolved_log_path = Path(measurement_log_path).resolve()
    log = load_measurement_log(resolved_log_path)
    batch = observation_batch_from_log(log)
    mle_config = _resolve_mle_config(config, batch)
    if config_source_sha256 is not None:
        config_sha256 = str(config_source_sha256).lower()
        if len(config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in config_sha256
        ):
            raise ValueError("config_source_sha256 must be a lowercase SHA-256 digest.")
    elif isinstance(config, (str, Path)):
        config_sha256 = sha256(Path(config).read_bytes()).hexdigest()
    else:
        config_sha256 = sha256(canonical_json_bytes(mle_config.to_dict())).hexdigest()
    resolved_estimator_config_sha256 = canonical_json_sha256(mle_config.to_dict())
    forward_context = ResolvedForwardContext.from_log(log)
    kernel = forward_context.build_continuous_kernel(
        use_gpu=bool(mle_config.use_gpu),
        gpu_device=str(mle_config.gpu_device),
        gpu_dtype=str(mle_config.gpu_dtype),
    )
    return EstimatorContext(
        measurement_log_path=resolved_log_path,
        log=log,
        batch=batch,
        config=mle_config,
        forward_context=forward_context,
        environment=forward_context.environment,
        obstacle_grid=forward_context.obstacle_grid,
        resolved_obstacle_path=forward_context.resolved_obstacle_path,
        observation_model=forward_context.observation_model,
        kernel=kernel,
        config_sha256=config_sha256,
        resolved_estimator_config_sha256=resolved_estimator_config_sha256,
    )


def _report_directory(path: str | Path) -> Path:
    """Resolve an MLE report directory from a directory or NPZ member path."""
    candidate = Path(path).resolve()
    return candidate.parent if candidate.name == ESTIMATE_FILENAME else candidate


def _lineage_integer(value: object, *, name: str) -> int:
    """Return one nonnegative lineage integer without coercing booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Warm-start causal_lineage.{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"Warm-start causal_lineage.{name} must be nonnegative.")
    return result


def _warm_start_lineage(
    context: EstimatorContext,
    estimate: MLEEstimate,
) -> dict[str, object]:
    """Validate causal prefix ancestry and return normalized prior lineage."""
    raw = estimate.diagnostics.get("causal_lineage")
    if not isinstance(raw, Mapping):
        raise ValueError("Warm-start report lacks causal_lineage diagnostics.")
    if raw.get("schema_version") != 2:
        raise ValueError("Warm-start causal_lineage must use schema version 2.")
    raw_steps = raw.get("covered_step_ids")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise ValueError("Warm-start covered_step_ids must be an integer array.")
    covered_steps = tuple(
        _lineage_integer(value, name="covered_step_ids") for value in raw_steps
    )
    if not covered_steps or any(
        right <= left for left, right in zip(covered_steps, covered_steps[1:])
    ):
        raise ValueError(
            "Warm-start covered_step_ids must be nonempty and strictly increasing."
        )
    record_count = _lineage_integer(raw.get("record_count"), name="record_count")
    cutoff_step = _lineage_integer(raw.get("data_cutoff_step"), name="data_cutoff_step")
    cutoff_station = _lineage_integer(
        raw.get("data_cutoff_station"), name="data_cutoff_station"
    )
    if record_count != len(covered_steps) or cutoff_step != covered_steps[-1]:
        raise ValueError("Warm-start cutoff and record_count lineage are inconsistent.")
    if record_count >= len(context.log.records):
        raise ValueError(
            "Warm-start input must be a strict causal prefix of current data."
        )
    current_prefix = context.log.records[:record_count]
    current_steps = tuple(record.step_id for record in current_prefix)
    if current_steps != covered_steps:
        raise ValueError(
            "Warm-start covered_step_ids are not an exact current-log prefix."
        )
    if current_prefix[-1].station_id != cutoff_station:
        raise ValueError("Warm-start cutoff station does not match current data.")
    if context.log.records[record_count].station_id == cutoff_station:
        raise ValueError("Warm-start cutoff is not a station-complete prefix.")
    expected_records_digest = validate_covered_records_lineage(
        raw,
        current_prefix,
        location="warm_start.causal_lineage",
    )
    attestation = raw.get("station_boundary_attestation")
    if attestation not in {
        "writer_metadata",
        "external_validated_schedule",
        "covered_prefix_markers_v1",
    }:
        raise ValueError(
            "Warm-start prefix lacks an accepted station boundary attestation."
        )
    fit_kind = raw.get("fit_kind")
    if fit_kind not in {"cold_start_all_history", "warm_start_all_history"}:
        raise ValueError("Warm-start causal_lineage fit_kind is unsupported.")
    return {
        "covered_step_ids": list(covered_steps),
        "data_cutoff_step": cutoff_step,
        "data_cutoff_station": cutoff_station,
        "record_count": record_count,
        "covered_records_digest": expected_records_digest.to_payload(),
        "covered_records_sha256": expected_records_digest.sha256,
        "station_boundary_attestation": attestation,
        "fit_kind": fit_kind,
    }


def validate_warm_start_artifact(
    context: EstimatorContext,
    initial_estimate_path: str | Path,
) -> WarmStartArtifact:
    """Load a prior estimate and fail closed on every identity boundary."""
    directory = _report_directory(initial_estimate_path)
    estimate = load_mle_estimate(directory)
    diagnostics = estimate.diagnostics
    provenance = diagnostics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Warm-start report lacks provenance diagnostics.")
    if provenance.get("causal_lineage") != diagnostics.get("causal_lineage"):
        raise ValueError(
            "Warm-start provenance and diagnostics causal lineage do not match."
        )
    expected_contract = {
        "estimator_family": "surface_mle",
        "candidate_domain": "complete_surface_dictionary",
        "uses_pf_state": False,
        "uses_pf_candidates": False,
    }
    for name, expected in expected_contract.items():
        if provenance.get(name) != expected:
            raise ValueError(f"Warm-start provenance {name} is incompatible.")
    if (
        diagnostics.get("mode") != context.config.mode
        or provenance.get("estimator_variant") != context.config.mode
    ):
        raise ValueError("Warm-start estimator mode is incompatible with this context.")
    if tuple(estimate.isotope_names) != tuple(context.batch.isotope_names):
        raise ValueError(
            "Warm-start isotope ordering is incompatible with this context."
        )
    stored_config = load_mle_config_payload(directory)
    if stored_config is None:
        raise ValueError("Warm-start report lacks a resolved MLE configuration.")
    stored_config_digest = canonical_json_sha256(stored_config)
    if stored_config_digest != context.resolved_estimator_config_sha256:
        raise ValueError("Warm-start resolved MLE configuration is incompatible.")
    forward_digest = sha256(
        (context.measurement_log_path / "forward_model_manifest.json").read_bytes()
    ).hexdigest()
    expected_identities = {
        "measurement_log_schema_version": context.log.context.schema_version,
        "measurement_run_id": context.log.context.run_id,
        "measurement_repository_commit": context.log.context.repository_commit,
        "resolved_config_sha256": context.log.context.runtime_config_sha256,
        "forward_model_manifest_sha256": forward_digest,
        "resolved_estimator_config_sha256": context.resolved_estimator_config_sha256,
    }
    for name, expected in expected_identities.items():
        if provenance.get(name) != expected:
            raise ValueError(f"Warm-start provenance {name} is incompatible.")
    measurement_digest = provenance.get("measurement_log_sha256")
    if not isinstance(measurement_digest, str) or (
        len(measurement_digest) != 64
        or any(character not in "0123456789abcdef" for character in measurement_digest)
    ):
        raise ValueError("Warm-start measurement_log_sha256 is invalid.")
    lineage = _warm_start_lineage(context, estimate)
    estimate_path = directory / ESTIMATE_FILENAME
    diagnostics_path = directory / DIAGNOSTICS_FILENAME
    return WarmStartArtifact(
        estimate=estimate,
        report_sha256=mle_report_sha256(directory),
        estimate_sha256=sha256(estimate_path.read_bytes()).hexdigest(),
        diagnostics_sha256=sha256(diagnostics_path.read_bytes()).hexdigest(),
        measurement_log_sha256=measurement_digest,
        causal_lineage=lineage,
    )


__all__ = [
    "EstimatorContext",
    "WarmStartArtifact",
    "prepare_estimator_context",
    "validate_warm_start_artifact",
]
