"""Standalone orchestration for replaying versioned measurement logs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import import_module
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
from .estimator import SurfaceMLEEstimator
from runtime.prefix import (
    covered_station_boundaries_sha256,
)
from .lineage import covered_records_lineage, validate_covered_records_lineage
from .observation_batch import observation_batch_from_log
from .provenance import estimator_provenance
from .reporting import (
    DIAGNOSTICS_FILENAME,
    ESTIMATE_FILENAME,
    load_mle_config_payload,
    load_mle_estimate,
    mle_report_sha256,
)
from .types import MLEEstimate, ObservationBatch


ReplaySaveHook = Callable[[MLEEstimate, Path], object]


@dataclass(frozen=True)
class ReplayContext:
    """All validated local objects required for one all-history replay fit."""

    run_dir: Path
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


@dataclass(frozen=True)
class ReplayResult:
    """Return a fitted estimate together with its fully resolved replay context."""

    estimate: MLEEstimate
    context: ReplayContext
    saved_output: object | None = None


@dataclass(frozen=True)
class WarmStartArtifact:
    """Validated warm-start estimate and immutable artifact lineage."""

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
    """Resolve a public MLEConfig and enforce logged isotope ordering."""
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


def prepare_replay(
    run_dir: str | Path,
    *,
    config: MLEConfig | Mapping[str, Any] | str | Path | None = None,
    config_source_sha256: str | None = None,
) -> ReplayContext:
    """Load a measurement log and resolve shared runtime forward physics."""
    resolved_run_dir = Path(run_dir).resolve()
    log = load_measurement_log(resolved_run_dir)
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
    return ReplayContext(
        run_dir=resolved_run_dir,
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


def _available_reporting_hook() -> ReplaySaveHook | None:
    """Return an optional reporting hook without requiring a reporting module."""
    try:
        module = import_module("three_d_estimation.reporting")
    except ModuleNotFoundError as exc:
        if exc.name != "three_d_estimation.reporting":
            raise
        return None
    for name in ("save_estimate", "save_mle_estimate"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


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


def _prefix_attestation(log: MeasurementLog) -> str:
    """Return and validate an optional materialized-prefix attestation."""
    raw = log.context.metadata.get("measurement_log_prefix")
    if raw is None:
        return "finalized_measurement_log"
    if not isinstance(raw, Mapping):
        raise ValueError("measurement_log_prefix metadata must be an object.")
    final = log.records[-1]
    expected = {
        "schema_version": 1,
        "source_run_id": log.context.run_id,
        "data_cutoff_step": final.step_id,
        "data_cutoff_station": final.station_id,
    }
    for name, value in expected.items():
        if raw.get(name) != value:
            raise ValueError(
                f"measurement_log_prefix metadata {name} does not match the log."
            )
    validate_covered_records_lineage(
        raw,
        log.records,
        location="measurement_log_prefix",
    )
    attestation = raw.get("station_boundary_attestation")
    if attestation not in {
        "writer_metadata",
        "external_validated_schedule",
        "covered_prefix_markers_v1",
    }:
        raise ValueError(
            "measurement_log_prefix has an unsupported station boundary attestation."
        )
    if attestation == "covered_prefix_markers_v1":
        expected_boundary_hash = covered_station_boundaries_sha256(
            log.records,
            source_run_id=log.context.run_id,
        )
        if raw.get("covered_station_boundaries_sha256") != expected_boundary_hash:
            raise ValueError(
                "measurement_log_prefix covered station-boundary hash is invalid."
            )
    return str(attestation)


def _warm_start_lineage(
    context: ReplayContext,
    estimate: MLEEstimate,
) -> dict[str, object]:
    """Validate causal prefix ancestry and return normalized warm lineage."""
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
            "Warm-start input must be a strict causal prefix of the replay log."
        )
    current_prefix = context.log.records[:record_count]
    current_steps = tuple(record.step_id for record in current_prefix)
    if current_steps != covered_steps:
        raise ValueError(
            "Warm-start covered_step_ids are not an exact replay-log prefix."
        )
    if current_prefix[-1].station_id != cutoff_station:
        raise ValueError("Warm-start cutoff station does not match the replay log.")
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
    context: ReplayContext,
    initial_estimate_path: str | Path,
) -> WarmStartArtifact:
    """Load an artifact warm start and fail closed on every identity boundary."""
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
        raise ValueError("Warm-start estimator mode is incompatible with this fit.")
    if tuple(estimate.isotope_names) != tuple(context.batch.isotope_names):
        raise ValueError("Warm-start isotope ordering is incompatible with this fit.")
    stored_config = load_mle_config_payload(directory)
    if stored_config is None:
        raise ValueError("Warm-start report lacks a resolved MLE configuration.")
    stored_config_digest = canonical_json_sha256(stored_config)
    if stored_config_digest != context.resolved_estimator_config_sha256:
        raise ValueError("Warm-start resolved MLE configuration is incompatible.")
    forward_digest = sha256(
        (context.run_dir / "forward_model_manifest.json").read_bytes()
    ).hexdigest()
    expected_identities = {
        "measurement_log_schema_version": context.log.context.schema_version,
        "measurement_run_id": context.log.context.run_id,
        "measurement_repository_commit": context.log.context.repository_commit,
        "resolved_config_sha256": context.log.context.runtime_config_sha256,
        "forward_model_manifest_sha256": forward_digest,
        "resolved_estimator_config_sha256": (context.resolved_estimator_config_sha256),
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


def _causal_lineage(
    context: ReplayContext,
    warm_start: WarmStartArtifact | None,
) -> dict[str, object]:
    """Build deterministic all-history fit lineage for one replay result."""
    records = context.log.records
    records_lineage = covered_records_lineage(records)
    lineage: dict[str, object] = {
        "schema_version": 2,
        "covered_step_ids": [record.step_id for record in records],
        "data_cutoff_step": records[-1].step_id,
        "data_cutoff_station": records[-1].station_id,
        "record_count": len(records),
        **records_lineage,
        "station_boundary_attestation": _prefix_attestation(context.log),
        "fit_kind": (
            "cold_start_all_history" if warm_start is None else "warm_start_all_history"
        ),
        "warm_start": None,
    }
    if warm_start is not None:
        prior = warm_start.causal_lineage
        lineage["warm_start"] = {
            "report_sha256": warm_start.report_sha256,
            "estimate_sha256": warm_start.estimate_sha256,
            "diagnostics_sha256": warm_start.diagnostics_sha256,
            "data_cutoff_step": prior["data_cutoff_step"],
            "data_cutoff_station": prior["data_cutoff_station"],
            "record_count": prior["record_count"],
            "covered_records_digest": prior["covered_records_digest"],
            "covered_records_sha256": prior["covered_records_sha256"],
            "measurement_log_sha256": warm_start.measurement_log_sha256,
        }
    return lineage


def run_replay(
    run_dir: str | Path,
    *,
    config: MLEConfig | Mapping[str, Any] | str | Path | None = None,
    output_dir: str | Path | None = None,
    save_hook: ReplaySaveHook | None = None,
    config_source_sha256: str | None = None,
    initial_estimate_path: str | Path | None = None,
) -> ReplayResult:
    """Prepare, fit, and optionally persist one standalone replay result.

    ``initial_estimate_path`` is used only to initialize the optimizer after
    exact causal-lineage and model compatibility checks. The objective is
    always recomputed over the complete current MeasurementLog.
    """
    context = prepare_replay(
        run_dir,
        config=config,
        config_source_sha256=config_source_sha256,
    )
    _prefix_attestation(context.log)
    warm_start = (
        None
        if initial_estimate_path is None
        else validate_warm_start_artifact(context, initial_estimate_path)
    )
    estimate = SurfaceMLEEstimator(context.config).fit(
        context.batch,
        context.environment,
        context.kernel,
        obstacle_grid=context.obstacle_grid,
        initial_estimate=None if warm_start is None else warm_start.estimate,
    )
    if isinstance(estimate, MLEEstimate):
        measurement_log_digest = context.log.content_sha256
        if measurement_log_digest is None:
            raise ValueError("Loaded MeasurementLog is missing its content SHA-256.")
        forward_model_manifest_sha256 = sha256(
            (context.run_dir / "forward_model_manifest.json").read_bytes()
        ).hexdigest()
        provenance = estimator_provenance(
            variant=context.config.mode,
            measurement_log_schema_version=context.log.context.schema_version,
            measurement_run_id=context.log.context.run_id,
            measurement_repository_commit=context.log.context.repository_commit,
            resolved_config_sha256=context.log.context.runtime_config_sha256,
            forward_model_manifest_sha256=forward_model_manifest_sha256,
            measurement_log_sha256=measurement_log_digest,
            config_sha256=context.config_sha256,
            resolved_estimator_config_sha256=(context.resolved_estimator_config_sha256),
        )
        lineage = _causal_lineage(context, warm_start)
        provenance["causal_lineage"] = lineage
        estimate = replace(
            estimate,
            diagnostics={
                **estimate.diagnostics,
                "provenance": provenance,
                "causal_lineage": lineage,
                "estimator_family": provenance["estimator_family"],
                "estimator_variant": provenance["estimator_variant"],
                "candidate_domain": provenance["candidate_domain"],
                "uses_pf_state": provenance["uses_pf_state"],
                "uses_pf_candidates": provenance["uses_pf_candidates"],
                "measurement_run_id": context.log.context.run_id,
                "measurement_log_schema_version": context.log.context.schema_version,
            },
        )
    saved_output = None
    if output_dir is not None:
        hook = save_hook if save_hook is not None else _available_reporting_hook()
        if hook is not None:
            saved_output = hook(estimate, Path(output_dir))
    elif save_hook is not None:
        raise ValueError("output_dir is required when save_hook is provided.")
    return ReplayResult(
        estimate=estimate,
        context=context,
        saved_output=saved_output,
    )


__all__ = [
    "ReplayContext",
    "ReplayResult",
    "WarmStartArtifact",
    "prepare_replay",
    "run_replay",
    "validate_warm_start_artifact",
]
