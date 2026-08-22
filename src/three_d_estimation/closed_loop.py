"""Live MLE control of a private shared-runtime acquisition session."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from runtime.adaptive_client import (
    AdaptiveCandidateSnapshot,
    AdaptiveRefineRequest,
    AdaptiveRuntimeClient,
    AdaptiveStepRequest,
    candidate_index_for_pose,
)
from runtime.defaults import (
    DEFAULT_CUI_SPLIT_VIEW_HOST,
    DEFAULT_CUI_SPLIT_VIEW_PORT,
)
from runtime.measurement_log import MeasurementLogRecord

from .config import MLEConfig
from .information_planner import MLEPlanningConfig, MLEPlanningResult
from .online import OnlineMLESession
from .ral import validate_ral_measurement_log

RAL_PRIVATE_SCENE_PROFILES = (
    "ral-mix9",
    "ral-cs4-co3-eu0",
)


@dataclass(frozen=True, slots=True)
class RALClosedLoopResult:
    """Describe a completed MLE-controlled physical acquisition."""

    measurement_log_path: Path
    mle_output_dir: Path
    run_id: str
    record_count: int
    station_count: int
    stop_reason: str
    dashboard_url: str | None
    private_scene_profile: str

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-safe result payload."""
        return {
            "schema_version": 1,
            "status": "complete",
            "profile": "ral_surface_mle_closed_loop_v1",
            "private_scene_profile": self.private_scene_profile,
            "control_mode": "mle_closed_loop",
            "measurement_log_path": self.measurement_log_path.as_posix(),
            "mle_output_dir": self.mle_output_dir.as_posix(),
            "run_id": self.run_id,
            "record_count": self.record_count,
            "station_count": self.station_count,
            "stop_reason": self.stop_reason,
            "dashboard_url": self.dashboard_url,
        }


@dataclass(frozen=True, slots=True)
class MLEStopConfig:
    """Configure the compound, fail-closed MLE mission stop rule."""

    minimum_measurements: int = 12
    minimum_independent_poses: int = 4
    minimum_height_levels: int = 2
    minimum_elevation_span_rad: float = 0.35
    pose_tolerance_m: float = 1.0e-3
    height_tolerance_m: float = 1.0e-3
    stability_window: int = 3
    maximum_kkt_residual: float = 1.0e-3
    maximum_relative_deviance_range: float = 0.02
    maximum_relative_map_change: float = 0.05
    maximum_cluster_centroid_change_m: float = 0.5
    maximum_response_correlation: float = 0.98
    minimum_floor_ceiling_separation: float = 1.0e-3
    maximum_systematic_residual_z: float = 5.0
    maximum_expected_information_gain_nats: float = 1.0e-3
    low_information_patience: int = 3

    def __post_init__(self) -> None:
        """Validate every stopping threshold and coverage requirement."""
        positive_integers = {
            "minimum_measurements": self.minimum_measurements,
            "minimum_independent_poses": self.minimum_independent_poses,
            "minimum_height_levels": self.minimum_height_levels,
            "stability_window": self.stability_window,
            "low_information_patience": self.low_information_patience,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        nonnegative = {
            "minimum_elevation_span_rad": self.minimum_elevation_span_rad,
            "pose_tolerance_m": self.pose_tolerance_m,
            "height_tolerance_m": self.height_tolerance_m,
            "maximum_kkt_residual": self.maximum_kkt_residual,
            "maximum_relative_deviance_range": self.maximum_relative_deviance_range,
            "maximum_relative_map_change": self.maximum_relative_map_change,
            "maximum_cluster_centroid_change_m": (
                self.maximum_cluster_centroid_change_m
            ),
            "maximum_response_correlation": self.maximum_response_correlation,
            "minimum_floor_ceiling_separation": (self.minimum_floor_ceiling_separation),
            "maximum_systematic_residual_z": self.maximum_systematic_residual_z,
            "maximum_expected_information_gain_nats": (
                self.maximum_expected_information_gain_nats
            ),
        }
        for name, value in nonnegative.items():
            parsed = float(value)
            if not np.isfinite(parsed) or parsed < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if float(self.pose_tolerance_m) <= 0.0 or float(self.height_tolerance_m) <= 0.0:
            raise ValueError("Pose and height tolerances must be positive.")
        if not 0.0 <= float(self.maximum_response_correlation) <= 1.0:
            raise ValueError("maximum_response_correlation must lie in [0, 1].")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-safe stop configuration."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MLEStopConfig":
        """Build one strict stop configuration from a mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("MLE stop configuration must be a mapping.")
        return cls(**dict(payload))

    @classmethod
    def load(cls, path: str | Path) -> "MLEStopConfig":
        """Load one stop configuration from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("MLE stop configuration root must be an object.")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class MLEStopDecision:
    """Report whether all scientific and operational stop gates passed."""

    should_stop: bool
    gates: dict[str, bool]
    diagnostics: dict[str, object]


def _cluster_signature(estimate: object) -> tuple[tuple[str, np.ndarray], ...]:
    """Return deterministic isotope/centroid entries from MLE diagnostics."""
    diagnostics = getattr(estimate, "diagnostics", {})
    raw = (
        diagnostics.get("hotspot_clusters", []) if isinstance(diagnostics, dict) else []
    )
    result: list[tuple[str, np.ndarray]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for cluster in raw:
            if not isinstance(cluster, Mapping):
                continue
            centroid = np.asarray(cluster.get("centroid_xyz"), dtype=float)
            if centroid.shape == (3,) and np.all(np.isfinite(centroid)):
                result.append((str(cluster.get("isotope", "")), centroid))
    return tuple(sorted(result, key=lambda item: (item[0], *item[1].tolist())))


def _relative_map_change(first: object, second: object) -> float:
    """Return relative L1 change over the union of stable patch/isotope IDs."""
    first_names = tuple(getattr(first, "isotope_names"))
    second_names = tuple(getattr(second, "isotope_names"))
    if first_names != second_names:
        return float("inf")

    def values(estimate: object) -> dict[tuple[str, int], float]:
        """Map one estimate to stable isotope/patch integrated strengths."""
        patches = tuple(getattr(estimate, "patches"))
        strengths = np.asarray(getattr(estimate, "patch_strength_by_isotope"))
        return {
            (isotope, int(patch.patch_id)): float(strengths[i, g])
            for i, isotope in enumerate(first_names)
            for g, patch in enumerate(patches)
        }

    left = values(first)
    right = values(second)
    keys = set(left) | set(right)
    numerator = sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
    denominator = max(sum(abs(left.get(key, 0.0)) for key in keys), 1.0e-12)
    return float(numerator / denominator)


def _cluster_change(first: object, second: object) -> float:
    """Return worst matched centroid displacement or infinity on count change."""
    left = _cluster_signature(first)
    right = _cluster_signature(second)
    if len(left) != len(right) or [item[0] for item in left] != [
        item[0] for item in right
    ]:
        return float("inf")
    if not left:
        return 0.0
    return float(
        max(
            np.linalg.norm(first_item[1] - second_item[1])
            for first_item, second_item in zip(left, right, strict=True)
        )
    )


def _elevation_span(records: Sequence[object], estimate: object) -> float:
    """Return detector-to-estimated-mass elevation span in radians."""
    strengths = np.sum(
        np.asarray(getattr(estimate, "patch_strength_by_isotope"), dtype=float),
        axis=0,
    )
    centroids = np.vstack(
        [patch.centroid_xyz for patch in getattr(estimate, "patches")]
    )
    source = (
        np.average(centroids, axis=0, weights=strengths)
        if np.any(strengths > 0.0)
        else np.mean(centroids, axis=0)
    )
    angles = []
    for record in records:
        detector = np.asarray(record.detector_pose_xyz, dtype=float)
        delta = source - detector
        angles.append(
            float(np.arctan2(delta[2], max(np.linalg.norm(delta[:2]), 1.0e-12)))
        )
    return 0.0 if len(angles) < 2 else float(np.ptp(angles))


def _mean_deviance(estimate: object) -> float:
    """Return deviance normalized by the fitted observation-channel count."""
    prediction = getattr(estimate, "predicted_spectra", None)
    if prediction is None:
        prediction = getattr(estimate, "predicted_isotope_counts", None)
    channel_count = 1 if prediction is None else int(np.asarray(prediction).size)
    return float(getattr(estimate, "poisson_deviance")) / max(channel_count, 1)


def evaluate_mle_stop(
    estimates: Sequence[object],
    records: Sequence[object],
    plans: Sequence[MLEPlanningResult],
    config: MLEStopConfig,
) -> MLEStopDecision:
    """Evaluate all convergence, coverage, mismatch, and information gates."""
    if not isinstance(config, MLEStopConfig):
        raise TypeError("config must be MLEStopConfig.")
    if not estimates or not records or not plans:
        return MLEStopDecision(False, {"history_available": False}, {})
    latest = estimates[-1]
    diagnostics = getattr(latest, "diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    window = min(int(config.stability_window), len(estimates))
    recent = tuple(estimates[-window:])
    deviances = np.asarray([_mean_deviance(item) for item in recent])
    deviance_range = (
        float(np.ptp(deviances) / max(abs(float(np.mean(deviances))), 1.0e-12))
        if len(recent) >= int(config.stability_window)
        else float("inf")
    )
    map_change = max(
        (
            _relative_map_change(first, second)
            for first, second in zip(recent, recent[1:])
        ),
        default=float("inf"),
    )
    cluster_change = max(
        (_cluster_change(first, second) for first, second in zip(recent, recent[1:])),
        default=float("inf"),
    )
    positions = np.asarray(
        [record.detector_pose_xyz for record in records], dtype=float
    )
    pose_keys = np.round(positions / float(config.pose_tolerance_m)).astype(np.int64)
    height_keys = np.round(positions[:, 2] / float(config.height_tolerance_m)).astype(
        np.int64
    )
    independent_poses = int(np.unique(pose_keys, axis=0).shape[0])
    height_levels = int(np.unique(height_keys).size)
    identifiability = diagnostics.get("identifiability", {})
    maximum_correlation = (
        identifiability.get("maximum_column_correlation")
        if isinstance(identifiability, Mapping)
        else None
    )
    correlation = 1.0 if maximum_correlation is None else float(maximum_correlation)
    kkt = float(diagnostics.get("kkt_residual", float("inf")))
    predicted = getattr(latest, "predicted_spectra", None)
    residual_z = float("inf")
    if predicted is not None and np.asarray(predicted).shape[0] == len(records):
        observed = np.vstack([record.spectrum_counts for record in records]).astype(
            float
        )
        expected = np.maximum(np.asarray(predicted, dtype=float), 1.0)
        station_scores = []
        station_ids = sorted({int(record.station_id) for record in records})
        for station_id in station_ids:
            indices = [
                index
                for index, record in enumerate(records)
                if int(record.station_id) == station_id
            ]
            numerator = float(np.sum(observed[indices] - expected[indices]))
            denominator = float(np.sqrt(np.sum(expected[indices])))
            station_scores.append(abs(numerator) / max(denominator, 1.0e-12))
        residual_z = max(station_scores, default=float("inf"))
    recent_plans = tuple(plans[-int(config.low_information_patience) :])
    gains = [plan.selected_action.information_gain_nats for plan in recent_plans]
    floor_ceiling = float(plans[-1].selected_action.floor_ceiling_separation)
    gates = {
        "history_available": len(recent) >= int(config.stability_window),
        "mle_converged": bool(getattr(latest, "converged", False)),
        "kkt_residual": kkt <= float(config.maximum_kkt_residual),
        "minimum_measurements": len(records) >= int(config.minimum_measurements),
        "minimum_independent_poses": independent_poses
        >= int(config.minimum_independent_poses),
        "minimum_height_levels": height_levels >= int(config.minimum_height_levels),
        "minimum_elevation_span": _elevation_span(records, latest)
        >= float(config.minimum_elevation_span_rad),
        "deviance_stable": deviance_range
        <= float(config.maximum_relative_deviance_range),
        "surface_map_stable": map_change <= float(config.maximum_relative_map_change),
        "clusters_stable": cluster_change
        <= float(config.maximum_cluster_centroid_change_m),
        "correlated_hypotheses_resolved": correlation
        <= float(config.maximum_response_correlation),
        "floor_ceiling_resolved": floor_ceiling
        >= float(config.minimum_floor_ceiling_separation),
        "systematic_residual_absent": residual_z
        <= float(config.maximum_systematic_residual_z),
        "information_gain_low": len(recent_plans)
        >= int(config.low_information_patience)
        and all(
            gain <= float(config.maximum_expected_information_gain_nats)
            for gain in gains
        ),
    }
    details: dict[str, object] = {
        "kkt_residual": kkt,
        "independent_pose_count": independent_poses,
        "height_level_count": height_levels,
        "elevation_span_rad": _elevation_span(records, latest),
        "relative_deviance_range": deviance_range,
        "recent_mean_deviance_per_channel": deviances.tolist(),
        "relative_map_change": map_change,
        "cluster_centroid_change_m": cluster_change,
        "maximum_response_correlation": correlation,
        "floor_ceiling_separation": floor_ceiling,
        "maximum_station_systematic_residual_z": residual_z,
        "recent_information_gains_nats": [float(value) for value in gains],
        "config": config.to_dict(),
    }
    return MLEStopDecision(all(gates.values()), gates, details)


def run_ral_closed_loop(
    scenario_path: str | Path,
    *,
    runtime_root: str | Path,
    mle_config_path: str | Path,
    planning_config_path: str | Path,
    output_dir: str | Path,
    private_scene_profile: str = "ral-mix9",
    resume_stage_path: str | Path | None = None,
    resume_compatibility_path: str | Path | None = None,
    max_measurements: int = 256,
    minimum_information_gain_nats: float = 1.0e-3,
    low_information_patience: int = 3,
    stop_config: MLEStopConfig | None = None,
    overwrite: bool = False,
    enable_dashboard: bool = True,
    serve_dashboard: bool = True,
    dashboard_host: str = DEFAULT_CUI_SPLIT_VIEW_HOST,
    dashboard_port: int = DEFAULT_CUI_SPLIT_VIEW_PORT,
    dashboard_public_host: str | None = None,
    dashboard_url_hook: Callable[[str], None] | None = None,
    output_hook: Callable[[str], None] = print,
) -> RALClosedLoopResult:
    """Run observation, MLE fit, Fisher selection, and runtime action in a loop."""
    if private_scene_profile not in RAL_PRIVATE_SCENE_PROFILES:
        raise ValueError(
            f"private_scene_profile must be one of {RAL_PRIVATE_SCENE_PROFILES}."
        )
    if isinstance(max_measurements, bool) or int(max_measurements) < 1:
        raise ValueError("max_measurements must be a positive safety bound.")
    if isinstance(low_information_patience, bool) or int(low_information_patience) < 1:
        raise ValueError("low_information_patience must be positive.")
    threshold = float(minimum_information_gain_nats)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum_information_gain_nats must be nonnegative.")
    mle_path = Path(mle_config_path).expanduser().resolve()
    planning_path = Path(planning_config_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    mle_config = MLEConfig.load(mle_path)
    planning_config = MLEPlanningConfig.load(planning_path)
    resolved_stop = stop_config or MLEStopConfig(
        maximum_expected_information_gain_nats=threshold,
        low_information_patience=int(low_information_patience),
    )
    progress_last_elapsed: dict[str, float] = {}

    def relay_progress(progress: Mapping[str, object]) -> None:
        """Emit one compact long-running phase progress line with an ETA."""
        completed = int(
            progress.get("completed", progress.get("completed_candidates", 0))
        )
        total = int(progress.get("total", progress.get("total_candidates", 0)))
        elapsed = float(progress.get("elapsed_seconds", 0.0))
        phase = str(progress.get("phase", "unknown"))
        if phase.startswith("mle_solver_iterations:"):
            last_elapsed = progress_last_elapsed.get(phase, 0.0)
            if completed not in {0, total} and elapsed - last_elapsed < 5.0:
                return
            progress_last_elapsed[phase] = elapsed
        eta_value = progress.get("eta_seconds")
        eta = "estimating" if eta_value is None else f"{float(eta_value):.1f}s"
        percent = 0.0 if total < 1 else 100.0 * completed / total
        output_hook(
            "Progress: "
            f"phase={phase} "
            f"completed={completed}/{total} ({percent:.1f}%) "
            f"elapsed={elapsed:.1f}s eta={eta}"
        )

    def run_session(client: AdaptiveRuntimeClient) -> RALClosedLoopResult:
        """Drive one context-owned adaptive session to publication."""
        ready = client.handshake()
        context = ready.context
        candidates = ready.candidates
        bootstrap = ready.bootstrap
        resume_prefix = ready.resume
        if tuple(mle_config.isotope_names) != tuple(context.isotopes):
            raise ValueError(
                "RAL MLE isotopes must match the adaptive runtime scenario."
            )
        online = OnlineMLESession(
            context=context,
            config=mle_config,
            output_dir=target,
            run_root=runtime_root,
            config_source_sha256=sha256(mle_path.read_bytes()).hexdigest(),
            measurement_log_sha256=None,
            overwrite=overwrite,
            enable_dashboard=enable_dashboard,
            serve_dashboard=serve_dashboard,
            dashboard_host=dashboard_host,
            dashboard_port=dashboard_port,
            dashboard_public_host=dashboard_public_host,
            progress_hook=relay_progress,
        )
        if online.dashboard_url is not None and dashboard_url_hook is not None:
            dashboard_url_hook(online.dashboard_url)
        pending_program: list[tuple[int, int, float]] = []
        pending_pose: tuple[float, float, float] | None = None
        pending_station_id: int | None = None
        plan_history: list[MLEPlanningResult] = []
        stop_reason = "maximum_measurement_safety_bound"

        def plan_next_station(
            record: MeasurementLogRecord,
            snapshot: AdaptiveCandidateSnapshot,
        ) -> AdaptiveStepRequest | None:
            """Plan one station after a durable boundary and return its first view."""
            nonlocal candidates
            nonlocal pending_pose
            nonlocal pending_program
            nonlocal pending_station_id
            nonlocal stop_reason
            plan = online.plan_next_action(
                snapshot.candidate_poses_xyz,
                planning_config=planning_config,
                allowed_pair_ids=snapshot.allowed_pair_ids,
                travel_costs=snapshot.travel_costs,
                current_pair_id=snapshot.current_pair_id,
                progress_hook=relay_progress,
                screening_only=(
                    bool(getattr(planning_config, "two_stage_screening", False))
                    and int(getattr(planning_config, "local_refinement_top_k", 0)) > 0
                ),
            )
            refinement_count = int(
                getattr(planning_config, "local_refinement_top_k", 0)
            )
            if refinement_count > 0:
                seed_indices: list[int] = []
                for ranked_action in plan.ranked_actions:
                    if ranked_action.candidate_index not in seed_indices:
                        seed_indices.append(ranked_action.candidate_index)
                    if len(seed_indices) >= refinement_count:
                        break
                refined_event = client.refine_candidates(
                    AdaptiveRefineRequest.from_indices(seed_indices)
                )
                candidates = refined_event.candidates
                snapshot = candidates
                plan = online.plan_next_action(
                    snapshot.candidate_poses_xyz,
                    planning_config=planning_config,
                    allowed_pair_ids=snapshot.allowed_pair_ids,
                    travel_costs=snapshot.travel_costs,
                    current_pair_id=snapshot.current_pair_id,
                    overwrite=True,
                    progress_hook=relay_progress,
                )
            plan_history.append(plan)
            selected = plan.selected_action
            estimates = tuple(getattr(online, "station_estimates", ()))
            decision = evaluate_mle_stop(
                estimates,
                tuple(online.records),
                plan_history,
                resolved_stop,
            )
            if decision.should_stop:
                stop_reason = "compound_mle_stop_satisfied"
                return None
            remaining = int(max_measurements) - len(online.records)
            program_length = min(len(selected.shield_pair_ids), remaining)
            if program_length < 1:
                return None
            pending_program = [
                (
                    int(selected.fe_orientation_indices[index]),
                    int(selected.pb_orientation_indices[index]),
                    float(selected.live_time_s_by_view[index]),
                )
                for index in range(program_length)
            ]
            pending_pose = selected.detector_pose_xyz
            pending_station_id = int(record.station_id) + 1
            fe_index, pb_index, dwell_time = pending_program.pop(0)
            return AdaptiveStepRequest(
                candidate_index=selected.candidate_index,
                fe_orientation_index=fe_index,
                pb_orientation_index=pb_index,
                dwell_time_s=dwell_time,
                station_id=pending_station_id,
                station_complete=not pending_program,
            )

        if resume_prefix is None:
            if bootstrap is None:
                raise RuntimeError("Fresh adaptive handshake has no bootstrap.")
            request: AdaptiveStepRequest | None = AdaptiveStepRequest(
                candidate_index=bootstrap.candidate_index,
                fe_orientation_index=bootstrap.fe_orientation_index,
                pb_orientation_index=bootstrap.pb_orientation_index,
                dwell_time_s=planning_config.live_time_s,
                station_id=0,
                station_complete=True,
            )
        else:
            for prefix_record in resume_prefix.records:
                station_complete = (
                    prefix_record.metadata.get("station_complete") is True
                )
                online.receive_persisted(
                    prefix_record,
                    station_complete=station_complete,
                )
            request = (
                None
                if len(online.records) >= int(max_measurements)
                else plan_next_station(resume_prefix.records[-1], candidates)
            )

        while request is not None:
            event = client.acquire(request)
            record = event.record
            candidates = event.candidates
            station_complete = record.metadata.get("station_complete") is True
            online.receive_persisted(record, station_complete=station_complete)
            if len(online.records) >= int(max_measurements):
                break
            if pending_program:
                if pending_pose is None or pending_station_id is None:
                    raise RuntimeError(
                        "Pending station program lost its pose metadata."
                    )
                fe_index, pb_index, dwell_time = pending_program.pop(0)
                request = AdaptiveStepRequest(
                    candidate_index=candidate_index_for_pose(candidates, pending_pose),
                    fe_orientation_index=fe_index,
                    pb_orientation_index=pb_index,
                    dwell_time_s=dwell_time,
                    station_id=pending_station_id,
                    station_complete=not pending_program,
                )
                continue
            if not station_complete:
                raise RuntimeError("A runtime station ended without its final marker.")
            request = plan_next_station(record, candidates)
        published = client.finalize_log()
        log = validate_ral_measurement_log(published.path)
        if published.record_count != len(log.records):
            raise RuntimeError("Published runtime record count is inconsistent.")
        online.bind_finalized_measurement_log(log.path)
        completed = online.finalize()
        station_count = len({record.station_id for record in log.records})
        return RALClosedLoopResult(
            measurement_log_path=log.path.resolve(),
            mle_output_dir=target,
            run_id=log.run_id,
            record_count=len(log.records),
            station_count=station_count,
            stop_reason=stop_reason,
            dashboard_url=completed.dashboard_url,
            private_scene_profile=private_scene_profile,
        )

    with AdaptiveRuntimeClient(
        scenario_path,
        runtime_root=runtime_root,
        private_scene_profile=private_scene_profile,
        resume_stage_path=resume_stage_path,
        resume_compatibility_path=resume_compatibility_path,
        output_hook=output_hook,
    ) as client:
        return run_session(client)


__all__ = [
    "RALClosedLoopResult",
    "RAL_PRIVATE_SCENE_PROFILES",
    "run_ral_closed_loop",
]
