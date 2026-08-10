"""Station-causal online surface MLE over shared runtime records."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from runtime.measurement_log import MeasurementLog, load_measurement_log
from runtime.prefix import measurement_records_sha256
from runtime.records import MeasurementRecord, RunContext, canonical_json_sha256

from .backend_contracts import EstimatorResult, EstimatorSnapshot
from .config import MLEConfig
from .dashboard import (
    DEFAULT_DASHBOARD_PORT,
    OnlineMLEDashboard,
    ensure_dashboard_server,
)
from .estimator_backend import SurfaceMLEBackend
from .information_planner import (
    MLEPlanningConfig,
    MLEPlanningResult,
    save_mle_planning_result,
)
from .provenance import estimator_provenance
from .reporting import (
    MLEReportPaths,
    mle_report_sha256,
    save_mle_estimate,
)
from .session import LiveEstimationSession
from .types import MLEEstimate

ONLINE_STATE_FILENAME = "online_state.json"


class OnlineBackend:
    """Structural surface needed from an online MLE backend."""

    latest_estimate: MLEEstimate | None

    def initialize(self, context: RunContext) -> None:
        """Initialize the backend for one runtime context."""
        raise NotImplementedError

    def update(self, measurement: MeasurementRecord) -> None:
        """Consume one already persisted runtime record."""
        raise NotImplementedError

    def on_station_complete(
        self,
        station_id: int,
        measurements: tuple[MeasurementRecord, ...],
    ) -> None:
        """Fit the station-complete all-history prefix."""
        raise NotImplementedError

    def snapshot(self) -> EstimatorSnapshot:
        """Return the latest estimator-neutral snapshot."""
        raise NotImplementedError

    def finalize(self) -> EstimatorResult:
        """Return the final all-history result."""
        raise NotImplementedError


SaveReportHook = Callable[[MLEEstimate, Path, MLEConfig], MLEReportPaths]


@dataclass(frozen=True, slots=True)
class OnlineStationReport:
    """Describe one durably published station-complete MLE report."""

    station_id: int
    data_cutoff_step: int
    record_count: int
    covered_records_sha256: str
    report_paths: MLEReportPaths
    report_sha256: str

    def to_dict(self, *, relative_to: Path) -> dict[str, object]:
        """Return deterministic state-manifest data for this report."""
        return {
            "station_id": int(self.station_id),
            "data_cutoff_step": int(self.data_cutoff_step),
            "record_count": int(self.record_count),
            "covered_records_sha256": self.covered_records_sha256,
            "report_dir": self.report_paths.output_dir.relative_to(
                relative_to
            ).as_posix(),
            "report_sha256": self.report_sha256,
        }


@dataclass(frozen=True, slots=True)
class OnlineMLERunResult:
    """Return station reports and the final online all-history estimate."""

    result: EstimatorResult
    final_estimate: MLEEstimate
    final_report_paths: MLEReportPaths
    station_reports: tuple[OnlineStationReport, ...]
    state_path: Path
    dashboard_url: str | None


def _save_report(
    estimate: MLEEstimate,
    output_dir: Path,
    config: MLEConfig,
) -> MLEReportPaths:
    """Persist one online report through the canonical MLE writer."""
    return save_mle_estimate(estimate, output_dir, config=config)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Durably replace one strict deterministic JSON state file."""
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Online state staging file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _forward_manifest_sha256(run_root: Path | None) -> str | None:
    """Return the exact finalized forward-manifest hash when available."""
    if run_root is None:
        return None
    path = run_root / "forward_model_manifest.json"
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _resolved_config(
    config: MLEConfig | Mapping[str, Any] | str | Path | None,
    log: MeasurementLog,
) -> tuple[MLEConfig, str]:
    """Resolve online configuration and its caller-visible source digest."""
    if config is None:
        resolved = MLEConfig(
            mode="spectral",
            isotope_names=log.context.isotopes,
        )
        source_sha256 = canonical_json_sha256(resolved.to_dict())
    elif isinstance(config, MLEConfig):
        resolved = config
        source_sha256 = canonical_json_sha256(resolved.to_dict())
    elif isinstance(config, Mapping):
        resolved = MLEConfig.from_dict(config)
        source_sha256 = canonical_json_sha256(dict(config))
    elif isinstance(config, (str, Path)):
        path = Path(config)
        resolved = MLEConfig.load(path)
        source_sha256 = sha256(path.read_bytes()).hexdigest()
    else:
        raise TypeError("config must be MLEConfig, a mapping, a JSON path, or None.")
    if tuple(resolved.isotope_names) != tuple(log.context.isotopes):
        raise ValueError(
            "Online MLE isotope_names must exactly match the runtime log order."
        )
    return resolved, source_sha256


def _station_boundary(
    records: Sequence[MeasurementRecord],
    index: int,
) -> bool:
    """Validate and return whether one record closes its runtime station."""
    record = records[index]
    expected = index + 1 == len(records) or (
        records[index + 1].station_id != record.station_id
    )
    marker = record.metadata.get("station_complete")
    if marker is not None and not isinstance(marker, bool):
        raise ValueError("Runtime station_complete metadata must be boolean.")
    actual = marker is True
    if actual != expected:
        raise ValueError(
            "Runtime records must carry one station_complete=true marker on "
            "the final record of every station."
        )
    return expected


def _dashboard_trajectory(
    records: Sequence[MeasurementRecord],
) -> dict[str, object]:
    """Return PF-compatible route and station history from runtime records."""
    path_segments: list[list[list[float]]] = []
    measurement_stations: list[dict[str, object]] = []
    for record in records:
        raw_waypoints = record.metadata.get("travel_waypoints_xyz")
        if raw_waypoints is not None:
            waypoints = np.asarray(raw_waypoints, dtype=np.float64)
            if (
                waypoints.ndim == 2
                and waypoints.shape[1] == 3
                and waypoints.shape[0] >= 2
                and np.all(np.isfinite(waypoints))
            ):
                segment = waypoints.tolist()
                if not path_segments or segment != path_segments[-1]:
                    path_segments.append(segment)

        point = np.asarray(record.detector_pose_xyz, dtype=np.float64).reshape(3)
        if measurement_stations:
            latest = measurement_stations[-1]
            latest_point = np.asarray(latest["position_xyz"], dtype=np.float64)
            if float(np.linalg.norm(point - latest_point)) <= 1.0e-6:
                latest["visit_count"] = int(latest["visit_count"]) + 1
                continue
        measurement_stations.append(
            {
                "station_id": int(record.station_id),
                "position_xyz": point.tolist(),
                "visit_count": 1,
            }
        )
    return {
        "travel_path_segments_xyz": path_segments,
        "measurement_stations": measurement_stations,
        "current_detector_position_xyz": (
            None if not records else list(map(float, records[-1].detector_pose_xyz))
        ),
    }


class OnlineMLESession:
    """Update and publish an all-history MLE at durable station boundaries.

    The caller must pass records only after the shared runtime has durably staged
    them. This class never creates observations or writes a MeasurementLog.
    """

    def __init__(
        self,
        *,
        context: RunContext,
        config: MLEConfig,
        output_dir: str | Path,
        run_root: str | Path | None = None,
        config_source_sha256: str | None = None,
        measurement_log_sha256: str | None = None,
        backend: OnlineBackend | None = None,
        save_report_hook: SaveReportHook = _save_report,
        overwrite: bool = False,
        enable_dashboard: bool = True,
        serve_dashboard: bool = False,
        dashboard_host: str = "0.0.0.0",
        dashboard_port: int = DEFAULT_DASHBOARD_PORT,
        dashboard_public_host: str | None = None,
        dashboard_cui_overlay: Mapping[str, object] | None = None,
        progress_hook: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        """Initialize one station-causal session and its output directory."""
        if not isinstance(context, RunContext):
            raise TypeError("context must be a shared runtime RunContext.")
        if not isinstance(config, MLEConfig):
            raise TypeError("config must be MLEConfig.")
        if tuple(config.isotope_names) != tuple(context.isotopes):
            raise ValueError(
                "MLEConfig isotope_names must exactly match RunContext isotopes."
            )
        if not callable(save_report_hook):
            raise TypeError("save_report_hook must be callable.")

        target = Path(output_dir).resolve()
        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Online MLE output exists; pass overwrite=True: {target}"
                )
            if not target.is_dir():
                raise NotADirectoryError(
                    f"Online MLE output is not a directory: {target}"
                )
            shutil.rmtree(target)
        target.mkdir(parents=True)

        resolved_run_root = None if run_root is None else Path(run_root).resolve()
        active_backend = backend or SurfaceMLEBackend(
            config,
            run_root=resolved_run_root,
            progress_hook=progress_hook,
        )
        self.context = context
        self.config = config
        self.output_dir = target
        self.run_root = resolved_run_root
        self.config_source_sha256 = (
            canonical_json_sha256(config.to_dict())
            if config_source_sha256 is None
            else str(config_source_sha256)
        )
        self.resolved_estimator_config_sha256 = canonical_json_sha256(config.to_dict())
        self.measurement_log_sha256 = measurement_log_sha256
        self.forward_model_manifest_sha256 = _forward_manifest_sha256(resolved_run_root)
        self.backend = active_backend
        self._save_report_hook = save_report_hook
        self._session = LiveEstimationSession(
            context=context,
            backend=active_backend,
        )
        self._station_reports: list[OnlineStationReport] = []
        self._station_estimates: list[MLEEstimate] = []
        self._last_completed_record_count = 0
        self._final_result: OnlineMLERunResult | None = None
        self._failed = False
        self._latest_published_estimate: MLEEstimate | None = None
        self._latest_planning_state: dict[str, object] | None = None
        self._planning_paths: list[Path] = []
        self.dashboard = (
            OnlineMLEDashboard(
                self.output_dir,
                environment=context.environment,
                cui_overlay=dashboard_cui_overlay,
            )
            if enable_dashboard
            else None
        )
        self.dashboard_url = (
            ensure_dashboard_server(
                self.output_dir,
                host=dashboard_host,
                port=dashboard_port,
                public_host=dashboard_public_host,
            )
            if serve_dashboard and self.dashboard is not None
            else None
        )
        self._persist_state(status="running")

    @property
    def records(self) -> tuple[MeasurementRecord, ...]:
        """Return all records accepted from the durable runtime stream."""
        return self._session.records

    @property
    def station_reports(self) -> tuple[OnlineStationReport, ...]:
        """Return reports published at completed stations."""
        return tuple(self._station_reports)

    @property
    def station_estimates(self) -> tuple[MLEEstimate, ...]:
        """Return station-complete estimates in causal publication order."""
        return tuple(self._station_estimates)

    @property
    def latest_estimate(self) -> MLEEstimate | None:
        """Return the latest truth-free MLE estimate available for control."""
        estimate = self.backend.latest_estimate
        return estimate if isinstance(estimate, MLEEstimate) else None

    def _ensure_active(self) -> None:
        """Reject work after failure or finalization."""
        if self._failed:
            raise RuntimeError("OnlineMLESession is failed and cannot continue.")
        if self._final_result is not None:
            raise RuntimeError("OnlineMLESession is already finalized.")

    def _online_estimate(
        self,
        estimate: MLEEstimate,
        *,
        fit_kind: str,
        include_final_log_hash: bool,
    ) -> MLEEstimate:
        """Attach runtime identity and causal prefix lineage to an estimate."""
        records = self.records
        if not records:
            raise RuntimeError("Online lineage requires at least one record.")
        lineage = {
            "schema_version": 1,
            "fit_kind": fit_kind,
            "update_policy": "station_complete_all_history",
            "covered_step_ids": [record.step_id for record in records],
            "data_cutoff_step": records[-1].step_id,
            "data_cutoff_station": records[-1].station_id,
            "record_count": len(records),
            "covered_records_sha256": measurement_records_sha256(records),
        }
        provenance = estimator_provenance(
            variant=self.config.mode,
            measurement_log_schema_version=self.context.schema_version,
            measurement_run_id=self.context.run_id,
            measurement_repository_commit=self.context.repository_commit,
            resolved_config_sha256=self.context.runtime_config_sha256,
            forward_model_manifest_sha256=(self.forward_model_manifest_sha256),
            measurement_log_sha256=(
                self.measurement_log_sha256 if include_final_log_hash else None
            ),
            config_sha256=self.config_source_sha256,
            resolved_estimator_config_sha256=(self.resolved_estimator_config_sha256),
        )
        provenance["execution_mode"] = "online_station_complete"
        provenance["online_lineage"] = lineage
        return replace(
            estimate,
            diagnostics={
                **estimate.diagnostics,
                "provenance": provenance,
                "online_lineage": lineage,
                "estimator_family": provenance["estimator_family"],
                "estimator_variant": provenance["estimator_variant"],
                "candidate_domain": provenance["candidate_domain"],
                "uses_pf_state": provenance["uses_pf_state"],
                "uses_pf_candidates": provenance["uses_pf_candidates"],
                "measurement_run_id": self.context.run_id,
                "measurement_log_schema_version": self.context.schema_version,
            },
        )

    def _state_payload(
        self,
        *,
        status: str,
        final_report: MLEReportPaths | None = None,
    ) -> dict[str, object]:
        """Build the current durable online state manifest."""
        records = self.records
        return {
            "schema_version": 1,
            "status": status,
            "execution_mode": "online_station_complete",
            "update_policy": "station_complete_all_history",
            "run_id": self.context.run_id,
            "measurement_log_schema_version": self.context.schema_version,
            "measurement_repository_commit": self.context.repository_commit,
            "source_rate_model": self.context.source_rate_model,
            "isotopes": list(self.context.isotopes),
            "mode": self.config.mode,
            "config_sha256": self.config_source_sha256,
            "resolved_estimator_config_sha256": (self.resolved_estimator_config_sha256),
            "measurement_log_sha256": self.measurement_log_sha256,
            "record_count": len(records),
            "latest_step_id": None if not records else records[-1].step_id,
            "latest_station_id": None if not records else records[-1].station_id,
            "station_reports": [
                report.to_dict(relative_to=self.output_dir)
                for report in self._station_reports
            ],
            "final_report_dir": (
                None
                if final_report is None
                else final_report.output_dir.relative_to(self.output_dir).as_posix()
            ),
            "dashboard_url": self.dashboard_url,
            "latest_planning": self._latest_planning_state,
        }

    def _persist_state(
        self,
        *,
        status: str,
        final_report: MLEReportPaths | None = None,
    ) -> None:
        """Publish the latest station progress or finalized state."""
        payload = self._state_payload(status=status, final_report=final_report)
        _write_json_atomic(
            self.output_dir / ONLINE_STATE_FILENAME,
            payload,
        )
        if self.dashboard is not None:
            dashboard_payload = dict(payload)
            records = self.records
            dashboard_payload.update(_dashboard_trajectory(records))
            if records:
                dashboard_payload["latest_observed_spectrum_counts"] = records[
                    -1
                ].spectrum_counts.tolist()
                dashboard_payload["energy_bin_edges_keV"] = records[
                    -1
                ].energy_bin_edges_keV.tolist()
            self.dashboard.publish(
                self._latest_published_estimate,
                dashboard_payload,
            )

    def receive_persisted(
        self,
        measurement: MeasurementRecord,
        *,
        station_complete: bool | None = None,
    ) -> EstimatorSnapshot | None:
        """Consume one durable runtime record and fit at a station boundary."""
        self._ensure_active()
        marker = measurement.metadata.get("station_complete")
        if marker is not None and not isinstance(marker, bool):
            raise ValueError("station_complete record metadata must be boolean.")
        if station_complete is None:
            complete = marker is True
        elif not isinstance(station_complete, bool):
            raise TypeError("station_complete must be boolean or None.")
        else:
            complete = station_complete
        if marker is not None and marker is not complete:
            raise ValueError(
                "station_complete argument disagrees with durable record metadata."
            )

        try:
            snapshot = self._session.receive(
                measurement,
                station_complete=complete,
            )
            if not complete:
                return snapshot
            estimate = self.backend.latest_estimate
            if not isinstance(estimate, MLEEstimate):
                raise TypeError(
                    "Online backend must expose an MLEEstimate after station fit."
                )
            annotated = self._online_estimate(
                estimate,
                fit_kind="online_station_complete",
                include_final_log_hash=False,
            )
            self._latest_published_estimate = annotated
            self._station_estimates.append(annotated)
            report_dir = (
                self.output_dir
                / "stations"
                / (
                    f"station_{measurement.station_id:06d}_"
                    f"step_{measurement.step_id:08d}"
                )
            )
            report_paths = self._save_report_hook(
                annotated,
                report_dir,
                self.config,
            )
            records = self.records
            self._station_reports.append(
                OnlineStationReport(
                    station_id=measurement.station_id,
                    data_cutoff_step=measurement.step_id,
                    record_count=len(records),
                    covered_records_sha256=measurement_records_sha256(records),
                    report_paths=report_paths,
                    report_sha256=mle_report_sha256(report_paths.output_dir),
                )
            )
            self._last_completed_record_count = len(records)
            self._persist_state(status="running")
            return snapshot
        except BaseException:
            self._failed = True
            raise

    def finalize(self) -> OnlineMLERunResult:
        """Fit, publish, and return the final complete online history."""
        if self._final_result is not None:
            return self._final_result
        self._ensure_active()
        if not self.records:
            raise RuntimeError("OnlineMLESession has no measurements to finalize.")
        if self._last_completed_record_count != len(self.records):
            raise RuntimeError(
                "Cannot finalize before the current runtime station is complete."
            )
        try:
            base_result = self._session.finalize()
            estimate = self.backend.latest_estimate
            if not isinstance(estimate, MLEEstimate):
                raise TypeError(
                    "Online backend must expose an MLEEstimate after final fit."
                )
            annotated = self._online_estimate(
                estimate,
                fit_kind="online_final_all_history",
                include_final_log_hash=True,
            )
            self._latest_published_estimate = annotated
            final_paths = self._save_report_hook(
                annotated,
                self.output_dir / "final",
                self.config,
            )
            state_path = self.output_dir / ONLINE_STATE_FILENAME
            result = EstimatorResult(
                final_snapshot=base_result.final_snapshot,
                diagnostics={
                    **base_result.diagnostics,
                    "execution_mode": "online_station_complete",
                    "station_report_count": len(self._station_reports),
                    "measurement_log_sha256": self.measurement_log_sha256,
                },
                artifacts={
                    **base_result.artifacts,
                    "online_state": str(state_path),
                    "final_mle_report": str(final_paths.output_dir),
                    **(
                        {}
                        if self.dashboard is None
                        else {"dashboard": str(self.dashboard.index_path)}
                    ),
                    **(
                        {}
                        if not self._planning_paths
                        else {"latest_mle_planning": str(self._planning_paths[-1])}
                    ),
                },
            )
            self._persist_state(status="finalized", final_report=final_paths)
            completed = OnlineMLERunResult(
                result=result,
                final_estimate=annotated,
                final_report_paths=final_paths,
                station_reports=tuple(self._station_reports),
                state_path=state_path,
                dashboard_url=self.dashboard_url,
            )
            self._final_result = completed
            return completed
        except BaseException:
            self._failed = True
            raise

    process_persisted_measurement = receive_persisted

    def bind_finalized_measurement_log(
        self,
        run_dir: str | Path,
    ) -> MeasurementLog:
        """Bind and verify the immutable log produced by this live session."""
        self._ensure_active()
        log = load_measurement_log(Path(run_dir).expanduser().resolve())
        if log.run_id != self.context.run_id:
            raise ValueError("Finalized MeasurementLog belongs to another run_id.")
        if log.context.runtime_config_sha256 != self.context.runtime_config_sha256:
            raise ValueError("Finalized MeasurementLog runtime context changed.")
        if len(log.records) != len(self.records) or (
            measurement_records_sha256(log.records)
            != measurement_records_sha256(self.records)
        ):
            raise ValueError(
                "Finalized MeasurementLog differs from the persisted live history."
            )
        self.measurement_log_sha256 = log.content_sha256
        self.forward_model_manifest_sha256 = _forward_manifest_sha256(log.path)
        return log

    def plan_next_action(
        self,
        candidate_poses_xyz: object,
        *,
        planning_config: MLEPlanningConfig | None = None,
        allowed_pair_ids: Sequence[int] | None = None,
        travel_costs: object | None = None,
        current_pair_id: int | None = None,
        overwrite: bool = False,
        progress_hook: Callable[[Mapping[str, object]], None] | None = None,
        screening_only: bool = False,
    ) -> MLEPlanningResult:
        """Plan and publish the next runtime action after a completed station."""
        self._ensure_active()
        if not self.records or self._last_completed_record_count != len(self.records):
            raise RuntimeError(
                "Online MLE planning requires a current station-complete fit."
            )
        planner = getattr(self.backend, "plan_next_action", None)
        if not callable(planner):
            raise TypeError("Online backend does not provide MLE action planning.")
        planned = planner(
            candidate_poses_xyz,
            planning_config=planning_config,
            allowed_pair_ids=allowed_pair_ids,
            travel_costs=travel_costs,
            current_pair_id=current_pair_id,
            progress_hook=progress_hook,
            screening_only=screening_only,
        )
        if not isinstance(planned, MLEPlanningResult):
            raise TypeError("Online backend planning must return MLEPlanningResult.")
        latest = self.records[-1]
        annotated = MLEPlanningResult(
            selected_action=planned.selected_action,
            ranked_actions=planned.ranked_actions,
            diagnostics={
                **planned.diagnostics,
                "measurement_run_id": self.context.run_id,
                "data_cutoff_step": int(latest.step_id),
                "data_cutoff_station": int(latest.station_id),
                "record_count": len(self.records),
                "covered_step_ids": [int(record.step_id) for record in self.records],
                "covered_records_sha256": measurement_records_sha256(self.records),
                "resolved_estimator_config_sha256": (
                    self.resolved_estimator_config_sha256
                ),
            },
        )
        path = self.output_dir / "planning" / f"after_step_{latest.step_id:08d}.json"
        save_mle_planning_result(annotated, path, overwrite=overwrite)
        if path not in self._planning_paths:
            self._planning_paths.append(path)
        self._latest_planning_state = {
            "planning_method": annotated.to_dict()["planning_method"],
            "data_cutoff_step": int(latest.step_id),
            "path": path.relative_to(self.output_dir).as_posix(),
            "selected_action": annotated.selected_action.to_dict(),
            "preliminary_screening": bool(screening_only),
        }
        self._persist_state(status="running")
        return annotated


def run_online_replay(
    run_dir: str | Path,
    *,
    config: MLEConfig | Mapping[str, Any] | str | Path | None = None,
    output_dir: str | Path,
    overwrite: bool = False,
    enable_dashboard: bool = True,
    serve_dashboard: bool = False,
    dashboard_host: str = "0.0.0.0",
    dashboard_port: int = DEFAULT_DASHBOARD_PORT,
    dashboard_public_host: str | None = None,
    dashboard_url_hook: Callable[[str], None] | None = None,
) -> OnlineMLERunResult:
    """Replay a finalized runtime log through the live station-update path."""
    resolved_run_dir = Path(run_dir).resolve()
    log = load_measurement_log(resolved_run_dir)
    resolved_config, config_source_sha256 = _resolved_config(config, log)
    if log.content_sha256 is None:
        raise ValueError("Runtime MeasurementLog is missing its content SHA-256.")
    session = OnlineMLESession(
        context=log.context,
        config=resolved_config,
        output_dir=output_dir,
        run_root=resolved_run_dir,
        config_source_sha256=config_source_sha256,
        measurement_log_sha256=log.content_sha256,
        overwrite=overwrite,
        enable_dashboard=enable_dashboard,
        serve_dashboard=serve_dashboard,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        dashboard_public_host=dashboard_public_host,
    )
    if session.dashboard_url is not None and dashboard_url_hook is not None:
        dashboard_url_hook(session.dashboard_url)
    for index, record in enumerate(log.records):
        session.receive_persisted(
            record,
            station_complete=_station_boundary(log.records, index),
        )
    return session.finalize()


__all__ = [
    "ONLINE_STATE_FILENAME",
    "OnlineMLERunResult",
    "OnlineMLESession",
    "OnlineStationReport",
    "run_online_replay",
]
