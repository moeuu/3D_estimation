"""Station-causal online surface MLE over shared runtime records."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from runtime import CUIScene, DigestIdentity, ResolvedForwardContext
from runtime.cui import CUIRoute, cui_route_from_records
from runtime.defaults import DEFAULT_CUI_SPLIT_VIEW_HOST
from runtime.measurement_log import MeasurementLog, load_measurement_log
from runtime.prefix import measurement_records_digest
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
from .lineage import covered_records_lineage
from .live_snapshot import MLELiveSurfaceSnapshot
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
    covered_records_digest: DigestIdentity
    covered_records_sha256: str
    report_paths: MLEReportPaths
    report_sha256: str

    def __post_init__(self) -> None:
        """Require the transitional SHA alias to match the primary digest."""
        if not isinstance(self.covered_records_digest, DigestIdentity):
            raise TypeError("covered_records_digest must be a DigestIdentity.")
        if self.covered_records_sha256 != self.covered_records_digest.sha256:
            raise ValueError(
                "covered_records_sha256 must equal covered_records_digest.sha256."
            )

    def to_dict(self, *, relative_to: Path) -> dict[str, object]:
        """Return deterministic state-manifest data for this report."""
        return {
            "station_id": int(self.station_id),
            "data_cutoff_step": int(self.data_cutoff_step),
            "record_count": int(self.record_count),
            "covered_records_digest": self.covered_records_digest.to_payload(),
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


@dataclass(frozen=True, slots=True)
class CompletedOnlineMLEState:
    """Hold the sealed scientific result before final-log publication."""

    result: EstimatorResult
    final_estimate: MLEEstimate


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


def _dashboard_trajectory(
    records: Sequence[MeasurementRecord],
) -> dict[str, object]:
    """Return the shared runtime route payload for compatibility callers."""
    return cui_route_from_records(records).to_payload()


def _resolved_dashboard_scene(
    context: RunContext,
    *,
    run_root: Path | None,
    forward_context: ResolvedForwardContext | None = None,
) -> CUIScene:
    """Resolve one authenticated runtime scene, including file-backed obstacles."""
    resolved = forward_context
    if resolved is None:
        if run_root is None:
            raise ValueError(
                "Dashboard scene resolution requires an explicit runtime asset root."
            )
        resolved = ResolvedForwardContext.from_run_context(
            context,
            run_root=run_root,
        )
    return CUIScene.from_environment(
        resolved.environment,
        resolved.obstacle_grid,
        obstacle_height_m=min(2.0, resolved.environment.size_z),
    )


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
        dashboard_host: str = DEFAULT_CUI_SPLIT_VIEW_HOST,
        dashboard_port: int = DEFAULT_DASHBOARD_PORT,
        dashboard_public_host: str | None = None,
        dashboard_scene: CUIScene | None = None,
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
        if dashboard_scene is not None and not isinstance(dashboard_scene, CUIScene):
            raise TypeError("dashboard_scene must be a runtime CUIScene or None.")

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
        self._completed_state: CompletedOnlineMLEState | None = None
        self._bound_log: MeasurementLog | None = None
        self._final_result: OnlineMLERunResult | None = None
        self._failed = False
        self._latest_published_estimate: MLEEstimate | None = None
        self._latest_planning_state: dict[str, object] | None = None
        self._planning_paths: list[Path] = []
        resolved_dashboard_scene = dashboard_scene
        if enable_dashboard and resolved_dashboard_scene is None:
            backend_forward_context = getattr(
                active_backend,
                "forward_context",
                None,
            )
            if backend_forward_context is not None and not isinstance(
                backend_forward_context,
                ResolvedForwardContext,
            ):
                raise TypeError(
                    "backend.forward_context must be a ResolvedForwardContext or None."
                )
            resolved_dashboard_scene = _resolved_dashboard_scene(
                context,
                run_root=resolved_run_root,
                forward_context=backend_forward_context,
            )
        self.dashboard = (
            OnlineMLEDashboard(
                self.output_dir,
                scene=resolved_dashboard_scene,
                environment=context.environment,
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
        """Reject estimator work after failure or scientific completion."""
        if self._failed:
            raise RuntimeError("OnlineMLESession is failed and cannot continue.")
        if self._completed_state is not None:
            raise RuntimeError("OnlineMLESession live state is already complete.")

    def _ensure_healthy(self) -> None:
        """Reject lifecycle transitions after an estimator failure."""
        if self._failed:
            raise RuntimeError("OnlineMLESession is failed and cannot continue.")

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
        records_lineage = covered_records_lineage(records)
        lineage = {
            "schema_version": 2,
            "fit_kind": fit_kind,
            "update_policy": "station_complete_all_history",
            "covered_step_ids": [record.step_id for record in records],
            "data_cutoff_step": records[-1].step_id,
            "data_cutoff_station": records[-1].station_id,
            "record_count": len(records),
            **records_lineage,
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
        records_lineage = covered_records_lineage(records) if records else {
            "covered_records_digest": None,
            "covered_records_sha256": None,
        }
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
            **records_lineage,
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
            records = self.records
            route: CUIRoute = cui_route_from_records(records)
            self.dashboard.publish(
                self._latest_published_estimate,
                payload,
                route=route,
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
            records_digest = measurement_records_digest(records)
            self._station_reports.append(
                OnlineStationReport(
                    station_id=measurement.station_id,
                    data_cutoff_step=measurement.step_id,
                    record_count=len(records),
                    covered_records_digest=records_digest,
                    covered_records_sha256=records_digest.sha256,
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

    def complete_live_state(self) -> CompletedOnlineMLEState:
        """Run and seal the final fit before runtime log publication."""
        if self._completed_state is not None:
            return self._completed_state
        self._ensure_healthy()
        if not self.records:
            raise RuntimeError("OnlineMLESession has no measurements to complete.")
        if self._last_completed_record_count != len(self.records):
            raise RuntimeError(
                "Cannot complete before the current runtime station is complete."
            )
        try:
            base_result = self._session.finalize()
            estimate = self.backend.latest_estimate
            if not isinstance(estimate, MLEEstimate):
                raise TypeError(
                    "Online backend must expose an MLEEstimate after final fit."
                )
            completed = CompletedOnlineMLEState(
                result=base_result,
                final_estimate=estimate,
            )
            self._completed_state = completed
            return completed
        except BaseException:
            self._failed = True
            raise

    def live_surface_snapshot(self) -> MLELiveSurfaceSnapshot:
        """Copy the current station-complete MLE state for live rendering."""
        self._ensure_healthy()
        records = self.records
        if not records or self._last_completed_record_count != len(records):
            raise RuntimeError(
                "MLE live surface snapshots require a completed station fit."
            )
        estimate = (
            self._completed_state.final_estimate
            if self._completed_state is not None
            else self._latest_published_estimate
        )
        if not isinstance(estimate, MLEEstimate):
            raise RuntimeError(
                "MLE live surface snapshots require a current MLE estimate."
            )
        predicted = estimate.predicted_spectra
        prediction_covers_history = (
            predicted is not None and predicted.shape[0] == len(records)
        )
        latest = records[-1]
        return MLELiveSurfaceSnapshot.from_estimate(
            estimate,
            measurement_run_id=self.context.run_id,
            record_count=len(records),
            data_cutoff_step=latest.step_id,
            data_cutoff_station=latest.station_id,
            covered_records_digest=measurement_records_digest(records),
            include_latest_prediction=prediction_covers_history,
        )

    process_persisted_measurement = receive_persisted

    def bind_finalized_measurement_log(
        self,
        run_dir: str | Path,
    ) -> MeasurementLog:
        """Verify and bind the immutable log for the sealed live history."""
        self._ensure_healthy()
        if self._completed_state is None:
            raise RuntimeError(
                "Complete the live estimator state before binding a finalized log."
            )
        resolved_run_dir = Path(run_dir).expanduser().resolve()
        if self._bound_log is not None:
            if self._bound_log.path != resolved_run_dir:
                raise RuntimeError(
                    "OnlineMLESession is already bound to another MeasurementLog."
                )
            return self._bound_log
        log = load_measurement_log(resolved_run_dir)
        if log.run_id != self.context.run_id:
            raise ValueError("Finalized MeasurementLog belongs to another run_id.")
        if canonical_json_sha256(log.context.to_payload()) != canonical_json_sha256(
            self.context.to_payload()
        ):
            raise ValueError("Finalized MeasurementLog runtime context changed.")
        if len(log.records) != len(self.records) or (
            measurement_records_digest(log.records)
            != measurement_records_digest(self.records)
        ):
            raise ValueError(
                "Finalized MeasurementLog differs from the persisted live history."
            )
        self.measurement_log_sha256 = log.content_sha256
        self.forward_model_manifest_sha256 = _forward_manifest_sha256(log.path)
        self._bound_log = log
        return log

    def publish_bound_result(self) -> OnlineMLERunResult:
        """Annotate and serialize a sealed result bound to an immutable log."""
        if self._final_result is not None:
            return self._final_result
        self._ensure_healthy()
        if self._completed_state is None:
            raise RuntimeError(
                "Complete the live estimator state before publishing its result."
            )
        if self._bound_log is None or self.measurement_log_sha256 is None:
            raise RuntimeError(
                "Bind the finalized MeasurementLog before publishing its result."
            )
        try:
            annotated = self._online_estimate(
                self._completed_state.final_estimate,
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
            base_result = self._completed_state.result
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
            published = OnlineMLERunResult(
                result=result,
                final_estimate=annotated,
                final_report_paths=final_paths,
                station_reports=tuple(self._station_reports),
                state_path=state_path,
                dashboard_url=self.dashboard_url,
            )
            self._final_result = published
            return published
        except BaseException:
            self._failed = True
            raise

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
        records_lineage = covered_records_lineage(self.records)
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
                **records_lineage,
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


__all__ = [
    "CompletedOnlineMLEState",
    "ONLINE_STATE_FILENAME",
    "MLELiveSurfaceSnapshot",
    "OnlineMLERunResult",
    "OnlineMLESession",
    "OnlineStationReport",
]
