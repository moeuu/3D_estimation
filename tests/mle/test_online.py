"""Tests for station-causal online MLE publication."""

from __future__ import annotations

import argparse
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from runtime import CUIScene
from runtime.prefix import measurement_records_digest
from runtime.records import MeasurementRecord, RunContext
from three_d_estimation import MLELiveSurfaceSnapshot, online as online_module
from three_d_estimation.backend_contracts import EstimatorResult, EstimatorSnapshot
from three_d_estimation.cli import build_argument_parser
from three_d_estimation.config import MLEConfig
from three_d_estimation.information_planner import (
    MLEPlanningAction,
    MLEPlanningConfig,
    MLEPlanningResult,
)
from three_d_estimation.lineage import (
    covered_records_lineage,
    validate_covered_records_lineage,
)
from three_d_estimation.online import (
    ONLINE_STATE_FILENAME,
    OnlineMLESession,
    _dashboard_trajectory,
)
from three_d_estimation.reporting import load_mle_estimate
from three_d_estimation.types import MLEEstimate, SurfacePatch


def _context() -> RunContext:
    """Return a minimal estimator-neutral runtime context."""
    return RunContext(
        repository_commit="a" * 40,
        runtime_config={},
        environment={"size_x": 2.0, "size_y": 2.0, "size_z": 1.5},
        sim_backend="test",
        spectrum_count_method="joint_full_spectrum_generative",
        isotopes=("Cs-137",),
        obstacle_layout_path=None,
        source_layout_path=None,
        source_rate_model="detector_cps_1m",
        metadata={},
        run_id="online-test",
        source_rate_semantics={},
        forward_model_manifest={},
        runtime_config_sha256="b" * 64,
    )


def _dashboard_scene() -> CUIScene:
    """Return one explicit fake CUI scene without resolving runtime assets."""
    return CUIScene(
        bounds_min_xyz=np.zeros(3, dtype=np.float64),
        bounds_max_xyz=np.asarray([2.0, 2.0, 1.5], dtype=np.float64),
        obstacle_boxes_xyz=np.zeros((0, 6), dtype=np.float64),
    )


def _record(
    step_id: int,
    station_id: int,
    *,
    station_complete: bool,
    detector_pose_xyz: tuple[float, float, float] = (0.5, 0.5, 1.0),
    travel_waypoints_xyz: list[list[float]] | None = None,
) -> MeasurementRecord:
    """Return one finalized shared-runtime record."""
    metadata: dict[str, object] = {
        "full_spectrum_contract_hash_sha256": "c" * 64,
    }
    if station_complete:
        metadata["station_complete"] = True
    if travel_waypoints_xyz is not None:
        metadata["travel_waypoints_xyz"] = travel_waypoints_xyz
    return MeasurementRecord(
        step_id=step_id,
        action_id=step_id,
        station_id=station_id,
        detector_pose_xyz=detector_pose_xyz,
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=step_id % 8,
        pb_orientation_index=(step_id + 1) % 8,
        live_time_s=2.0,
        travel_time_s=0.1,
        shield_actuation_time_s=0.2,
        spectrum_counts=np.asarray([step_id + 1, 2], dtype=np.int64),
        energy_bin_edges_keV=np.asarray([0.0, 400.0, 800.0]),
        metadata=metadata,
    )


def test_dashboard_trajectory_preserves_runtime_route_and_station_visits() -> None:
    """Shared routes must retain waypoints and same-pose station identities."""
    segment = [
        [0.5, 0.5, 1.0],
        [0.5, 1.5, 0.25],
        [1.5, 1.5, 0.25],
        [1.5, 2.5, 1.0],
    ]
    records = (
        _record(0, 0, station_complete=True),
        _record(
            1,
            1,
            station_complete=False,
            detector_pose_xyz=(1.5, 2.5, 1.0),
            travel_waypoints_xyz=segment,
        ),
        _record(
            2,
            1,
            station_complete=True,
            detector_pose_xyz=(1.5, 2.5, 1.0),
        ),
        _record(
            3,
            2,
            station_complete=True,
            detector_pose_xyz=(1.5, 2.5, 1.0),
        ),
    )

    payload = _dashboard_trajectory(records)
    redraw_payload = _dashboard_trajectory(records)

    assert payload["travel_path_segments_xyz"] == [segment]
    assert payload["measurement_stations"] == [
        {
            "station_id": 0,
            "step_id": 0,
            "position_xyz": [0.5, 0.5, 1.0],
            "visit_count": 1,
        },
        {
            "station_id": 1,
            "step_id": 1,
            "position_xyz": [1.5, 2.5, 1.0],
            "visit_count": 2,
        },
        {
            "station_id": 2,
            "step_id": 3,
            "position_xyz": [1.5, 2.5, 1.0],
            "visit_count": 1,
        },
    ]
    assert payload["current_detector_position_xyz"] == [1.5, 2.5, 1.0]
    assert payload["latest_spectrum_counts"] == [4, 2]
    assert redraw_payload["measurement_stations"] == payload["measurement_stations"]


@pytest.mark.parametrize("tampering", ["missing-primary", "algorithm", "alias"])
def test_active_lineage_rejects_legacy_or_disagreeing_digest_fields(
    tampering: str,
) -> None:
    """Active v2 lineage must never self-downgrade to an unbound SHA alias."""
    records = (_record(0, 0, station_complete=True),)
    payload = covered_records_lineage(records)
    if tampering == "missing-primary":
        payload.pop("covered_records_digest")
    elif tampering == "algorithm":
        payload["covered_records_digest"] = {
            "algorithm": "legacy.measurement-records-v1+sha256",
            "sha256": payload["covered_records_sha256"],
        }
    else:
        payload["covered_records_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="covered_records"):
        validate_covered_records_lineage(
            payload,
            records,
            location="test lineage",
        )


def _patch() -> SurfacePatch:
    """Return one valid floor patch for deterministic fake estimates."""
    return SurfacePatch(
        patch_id=0,
        centroid_xyz=np.asarray([0.5, 0.5, 0.0]),
        normal_xyz=np.asarray([0.0, 0.0, 1.0]),
        area_m2=1.0,
        surface_kind="floor",
        object_id="room:floor",
        vertices_xyz=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        quadrature_points_xyz=np.asarray([[0.5, 0.5, 0.0]]),
        quadrature_weights=np.asarray([1.0]),
    )


def _estimate(record_count: int) -> MLEEstimate:
    """Return one shaped estimate for a buffered record prefix."""
    rows = np.arange(record_count, dtype=float)[:, None]
    return MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(_patch(),),
        density_by_isotope=np.asarray([[3.0]]),
        patch_strength_by_isotope=np.asarray([[3.0]]),
        predicted_spectra=np.hstack([rows + 1.0, rows + 2.0]),
        predicted_isotope_counts=rows + 3.0,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=float(record_count),
        poisson_deviance=0.5,
        iterations=2,
        converged=True,
        diagnostics={
            "hotspot_clusters": [
                {
                    "isotope": "Cs-137",
                    "cluster_id": np.int64(4),
                    "patch_ids": [0],
                    "centroid_xyz": [0.5, 0.5, 0.0],
                    "integrated_strength_cps_1m": 3.0,
                    "peak_density_cps_1m_m2": np.float64(3.0),
                    "surface_kinds": ["floor"],
                    "centroid_covariance_xyz_m2": np.eye(3).tolist(),
                }
            ]
        },
    )


class _FakeOnlineBackend:
    """Expose deterministic station and final all-history fits."""

    def __init__(self) -> None:
        """Initialize empty runtime state."""
        self.records: list[MeasurementRecord] = []
        self.latest_estimate: MLEEstimate | None = None
        self.finalize_calls = 0

    def initialize(self, context: RunContext) -> None:
        """Accept the runtime context without constructing physics."""
        assert context.run_id == "online-test"

    def update(self, measurement: MeasurementRecord) -> None:
        """Buffer one finalized record."""
        self.records.append(measurement)

    def on_station_complete(
        self,
        station_id: int,
        measurements: tuple[MeasurementRecord, ...],
    ) -> None:
        """Fit the complete buffered prefix at one station boundary."""
        assert measurements[-1].station_id == station_id
        self.latest_estimate = _estimate(len(self.records))

    def snapshot(self) -> EstimatorSnapshot:
        """Return a minimal estimator-neutral snapshot."""
        step_id = -1 if not self.records else self.records[-1].step_id
        return EstimatorSnapshot(
            step_id=step_id,
            source_modes_by_isotope={"Cs-137": ()},
            surface_map_by_isotope=None,
            predicted_spectrum=None,
            diagnostics={"record_count": len(self.records)},
        )

    def finalize(self) -> EstimatorResult:
        """Fit and return the final buffered history."""
        self.finalize_calls += 1
        self.latest_estimate = _estimate(len(self.records))
        return EstimatorResult(
            final_snapshot=self.snapshot(),
            diagnostics={"record_count": len(self.records)},
        )

    def plan_next_action(
        self,
        candidate_poses_xyz: object,
        **kwargs: object,
    ) -> MLEPlanningResult:
        """Return a deterministic recommendation for online publication tests."""
        del kwargs
        pose = tuple(
            float(value) for value in np.asarray(candidate_poses_xyz, dtype=float)[0]
        )
        action = MLEPlanningAction(
            candidate_index=0,
            detector_pose_xyz=pose,
            shield_pair_ids=(3,),
            fe_orientation_indices=(0,),
            pb_orientation_indices=(3,),
            information_gain_nats=0.75,
            travel_cost=0.0,
            rotation_radians=0.0,
            score=0.75,
            live_time_s_by_view=(2.0,),
            expected_total_counts_by_view=(12.0,),
        )
        return MLEPlanningResult(
            selected_action=action,
            ranked_actions=(action,),
            diagnostics={"criterion": "test"},
        )


def test_online_session_publishes_each_causal_station_and_final_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Station reports must cover only records available at their cutoff."""
    output_dir = tmp_path / "online"
    backend = _FakeOnlineBackend()
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=output_dir,
        backend=backend,
        dashboard_scene=_dashboard_scene(),
    )

    assert session.receive_persisted(_record(0, 0, station_complete=False)) is None
    first_snapshot = session.receive_persisted(_record(1, 0, station_complete=True))
    second_snapshot = session.receive_persisted(_record(2, 1, station_complete=True))
    planned = session.plan_next_action(
        np.asarray([[1.0, 0.5, 1.0]]),
        planning_config=MLEPlanningConfig(shield_program_length=1),
    )
    sealed = session.complete_live_state()
    assert session.complete_live_state() is sealed
    assert backend.finalize_calls == 1
    assert not (output_dir / "final").exists()

    run_dir = (tmp_path / "measurement-log").resolve()
    fake_log = SimpleNamespace(
        path=run_dir,
        run_id=session.context.run_id,
        context=session.context,
        records=session.records,
        content_sha256="d" * 64,
    )
    monkeypatch.setattr(online_module, "load_measurement_log", lambda path: fake_log)
    assert session.bind_finalized_measurement_log(run_dir) is fake_log
    completed = session.publish_bound_result()
    assert session.publish_bound_result() is completed
    assert backend.finalize_calls == 1

    assert first_snapshot is not None
    assert second_snapshot is not None
    assert planned.selected_action.shield_pair_ids == (3,)
    assert [report.data_cutoff_step for report in completed.station_reports] == [
        1,
        2,
    ]
    first = load_mle_estimate(completed.station_reports[0].report_paths.output_dir)
    final = load_mle_estimate(completed.final_report_paths.output_dir)
    assert first.diagnostics["online_lineage"]["schema_version"] == 2
    assert first.diagnostics["online_lineage"]["covered_step_ids"] == [0, 1]
    first_digest = measurement_records_digest(session.records[:2])
    assert first.diagnostics["online_lineage"]["covered_records_digest"] == (
        first_digest.to_payload()
    )
    assert first.diagnostics["online_lineage"]["covered_records_sha256"] == (
        first_digest.sha256
    )
    assert first.diagnostics["provenance"]["measurement_log_sha256"] is None
    assert final.diagnostics["online_lineage"]["covered_step_ids"] == [0, 1, 2]
    assert final.diagnostics["provenance"]["measurement_log_sha256"] == "d" * 64

    state = json.loads((output_dir / ONLINE_STATE_FILENAME).read_text())
    dashboard = json.loads(
        (output_dir / "dashboard_data.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "finalized"
    assert state["record_count"] == 3
    final_digest = measurement_records_digest(session.records)
    assert state["covered_records_digest"] == final_digest.to_payload()
    assert state["covered_records_sha256"] == final_digest.sha256
    assert state["station_reports"][0]["covered_records_digest"] == (
        first_digest.to_payload()
    )
    assert state["station_reports"][0]["covered_records_sha256"] == (
        first_digest.sha256
    )
    assert state["final_report_dir"] == "final"
    assert (output_dir / "index.html").is_file()
    assert dashboard["status"] == "finalized"
    assert dashboard["record_count"] == 3
    assert dashboard["latest_observed_spectrum_counts"] == [3, 2]
    assert dashboard["energy_bin_edges_keV"] == [0.0, 400.0, 800.0]
    assert [station["station_id"] for station in dashboard["measurement_stations"]] == [
        0,
        1,
    ]
    assert [
        station["visit_count"] for station in dashboard["measurement_stations"]
    ] == [2, 1]
    assert (
        dashboard["measurement_stations"][0]["position_xyz"]
        == dashboard["measurement_stations"][1]["position_xyz"]
    )
    assert dashboard["density_by_isotope"]["Cs-137"] == [3.0]
    assert dashboard["detector_positions_xyz"] == []
    assert dashboard["planning"]["selected_action"]["shield_pair_ids"] == [3]
    assert "truth" not in json.dumps(dashboard, sort_keys=True).lower()
    assert "truth" not in state
    assert "cui" not in state
    planning_path = output_dir / "planning" / "after_step_00000002.json"
    assert planning_path.is_file()
    planning = json.loads(planning_path.read_text(encoding="utf-8"))
    assert planning["diagnostics"]["data_cutoff_step"] == 2
    assert planning["diagnostics"]["covered_step_ids"] == [0, 1, 2]
    assert planning["diagnostics"]["covered_records_digest"] == (
        final_digest.to_payload()
    )
    assert planning["diagnostics"]["covered_records_sha256"] == final_digest.sha256
    assert planning["selected_action"]["measurement_program"] == [
        {
            "fe_orientation_index": 0,
            "live_time_s": 2.0,
            "pb_orientation_index": 3,
            "sequence_index": 0,
            "shield_pair_id": 3,
            "station_complete": True,
        }
    ]
    assert completed.result.artifacts["latest_mle_planning"] == str(planning_path)
    assert completed.result.artifacts["online_state"] == str(
        output_dir / ONLINE_STATE_FILENAME
    )


def test_online_session_rejects_station_marker_disagreement(
    tmp_path: Path,
) -> None:
    """The controller cannot contradict a durable runtime boundary marker."""
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=tmp_path / "online",
        backend=_FakeOnlineBackend(),
        dashboard_scene=_dashboard_scene(),
    )

    with pytest.raises(ValueError, match="disagrees"):
        session.receive_persisted(
            _record(0, 0, station_complete=True),
            station_complete=False,
        )


def test_online_session_exposes_no_truth_overlay_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Estimator-owned session and dashboard APIs must have no truth channel."""
    output_dir = tmp_path / "online"
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=output_dir,
        backend=_FakeOnlineBackend(),
        dashboard_scene=_dashboard_scene(),
    )
    assert not hasattr(session, "set_dashboard_cui_overlay")
    assert session.dashboard is not None
    assert not hasattr(session.dashboard, "set_cui_overlay")

    session.receive_persisted(_record(0, 0, station_complete=True))
    session.complete_live_state()
    run_dir = (tmp_path / "measurement-log").resolve()
    fake_log = SimpleNamespace(
        path=run_dir,
        run_id=session.context.run_id,
        context=session.context,
        records=session.records,
        content_sha256="d" * 64,
    )
    monkeypatch.setattr(online_module, "load_measurement_log", lambda path: fake_log)
    session.bind_finalized_measurement_log(run_dir)
    session.publish_bound_result()
    final_dashboard = json.loads((output_dir / "dashboard_data.json").read_text())
    state = json.loads((output_dir / ONLINE_STATE_FILENAME).read_text())

    assert final_dashboard["status"] == "finalized"
    assert "truth" not in json.dumps(final_dashboard, sort_keys=True).lower()
    assert "truth" not in json.dumps(state, sort_keys=True).lower()


def test_online_session_enforces_three_phase_finalization_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Binding and publication must surround no scientific estimator work."""
    backend = _FakeOnlineBackend()
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=tmp_path / "online",
        backend=backend,
        enable_dashboard=False,
    )
    session.receive_persisted(_record(0, 0, station_complete=True))
    run_dir = (tmp_path / "measurement-log").resolve()
    fake_log = SimpleNamespace(
        path=run_dir,
        run_id=session.context.run_id,
        context=session.context,
        records=session.records,
        content_sha256="d" * 64,
    )
    monkeypatch.setattr(online_module, "load_measurement_log", lambda path: fake_log)

    assert not hasattr(session, "finalize")
    with pytest.raises(RuntimeError, match="Complete the live estimator state"):
        session.bind_finalized_measurement_log(run_dir)
    with pytest.raises(RuntimeError, match="Complete the live estimator state"):
        session.publish_bound_result()

    session.complete_live_state()
    assert backend.finalize_calls == 1
    with pytest.raises(RuntimeError, match="Bind the finalized MeasurementLog"):
        session.publish_bound_result()

    session.bind_finalized_measurement_log(run_dir)
    session.publish_bound_result()
    assert backend.finalize_calls == 1

    with pytest.raises(RuntimeError, match="already complete"):
        session.receive_persisted(_record(1, 1, station_complete=True))
    with pytest.raises(RuntimeError, match="already complete"):
        session.plan_next_action(np.asarray([[1.0, 0.5, 1.0]]))


def test_live_surface_snapshot_copies_immutable_mle_grid_and_lineage(
    tmp_path: Path,
) -> None:
    """The live API must expose an immutable MLE grid, never PF-shaped state."""
    backend = _FakeOnlineBackend()
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=tmp_path / "online",
        backend=backend,
        enable_dashboard=False,
    )
    session.receive_persisted(_record(0, 0, station_complete=True))

    snapshot = session.live_surface_snapshot()

    assert backend.finalize_calls == 0
    assert isinstance(snapshot, MLELiveSurfaceSnapshot)
    assert snapshot.measurement_run_id == "online-test"
    assert snapshot.record_count == 1
    assert snapshot.data_cutoff_step == 0
    assert snapshot.data_cutoff_station == 0
    assert snapshot.covered_records_digest == measurement_records_digest(
        session.records
    )
    assert snapshot.covered_records_sha256 == snapshot.covered_records_digest.sha256
    assert snapshot.representation == "mle_surface_density_grid"
    assert snapshot.density_unit == "detector_cps_1m_per_m2"
    assert snapshot.isotope_names == ("Cs-137",)
    assert snapshot.patch_ids == (0,)
    assert snapshot.patch_surface_kinds == ("floor",)
    assert snapshot.patch_object_ids == ("room:floor",)
    np.testing.assert_array_equal(snapshot.patch_centroids_xyz, [[0.5, 0.5, 0.0]])
    np.testing.assert_array_equal(snapshot.density_by_isotope, [[3.0]])
    np.testing.assert_array_equal(snapshot.latest_predicted_spectrum, [1.0, 2.0])
    assert snapshot.hotspot_clusters[0]["patch_ids"] == (0,)
    assert snapshot.hotspot_clusters[0]["centroid_covariance_xyz_m2"] == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert not hasattr(snapshot, "particles")
    assert not hasattr(snapshot, "particle_weights")
    assert not np.shares_memory(
        snapshot.density_by_isotope,
        backend.latest_estimate.density_by_isotope,
    )
    assert not np.shares_memory(
        snapshot.latest_predicted_spectrum,
        backend.latest_estimate.predicted_spectra,
    )

    with pytest.raises(ValueError, match="read-only"):
        snapshot.density_by_isotope[0, 0] = 9.0
    with pytest.raises(ValueError, match="read-only"):
        snapshot.patch_centroids_xyz[0, 0] = 9.0
    with pytest.raises(TypeError):
        snapshot.hotspot_clusters[0]["isotope"] = "Co-60"
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, "record_count", 2)


def test_live_surface_snapshot_remains_available_across_finalization_phases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Completion, binding, and publication must retain the same live grid API."""
    backend = _FakeOnlineBackend()
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=tmp_path / "online",
        backend=backend,
        enable_dashboard=False,
    )
    with pytest.raises(RuntimeError, match="completed station fit"):
        session.live_surface_snapshot()
    session.receive_persisted(_record(0, 0, station_complete=False))
    with pytest.raises(RuntimeError, match="completed station fit"):
        session.live_surface_snapshot()
    session.receive_persisted(_record(1, 0, station_complete=True))
    station_snapshot = session.live_surface_snapshot()
    session.complete_live_state()
    complete_snapshot = session.live_surface_snapshot()
    assert backend.finalize_calls == 1

    run_dir = (tmp_path / "measurement-log").resolve()
    fake_log = SimpleNamespace(
        path=run_dir,
        run_id=session.context.run_id,
        context=session.context,
        records=session.records,
        content_sha256="d" * 64,
    )
    monkeypatch.setattr(online_module, "load_measurement_log", lambda path: fake_log)
    session.bind_finalized_measurement_log(run_dir)

    def reject_completed_log_read(path: Path) -> object:
        """Fail if snapshot construction reopens the completed log."""
        del path
        raise AssertionError("live_surface_snapshot must not read the completed log")

    monkeypatch.setattr(
        online_module,
        "load_measurement_log",
        reject_completed_log_read,
    )
    bound_snapshot = session.live_surface_snapshot()
    session.publish_bound_result()
    published_snapshot = session.live_surface_snapshot()

    assert station_snapshot.covered_records_digest == (
        complete_snapshot.covered_records_digest
    )
    assert bound_snapshot.covered_records_digest == (
        published_snapshot.covered_records_digest
    )
    assert backend.finalize_calls == 1


def test_live_surface_snapshot_rejects_failed_state_and_stale_prediction(
    tmp_path: Path,
) -> None:
    """Failed sessions are unreadable and stale predictions are not exposed."""

    class StalePredictionBackend(_FakeOnlineBackend):
        """Publish a valid surface estimate with no current prediction row."""

        def on_station_complete(
            self,
            station_id: int,
            measurements: tuple[MeasurementRecord, ...],
        ) -> None:
            """Fit density while leaving prediction rows behind live history."""
            del station_id, measurements
            self.latest_estimate = _estimate(0)

    stale = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=tmp_path / "stale",
        backend=StalePredictionBackend(),
        enable_dashboard=False,
    )
    stale.receive_persisted(_record(0, 0, station_complete=True))
    assert stale.live_surface_snapshot().latest_predicted_spectrum is None

    class FailingBackend(_FakeOnlineBackend):
        """Raise during a station fit to place the online session in failure."""

        def on_station_complete(
            self,
            station_id: int,
            measurements: tuple[MeasurementRecord, ...],
        ) -> None:
            """Inject a scientific station-fit failure."""
            del station_id, measurements
            raise RuntimeError("injected station fit failure")

    failed = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=tmp_path / "failed",
        backend=FailingBackend(),
        enable_dashboard=False,
    )
    with pytest.raises(RuntimeError, match="injected station fit failure"):
        failed.receive_persisted(_record(0, 0, station_complete=True))
    with pytest.raises(RuntimeError, match="failed"):
        failed.live_surface_snapshot()


@pytest.mark.parametrize("tampering", ("context", "records"))
def test_finalized_log_binding_rejects_exact_live_history_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tampering: str,
) -> None:
    """Final-log binding must reject any changed context or record digest."""
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=tmp_path / "online",
        backend=_FakeOnlineBackend(),
        enable_dashboard=False,
    )
    session.receive_persisted(_record(0, 0, station_complete=True))
    session.complete_live_state()
    run_dir = (tmp_path / "measurement-log").resolve()
    changed_context = RunContext.from_payload(
        {
            **session.context.to_payload(),
            "repository_commit": "f" * 40,
        }
    )
    fake_log = SimpleNamespace(
        path=run_dir,
        run_id=session.context.run_id,
        context=(
            changed_context if tampering == "context" else session.context
        ),
        records=(
            (_record(0, 0, station_complete=True, detector_pose_xyz=(1.5, 0.5, 1.0)),)
            if tampering == "records"
            else session.records
        ),
        content_sha256="d" * 64,
    )
    monkeypatch.setattr(online_module, "load_measurement_log", lambda path: fake_log)

    expected = "runtime context changed" if tampering == "context" else "live history"
    with pytest.raises(ValueError, match=expected):
        session.bind_finalized_measurement_log(run_dir)


def test_online_dashboard_uses_runtime_resolved_file_obstacle_scene(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Production scene construction must consume the resolved runtime obstacle."""
    obstacle_path = tmp_path / "obstacles.json"
    obstacle_path.write_text("{}\n", encoding="utf-8")
    environment = EnvironmentConfig(size_x=10.0, size_y=20.0, size_z=3.0)
    obstacle_grid = ObstacleGrid.from_dict(
        {
            "version": 1,
            "cell_size": 0.5,
            "origin": [0.25, 1.5],
            "grid_shape": [10, 12],
            "blocked_cells": [[6, 3]],
            "blocked_fraction": 1.0 / 120.0,
        }
    )

    class FakeResolvedForwardContext:
        """Return the already loaded file-backed runtime scene inputs."""

        @classmethod
        def from_run_context(
            cls,
            context: RunContext,
            *,
            run_root: str | Path,
        ) -> object:
            """Verify the asset root and expose resolved geometry."""
            del cls
            assert context.run_id == "online-test"
            assert Path(run_root) == tmp_path
            return type(
                "Resolved",
                (),
                {
                    "environment": environment,
                    "obstacle_grid": obstacle_grid,
                    "resolved_obstacle_path": obstacle_path,
                },
            )()

    monkeypatch.setattr(
        online_module,
        "ResolvedForwardContext",
        FakeResolvedForwardContext,
    )
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=tmp_path / "online",
        run_root=tmp_path,
        backend=_FakeOnlineBackend(),
    )

    assert session.dashboard is not None
    np.testing.assert_array_equal(
        session.dashboard.scene.obstacle_boxes_xyz,
        [[3.25, 3.0, 0.0, 3.75, 3.5, 2.0]],
    )


@pytest.mark.parametrize(
    "command",
    (
        "replay",
        "fit-spectrum",
        "online-replay",
        "online",
        "ral-holdout",
        "plan-next",
        "score-future",
    ),
)
def test_offline_commands_are_not_cli_surfaces(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Offline fitting, planning, and scoring commands must remain absent."""
    with pytest.raises(SystemExit) as exc_info:
        build_argument_parser().parse_args([command])
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_exposes_only_live_and_read_only_commands() -> None:
    """The installed CLI must keep one live launcher and read-only utilities."""
    parser = build_argument_parser()
    subparser_actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]

    assert len(subparser_actions) == 1
    assert set(subparser_actions[0].choices) == {
        "ral-full-simulation",
        "report",
        "forward-conformance",
    }
