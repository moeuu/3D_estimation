"""Tests for station-causal online MLE publication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from runtime.records import MeasurementRecord, RunContext
from three_d_estimation.backend_contracts import EstimatorResult, EstimatorSnapshot
from three_d_estimation.cli import build_argument_parser
from three_d_estimation.config import MLEConfig
from three_d_estimation.information_planner import (
    MLEPlanningAction,
    MLEPlanningConfig,
    MLEPlanningResult,
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
    """The MLE CUI must use runtime waypoints instead of obstacle-crossing chords."""
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
    )

    payload = _dashboard_trajectory(records)

    assert payload["travel_path_segments_xyz"] == [segment]
    assert payload["measurement_stations"] == [
        {
            "station_id": 0,
            "position_xyz": [0.5, 0.5, 1.0],
            "visit_count": 1,
        },
        {
            "station_id": 1,
            "position_xyz": [1.5, 2.5, 1.0],
            "visit_count": 2,
        },
    ]
    assert payload["current_detector_position_xyz"] == [1.5, 2.5, 1.0]


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
        diagnostics={"hotspot_clusters": []},
    )


class _FakeOnlineBackend:
    """Expose deterministic station and final all-history fits."""

    def __init__(self) -> None:
        """Initialize empty runtime state."""
        self.records: list[MeasurementRecord] = []
        self.latest_estimate: MLEEstimate | None = None

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
    tmp_path: Path,
) -> None:
    """Station reports must cover only records available at their cutoff."""
    output_dir = tmp_path / "online"
    session = OnlineMLESession(
        context=_context(),
        config=MLEConfig(mode="spectral", isotope_names=("Cs-137",)),
        output_dir=output_dir,
        backend=_FakeOnlineBackend(),
        measurement_log_sha256="d" * 64,
        dashboard_cui_overlay={
            "type": "cui_overlay",
            "schema_version": 1,
            "truth": {
                "schema_version": 1,
                "semantics": "evaluation_cui_overlay_only_not_estimator_input",
                "true_sources": {"Cs-137": [[0.5, 0.5, 0.0]]},
                "true_strengths": {"Cs-137": [1200.0]},
            },
        },
    )

    assert session.receive_persisted(_record(0, 0, station_complete=False)) is None
    first_snapshot = session.receive_persisted(_record(1, 0, station_complete=True))
    second_snapshot = session.receive_persisted(_record(2, 1, station_complete=True))
    planned = session.plan_next_action(
        np.asarray([[1.0, 0.5, 1.0]]),
        planning_config=MLEPlanningConfig(shield_program_length=1),
    )
    completed = session.finalize()

    assert first_snapshot is not None
    assert second_snapshot is not None
    assert planned.selected_action.shield_pair_ids == (3,)
    assert [report.data_cutoff_step for report in completed.station_reports] == [
        1,
        2,
    ]
    first = load_mle_estimate(completed.station_reports[0].report_paths.output_dir)
    final = load_mle_estimate(completed.final_report_paths.output_dir)
    assert first.diagnostics["online_lineage"]["covered_step_ids"] == [0, 1]
    assert first.diagnostics["provenance"]["measurement_log_sha256"] is None
    assert final.diagnostics["online_lineage"]["covered_step_ids"] == [0, 1, 2]
    assert final.diagnostics["provenance"]["measurement_log_sha256"] == "d" * 64

    state = json.loads((output_dir / ONLINE_STATE_FILENAME).read_text())
    dashboard = json.loads(
        (output_dir / "dashboard_data.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "finalized"
    assert state["record_count"] == 3
    assert state["final_report_dir"] == "final"
    assert (output_dir / "index.html").is_file()
    assert dashboard["status"] == "finalized"
    assert dashboard["record_count"] == 3
    assert dashboard["latest_observed_spectrum_counts"] == [3, 2]
    assert dashboard["energy_bin_edges_keV"] == [0.0, 400.0, 800.0]
    assert dashboard["density_by_isotope"]["Cs-137"] == [3.0]
    assert dashboard["detector_positions_xyz"] == []
    assert dashboard["planning"]["selected_action"]["shield_pair_ids"] == [3]
    assert dashboard["cui"]["truth"]["true_sources"]["Cs-137"] == [[0.5, 0.5, 0.0]]
    assert "truth" not in state
    assert "cui" not in state
    planning_path = output_dir / "planning" / "after_step_00000002.json"
    assert planning_path.is_file()
    planning = json.loads(planning_path.read_text(encoding="utf-8"))
    assert planning["diagnostics"]["data_cutoff_step"] == 2
    assert planning["diagnostics"]["covered_step_ids"] == [0, 1, 2]
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
    )

    with pytest.raises(ValueError, match="disagrees"):
        session.receive_persisted(
            _record(0, 0, station_complete=True),
            station_complete=False,
        )


def test_online_cli_serves_dashboard_by_default() -> None:
    """The PF-style online command must expose explicit URL controls."""
    args = build_argument_parser().parse_args(
        ["online-replay", "--run-dir", "/tmp/runtime-log"]
    )

    assert args.no_dashboard is False
    assert args.no_serve is False
    assert args.dashboard_host == "0.0.0.0"
    assert args.dashboard_port == 8878
    assert args.dashboard_public_host is None
