"""Tests for live MLE-to-runtime closed-loop protocol handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from runtime.adaptive_client import (
    AdaptiveCandidatesEvent,
    AdaptivePublishedEvent,
    AdaptiveReadyEvent,
    AdaptiveRecordEvent,
    AdaptiveRefineRequest,
    AdaptiveStepRequest,
    adaptive_step_request,
    parse_adaptive_record,
    parse_candidate_snapshot,
    parse_run_context,
)
from runtime.measurement_log import MeasurementLogValidationError

from three_d_estimation.closed_loop import (
    MLEStopConfig,
    RALClosedLoopResult,
    evaluate_mle_stop,
    run_ral_closed_loop,
)
from three_d_estimation.information_planner import (
    MLEPlanningAction,
    MLEPlanningResult,
)


def _context_payload() -> dict[str, object]:
    """Return one minimal truth-free adaptive handshake context."""
    return {
        "repository_commit": "a" * 40,
        "runtime_config": {},
        "environment": {"size_x": 2.0, "size_y": 2.0, "size_z": 1.5},
        "sim_backend": "geant4",
        "spectrum_count_method": "joint_full_spectrum_generative",
        "isotopes": ["Co-60", "Cs-137", "Eu-154"],
        "obstacle_layout_path": None,
        "source_rate_model": "detector_cps_1m",
        "metadata": {},
        "run_id": "adaptive-test",
        "source_rate_semantics": {},
        "forward_model_manifest": {},
        "runtime_config_sha256": "b" * 64,
        "schema_version": 2,
    }


def test_live_context_has_no_source_layout_or_precomputed_actions() -> None:
    """Estimator input should contain only truth-free context, never a plan."""
    context = parse_run_context(_context_payload())

    assert context.run_id == "adaptive-test"
    assert context.source_layout_path is None
    assert "actions" not in _context_payload()


def test_live_context_rejects_realized_source_truth() -> None:
    """A private source realization must not cross the runtime boundary."""
    payload = _context_payload()
    payload["sources"] = [{"position": [1.0, 1.0, 0.0]}]

    with pytest.raises(MeasurementLogValidationError, match="realized truth"):
        parse_run_context(payload)


def test_runtime_candidate_snapshot_is_shape_and_cost_checked() -> None:
    """Only aligned runtime-owned reachable candidates should reach planning."""
    payload = {
        "candidate_poses_xyz": [[0.5, 0.5, 1.0], [1.5, 0.5, 1.0]],
        "travel_costs": [0.0, 1.0],
        "allowed_pair_ids": list(range(64)),
        "current_pair_id": 7,
    }

    parsed = parse_candidate_snapshot(payload)

    assert parsed == payload
    with pytest.raises(ValueError, match="align"):
        parse_candidate_snapshot({**payload, "travel_costs": [0.0]})


def test_persisted_record_parser_preserves_exact_integer_spectrum() -> None:
    """Closed-loop fitting must consume only the runtime's persisted raw counts."""
    payload = {
        "step_id": 0,
        "action_id": 0,
        "station_id": 0,
        "detector_pose_xyz": [0.5, 0.5, 1.0],
        "detector_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "fe_orientation_index": 1,
        "pb_orientation_index": 2,
        "live_time_s": 30.0,
        "travel_time_s": 0.0,
        "shield_actuation_time_s": 0.0,
        "energy_bin_edges_keV": [0.0, 1.0, 2.0],
        "spectrum_counts": [2, 3],
        "metadata": {
            "station_complete": True,
            "full_spectrum_contract_hash_sha256": "c" * 64,
        },
    }

    record = parse_adaptive_record(payload)

    assert record.spectrum_counts.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(record.spectrum_counts, [2, 3])


def test_each_closed_loop_request_contains_exactly_one_observation() -> None:
    """MLE should re-fit after each runtime observation, not pre-plan a program."""
    request = adaptive_step_request(
        candidate_index=4,
        fe_orientation_index=3,
        pb_orientation_index=6,
        dwell_time_s=30.0,
        station_id=9,
        station_complete=True,
    )

    assert request == {
        "type": "step",
        "candidate_index": 4,
        "fe_orientation_index": 3,
        "pb_orientation_index": 6,
        "dwell_time_s": 30.0,
        "station_id": 9,
        "station_complete": True,
    }
    assert "actions" not in request


def test_compound_stop_requires_every_scientific_gate() -> None:
    """Low EIG alone must not stop when convergence or ambiguity is unresolved."""
    patch = SimpleNamespace(patch_id=0, centroid_xyz=np.asarray([1.0, 1.0, 1.0]))
    diagnostics = {
        "kkt_residual": 1.0e-6,
        "identifiability": {"maximum_column_correlation": 0.2},
        "hotspot_clusters": [{"isotope": "Cs-137", "centroid_xyz": [1.0, 1.0, 1.0]}],
    }
    estimates = tuple(
        SimpleNamespace(
            isotope_names=("Cs-137",),
            patches=(patch,),
            patch_strength_by_isotope=np.asarray([[1.0]]),
            predicted_spectra=np.ones((4, 2)),
            poisson_deviance=10.0,
            converged=True,
            diagnostics=diagnostics,
        )
        for _ in range(2)
    )
    records = tuple(
        SimpleNamespace(
            detector_pose_xyz=pose,
            spectrum_counts=np.ones(2),
            station_id=index,
        )
        for index, pose in enumerate(
            ((0.0, 0.0, 0.2), (2.0, 0.0, 0.8), (0.0, 2.0, 1.4), (2.0, 2.0, 1.8))
        )
    )
    action = MLEPlanningAction(
        candidate_index=0,
        detector_pose_xyz=(0.0, 0.0, 1.0),
        shield_pair_ids=(0,),
        fe_orientation_indices=(0,),
        pb_orientation_indices=(0,),
        information_gain_nats=1.0e-5,
        travel_cost=0.0,
        rotation_radians=0.0,
        score=0.0,
        live_time_s_by_view=(1.0,),
        expected_total_counts_by_view=(1.0,),
        floor_ceiling_separation=0.1,
    )
    plans = (MLEPlanningResult(action, (action,), {}),) * 2
    config = MLEStopConfig(
        minimum_measurements=4,
        minimum_independent_poses=4,
        minimum_height_levels=2,
        minimum_elevation_span_rad=0.0,
        stability_window=2,
        low_information_patience=2,
    )

    decision = evaluate_mle_stop(estimates, records, plans, config)
    assert decision.should_stop is True

    unresolved = tuple(
        SimpleNamespace(
            **{
                **estimate.__dict__,
                "diagnostics": {
                    **diagnostics,
                    "identifiability": {"maximum_column_correlation": 0.999},
                },
            }
        )
        for estimate in estimates
    )
    decision = evaluate_mle_stop(unresolved, records, plans, config)
    assert decision.should_stop is False
    assert decision.gates["correlated_hypotheses_resolved"] is False


def test_compound_stop_compares_mean_not_growing_total_deviance() -> None:
    """A longer stable prefix should not fail merely because total deviance grows."""
    patch = SimpleNamespace(patch_id=0, centroid_xyz=np.asarray([1.0, 1.0, 1.0]))
    diagnostics = {
        "kkt_residual": 0.0,
        "identifiability": {"maximum_column_correlation": 0.0},
        "hotspot_clusters": [],
    }
    estimates = (
        SimpleNamespace(
            isotope_names=("Cs-137",),
            patches=(patch,),
            patch_strength_by_isotope=np.asarray([[1.0]]),
            predicted_spectra=np.ones((2, 2)),
            poisson_deviance=4.0,
            converged=True,
            diagnostics=diagnostics,
        ),
        SimpleNamespace(
            isotope_names=("Cs-137",),
            patches=(patch,),
            patch_strength_by_isotope=np.asarray([[1.0]]),
            predicted_spectra=np.ones((4, 2)),
            poisson_deviance=8.0,
            converged=True,
            diagnostics=diagnostics,
        ),
    )
    records = tuple(
        SimpleNamespace(
            detector_pose_xyz=(float(index), 0.0, float(index)),
            spectrum_counts=np.ones(2),
            station_id=index,
        )
        for index in range(4)
    )
    action = MLEPlanningAction(
        candidate_index=0,
        detector_pose_xyz=(0.0, 0.0, 1.0),
        shield_pair_ids=(0,),
        fe_orientation_indices=(0,),
        pb_orientation_indices=(0,),
        information_gain_nats=0.0,
        travel_cost=0.0,
        rotation_radians=0.0,
        score=0.0,
        live_time_s_by_view=(1.0,),
        expected_total_counts_by_view=(1.0,),
        floor_ceiling_separation=1.0,
    )
    plan = MLEPlanningResult(action, (action,), {})
    config = MLEStopConfig(
        minimum_measurements=4,
        minimum_independent_poses=4,
        minimum_height_levels=2,
        minimum_elevation_span_rad=0.0,
        stability_window=2,
        low_information_patience=2,
    )

    decision = evaluate_mle_stop(estimates, records, (plan, plan), config)

    assert decision.gates["deviance_stable"] is True
    assert decision.diagnostics["recent_mean_deviance_per_channel"] == [1.0, 1.0]


def _record_payload(
    step_id: int,
    station_id: int,
    *,
    station_complete: bool = True,
) -> dict[str, object]:
    """Return one fake persisted runtime response for loop orchestration."""
    return {
        "step_id": step_id,
        "action_id": step_id,
        "station_id": station_id,
        "detector_pose_xyz": [0.5, 0.5, 1.0],
        "detector_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "fe_orientation_index": 0,
        "pb_orientation_index": 0,
        "live_time_s": 30.0,
        "travel_time_s": 0.0,
        "shield_actuation_time_s": 0.0,
        "energy_bin_edges_keV": [0.0, 1.0, 2.0],
        "spectrum_counts": [2, 3],
        "metadata": {
            **({"station_complete": True} if station_complete else {}),
            "full_spectrum_contract_hash_sha256": "c" * 64,
        },
    }


class _FakeRuntimeClient:
    """Return two causal observations and capture incremental requests."""

    instance: _FakeRuntimeClient | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize a deterministic runtime handshake."""
        del args
        type(self).instance = self
        _FakeRuntimeClient.instance = self
        self.private_scene_profile = kwargs.get("private_scene_profile")
        self.resume_stage_path = kwargs.get("resume_stage_path")
        self.resume_compatibility_path = kwargs.get("resume_compatibility_path")
        self.requests: list[dict[str, object]] = []
        self.refinement_requests: list[dict[str, object]] = []
        self.cui_overlay_requests: list[bool] = []
        self.lifecycle_events: list[str] = []
        self.closed = False
        self.context = _context_payload()
        self.candidates = {
            "candidate_poses_xyz": [[0.5, 0.5, 1.0]],
            "travel_costs": [0.0],
            "allowed_pair_ids": list(range(64)),
            "current_pair_id": 0,
        }

    def read_ready_event(self) -> AdaptiveReadyEvent:
        """Return the typed initial runtime event."""
        return AdaptiveReadyEvent.from_payload(
            {
                "type": "ready",
                "schema_version": 1,
                "context": self.context,
                "candidates": self.candidates,
                "bootstrap": {
                    "candidate_index": 0,
                    "fe_orientation_index": 0,
                    "pb_orientation_index": 0,
                },
            }
        )

    def handshake(self) -> AdaptiveReadyEvent:
        """Return the typed handshake through the concise client API."""
        return self.read_ready_event()

    def request_refinement(
        self,
        request: AdaptiveRefineRequest,
    ) -> AdaptiveCandidatesEvent:
        """Return typed candidates for one typed refinement request."""
        assert isinstance(request, AdaptiveRefineRequest)
        self.refinement_requests.append(request.to_payload())
        return AdaptiveCandidatesEvent.from_payload(
            {"type": "candidates", "candidates": self.candidates}
        )

    def refine_candidates(
        self,
        request: AdaptiveRefineRequest,
    ) -> AdaptiveCandidatesEvent:
        """Return refined candidates through the concise client API."""
        return self.request_refinement(request)

    def request_step(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Return a typed record for exactly the supplied typed action."""
        assert isinstance(request, AdaptiveStepRequest)
        payload = request.to_payload()
        self.requests.append(payload)
        self.lifecycle_events.append("record")
        step_id = len(self.requests) - 1
        return AdaptiveRecordEvent.from_payload(
            {
                "type": "record",
                "record": _record_payload(
                    step_id,
                    request.station_id,
                    station_complete=request.station_complete,
                ),
                "candidates": self.candidates,
            }
        )

    def acquire(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Return one record through the concise client API."""
        return self.request_step(request)

    def request_cui_overlay(self, *, include_truth: bool) -> dict[str, object]:
        """Fail if the estimator-owned controller requests realized truth."""
        self.cui_overlay_requests.append(include_truth)
        self.lifecycle_events.append("cui_overlay")
        raise AssertionError("MLE closed loop must not request a truth overlay.")

    def finalize_event(self) -> AdaptivePublishedEvent:
        """Return the typed final immutable log location."""
        self.lifecycle_events.append("finalize")
        return AdaptivePublishedEvent(
            path="/tmp/adaptive-log",
            record_count=len(self.requests),
        )

    def finalize_log(self) -> AdaptivePublishedEvent:
        """Publish the log and close the fake client session."""
        event = self.finalize_event()
        self.closed = True
        return event

    def terminate(self) -> None:
        """Record bounded cleanup of an incomplete fake client session."""
        if not self.closed:
            self.lifecycle_events.append("terminate")
            self.closed = True

    def __enter__(self) -> _FakeRuntimeClient:
        """Return this fake under deterministic context ownership."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Terminate this fake when it has not already finalized."""
        del exc_type, exc, traceback
        self.terminate()


class _FakeResumeRuntimeClient(_FakeRuntimeClient):
    """Return a one-station prefix before accepting continued acquisition."""

    def read_ready_event(self) -> AdaptiveReadyEvent:
        """Return one typed schema-v2 resumed runtime event."""
        prefix = _record_payload(0, 0, station_complete=True)
        return AdaptiveReadyEvent.from_payload(
            {
                "type": "ready",
                "schema_version": 2,
                "context": self.context,
                "candidates": self.candidates,
                "resume": {
                    "record_count": 1,
                    "records": [prefix],
                    "next_station_id": 1,
                },
            }
        )

    def request_step(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Continue typed record identifiers after the resumed prefix."""
        assert isinstance(request, AdaptiveStepRequest)
        self.requests.append(request.to_payload())
        step_id = len(self.requests)
        return AdaptiveRecordEvent.from_payload(
            {
                "type": "record",
                "record": _record_payload(
                    step_id,
                    request.station_id,
                    station_complete=request.station_complete,
                ),
                "candidates": self.candidates,
            }
        )

    def finalize_event(self) -> AdaptivePublishedEvent:
        """Return a typed published count including the resumed prefix."""
        self.lifecycle_events.append("finalize")
        return AdaptivePublishedEvent(
            path="/tmp/adaptive-log",
            record_count=1 + len(self.requests),
        )


class _FailingRuntimeClient(_FakeRuntimeClient):
    """Fail one acquisition to exercise context-owned bounded cleanup."""

    def acquire(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Raise after recording that acquisition reached the runtime boundary."""
        assert isinstance(request, AdaptiveStepRequest)
        self.lifecycle_events.append("acquire-failed")
        raise RuntimeError("injected adaptive acquisition failure")


class _FakeOnlineSession:
    """Expose the online MLE operations used by the controller."""

    last_init_kwargs: dict[str, object] = {}
    last_instance: _FakeOnlineSession | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize record capture and a dashboard sentinel."""
        del args
        type(self).last_init_kwargs = dict(kwargs)
        type(self).last_instance = self
        self.records: list[object] = []
        self.planning_calls: list[dict[str, object]] = []
        self.dashboard_url = "http://127.0.0.1:8877/index.html"
        self.bound_path: Path | None = None

    def receive_persisted(self, record: object, *, station_complete: bool) -> None:
        """Capture one persisted record before fitting."""
        assert station_complete is True
        self.records.append(record)

    def plan_next_action(self, *args: object, **kwargs: object) -> MLEPlanningResult:
        """Select one next observation after the first fit."""
        del args
        self.planning_calls.append(dict(kwargs))
        progress_hook = kwargs.get("progress_hook")
        if callable(progress_hook):
            for completed, elapsed in ((0, 0.0), (20, 1.0), (40, 5.1), (100, 6.0)):
                progress_hook(
                    {
                        "phase": "mle_solver_iterations:test",
                        "completed": completed,
                        "total": 100,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": 0.0,
                    }
                )
            progress_hook(
                {
                    "phase": "candidate_search",
                    "completed_candidates": 1,
                    "total_candidates": 2,
                    "elapsed_seconds": 3.0,
                    "eta_seconds": 3.0,
                }
            )
        action = MLEPlanningAction(
            candidate_index=0,
            detector_pose_xyz=(0.5, 0.5, 1.0),
            shield_pair_ids=(0,),
            fe_orientation_indices=(0,),
            pb_orientation_indices=(0,),
            information_gain_nats=1.0,
            travel_cost=0.0,
            rotation_radians=0.0,
            score=1.0,
            live_time_s_by_view=(30.0,),
            expected_total_counts_by_view=(5.0,),
        )
        return MLEPlanningResult(action, (action,), {})

    def bind_finalized_measurement_log(self, path: Path) -> None:
        """Capture final-log binding."""
        client = _FakeRuntimeClient.instance
        assert client is not None
        assert client.lifecycle_events[-1] == "finalize"
        client.lifecycle_events.append("bind")
        self.bound_path = Path(path)

    def complete_live_state(self) -> SimpleNamespace:
        """Seal the scientific fit before runtime log publication."""
        client = _FakeRuntimeClient.instance
        assert client is not None
        assert "finalize" not in client.lifecycle_events
        client.lifecycle_events.append("complete_live_state")
        return SimpleNamespace()

    def publish_bound_result(self) -> SimpleNamespace:
        """Publish the bound result without running another scientific fit."""
        client = _FakeRuntimeClient.instance
        assert client is not None
        assert client.lifecycle_events[-1] == "bind"
        client.lifecycle_events.append("publish_bound_result")
        return SimpleNamespace(dashboard_url=self.dashboard_url)


class _FakeOnlineProgramSession(_FakeOnlineSession):
    """Return a two-view shield program at one selected detector pose."""

    def receive_persisted(self, record: object, *, station_complete: bool) -> None:
        """Capture every row while allowing an open dynamic station."""
        del station_complete
        self.records.append(record)

    def plan_next_action(self, *args: object, **kwargs: object) -> MLEPlanningResult:
        """Select two shield angles at the same candidate pose."""
        del args, kwargs
        action = MLEPlanningAction(
            candidate_index=0,
            detector_pose_xyz=(0.5, 0.5, 1.0),
            shield_pair_ids=(0, 1),
            fe_orientation_indices=(0, 0),
            pb_orientation_indices=(0, 1),
            information_gain_nats=1.0,
            travel_cost=0.0,
            rotation_radians=1.0,
            score=1.0,
            live_time_s_by_view=(30.0, 30.0),
            expected_total_counts_by_view=(5.0, 5.0),
        )
        return MLEPlanningResult(action, (action,), {})


def test_closed_loop_sends_bootstrap_then_one_mle_selected_action(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """No future action should reach runtime before the preceding MLE fit."""
    from three_d_estimation import closed_loop

    mle_path = tmp_path / "mle.json"
    planning_path = tmp_path / "planning.json"
    mle_path.write_text("{}\n", encoding="utf-8")
    planning_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(closed_loop, "OnlineMLESession", _FakeOnlineSession)
    monkeypatch.setattr(
        closed_loop.MLEConfig,
        "load",
        lambda path: SimpleNamespace(isotope_names=("Co-60", "Cs-137", "Eu-154")),
    )
    monkeypatch.setattr(
        closed_loop.MLEPlanningConfig,
        "load",
        lambda path: SimpleNamespace(shield_program_length=1, live_time_s=30.0),
    )
    fake_log = SimpleNamespace(
        path=Path("/tmp/adaptive-log"),
        run_id="adaptive-test",
        records=(SimpleNamespace(station_id=0), SimpleNamespace(station_id=1)),
    )
    monkeypatch.setattr(
        closed_loop, "validate_ral_measurement_log", lambda path: fake_log
    )
    output_lines: list[str] = []

    result = run_ral_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        mle_config_path=mle_path,
        planning_config_path=planning_path,
        output_dir=tmp_path / "output",
        private_scene_profile="ral-cs4-co3-eu0",
        max_measurements=2,
        output_hook=output_lines.append,
    )

    client = _FakeRuntimeClient.instance
    assert isinstance(result, RALClosedLoopResult)
    assert client is not None
    assert client.private_scene_profile == "ral-cs4-co3-eu0"
    assert client.cui_overlay_requests == []
    assert client.lifecycle_events[-4:] == [
        "complete_live_state",
        "finalize",
        "bind",
        "publish_bound_result",
    ]
    assert all(
        "truth" not in key and "overlay" not in key
        for key in _FakeOnlineSession.last_init_kwargs
    )
    assert len(client.requests) == 2
    assert client.requests[0]["station_id"] == 0
    assert client.requests[1]["station_id"] == 1
    assert all("actions" not in request for request in client.requests)
    assert result.record_count == 2
    assert result.stop_reason == "maximum_measurement_safety_bound"
    assert any(
        "Progress: phase=candidate_search completed=1/2 (50.0%)" in line
        and "elapsed=3.0s eta=3.0s" in line
        for line in output_lines
    )
    solver_lines = [
        line for line in output_lines if "phase=mle_solver_iterations:test" in line
    ]
    assert [
        line.split("completed=", 1)[1].split(" ", 1)[0] for line in solver_lines
    ] == [
        "0/100",
        "40/100",
        "100/100",
    ]


def test_closed_loop_context_terminates_an_incomplete_runtime(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An acquisition exception must leave cleanup to the client context."""
    from three_d_estimation import closed_loop

    mle_path = tmp_path / "mle.json"
    planning_path = tmp_path / "planning.json"
    mle_path.write_text("{}\n", encoding="utf-8")
    planning_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        closed_loop,
        "AdaptiveRuntimeClient",
        _FailingRuntimeClient,
    )
    monkeypatch.setattr(closed_loop, "OnlineMLESession", _FakeOnlineSession)
    monkeypatch.setattr(
        closed_loop.MLEConfig,
        "load",
        lambda path: SimpleNamespace(isotope_names=("Co-60", "Cs-137", "Eu-154")),
    )
    monkeypatch.setattr(
        closed_loop.MLEPlanningConfig,
        "load",
        lambda path: SimpleNamespace(shield_program_length=1, live_time_s=30.0),
    )

    with pytest.raises(RuntimeError, match="injected adaptive acquisition failure"):
        run_ral_closed_loop(
            tmp_path / "private-scenario.json",
            runtime_root=tmp_path,
            mle_config_path=mle_path,
            planning_config_path=planning_path,
            output_dir=tmp_path / "output",
            max_measurements=1,
            output_hook=lambda line: None,
        )

    client = _FailingRuntimeClient.instance
    assert client is not None
    assert client.lifecycle_events == ["acquire-failed", "terminate"]


def test_closed_loop_cui_remains_truth_free(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The estimator-owned CUI must never request or persist realized truth."""
    from three_d_estimation import closed_loop

    mle_path = tmp_path / "mle.json"
    planning_path = tmp_path / "planning.json"
    mle_path.write_text("{}\n", encoding="utf-8")
    planning_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(closed_loop, "OnlineMLESession", _FakeOnlineSession)
    monkeypatch.setattr(
        closed_loop.MLEConfig,
        "load",
        lambda path: SimpleNamespace(isotope_names=("Co-60", "Cs-137", "Eu-154")),
    )
    monkeypatch.setattr(
        closed_loop.MLEPlanningConfig,
        "load",
        lambda path: SimpleNamespace(shield_program_length=1, live_time_s=30.0),
    )
    fake_log = SimpleNamespace(
        path=Path("/tmp/adaptive-log"),
        run_id="adaptive-test",
        records=(SimpleNamespace(station_id=0),),
    )
    monkeypatch.setattr(
        closed_loop, "validate_ral_measurement_log", lambda path: fake_log
    )

    run_ral_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        mle_config_path=mle_path,
        planning_config_path=planning_path,
        output_dir=tmp_path / "output",
        max_measurements=1,
        output_hook=lambda line: None,
    )

    client = _FakeRuntimeClient.instance
    assert client is not None
    assert client.cui_overlay_requests == []
    assert all(
        "truth" not in key and "overlay" not in key
        for key in _FakeOnlineSession.last_init_kwargs
    )
    assert all("truth" not in request for request in client.requests)


def test_closed_loop_restores_resume_prefix_then_plans_next_station(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A resumed controller must rebuild state before requesting a new record."""
    from three_d_estimation import closed_loop

    mle_path = tmp_path / "mle.json"
    planning_path = tmp_path / "planning.json"
    mle_path.write_text("{}\n", encoding="utf-8")
    planning_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        closed_loop,
        "AdaptiveRuntimeClient",
        _FakeResumeRuntimeClient,
    )
    monkeypatch.setattr(closed_loop, "OnlineMLESession", _FakeOnlineSession)
    monkeypatch.setattr(
        closed_loop.MLEConfig,
        "load",
        lambda path: SimpleNamespace(isotope_names=("Co-60", "Cs-137", "Eu-154")),
    )
    monkeypatch.setattr(
        closed_loop.MLEPlanningConfig,
        "load",
        lambda path: SimpleNamespace(shield_program_length=1, live_time_s=30.0),
    )
    fake_log = SimpleNamespace(
        path=Path("/tmp/adaptive-log"),
        run_id="adaptive-test",
        records=(SimpleNamespace(station_id=0), SimpleNamespace(station_id=1)),
    )
    monkeypatch.setattr(
        closed_loop,
        "validate_ral_measurement_log",
        lambda path: fake_log,
    )
    stage = tmp_path / ".measurement-log.stream-17"
    compatibility = tmp_path / "resume-compatibility.json"

    result = run_ral_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        mle_config_path=mle_path,
        planning_config_path=planning_path,
        output_dir=tmp_path / "output",
        resume_stage_path=stage,
        resume_compatibility_path=compatibility,
        max_measurements=2,
    )

    client = _FakeResumeRuntimeClient.instance
    session = _FakeOnlineSession.last_instance
    assert client is not None
    assert session is not None
    assert client.resume_stage_path == stage
    assert client.resume_compatibility_path == compatibility
    assert len(client.requests) == 1
    assert client.requests[0]["station_id"] == 1
    assert len(session.records) == 2
    assert result.record_count == 2


def test_two_stage_closed_loop_screens_before_runtime_refinement(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Runtime refinement must sit between screening and exact planning calls."""
    from three_d_estimation import closed_loop

    mle_path = tmp_path / "mle.json"
    planning_path = tmp_path / "planning.json"
    mle_path.write_text("{}\n", encoding="utf-8")
    planning_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(closed_loop, "OnlineMLESession", _FakeOnlineSession)
    monkeypatch.setattr(
        closed_loop.MLEConfig,
        "load",
        lambda path: SimpleNamespace(isotope_names=("Co-60", "Cs-137", "Eu-154")),
    )
    monkeypatch.setattr(
        closed_loop.MLEPlanningConfig,
        "load",
        lambda path: SimpleNamespace(
            shield_program_length=1,
            live_time_s=30.0,
            local_refinement_top_k=1,
            two_stage_screening=True,
        ),
    )
    fake_log = SimpleNamespace(
        path=Path("/tmp/adaptive-log"),
        run_id="adaptive-test",
        records=(SimpleNamespace(station_id=0), SimpleNamespace(station_id=1)),
    )
    monkeypatch.setattr(
        closed_loop,
        "validate_ral_measurement_log",
        lambda path: fake_log,
    )

    run_ral_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        mle_config_path=mle_path,
        planning_config_path=planning_path,
        output_dir=tmp_path / "output",
        max_measurements=2,
    )

    client = _FakeRuntimeClient.instance
    assert client is not None
    assert client.refinement_requests == [{"type": "refine", "candidate_indices": [0]}]
    session = _FakeOnlineSession.last_instance
    assert session is not None
    session_calls = session.planning_calls
    assert session_calls[0]["screening_only"] is True
    assert "screening_only" not in session_calls[1]


def test_closed_loop_groups_same_pose_shield_views_into_one_station(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Multiple selected shield angles must share one runtime pose block."""
    from three_d_estimation import closed_loop

    mle_path = tmp_path / "mle.json"
    planning_path = tmp_path / "planning.json"
    mle_path.write_text("{}\n", encoding="utf-8")
    planning_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(closed_loop, "AdaptiveRuntimeClient", _FakeRuntimeClient)
    monkeypatch.setattr(closed_loop, "OnlineMLESession", _FakeOnlineProgramSession)
    monkeypatch.setattr(
        closed_loop.MLEConfig,
        "load",
        lambda path: SimpleNamespace(isotope_names=("Co-60", "Cs-137", "Eu-154")),
    )
    monkeypatch.setattr(
        closed_loop.MLEPlanningConfig,
        "load",
        lambda path: SimpleNamespace(
            shield_program_length=2,
            live_time_s=30.0,
            local_refinement_top_k=0,
        ),
    )
    fake_log = SimpleNamespace(
        path=Path("/tmp/adaptive-log"),
        run_id="adaptive-test",
        records=(
            SimpleNamespace(station_id=0),
            SimpleNamespace(station_id=1),
            SimpleNamespace(station_id=1),
        ),
    )
    monkeypatch.setattr(
        closed_loop,
        "validate_ral_measurement_log",
        lambda path: fake_log,
    )

    result = run_ral_closed_loop(
        tmp_path / "private-scenario.json",
        runtime_root=tmp_path,
        mle_config_path=mle_path,
        planning_config_path=planning_path,
        output_dir=tmp_path / "output",
        max_measurements=3,
    )

    client = _FakeRuntimeClient.instance
    assert client is not None
    assert [request["station_id"] for request in client.requests] == [0, 1, 1]
    assert [request["station_complete"] for request in client.requests] == [
        True,
        False,
        True,
    ]
    assert result.station_count == 2
