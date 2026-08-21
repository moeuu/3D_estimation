"""Tests for the live runtime adapter around the standalone surface MLE."""

from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from three_d_estimation.backend_contracts import (
    EstimatorBackend,
    StationCompleteEstimatorBackend,
)
from runtime.records import MeasurementRecord, RunContext
from spectrum.transport_spectral import GeometryConditionedSpectralModel
from three_d_estimation.config import MLEConfig
from three_d_estimation.estimator_backend import SurfaceMLEBackend
from three_d_estimation.information_planner import (
    MLEPlanningAction,
    MLEPlanningConfig,
    MLEPlanningResult,
)
from three_d_estimation.types import MLEEstimate, ObservationBatch, SurfacePatch


def _context(
    *,
    spectrum_count_method: str = "joint_full_spectrum_generative",
    obstacle_layout_path: str | None = None,
    embedded_obstacle: bool = True,
) -> RunContext:
    """Return a portable one-isotope runtime context."""
    environment: dict[str, object] = {
        "size_x": 3.0,
        "size_y": 4.0,
        "size_z": 2.5,
        "detector_position": [0.5, 0.5, 1.0],
    }
    if embedded_obstacle:
        environment["obstacle_grid"] = {
            "version": 1,
            "origin": [0.0, 0.0],
            "cell_size": 1.0,
            "grid_shape": [3, 4],
            "blocked_cells": [[2, 3]],
            "blocked_fraction": 1.0 / 12.0,
        }
    return RunContext(
        repository_commit="a" * 40,
        runtime_config={
            "source_rate_model": "detector_cps_1m",
            "line_resolved_shield_attenuation": False,
            "full_spectrum_generative_model": (
                GeometryConditionedSpectralModel.standard_native(
                    ("Cs-137",),
                    dead_time_tau_s=0.0,
                    background_rate_cps=0.0,
                ).manifest_payload()
            ),
        },
        environment=environment,
        sim_backend="analytic",
        spectrum_count_method=spectrum_count_method,
        isotopes=("Cs-137",),
        obstacle_layout_path=obstacle_layout_path,
        source_rate_model="detector_cps_1m",
        source_layout_path=None,
        metadata={},
        run_id="surface-mle-backend-test",
        source_rate_semantics={},
        forward_model_manifest={},
        runtime_config_sha256="b" * 64,
    )


def _record(step_id: int, station_id: int) -> MeasurementRecord:
    """Return one complete finalized runtime observation."""
    return MeasurementRecord(
        step_id=step_id,
        action_id=step_id,
        station_id=station_id,
        detector_pose_xyz=(0.5 + 0.1 * step_id, 0.5, 1.0),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=step_id % 4,
        pb_orientation_index=(step_id + 1) % 4,
        live_time_s=2.0,
        travel_time_s=0.25,
        shield_actuation_time_s=0.1,
        spectrum_counts=np.array([3 + step_id, 2], dtype=np.int64),
        energy_bin_edges_keV=np.array([0.0, 400.0, 800.0]),
        metadata={
            "finalized": True,
            "full_spectrum_contract_hash_sha256": "c" * 64,
        },
    )


def _patch() -> SurfacePatch:
    """Return a single exact square floor patch."""
    return SurfacePatch(
        patch_id=10,
        centroid_xyz=np.array([0.5, 0.5, 0.0]),
        normal_xyz=np.array([0.0, 0.0, 1.0]),
        area_m2=1.0,
        surface_kind="floor",
        object_id="room:floor",
        vertices_xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        quadrature_points_xyz=np.array([[0.5, 0.5, 0.0]]),
        quadrature_weights=np.array([1.0]),
    )


def _estimate(measurement_count: int) -> MLEEstimate:
    """Return a shaped estimate whose predictions expose all-history fitting."""
    row = np.arange(measurement_count, dtype=float)[:, None]
    return MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(_patch(),),
        density_by_isotope=np.array([[4.0]]),
        patch_strength_by_isotope=np.array([[4.0]]),
        predicted_spectra=np.hstack([row + 1.0, row + 2.0]),
        predicted_isotope_counts=row + 3.0,
        background_parameters=np.array([0.2]),
        nuisance_parameters=np.array([0.1]),
        objective_value=12.5,
        poisson_deviance=2.25,
        iterations=7,
        converged=True,
        diagnostics={
            "hotspot_clusters": [
                {
                    "isotope": "Cs-137",
                    "cluster_id": np.int64(3),
                    "patch_ids": (10,),
                    "centroid_xyz": (0.5, 0.5, 0.0),
                    "integrated_strength_cps_1m": 4.0,
                    "peak_density_cps_1m_m2": np.float64(4.0),
                    "surface_kinds": ("floor",),
                }
            ],
            "identifiability": {
                "singular_values": np.array([2.0, 1.0]),
                "condition_number_nonzero": float("inf"),
            },
        },
    )


class _RecordingEstimator:
    """Record adapter inputs while returning deterministic MLE estimates."""

    def __init__(self) -> None:
        """Initialize an empty fit-call history."""
        self.measurement_counts: list[int] = []
        self.environments: list[EnvironmentConfig] = []
        self.kernels: list[ContinuousKernel] = []
        self.obstacles: list[ObstacleGrid | None] = []
        self.initial_estimates: list[MLEEstimate | None] = []

    def fit(
        self,
        batch: ObservationBatch,
        environment: EnvironmentConfig,
        kernel: ContinuousKernel,
        *,
        obstacle_grid: ObstacleGrid | None,
        initial_estimate: MLEEstimate | None = None,
    ) -> MLEEstimate:
        """Record one all-history invocation and return a shaped estimate."""
        self.measurement_counts.append(batch.measurement_count)
        self.environments.append(environment)
        self.kernels.append(kernel)
        self.obstacles.append(obstacle_grid)
        self.initial_estimates.append(initial_estimate)
        return _estimate(batch.measurement_count)


def _backend(recorder: _RecordingEstimator) -> SurfaceMLEBackend:
    """Return a spectral adapter using the deterministic recording estimator."""
    config = MLEConfig(mode="spectral", isotope_names=("Cs-137",))
    return SurfaceMLEBackend(config, estimator_factory=lambda _config: recorder)


def _initialize_with_test_kernel(
    backend: SurfaceMLEBackend,
    context: RunContext,
) -> None:
    """Initialize the adapter without constructing the production physics kernel."""
    kernel = ContinuousKernel(use_gpu=False)
    with (
        patch(
            "three_d_estimation.estimator_backend.build_runtime_observation_model",
            return_value=object(),
        ),
        patch(
            "three_d_estimation.estimator_backend."
            "continuous_kernel_from_observation_model",
            return_value=kernel,
        ),
    ):
        backend.initialize(context)


class SurfaceMLEBackendTests(unittest.TestCase):
    """Exercise runtime contracts, station fits, conversion, and isolation."""

    def test_station_and_final_fits_use_complete_buffered_history(self) -> None:
        """Station fitting should be periodic while finalization refits all rows."""
        recorder = _RecordingEstimator()
        backend = _backend(recorder)

        self.assertIsInstance(backend, EstimatorBackend)
        self.assertIsInstance(backend, StationCompleteEstimatorBackend)
        _initialize_with_test_kernel(backend, _context())
        self.assertEqual(backend._kernel.gpu_device, backend.config.gpu_device)
        self.assertEqual(backend._kernel.gpu_dtype, backend.config.gpu_dtype)
        empty = backend.snapshot()
        self.assertEqual(empty.step_id, -1)
        self.assertEqual(empty.diagnostics["fit_kind"], "not_fitted")

        first = _record(0, 7)
        second = _record(1, 7)
        backend.update(first)
        self.assertEqual(recorder.measurement_counts, [])
        backend.update(second)
        self.assertEqual(recorder.measurement_counts, [])
        backend.on_station_complete(7, (first, second))

        self.assertEqual(recorder.measurement_counts, [2])
        environment = recorder.environments[0]
        self.assertEqual(
            (environment.size_x, environment.size_y, environment.size_z),
            (3.0, 4.0, 2.5),
        )
        self.assertIsInstance(recorder.kernels[0], ContinuousKernel)
        self.assertEqual(recorder.obstacles[0].blocked_cells, ((2, 3),))
        warm = backend.snapshot()
        station_estimate = backend.latest_estimate
        self.assertEqual(warm.step_id, 1)
        self.assertEqual(warm.diagnostics["fit_kind"], "station_warm")
        np.testing.assert_array_equal(warm.predicted_spectrum, [2.0, 3.0])

        third = _record(2, 8)
        backend.update(third)
        stale = backend.snapshot()
        self.assertEqual(stale.step_id, 1)
        self.assertEqual(stale.diagnostics["measurement_count"], 2)

        result = backend.finalize()
        self.assertEqual(recorder.measurement_counts, [2, 3])
        self.assertIsNone(recorder.initial_estimates[0])
        self.assertIs(recorder.initial_estimates[1], station_estimate)
        self.assertEqual(backend.records, (first, second, third))
        self.assertIs(backend.finalize(), result)
        self.assertEqual(recorder.measurement_counts, [2, 3])
        self.assertEqual(result.final_snapshot.step_id, 2)
        self.assertEqual(result.final_snapshot.diagnostics["fit_kind"], "final")
        np.testing.assert_array_equal(
            result.final_snapshot.predicted_spectrum, [3.0, 4.0]
        )

    def test_final_fit_warm_starts_across_online_and_final_patch_spacings(self) -> None:
        """The fine final dictionary should inherit the latest coarse estimate."""
        recorder = _RecordingEstimator()
        config = MLEConfig(
            mode="spectral",
            isotope_names=("Cs-137",),
            patch_spacing_m=(1.0, 1.0, 1.0),
            online_patch_spacing_m=(2.0, 2.0, 2.0),
        )
        backend = SurfaceMLEBackend(
            config,
            estimator_factory=lambda _config: recorder,
        )
        _initialize_with_test_kernel(backend, _context())
        record = _record(0, 4)
        backend.update(record)
        backend.on_station_complete(4, (record,))
        station_estimate = backend.latest_estimate

        backend.finalize()

        self.assertEqual(recorder.measurement_counts, [1, 1])
        self.assertIsNone(recorder.initial_estimates[0])
        self.assertIs(recorder.initial_estimates[1], station_estimate)

    def test_runtime_mapping_proxy_is_normalized_before_model_build(self) -> None:
        """A RunContext produced by MeasurementLog must initialize directly."""
        context = _context()
        readonly_context = replace(
            context,
            runtime_config=MappingProxyType(dict(context.runtime_config)),
        )
        backend = _backend(_RecordingEstimator())

        _initialize_with_test_kernel(backend, readonly_context)

        self.assertEqual(backend.snapshot().step_id, -1)

    def test_estimate_is_converted_to_surface_map_modes_and_strict_json(self) -> None:
        """MLE-specific arrays and clusters should satisfy generic output contracts."""
        recorder = _RecordingEstimator()
        backend = _backend(recorder)
        _initialize_with_test_kernel(backend, _context())
        record = _record(0, 4)
        backend.update(record)
        result = backend.finalize()
        snapshot = result.final_snapshot

        surface = snapshot.surface_map_by_isotope["Cs-137"]
        self.assertEqual(surface.patch_ids, (10,))
        np.testing.assert_array_equal(surface.patch_centroids_xyz, [[0.5, 0.5, 0.0]])
        np.testing.assert_array_equal(surface.density_cps_1m_per_m2, [4.0])
        self.assertEqual(surface.metadata["patch_strengths_cps_1m"], [4.0])
        mode = snapshot.source_modes_by_isotope["Cs-137"][0]
        self.assertEqual(mode.position_xyz, (0.5, 0.5, 0.0))
        self.assertEqual(mode.strength_cps_1m, 4.0)
        self.assertEqual(mode.metadata["patch_ids"], [10])
        self.assertIsNone(
            snapshot.diagnostics["identifiability"]["condition_number_nonzero"]
        )
        json.dumps(snapshot.diagnostics, allow_nan=False)
        json.dumps(result.diagnostics, allow_nan=False)

    def test_planning_uses_the_latest_station_complete_mle(self) -> None:
        """Backend planning should pass only fitted durable history and candidates."""
        recorder = _RecordingEstimator()
        backend = _backend(recorder)
        _initialize_with_test_kernel(backend, _context())
        record = _record(0, 4)
        backend.update(record)
        backend.on_station_complete(4, (record,))
        action = MLEPlanningAction(
            candidate_index=0,
            detector_pose_xyz=(1.0, 1.0, 1.0),
            shield_pair_ids=(1,),
            fe_orientation_indices=(0,),
            pb_orientation_indices=(1,),
            information_gain_nats=1.0,
            travel_cost=0.0,
            rotation_radians=0.0,
            score=1.0,
            live_time_s_by_view=(2.0,),
            expected_total_counts_by_view=(10.0,),
        )
        planned = MLEPlanningResult(
            selected_action=action,
            ranked_actions=(action,),
            diagnostics={},
        )
        candidates = np.asarray([[1.0, 1.0, 1.0]])
        planning_config = MLEPlanningConfig(shield_program_length=1)

        with patch(
            "three_d_estimation.estimator_backend.plan_next_measurement",
            return_value=planned,
        ) as planner:
            result = backend.plan_next_action(
                candidates,
                planning_config=planning_config,
            )

        self.assertIs(result, planned)
        self.assertIs(planner.call_args.args[0], backend.latest_estimate)
        np.testing.assert_array_equal(planner.call_args.args[4], candidates)
        self.assertEqual(planner.call_args.kwargs["current_pair_id"], 1)

    def test_station_hook_requires_the_exact_new_buffer_suffix(self) -> None:
        """A caller cannot warm-fit fabricated or already-completed station rows."""
        recorder = _RecordingEstimator()
        backend = _backend(recorder)
        _initialize_with_test_kernel(backend, _context())
        record = _record(0, 9)
        backend.update(record)

        with self.assertRaisesRegex(ValueError, "buffered history suffix"):
            backend.on_station_complete(9, (replace(record),))
        backend.on_station_complete(9, (record,))
        with self.assertRaisesRegex(RuntimeError, "No new measurements"):
            backend.on_station_complete(9, (record,))

    def test_invalid_runtime_inputs_fail_before_any_fit(self) -> None:
        """The adapter should reject external paths, duplicates, and empty history."""
        recorder = _RecordingEstimator()
        backend = _backend(recorder)
        with self.assertRaisesRegex(ValueError, "requires the shared runtime"):
            _initialize_with_test_kernel(
                backend,
                _context(
                    obstacle_layout_path="/tmp/external-obstacles.json",
                    embedded_obstacle=False,
                ),
            )
        self.assertEqual(recorder.measurement_counts, [])

        backend = _backend(recorder)
        _initialize_with_test_kernel(backend, _context())
        with self.assertRaisesRegex(RuntimeError, "no measurements"):
            backend.finalize()
        record = _record(0, 1)
        backend.update(record)
        with self.assertRaisesRegex(ValueError, "Duplicate finalized step_id"):
            backend.update(record)

        count_backend = SurfaceMLEBackend(
            MLEConfig(mode="count", isotope_names=("Cs-137",)),
            estimator_factory=lambda _config: recorder,
        )
        with self.assertRaisesRegex(ValueError, "response_poisson"):
            _initialize_with_test_kernel(
                count_backend,
                _context(spectrum_count_method="joint_full_spectrum_generative"),
            )


if __name__ == "__main__":
    unittest.main()
