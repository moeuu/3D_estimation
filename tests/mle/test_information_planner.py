"""Tests for MLE-specific Fisher optimal experimental design."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from measurement.continuous_kernels import ContinuousKernel

import three_d_estimation.information_planner as information_planner
from three_d_estimation.cli import _estimate_history_indices, build_argument_parser
from three_d_estimation.config import MLEConfig
from three_d_estimation.information_planner import (
    PLANNING_METHOD,
    MLEPlanningAction,
    MLEPlanningConfig,
    MLEPlanningResult,
    _LineSpectralDesign,
    _ambiguity_metrics,
    _beam_precision_chunk_size,
    _factorized_fisher_information,
    _fisher_information,
    _grouped_overdispersed_variance,
    _historical_factorized_fisher_precision,
    _historical_factorized_spectral_design,
    _planning_prior_precision,
    _plan_next_measurement_exact,
    _historical_fisher_precision,
    _historical_spectral_design,
    _representative_pair_ids,
    _screening_background_rate,
    _screen_candidate_measurements,
    _screening_fisher_information,
    _screening_pseudo_model,
    _screening_source_basis,
    _source_basis,
    _symmetric_spectral_separation,
    plan_next_measurement,
    select_fisher_action,
)
from three_d_estimation.types import MLEEstimate, ObservationBatch, SurfacePatch


def test_default_profiles_use_eight_measurements_per_station() -> None:
    """Default MLE planning profiles must retain eight-view station blocks."""
    root = Path(__file__).resolve().parents[2]

    assert MLEPlanningConfig().shield_program_length == 8
    for name in ("default_planning.json", "ral_full_planning.json"):
        config = MLEPlanningConfig.load(root / "configs" / "mle" / name)
        assert config.shield_program_length == 8

    legacy = MLEPlanningConfig(
        two_stage_screening=False,
        shield_program_length=2,
        screening_pair_limit=1,
        ranked_action_limit=1,
    )
    assert legacy.two_stage_screening is False


def test_planning_method_tracks_ambiguity_objective_version() -> None:
    """Planning artifacts must identify the nuisance-aware ambiguity objective."""
    assert PLANNING_METHOD.endswith("_v5")


def _planning_action(index: int, pose: np.ndarray, score: float) -> MLEPlanningAction:
    """Return one compact deterministic action for planner orchestration tests."""
    return MLEPlanningAction(
        candidate_index=index,
        detector_pose_xyz=tuple(float(value) for value in pose),
        shield_pair_ids=(0,),
        fe_orientation_indices=(0,),
        pb_orientation_indices=(0,),
        information_gain_nats=score,
        travel_cost=0.0,
        rotation_radians=0.0,
        score=score,
        live_time_s_by_view=(1.0,),
        expected_total_counts_by_view=(1.0,),
    )


def test_two_stage_planner_exactly_evaluates_only_adaptive_shortlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact stage must receive a bounded shortlist and restore global IDs."""
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(_floor_patch(0, 0.0),),
        density_by_isotope=np.asarray([[1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )
    history = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.5, 0.5, 0.5]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
        live_times_s=np.asarray([1.0]),
        spectrum_counts=np.ones((1, 2)),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        isotope_counts=None,
        isotope_covariances=None,
        station_ids=np.asarray([0]),
        isotope_names=("Cs-137",),
    )
    poses = np.asarray(
        [[float(index), float(index % 3), 1.0] for index in range(10)],
        dtype=np.float64,
    )
    screening_actions = tuple(
        _planning_action(index, pose, 10.0 - index) for index, pose in enumerate(poses)
    )

    def fake_screen(*args: object, **kwargs: object) -> MLEPlanningResult:
        """Return all candidates in a deterministic approximate order."""
        del args, kwargs
        return MLEPlanningResult(
            screening_actions[0],
            screening_actions,
            {"stage": "screening"},
        )

    exact_pose_batches: list[np.ndarray] = []

    def fake_exact(
        _estimate: MLEEstimate,
        _history: ObservationBatch,
        _kernel: ContinuousKernel,
        _mle_config: MLEConfig,
        exact_poses: object,
        **_kwargs: object,
    ) -> MLEPlanningResult:
        """Capture the exact shortlist and select its third local entry."""
        local_poses = np.asarray(exact_poses, dtype=np.float64)
        exact_pose_batches.append(local_poses.copy())
        actions = tuple(
            _planning_action(index, pose, float(index))
            for index, pose in enumerate(local_poses)
        )
        return MLEPlanningResult(actions[2], actions, {"stage": "exact"})

    monkeypatch.setattr(
        information_planner,
        "_screen_candidate_measurements",
        fake_screen,
    )
    monkeypatch.setattr(
        information_planner,
        "_plan_next_measurement_exact",
        fake_exact,
    )
    config = MLEPlanningConfig(
        shield_program_length=1,
        screening_pair_limit=1,
        ranked_action_limit=10,
        exact_candidate_min=4,
        exact_candidate_max=4,
    )

    result = plan_next_measurement(
        estimate,
        history,
        ContinuousKernel(use_gpu=False),
        MLEConfig(mode="spectral", isotope_names=("Cs-137",), use_gpu=False),
        poses,
        planning_config=config,
    )

    assert exact_pose_batches[0].shape == (4, 3)
    selected_global_index = int(result.selected_action.candidate_index)
    np.testing.assert_array_equal(
        poses[selected_global_index],
        exact_pose_batches[0][2],
    )
    assert result.diagnostics["exact_candidate_count"] == 4
    assert result.diagnostics["total_candidate_count"] == 10


def test_screening_background_excludes_fitted_source_counts() -> None:
    """Screening must not count fitted source photons twice in its denominator."""
    history = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.0, 0.0, 1.0]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
        live_times_s=np.asarray([2.0]),
        spectrum_counts=np.asarray([[10.0, 20.0, 30.0, 40.0]]),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        isotope_counts=None,
        isotope_covariances=None,
        station_ids=np.asarray([0]),
        isotope_names=("Cs-137",),
    )

    rate = _screening_background_rate(
        history,
        np.asarray([0.0, 2.0, 4.0]),
        np.asarray([[4.0, 8.0, 12.0, 16.0]]),
    )

    np.testing.assert_allclose(rate, [9.0, 21.0])


def test_refinement_screening_computes_only_new_candidate_poses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refined candidate set must reuse every unchanged screening action."""
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(_floor_patch(0, 0.0),),
        density_by_isotope=np.asarray([[1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )
    history = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.5, 0.5, 0.5]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
        live_times_s=np.asarray([1.0]),
        spectrum_counts=np.ones((1, 2)),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        isotope_counts=None,
        isotope_covariances=None,
        station_ids=np.asarray([0]),
        isotope_names=("Cs-137",),
    )

    def fake_history(
        *args: object,
        **kwargs: object,
    ) -> tuple[_LineSpectralDesign, dict[str, object]]:
        """Return one compact historical source and no nuisance columns."""
        del args, kwargs
        return (
            _LineSpectralDesign(
                spatial_factors=np.ones((1, 1, 1), dtype=np.float64),
                pulse_shapes=np.ones((1, 2), dtype=np.float64),
                line_isotope_indices=np.zeros(1, dtype=np.int64),
                nuisance_response=np.zeros((1, 2, 0), dtype=np.float64),
                nuisance_names=(),
                energy_chunk_size=1,
            ),
            {},
        )

    computed_rows: list[int] = []

    def fake_screening_design(
        observations: object,
        patches: object,
        isotopes: object,
        kernel: object,
        mle_config: object,
    ) -> np.ndarray:
        """Return deterministic response rows while recording new work only."""
        del isotopes, kernel, mle_config
        count = int(np.asarray(observations.detector_positions_xyz).shape[0])
        patch_count = int(np.asarray(patches.areas_m2).size)
        computed_rows.append(count)
        return np.ones((count, 2, patch_count, 1), dtype=np.float64)

    monkeypatch.setattr(
        information_planner,
        "_historical_factorized_spectral_design",
        fake_history,
    )
    monkeypatch.setattr(
        information_planner,
        "_screening_spectral_design",
        fake_screening_design,
    )
    kernel = ContinuousKernel(use_gpu=False)
    config = MLEPlanningConfig(
        shield_program_length=1,
        screening_pair_limit=1,
        screening_energy_bin_count=2,
        screening_pose_chunk_size=8,
        ranked_action_limit=3,
        exact_candidate_min=1,
        exact_candidate_max=1,
    )
    cache: dict[str, object] = {}
    first_poses = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    refined_poses = np.vstack((first_poses, [[0.5, 0.5, 1.0]]))
    common = (
        estimate,
        history,
        kernel,
        MLEConfig(mode="spectral", isotope_names=("Cs-137",), use_gpu=False),
    )

    _screen_candidate_measurements(
        *common,
        first_poses,
        np.asarray([0], dtype=np.int64),
        np.zeros(2),
        0,
        config,
        cache,
        None,
    )
    refined = _screen_candidate_measurements(
        *common,
        refined_poses,
        np.asarray([0], dtype=np.int64),
        np.zeros(3),
        0,
        config,
        cache,
        None,
    )
    uncached_refined = _screen_candidate_measurements(
        *common,
        refined_poses,
        np.asarray([0], dtype=np.int64),
        np.zeros(3),
        0,
        config,
        None,
        None,
    )
    changed_live_time = _screen_candidate_measurements(
        *common,
        refined_poses,
        np.asarray([0], dtype=np.int64),
        np.zeros(3),
        0,
        replace(config, live_time_s=2.0 * config.live_time_s),
        cache,
        None,
    )
    changed_history = replace(
        history,
        spectrum_counts=2.0 * history.spectrum_counts,
    )
    changed_counts = _screen_candidate_measurements(
        estimate,
        changed_history,
        kernel,
        common[3],
        refined_poses,
        np.asarray([0], dtype=np.int64),
        np.zeros(3),
        0,
        replace(config, live_time_s=2.0 * config.live_time_s),
        cache,
        None,
    )

    assert computed_rows == [2, 1, 3, 3, 3]
    assert refined.diagnostics["reused_candidates"] == 2
    assert refined.diagnostics["computed_candidates"] == 1
    assert changed_live_time.diagnostics["reused_candidates"] == 0
    assert changed_live_time.diagnostics["computed_candidates"] == 3
    assert changed_counts.diagnostics["reused_candidates"] == 0
    assert changed_counts.diagnostics["computed_candidates"] == 3
    cached_by_index = {
        action.candidate_index: action for action in refined.ranked_actions
    }
    for action in uncached_refined.ranked_actions:
        cached_action = cached_by_index[action.candidate_index]
        np.testing.assert_allclose(cached_action.score, action.score)
        np.testing.assert_allclose(
            cached_action.geometry_exploration,
            action.geometry_exploration,
        )


def test_kernel_cache_identity_changes_with_physical_mutation() -> None:
    """Planner caches must not survive an in-place physical-kernel change."""
    kernel = ContinuousKernel(use_gpu=False)
    original = information_planner._kernel_physical_identity(kernel)
    kernel.obstacle_height_m = float(kernel.obstacle_height_m) + 0.5
    changed = information_planner._kernel_physical_identity(kernel)
    kernel.gpu_device = "cuda:1"

    assert changed != original
    assert information_planner._kernel_physical_identity(kernel) == changed


def _orientations() -> np.ndarray:
    """Return two unit directions for four synthetic Fe/Pb pair IDs."""
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def _floor_patch(patch_id: int, x_offset: float) -> SurfacePatch:
    """Return one exact unit floor patch for basis tests."""
    return SurfacePatch(
        patch_id=patch_id,
        centroid_xyz=np.asarray([x_offset + 0.5, 0.5, 0.0]),
        normal_xyz=np.asarray([0.0, 0.0, 1.0]),
        area_m2=1.0,
        surface_kind="floor",
        object_id=f"floor:{patch_id}",
        vertices_xyz=np.asarray(
            [
                [x_offset, 0.0, 0.0],
                [x_offset + 1.0, 0.0, 0.0],
                [x_offset + 1.0, 1.0, 0.0],
                [x_offset, 1.0, 0.0],
            ]
        ),
        quadrature_points_xyz=np.asarray([[x_offset + 0.5, 0.5, 0.0]]),
        quadrature_weights=np.asarray([1.0]),
    )


def _ceiling_patch(patch_id: int, x_offset: float) -> SurfacePatch:
    """Return one exact unit ceiling patch for vertical ambiguity tests."""
    return SurfacePatch(
        patch_id=patch_id,
        centroid_xyz=np.asarray([x_offset + 0.5, 0.5, 2.0]),
        normal_xyz=np.asarray([0.0, 0.0, -1.0]),
        area_m2=1.0,
        surface_kind="ceiling",
        object_id=f"ceiling:{patch_id}",
        vertices_xyz=np.asarray(
            [
                [x_offset, 0.0, 2.0],
                [x_offset, 1.0, 2.0],
                [x_offset + 1.0, 1.0, 2.0],
                [x_offset + 1.0, 0.0, 2.0],
            ]
        ),
        quadrature_points_xyz=np.asarray([[x_offset + 0.5, 0.5, 2.0]]),
        quadrature_weights=np.asarray([1.0]),
    )


def _planner_regression_fixture() -> tuple[
    MLEEstimate,
    ObservationBatch,
    ContinuousKernel,
    MLEConfig,
    tuple[int, ...],
    MLEPlanningConfig,
]:
    """Return one small line-resolved physical planning regression fixture."""
    patches = (
        _floor_patch(0, 0.0),
        _floor_patch(1, 1.0),
        _ceiling_patch(2, 0.0),
        _ceiling_patch(3, 1.0),
    )
    strengths = np.asarray([[2.0, 0.5, 1.5, 0.2]])
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=strengths.copy(),
        patch_strength_by_isotope=strengths.copy(),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )
    historical_positions = np.asarray(
        [[0.5, -1.0, 1.0], [2.5, -1.0, 1.0], [1.0, 2.5, 0.5]],
        dtype=np.float64,
    )
    energy_edges = np.linspace(0.0, 800.0, 65)
    history = ObservationBatch(
        detector_positions_xyz=historical_positions,
        detector_quaternions_wxyz=np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            (3, 1),
        ),
        fe_indices=np.asarray([0, 1, 2]),
        pb_indices=np.asarray([0, 1, 2]),
        live_times_s=np.full(3, 10.0),
        spectrum_counts=np.ones((3, 64)),
        spectrum_variances=None,
        energy_bin_edges_keV=energy_edges,
        isotope_counts=None,
        isotope_covariances=None,
        station_ids=np.arange(3),
        isotope_names=("Cs-137",),
    )
    kernel = ContinuousKernel(
        use_gpu=False,
        mu_by_isotope={"Cs-137": {"fe": 0.1, "pb": 0.2}},
        line_mu_by_isotope={
            "Cs-137": (
                {
                    "energy_keV": 661.7,
                    "weight": 1.0,
                    "fe": 0.1,
                    "pb": 0.2,
                },
            )
        },
    )
    mle_config = MLEConfig(
        mode="spectral",
        isotope_names=("Cs-137",),
        fit_background_nuisance=False,
        fit_scatter_nuisance=False,
        response_energy_chunk_size=7,
        use_gpu=False,
    )
    planning_config = MLEPlanningConfig(
        shield_program_length=2,
        ranked_action_limit=32,
        candidate_pose_chunk_size=8,
        screening_pair_limit=4,
        screening_pose_chunk_size=64,
        exact_candidate_min=4,
        exact_candidate_max=8,
    )
    return (
        estimate,
        history,
        kernel,
        mle_config,
        (0, 7, 56, 63),
        planning_config,
    )


def _planner_regression_poses(seed: int) -> np.ndarray:
    """Return the fixed 40-pose candidate cloud used by planner regressions."""
    generator = np.random.default_rng(seed)
    return np.column_stack(
        (
            generator.uniform(-2.0, 4.0, 40),
            generator.uniform(-3.0, 4.0, 40),
            generator.uniform(0.15, 1.85, 40),
        )
    )


def test_exact_ambiguity_is_invariant_to_candidate_pose_chunk_size() -> None:
    """Performance chunks must not change the exact action, program, or score."""
    estimate, history, kernel, mle_config, pair_ids, config = (
        _planner_regression_fixture()
    )
    poses = _planner_regression_poses(17)
    results = tuple(
        _plan_next_measurement_exact(
            estimate,
            history,
            kernel,
            mle_config,
            poses,
            planning_config=replace(
                config,
                two_stage_screening=False,
                ranked_action_limit=40,
                candidate_pose_chunk_size=chunk_size,
            ),
            allowed_pair_ids=pair_ids,
        )
        for chunk_size in (1, 8, 40)
    )

    selected = tuple(
        (
            result.selected_action.candidate_index,
            result.selected_action.shield_pair_ids,
        )
        for result in results
    )
    assert selected == ((30, (7, 0)),) * 3
    np.testing.assert_allclose(
        [result.selected_action.score for result in results],
        results[0].selected_action.score,
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_two_stage_screening_retains_known_full_exact_winner() -> None:
    """Ambiguity-aware screening must retain the fixed full-exact winner."""
    estimate, history, kernel, mle_config, pair_ids, config = (
        _planner_regression_fixture()
    )
    poses = _planner_regression_poses(10)
    full = _plan_next_measurement_exact(
        estimate,
        history,
        kernel,
        mle_config,
        poses,
        planning_config=replace(
            config,
            two_stage_screening=False,
            ranked_action_limit=40,
            candidate_pose_chunk_size=40,
        ),
        allowed_pair_ids=pair_ids,
    )
    staged = plan_next_measurement(
        estimate,
        history,
        kernel,
        mle_config,
        poses,
        planning_config=config,
        allowed_pair_ids=pair_ids,
    )

    assert full.selected_action.candidate_index == 29
    assert 29 in staged.diagnostics["exact_candidate_indices"]
    assert staged.selected_action.candidate_index == 29
    assert (
        staged.selected_action.shield_pair_ids == full.selected_action.shield_pair_ids
    )
    np.testing.assert_allclose(
        staged.selected_action.score,
        full.selected_action.score,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    full_by_candidate = {
        action.candidate_index: action for action in full.ranked_actions
    }
    for action in staged.ranked_actions:
        oracle = full_by_candidate[action.candidate_index]
        assert action.shield_pair_ids == oracle.shield_pair_ids
        np.testing.assert_allclose(
            action.score,
            oracle.score,
            rtol=1.0e-14,
            atol=1.0e-14,
        )


def test_screening_cache_tracks_fitted_nuisance_state() -> None:
    """Candidate reuse must invalidate when fitted nuisance precision changes."""
    estimate, history, kernel, mle_config, pair_ids, config = (
        _planner_regression_fixture()
    )
    poses = _planner_regression_poses(3)[:4]
    config = replace(
        config,
        screening_pair_limit=2,
        screening_pose_chunk_size=4,
        ranked_action_limit=4,
        exact_candidate_min=2,
        exact_candidate_max=4,
    )
    mle_config = replace(mle_config, fit_background_nuisance=True)
    diagnostics = {"nuisance_names": ["background_rate_cps"]}
    low_background = replace(
        estimate,
        background_parameters=np.asarray([0.01]),
        diagnostics=diagnostics,
    )
    high_background = replace(
        estimate,
        background_parameters=np.asarray([100.0]),
        diagnostics=diagnostics,
    )
    common = (
        history,
        kernel,
        mle_config,
        poses,
        np.asarray(pair_ids[:2], dtype=np.int64),
        np.zeros(poses.shape[0]),
        None,
        config,
    )
    cache: dict[str, object] = {}

    low = _screen_candidate_measurements(
        low_background,
        *common,
        cache,
        None,
    )
    invalidated = _screen_candidate_measurements(
        high_background,
        *common,
        cache,
        None,
    )
    fresh = _screen_candidate_measurements(
        high_background,
        *common,
        None,
        None,
    )
    reused = _screen_candidate_measurements(
        high_background,
        *common,
        cache,
        None,
    )
    relabeled = replace(
        high_background,
        patches=tuple(
            replace(patch, patch_id=100 + index)
            for index, patch in enumerate(high_background.patches)
        ),
    )
    relabeled_result = _screen_candidate_measurements(
        relabeled,
        *common,
        cache,
        None,
    )

    assert invalidated.diagnostics["computed_candidates"] == poses.shape[0]
    assert invalidated.diagnostics["reused_candidates"] == 0
    assert reused.diagnostics["computed_candidates"] == 0
    assert reused.diagnostics["reused_candidates"] == poses.shape[0]
    assert relabeled_result.diagnostics["computed_candidates"] == poses.shape[0]
    assert relabeled_result.diagnostics["reused_candidates"] == 0
    assert invalidated.selected_action.to_dict() == fresh.selected_action.to_dict()
    assert reused.selected_action.to_dict() == fresh.selected_action.to_dict()
    assert tuple(action.to_dict() for action in invalidated.ranked_actions) == tuple(
        action.to_dict() for action in fresh.ranked_actions
    )
    assert tuple(action.to_dict() for action in reused.ranked_actions) == tuple(
        action.to_dict() for action in fresh.ranked_actions
    )
    assert not np.isclose(
        low.selected_action.score,
        fresh.selected_action.score,
    )


def test_screening_projects_alternative_support_to_pseudo_patches() -> None:
    """Screening must retain support alternatives after pseudo compression."""
    estimate, history, kernel, mle_config, pair_ids, config = (
        _planner_regression_fixture()
    )
    patches = (
        _floor_patch(0, 0.0),
        _floor_patch(1, 10.0),
        _floor_patch(2, 20.0),
    )
    poses = np.asarray(
        [
            [0.5, -1.0, 1.0],
            [10.5, -1.0, 1.0],
            [20.5, -1.0, 1.0],
            [5.5, 3.0, 1.0],
        ],
        dtype=np.float64,
    )
    config = replace(
        config,
        screening_pair_limit=2,
        screening_pose_chunk_size=4,
        screening_points_per_mode=2,
        ranked_action_limit=4,
        exact_candidate_min=2,
        exact_candidate_max=4,
    )
    estimate = replace(
        estimate,
        patches=patches,
        density_by_isotope=np.asarray([[5.0, 0.0, 0.0]]),
        patch_strength_by_isotope=np.asarray([[5.0, 0.0, 0.0]]),
    )
    alternative = replace(
        estimate,
        density_by_isotope=np.asarray([[0.0, 0.0, 5.0]]),
        patch_strength_by_isotope=np.asarray([[0.0, 0.0, 5.0]]),
    )
    view, strengths, pseudo_basis, _, _ = _screening_pseudo_model(estimate, config)
    base_projection = np.einsum(
        "rgi,gi->ri",
        view.strength_projection,
        estimate.patch_strength_by_isotope.T,
        optimize=True,
    )
    alternative_projection = np.einsum(
        "rgi,gi->ri",
        view.strength_projection,
        alternative.patch_strength_by_isotope.T,
        optimize=True,
    )
    base_row = int(np.argmax(base_projection[:, 0]))
    alternative_row = int(np.argmax(alternative_projection[:, 0]))
    common = (
        estimate,
        history,
        kernel,
        mle_config,
        poses,
        np.asarray(pair_ids[:2], dtype=np.int64),
        np.zeros(poses.shape[0]),
        None,
        config,
        None,
        None,
    )

    assert view.quadrature_points_xyz.shape == (2, 1, 3)
    np.testing.assert_allclose(
        np.sort(view.quadrature_points_xyz[:, 0, 0]),
        [0.5, 20.5],
    )
    assert base_row != alternative_row
    np.testing.assert_allclose(base_projection, strengths)
    assert pseudo_basis[alternative_row, 0, 0] == 0.0
    baseline = _screen_candidate_measurements(*common)
    projected = _screen_candidate_measurements(
        *common,
        alternative_estimates=(alternative,),
    )

    assert all(
        action.support_hypothesis_separation == 0.0
        for action in baseline.ranked_actions
    )
    assert any(
        action.support_hypothesis_separation > 0.0
        for action in projected.ranked_actions
    )


def test_d_optimal_pose_balances_source_parameter_information() -> None:
    """D-optimality should prefer a balanced reduction in source uncertainty."""
    information = np.zeros((2, 1, 2, 2), dtype=float)
    information[0, 0] = np.diag([9.0, 0.0])
    information[1, 0] = np.diag([3.0, 3.0])

    selected, ranked = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        (0,),
        information,
        np.ones((2, 1)),
        np.eye(2),
        _orientations(),
        nuisance_count=0,
        config=MLEPlanningConfig(shield_program_length=1),
    )

    assert selected.candidate_index == 1
    assert ranked[0] is selected
    assert selected.information_gain_nats > ranked[1].information_gain_nats


def test_joint_shield_program_chooses_complementary_orientations() -> None:
    """A jointly selected two-view station should span both sensitivities."""
    information = np.zeros((1, 3, 2, 2), dtype=float)
    information[0, 0] = np.diag([9.0, 0.0])
    information[0, 1] = np.diag([8.0, 0.0])
    information[0, 2] = np.diag([0.0, 9.0])

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0]]),
        (0, 1, 2),
        information,
        np.ones((1, 3)),
        np.eye(2),
        _orientations(),
        nuisance_count=0,
        config=MLEPlanningConfig(shield_program_length=2),
    )

    assert selected.shield_pair_ids == (0, 2)
    assert selected.fe_orientation_indices == (0, 1)
    assert selected.pb_orientation_indices == (0, 0)
    assert selected.live_time_s_by_view == (10.0, 10.0)
    assert selected.to_dict()["measurement_program"][-1]["station_complete"] is True


def test_joint_program_marginalizes_one_shared_station_rate() -> None:
    """A station block should prefer rate-independent shield contrast."""
    information = np.zeros((1, 3, 1, 1), dtype=float)
    information[0, :, 0, 0] = [100.0, 90.25, 10.0]
    station_cross = np.asarray([[[10.0], [9.5], [0.0]]])
    station_information = np.ones((1, 3), dtype=float)

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0]]),
        (0, 1, 2),
        information,
        np.ones((1, 3)),
        np.eye(1),
        _orientations(),
        nuisance_count=0,
        station_rate_cross_information=station_cross,
        station_rate_information=station_information,
        config=MLEPlanningConfig(
            shield_program_length=2,
            future_station_rate_prior_precision=0.01,
        ),
    )

    assert selected.shield_pair_ids == (0, 2)


def test_joint_program_uses_pair_specific_ambiguity_utility() -> None:
    """Shield choice should include ambiguity utility during optimization."""
    information = np.ones((1, 3, 1, 1), dtype=float)

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0]]),
        (0, 1, 2),
        information,
        np.ones((1, 3)),
        np.eye(1),
        _orientations(),
        nuisance_count=0,
        pair_utility_bonus=np.asarray([[0.0, 0.0, 2.0]]),
        config=MLEPlanningConfig(shield_program_length=1),
    )

    assert selected.shield_pair_ids == (2,)


def test_factorized_fisher_matches_dense_response_without_energy_expansion() -> None:
    """Line-factor Fisher reduction must equal the dense deterministic oracle."""
    rng = np.random.default_rng(20260822)
    action_count, bin_count, patch_count, isotope_count, line_count = (
        5,
        13,
        4,
        2,
        3,
    )
    spatial = rng.uniform(1.0e-4, 2.0e-2, (action_count, patch_count, line_count))
    pulses = rng.uniform(0.0, 1.0, (line_count, bin_count))
    line_isotopes = np.asarray([0, 1, 0], dtype=np.int64)
    nuisance = rng.uniform(1.0e-4, 1.0e-2, (action_count, bin_count, 2))
    design = _LineSpectralDesign(
        spatial_factors=spatial,
        pulse_shapes=pulses,
        line_isotope_indices=line_isotopes,
        nuisance_response=nuisance,
        nuisance_names=("background", "scatter"),
        energy_chunk_size=4,
    )
    dense = np.zeros(
        (action_count, bin_count, patch_count, isotope_count),
        dtype=np.float64,
    )
    for line_index, isotope_index in enumerate(line_isotopes):
        dense[:, :, :, isotope_index] += np.einsum(
            "mg,b->mbg",
            spatial[:, :, line_index],
            pulses[line_index],
        )
    basis = rng.uniform(0.0, 1.0, (patch_count, isotope_count, 3))
    strengths = rng.uniform(0.1, 3.0, (patch_count, isotope_count))
    coefficients = np.asarray([0.4, 0.8])
    scales = np.asarray([1.0, 1.5])

    expected = _fisher_information(
        dense,
        nuisance,
        basis,
        strengths,
        coefficients,
        scales,
        minimum_expected_count=1.0e-3,
    )
    actual = _factorized_fisher_information(
        design,
        basis,
        strengths,
        coefficients,
        scales,
        minimum_expected_count=1.0e-3,
    )

    for actual_values, expected_values in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=2.0e-13,
            atol=2.0e-14,
        )


def test_fisher_floor_and_overdispersion_preserve_station_derivatives() -> None:
    """A numerical mean floor must not invent station-rate information."""
    response = np.asarray([[[[1.0]], [[2.0]]]], dtype=np.float64)
    nuisance = np.zeros((1, 2, 0), dtype=np.float64)
    basis = np.ones((1, 1, 1), dtype=np.float64)
    strengths = np.asarray([[0.1]], dtype=np.float64)
    alpha = np.asarray([0.0, 1.0], dtype=np.float64)
    dense = _fisher_information(
        response,
        nuisance,
        basis,
        strengths,
        np.zeros(0),
        np.zeros(0),
        minimum_expected_count=1.0,
        overdispersion_alpha_by_bin=alpha,
    )
    factor = _factorized_fisher_information(
        _LineSpectralDesign(
            spatial_factors=np.ones((1, 1, 1), dtype=np.float64),
            pulse_shapes=np.asarray([[1.0, 2.0]], dtype=np.float64),
            line_isotope_indices=np.zeros(1, dtype=np.int64),
            nuisance_response=nuisance,
            nuisance_names=(),
            energy_chunk_size=1,
            overdispersion_alpha_by_bin=alpha,
        ),
        basis,
        strengths,
        np.zeros(0),
        np.zeros(0),
        minimum_expected_count=1.0,
    )

    for actual, expected in zip(factor, dense, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(dense[0], [[[3.0]]])
    np.testing.assert_allclose(dense[1], [0.3])
    np.testing.assert_allclose(dense[2], [[0.3]])
    np.testing.assert_allclose(dense[3], [0.03])


def test_planning_prior_scales_fitted_nuisance_regularization() -> None:
    """Normalized nuisance coordinates need L2 precision scaled by scale squared."""
    prior = _planning_prior_precision(
        2,
        np.asarray([2.0, 3.0]),
        np.asarray([4.0, 5.0]),
        laplace_prior_precision=0.1,
    )

    np.testing.assert_allclose(np.diag(prior), [0.1, 0.1, 16.1, 45.1])


def test_factorized_design_uses_effective_nuisance_weight_and_likelihood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner factors must match estimator regularization and Poisson variance."""
    operator = SimpleNamespace(
        spatial_factors=np.ones((1, 1, 1), dtype=np.float64),
        pulse_shapes=np.asarray([[0.25, 0.75]], dtype=np.float64),
        line_isotope_indices=np.asarray([0], dtype=np.int64),
    )
    details = SimpleNamespace(
        operator=operator,
        nuisance_response=np.ones((1, 2, 1), dtype=np.float64),
        nuisance_names=("background_rate_cps",),
        nuisance_l2_weights=np.asarray([2.0]),
        overdispersion_alpha_by_bin=np.asarray([0.1, 0.2]),
    )
    monkeypatch.setattr(
        information_planner,
        "build_spectral_response_operator",
        lambda *_args, **_kwargs: details,
    )
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(_floor_patch(0, 0.0),),
        density_by_isotope=np.asarray([[1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )

    design = information_planner._factorized_spectral_design(
        object(),
        estimate,
        object(),  # type: ignore[arg-type]
        MLEConfig(
            mode="spectral",
            isotope_names=("Cs-137",),
            nuisance_l2_weight=3.0,
        ),
    )

    np.testing.assert_array_equal(design.nuisance_l2_weights, [5.0])
    np.testing.assert_array_equal(design.overdispersion_alpha_by_bin, [0.0, 0.0])


def test_planner_cpu_gpu_equivalence_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Float64 CUDA Fisher and eight-view beam results must match CPU results."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(20260806)
    response = rng.uniform(1.0e-5, 1.0e-2, size=(9, 12, 3, 2))
    nuisance = rng.uniform(1.0e-4, 1.0e-2, size=(9, 12, 2))
    basis = rng.uniform(0.0, 1.0, size=(3, 2, 4))
    strengths = rng.uniform(0.1, 4.0, size=(3, 2))
    coefficients = np.asarray([0.5, 1.5])
    scales = np.asarray([1.0, 1.5])
    cpu_fisher = _fisher_information(
        response,
        nuisance,
        basis,
        strengths,
        coefficients,
        scales,
        minimum_expected_count=1.0e-3,
    )
    gpu_fisher = _fisher_information(
        response,
        nuisance,
        basis,
        strengths,
        coefficients,
        scales,
        minimum_expected_count=1.0e-3,
        use_gpu=True,
    )
    for gpu_values, cpu_values in zip(gpu_fisher, cpu_fisher, strict=True):
        np.testing.assert_allclose(
            gpu_values,
            cpu_values,
            rtol=2.0e-12,
            atol=2.0e-13,
        )

    line_isotopes = np.asarray([0, 1, 0], dtype=np.int64)
    factor_design = _LineSpectralDesign(
        spatial_factors=rng.uniform(1.0e-5, 1.0e-2, (9, 3, 3)),
        pulse_shapes=rng.uniform(0.0, 1.0, (3, 12)),
        line_isotope_indices=line_isotopes,
        nuisance_response=nuisance,
        nuisance_names=("background", "scatter"),
        energy_chunk_size=5,
        overdispersion_alpha_by_bin=rng.uniform(0.0, 0.2, size=12),
    )
    factor_cpu = _factorized_fisher_information(
        factor_design,
        basis,
        strengths,
        coefficients,
        scales,
        minimum_expected_count=1.0e-3,
    )
    factor_gpu = _factorized_fisher_information(
        factor_design,
        basis,
        strengths,
        coefficients,
        scales,
        minimum_expected_count=1.0e-3,
        use_gpu=True,
    )
    for gpu_values, cpu_values in zip(factor_gpu, factor_cpu, strict=True):
        np.testing.assert_allclose(
            gpu_values,
            cpu_values,
            rtol=2.0e-12,
            atol=2.0e-13,
        )

    pair_count = 9
    parameter_count = 24
    jacobian = rng.normal(size=(2, pair_count, parameter_count, 3))
    information = (
        np.einsum(
            "cpik,cpjk->cpij",
            jacobian,
            jacobian,
            optimize=True,
        )
        * 1.0e-3
    )
    poses = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    orientations = np.eye(3, dtype=np.float64)
    expected = rng.uniform(1.0, 10.0, size=(2, pair_count))
    bonuses = rng.uniform(0.0, 0.1, size=(2, pair_count))
    config = MLEPlanningConfig(
        shield_program_length=8,
        shield_program_beam_width=64,
    )
    cpu_selected, cpu_ranked = select_fisher_action(
        poses,
        range(pair_count),
        information,
        expected,
        np.eye(parameter_count),
        orientations,
        nuisance_count=2,
        config=config,
        pair_utility_bonus=bonuses,
    )
    monkeypatch.setattr(
        information_planner,
        "_BEAM_PRECISION_WORKSPACE_LIMIT_BYTES",
        3 * parameter_count**2 * 8 * 64,
    )
    gpu_selected, gpu_ranked = select_fisher_action(
        poses,
        range(pair_count),
        information,
        expected,
        np.eye(parameter_count),
        orientations,
        nuisance_count=2,
        config=config,
        pair_utility_bonus=bonuses,
        use_gpu=True,
    )

    assert gpu_selected.shield_pair_ids == cpu_selected.shield_pair_ids
    assert [action.shield_pair_ids for action in gpu_ranked] == [
        action.shield_pair_ids for action in cpu_ranked
    ]
    assert gpu_selected.score == pytest.approx(cpu_selected.score, abs=1.0e-12)
    assert gpu_selected.information_gain_nats == pytest.approx(
        cpu_selected.information_gain_nats,
        abs=1.0e-12,
    )

    small_information = information[:, :, :5, :5]
    small_cpu, _ = select_fisher_action(
        poses,
        range(pair_count),
        small_information,
        expected,
        np.eye(5),
        orientations,
        nuisance_count=0,
        config=config,
        pair_utility_bonus=bonuses,
    )
    small_gpu, _ = select_fisher_action(
        poses,
        range(pair_count),
        small_information,
        expected,
        np.eye(5),
        orientations,
        nuisance_count=0,
        config=config,
        pair_utility_bonus=bonuses,
        use_gpu=True,
    )
    assert small_gpu.shield_pair_ids == small_cpu.shield_pair_ids
    assert small_gpu.score == pytest.approx(small_cpu.score, abs=1.0e-12)


def test_historical_design_appends_only_new_station_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A causal history extension must compute only its new response rows."""
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(_floor_patch(0, 0.0),),
        density_by_isotope=np.asarray([[1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.asarray([0.5]),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )

    def history(count: int, *, position_offset: float = 0.0) -> ObservationBatch:
        """Return a deterministic causal spectrum prefix."""
        positions = np.zeros((count, 3), dtype=np.float64)
        positions[:, 0] = np.arange(count, dtype=np.float64) + position_offset
        return ObservationBatch(
            detector_positions_xyz=positions,
            detector_quaternions_wxyz=np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                (count, 1),
            ),
            fe_indices=np.zeros(count, dtype=np.int64),
            pb_indices=np.zeros(count, dtype=np.int64),
            live_times_s=np.ones(count),
            spectrum_counts=np.ones((count, 2)),
            spectrum_variances=None,
            energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
            isotope_counts=np.ones((count, 1)),
            isotope_covariances=np.ones((count, 1, 1)),
            station_ids=np.arange(count, dtype=np.int64),
            isotope_names=("Cs-137",),
        )

    computed_counts: list[int] = []

    def fake_design(
        observations: object,
        _estimate: MLEEstimate,
        _kernel: object,
        _config: MLEConfig,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Return row-identifiable arrays while recording computed rows."""
        positions = np.asarray(observations.detector_positions_xyz)
        count = int(positions.shape[0])
        computed_counts.append(count)
        values = positions[:, 0, None, None, None] + np.ones((count, 2, 1, 1))
        nuisance = positions[:, 0, None, None] + np.ones((count, 2, 1))
        return values, nuisance, ("background_rate_cps",)

    monkeypatch.setattr(information_planner, "_spectral_design", fake_design)
    cache: dict[str, object] = {}
    kernel = object()
    config = MLEConfig(mode="spectral", isotope_names=("Cs-137",))
    first = _historical_spectral_design(
        history(2),
        estimate,
        kernel,
        config,
        cache,  # type: ignore[arg-type]
    )
    extended = _historical_spectral_design(
        history(3),
        estimate,
        kernel,
        config,
        cache,  # type: ignore[arg-type]
    )
    hit = _historical_spectral_design(
        history(3),
        estimate,
        kernel,
        config,
        cache,  # type: ignore[arg-type]
    )
    changed_geometry = _historical_spectral_design(
        history(3, position_offset=0.25),
        estimate,
        kernel,
        config,
        cache,  # type: ignore[arg-type]
    )

    assert computed_counts == [2, 1, 3]
    assert first[3]["mode"] == "full_rebuild"
    assert extended[3] == {
        "mode": "prefix_append",
        "reused_measurements": 2,
        "computed_measurements": 1,
    }
    assert hit[3]["mode"] == "prefix_hit"
    assert changed_geometry[3]["mode"] == "full_rebuild"
    np.testing.assert_array_equal(extended[0], hit[0])
    np.testing.assert_array_equal(extended[0][:2], first[0])

    fisher_cache: dict[str, object] = {}
    basis = np.ones((1, 1, 1), dtype=np.float64)
    strengths = np.ones((1, 1), dtype=np.float64)
    coefficients = np.asarray([0.5])
    scales = np.asarray([1.0])
    first_fisher = _historical_fisher_precision(
        first[0],
        first[1],
        basis,
        strengths,
        coefficients,
        scales,
        (0, 1),
        model_identity="model-a",
        minimum_expected_count=1.0e-6,
        cache=fisher_cache,
    )
    extended_fisher = _historical_fisher_precision(
        extended[0],
        extended[1],
        basis,
        strengths,
        coefficients,
        scales,
        (0, 1, 2),
        model_identity="model-a",
        minimum_expected_count=1.0e-6,
        cache=fisher_cache,
    )
    full_fisher = _historical_fisher_precision(
        extended[0],
        extended[1],
        basis,
        strengths,
        coefficients,
        scales,
        (0, 1, 2),
        model_identity="model-a",
        minimum_expected_count=1.0e-6,
        cache=None,
    )
    changed_model_fisher = _historical_fisher_precision(
        extended[0],
        extended[1],
        basis,
        strengths,
        coefficients,
        scales,
        (0, 1, 2),
        model_identity="model-b",
        minimum_expected_count=1.0e-6,
        cache=fisher_cache,
    )
    changed_estimate_fisher = _historical_fisher_precision(
        extended[0],
        extended[1],
        basis,
        2.0 * strengths,
        coefficients,
        scales,
        (0, 1, 2),
        model_identity="model-b",
        minimum_expected_count=1.0e-6,
        cache=fisher_cache,
    )

    assert first_fisher[1]["mode"] == "full_rebuild"
    assert extended_fisher[1]["mode"] == "prefix_append"
    np.testing.assert_allclose(extended_fisher[0], full_fisher[0], rtol=1.0e-15)
    assert changed_model_fisher[1]["mode"] == "full_rebuild"
    assert changed_estimate_fisher[1]["mode"] == "full_rebuild"

    computed_counts.clear()
    drift_cache: dict[str, object] = {}
    drift_config = MLEConfig(
        mode="spectral",
        isotope_names=("Cs-137",),
        fit_gain_resolution_drift=True,
        discrepancy_calibration_path="/nonexistent/test-calibration.json",
    )
    _historical_spectral_design(
        history(2),
        estimate,
        kernel,
        drift_config,
        drift_cache,  # type: ignore[arg-type]
    )
    drift_extended = _historical_spectral_design(
        history(3),
        estimate,
        kernel,
        drift_config,
        drift_cache,  # type: ignore[arg-type]
    )

    assert computed_counts == [2, 3]
    assert drift_extended[3]["mode"] == "full_rebuild"


def test_factorized_history_and_fisher_append_only_new_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact exact planning caches must preserve causal prefix reuse."""
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(_floor_patch(0, 0.0),),
        density_by_isotope=np.asarray([[1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.asarray([0.5]),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )

    def history(count: int) -> ObservationBatch:
        """Return a deterministic factorized-planner history prefix."""
        positions = np.zeros((count, 3), dtype=np.float64)
        positions[:, 0] = np.arange(count, dtype=np.float64)
        return ObservationBatch(
            detector_positions_xyz=positions,
            detector_quaternions_wxyz=np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                (count, 1),
            ),
            fe_indices=np.zeros(count, dtype=np.int64),
            pb_indices=np.zeros(count, dtype=np.int64),
            live_times_s=np.ones(count),
            spectrum_counts=np.ones((count, 2)),
            spectrum_variances=None,
            energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
            isotope_counts=np.ones((count, 1)),
            isotope_covariances=np.ones((count, 1, 1)),
            station_ids=np.arange(count, dtype=np.int64),
            isotope_names=("Cs-137",),
        )

    computed_counts: list[int] = []

    def fake_design(
        observations: object,
        _estimate: MLEEstimate,
        _kernel: object,
        _config: MLEConfig,
    ) -> _LineSpectralDesign:
        """Return compact row-identifiable factors and record new work."""
        positions = np.asarray(observations.detector_positions_xyz)
        count = int(positions.shape[0])
        computed_counts.append(count)
        amplitudes = positions[:, 0] + 1.0
        return _LineSpectralDesign(
            spatial_factors=amplitudes[:, None, None],
            pulse_shapes=np.asarray([[0.25, 0.75]]),
            line_isotope_indices=np.asarray([0], dtype=np.int64),
            nuisance_response=np.broadcast_to(
                amplitudes[:, None, None],
                (count, 2, 1),
            ).copy(),
            nuisance_names=("background_rate_cps",),
            energy_chunk_size=1,
        )

    monkeypatch.setattr(
        information_planner,
        "_factorized_spectral_design",
        fake_design,
    )
    response_cache: dict[str, object] = {}
    kernel = object()
    config = MLEConfig(
        mode="spectral",
        isotope_names=("Cs-137",),
        discrepancy_calibration_path="/nonexistent/test-calibration.json",
    )
    first = _historical_factorized_spectral_design(
        history(2),
        estimate,
        kernel,  # type: ignore[arg-type]
        config,
        response_cache,
    )
    extended = _historical_factorized_spectral_design(
        history(3),
        estimate,
        kernel,  # type: ignore[arg-type]
        config,
        response_cache,
    )
    hit = _historical_factorized_spectral_design(
        history(3),
        estimate,
        kernel,  # type: ignore[arg-type]
        config,
        response_cache,
    )
    likelihood_rebuilt = _historical_factorized_spectral_design(
        history(3),
        estimate,
        kernel,  # type: ignore[arg-type]
        replace(config, spectral_likelihood="calibrated_overdispersed"),
        response_cache,
    )
    regularization_rebuilt = _historical_factorized_spectral_design(
        history(3),
        estimate,
        kernel,  # type: ignore[arg-type]
        replace(
            config,
            spectral_likelihood="calibrated_overdispersed",
            nuisance_l2_weight=1.0,
        ),
        response_cache,
    )

    assert computed_counts == [2, 1, 3, 3]
    assert first[1]["mode"] == "full_rebuild"
    assert extended[1]["mode"] == "prefix_append"
    assert hit[1]["mode"] == "prefix_hit"
    assert likelihood_rebuilt[1]["mode"] == "full_rebuild"
    assert regularization_rebuilt[1]["mode"] == "full_rebuild"
    np.testing.assert_array_equal(
        first[0].spatial_factors,
        extended[0].spatial_factors[:2],
    )

    fisher_cache: dict[str, object] = {}
    basis = np.ones((1, 1, 1), dtype=np.float64)
    strengths = np.ones((1, 1), dtype=np.float64)
    coefficients = np.asarray([0.5])
    scales = np.asarray([1.0])
    first_fisher = _historical_factorized_fisher_precision(
        first[0],
        basis,
        strengths,
        coefficients,
        scales,
        (0, 1),
        model_identity="model-a",
        minimum_expected_count=1.0e-6,
        cache=fisher_cache,
        use_gpu=False,
        gpu_device="cuda",
    )
    extended_fisher = _historical_factorized_fisher_precision(
        extended[0],
        basis,
        strengths,
        coefficients,
        scales,
        (0, 1, 2),
        model_identity="model-a",
        minimum_expected_count=1.0e-6,
        cache=fisher_cache,
        use_gpu=False,
        gpu_device="cuda",
    )
    full_fisher = _historical_factorized_fisher_precision(
        extended[0],
        basis,
        strengths,
        coefficients,
        scales,
        (0, 1, 2),
        model_identity="model-a",
        minimum_expected_count=1.0e-6,
        cache=None,
        use_gpu=False,
        gpu_device="cuda",
    )

    assert first_fisher[1]["mode"] == "full_rebuild"
    assert extended_fisher[1]["mode"] == "prefix_append"
    np.testing.assert_allclose(extended_fisher[0], full_fisher[0], rtol=1.0e-15)

    computed_counts.clear()
    drift_cache: dict[str, object] = {}
    drift_config = MLEConfig(
        mode="spectral",
        isotope_names=("Cs-137",),
        fit_gain_resolution_drift=True,
        discrepancy_calibration_path="/nonexistent/test-calibration.json",
    )
    _historical_factorized_spectral_design(
        history(2),
        estimate,
        kernel,  # type: ignore[arg-type]
        drift_config,
        drift_cache,
    )
    drift_extended = _historical_factorized_spectral_design(
        history(3),
        estimate,
        kernel,  # type: ignore[arg-type]
        drift_config,
        drift_cache,
    )

    assert computed_counts == [2, 3]
    assert drift_extended[1]["mode"] == "full_rebuild"


def test_zero_mle_regions_remain_in_the_exploration_basis() -> None:
    """A sparse MLE must not permanently remove zero-strength surface regions."""
    patches = (_floor_patch(0, 0.0), _floor_patch(1, 1.0))
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=np.asarray([[5.0, 0.0]]),
        patch_strength_by_isotope=np.asarray([[5.0, 0.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )

    basis, labels = _source_basis(
        estimate,
        MLEPlanningConfig(
            max_active_source_parameters=1,
            max_total_source_parameters=2,
        ),
    )

    assert basis.shape == (2, 1, 2)
    assert np.all(np.sum(basis, axis=2) > 0.0)
    assert {label["kind"] for label in labels} == {
        "active_patch",
        "residual_exploration",
    }


def test_screening_basis_retains_zero_strength_isotope_surface_modes() -> None:
    """Approximate screening must not remove an absent isotope or surface."""
    patches = (_floor_patch(0, 0.0), _ceiling_patch(1, 0.0))
    estimate = MLEEstimate(
        isotope_names=("Cs-137", "Eu-154"),
        patches=patches,
        density_by_isotope=np.asarray([[5.0, 0.0], [0.0, 0.0]]),
        patch_strength_by_isotope=np.asarray([[5.0, 0.0], [0.0, 0.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )

    basis, labels = _screening_source_basis(estimate, MLEPlanningConfig())

    assert basis.shape == (2, 2, 4)
    assert np.all(np.sum(basis, axis=(0, 1)) > 0.0)
    assert {label["isotope"] for label in labels} == {"Cs-137", "Eu-154"}
    assert {label["surface_kind"] for label in labels} == {"floor", "ceiling"}


def test_representative_pairs_are_deterministic_and_include_current_pair() -> None:
    """Farthest-first screening pairs must cover orientations reproducibly."""
    orientations = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pairs = np.arange(9, dtype=np.int64)

    first = _representative_pair_ids(pairs, orientations, 4, 8)
    second = _representative_pair_ids(pairs, orientations, 4, 8)

    np.testing.assert_array_equal(first, second)
    assert first.size == 4
    assert np.unique(first).size == 4
    assert 8 in first


def test_screening_pseudo_model_preserves_compact_mode_scales() -> None:
    """Representative source points must preserve every screening mode mass."""
    patches = tuple(_floor_patch(index, float(index)) for index in range(6))
    density = np.asarray([[1.0, 2.0, 0.0, 4.0, 0.0, 1.0]])
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=density,
        patch_strength_by_isotope=density.copy(),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )

    view, strengths, pseudo_basis, full_basis, labels = _screening_pseudo_model(
        estimate,
        MLEPlanningConfig(screening_points_per_mode=3),
    )

    assert view.quadrature_points_xyz.shape[0] <= 3 * len(labels)
    assert strengths.shape == (view.quadrature_points_xyz.shape[0], 1)
    np.testing.assert_allclose(
        np.sum(pseudo_basis, axis=(0, 1)),
        np.sum(full_basis, axis=(0, 1)),
    )
    np.testing.assert_array_equal(
        np.sum(view.strength_projection, axis=0),
        np.ones((len(patches), 1), dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.einsum(
            "rgi,gi->ri",
            view.strength_projection,
            density.T,
            optimize=True,
        ),
        strengths,
    )


def test_screening_fisher_cpu_gpu_equivalence_when_cuda_is_available() -> None:
    """Grouped float64 screening Fisher must be CPU/GPU equivalent."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(70806)
    response = rng.uniform(1.0e-5, 1.0e-2, size=(7, 16, 3, 2))
    basis = rng.uniform(0.0, 1.0, size=(3, 2, 5))
    strengths = rng.uniform(0.1, 3.0, size=(3, 2))
    background = rng.uniform(0.1, 4.0, size=(7, 16))
    variance = rng.uniform(0.2, 8.0, size=(7, 16))

    cpu = _screening_fisher_information(
        response,
        basis,
        strengths,
        background,
        minimum_expected_count=1.0e-3,
        observation_variance=variance,
    )
    gpu = _screening_fisher_information(
        response,
        basis,
        strengths,
        background,
        minimum_expected_count=1.0e-3,
        observation_variance=variance,
        use_gpu=True,
    )

    for actual, expected in zip(gpu, cpu, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=2.0e-13)


def test_grouped_overdispersion_preserves_fine_bin_variance() -> None:
    """Grouped NB2 variance must sum fine-bin variances, not average alpha."""
    variance = _grouped_overdispersed_variance(
        np.asarray([[2.0, 3.0]]),
        np.asarray([0.1, 0.2]),
        np.asarray([0, 0], dtype=np.int64),
        1,
    )

    np.testing.assert_allclose(variance, [[7.2]])


def test_grouped_nb2_ambiguity_uses_hypothesis_specific_fine_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fine-bin NB2 moments must prevent a base-alpha ranking reversal."""
    spatial = np.zeros((2, 3, 2), dtype=np.float64)
    spatial[0, 0, 0] = 10.0
    spatial[0, 1, 1] = 5.0
    spatial[0, 2, 1] = 10.0
    spatial[1, 0, 1] = 10.0
    spatial[1, 1, 1] = 5.0
    spatial[1, 2, 0] = 10.0
    grouped_response = np.sum(spatial, axis=2)[:, None, :, None]
    design = information_planner._GroupedNB2LineDesign(
        grouped_response=grouped_response,
        grouped_background=np.ones((2, 1), dtype=np.float64),
        spatial_factors=spatial,
        pulse_shapes=np.eye(2, dtype=np.float64),
        line_isotope_indices=np.zeros(2, dtype=np.int64),
        live_times_s=np.ones(2, dtype=np.float64),
        background_rate_by_bin=np.asarray([0.0, 1.0]),
        overdispersion_alpha_by_bin=np.asarray([1.0, 0.0]),
        fine_group_indices=np.zeros(2, dtype=np.int64),
        group_count=1,
        energy_chunk_size=1,
    )
    first_weights = np.asarray([[1.0], [0.0], [0.0]])
    second_weights = np.asarray([[0.0], [1.0], [0.0]])
    base_weights = np.asarray([[0.0], [0.0], [1.0]])
    grouped_shapes: list[tuple[int, ...]] = []
    original_group_last_axis = information_planner._group_last_axis

    def track_grouped_workspace(
        values: np.ndarray,
        group_indices: np.ndarray,
        group_count: int,
    ) -> np.ndarray:
        """Require hypothesis projection to group only action-by-chunk arrays."""
        grouped_shapes.append(values.shape)
        assert values.ndim == 2
        assert values.shape[0] == spatial.shape[0]
        assert values.shape[1] <= design.energy_chunk_size
        return original_group_last_axis(values, group_indices, group_count)

    monkeypatch.setattr(
        information_planner,
        "_group_last_axis",
        track_grouped_workspace,
    )
    first = design.project_hypothesis(first_weights)
    second = design.project_hypothesis(second_weights)
    base = design.project_hypothesis(base_weights)
    monkeypatch.setattr(
        information_planner,
        "_group_last_axis",
        original_group_last_axis,
    )

    def project(weights: np.ndarray) -> np.ndarray:
        """Return grouped source counts for one test hypothesis."""
        return design.project_hypothesis(weights).source_counts

    metrics = information_planner._response_ambiguity_metrics(
        project,
        np.ones((2, 1, 1), dtype=np.float64),
        first_weights,
        second_weights,
        base_weights,
        (),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        design.grouped_background,
        design.project_hypothesis,
    )
    expected = -np.expm1(
        -0.5
        * (np.asarray([10.0, 10.0]) - np.asarray([5.0, 5.0])) ** 2
        / np.asarray([117.0, 17.0])
    )
    base_mean = base.source_counts + design.grouped_background
    legacy_alpha = (base.total_variance - base_mean) / base_mean**2
    legacy = _symmetric_spectral_separation(
        first.source_counts,
        second.source_counts,
        legacy_alpha,
        design.grouped_background,
    )
    base_fine_mean = np.asarray([[0.0, 11.0], [10.0, 1.0]])
    base_variance_oracle = _grouped_overdispersed_variance(
        base_fine_mean,
        design.overdispersion_alpha_by_bin,
        design.fine_group_indices,
        design.group_count,
    )
    source_basis = np.zeros((3, 1, 2), dtype=np.float64)
    source_basis[0, 0, 0] = 1.0
    source_basis[2, 0, 1] = 1.0
    information, expected_totals = _screening_fisher_information(
        design.grouped_response,
        source_basis,
        base_weights,
        design.grouped_background,
        minimum_expected_count=1.0e-12,
        observation_variance=base.total_variance,
    )
    jacobian = np.einsum(
        "abgi,gik->abk",
        design.grouped_response,
        source_basis,
        optimize=True,
    )
    information_oracle = np.einsum(
        "abp,abq,ab->apq",
        jacobian,
        jacobian,
        1.0 / base_variance_oracle,
        optimize=True,
    )

    assert len(grouped_shapes) == 12
    np.testing.assert_allclose(first.source_counts, [[10.0], [10.0]])
    np.testing.assert_allclose(second.source_counts, [[5.0], [5.0]])
    np.testing.assert_allclose(first.total_variance, [[111.0], [11.0]])
    np.testing.assert_allclose(second.total_variance, [[6.0], [6.0]])
    np.testing.assert_allclose(base.total_variance, base_variance_oracle)
    np.testing.assert_allclose(information, information_oracle)
    np.testing.assert_allclose(expected_totals, [11.0, 11.0])
    np.testing.assert_allclose(metrics["floor_ceiling"], expected)
    assert legacy[0] > legacy[1]
    assert metrics["floor_ceiling"][1] > metrics["floor_ceiling"][0]


def test_screening_overdispersion_can_reverse_poisson_ranking() -> None:
    """High-count overdispersed candidates must not dominate screening."""
    response = np.asarray(
        [
            [[[[10.0]]]],
            [[[[6.0]]]],
        ]
    ).reshape(2, 1, 1, 1)
    basis = np.ones((1, 1, 1), dtype=np.float64)
    strengths = np.ones((1, 1), dtype=np.float64)
    background = np.zeros((2, 1), dtype=np.float64)
    poisson, totals = _screening_fisher_information(
        response,
        basis,
        strengths,
        background,
        minimum_expected_count=1.0e-6,
    )
    overdispersed, _ = _screening_fisher_information(
        response,
        basis,
        strengths,
        background,
        minimum_expected_count=1.0e-6,
        observation_variance=np.asarray([[110.0], [6.0]]),
    )
    common = (
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        (0,),
    )
    poisson_selected, _ = select_fisher_action(
        *common,
        poisson[:, None, :, :],
        totals[:, None],
        np.eye(1),
        _orientations(),
        nuisance_count=0,
        config=MLEPlanningConfig(shield_program_length=1),
    )
    nb_selected, _ = select_fisher_action(
        *common,
        overdispersed[:, None, :, :],
        totals[:, None],
        np.eye(1),
        _orientations(),
        nuisance_count=0,
        config=MLEPlanningConfig(shield_program_length=1),
    )

    assert poisson_selected.candidate_index == 0
    assert nb_selected.candidate_index == 1


def test_beam_precision_chunks_stay_within_workspace_limit() -> None:
    """Default-size beam expansion matrices must stay under 64 MiB."""
    parameter_count = 98
    chunk_size = _beam_precision_chunk_size(parameter_count)
    estimated_bytes = 3 * chunk_size * parameter_count**2 * 8

    assert chunk_size < 8 * 64 * 63
    assert estimated_bytes <= 64 * 1024 * 1024


def test_d_s_optimality_marginalizes_nuisance_confounding() -> None:
    """A source-specific action should beat a stronger but confounded action."""
    information = np.zeros((1, 2, 2, 2), dtype=float)
    confounded = np.asarray([10.0, 10.0])
    source_specific = np.asarray([5.0, 0.0])
    information[0, 0] = np.outer(confounded, confounded)
    information[0, 1] = np.outer(source_specific, source_specific)

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0]]),
        (0, 1),
        information,
        np.ones((1, 2)),
        np.eye(2),
        _orientations(),
        nuisance_count=1,
        config=MLEPlanningConfig(shield_program_length=1),
    )

    assert selected.shield_pair_ids == (1,)


def test_external_motion_cost_can_change_the_selected_pose() -> None:
    """Runtime-supplied travel cost should enter only through its explicit weight."""
    information = np.zeros((2, 1, 1, 1), dtype=float)
    information[0, 0, 0, 0] = 4.0
    information[1, 0, 0, 0] = 5.0

    selected, _ = select_fisher_action(
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        (0,),
        information,
        np.ones((2, 1)),
        np.eye(1),
        _orientations(),
        nuisance_count=0,
        travel_costs=np.asarray([0.0, 10.0]),
        config=MLEPlanningConfig(
            shield_program_length=1,
            motion_cost_weight=1.0,
        ),
    )

    assert selected.candidate_index == 0


def test_plan_next_cli_requires_runtime_candidates_and_output() -> None:
    """CLI should expose a separate runtime-candidate planning operation."""
    args = build_argument_parser().parse_args(
        [
            "plan-next",
            "--run-dir",
            "/tmp/runtime-log",
            "--estimate",
            "/tmp/mle-report",
            "--mle-config",
            "/tmp/mle.json",
            "--candidates",
            "/tmp/candidates.json",
            "--output",
            "/tmp/action.json",
        ]
    )

    assert args.command == "plan-next"
    assert args.cpu is False
    assert args.gpu is False
    assert args.planning_config is None


def test_planning_history_must_be_an_exact_causal_prefix() -> None:
    """An old station estimate cannot inspect later MeasurementLog records."""

    class _Estimate:
        """Expose only diagnostics required by the CLI lineage check."""

        diagnostics = {"online_lineage": {"covered_step_ids": [0, 1]}}

    indices = _estimate_history_indices(_Estimate(), np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(indices, [0, 1])

    _Estimate.diagnostics = {"online_lineage": {"covered_step_ids": [0, 2]}}
    with pytest.raises(ValueError, match="exact causal prefix"):
        _estimate_history_indices(_Estimate(), np.asarray([0, 1, 2]))


def test_floor_ceiling_competition_rewards_height_discrimination() -> None:
    """A candidate with distinct vertical signatures must receive a larger score."""
    patches = (_floor_patch(0, 0.0), _ceiling_patch(1, 0.0))
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=np.asarray([[1.0, 1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0, 1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=1.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )
    history = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.5, 0.5, 0.5]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
        live_times_s=np.asarray([1.0]),
        spectrum_counts=np.ones((1, 2)),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        isotope_counts=np.ones((1, 1)),
        isotope_covariances=np.ones((1, 1, 1)),
        station_ids=np.asarray([0]),
        isotope_names=("Cs-137",),
    )
    response = np.zeros((2, 2, 2, 1), dtype=float)
    response[0, :, 0, 0] = [1.0, 0.0]
    response[0, :, 1, 0] = [0.9, 0.1]
    response[1, :, 0, 0] = [1.0, 0.0]
    response[1, :, 1, 0] = [0.0, 1.0]
    source_basis = np.zeros((2, 1, 2), dtype=float)
    source_basis[0, 0, 0] = 1.0
    source_basis[1, 0, 1] = 1.0
    information = np.tile(np.eye(2), (2, 1, 1))

    metrics = _ambiguity_metrics(
        response,
        information,
        np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]]),
        estimate,
        history,
        source_basis,
        (),
    )
    factor_metrics = _ambiguity_metrics(
        _LineSpectralDesign(
            spatial_factors=np.transpose(response[:, :, :, 0], (0, 2, 1)),
            pulse_shapes=np.eye(2, dtype=np.float64),
            line_isotope_indices=np.zeros(2, dtype=np.int64),
            nuisance_response=np.zeros((2, 2, 0), dtype=np.float64),
            nuisance_names=(),
            energy_chunk_size=1,
        ),
        information,
        np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]]),
        estimate,
        history,
        source_basis,
        (),
    )

    assert metrics["floor_ceiling"][1] > metrics["floor_ceiling"][0]
    assert metrics["correlation"][1] > metrics["correlation"][0]
    assert metrics["surface_coverage"][1] > metrics["surface_coverage"][0]
    for name, values in metrics.items():
        np.testing.assert_allclose(factor_metrics[name], values)


def test_spectral_separation_respects_counts_and_overdispersion() -> None:
    """Low-count or overdispersed distinctions must receive less utility."""
    first = np.asarray([[1.0, 0.1]])
    second = np.asarray([[0.1, 1.0]])
    low_count = _symmetric_spectral_separation(first, second)
    high_count = _symmetric_spectral_separation(100.0 * first, 100.0 * second)
    overdispersed = _symmetric_spectral_separation(
        100.0 * first,
        100.0 * second,
        np.ones(2),
    )

    assert high_count[0] > low_count[0]
    assert overdispersed[0] < high_count[0]


def test_spectral_separation_includes_common_background_variance() -> None:
    """Common background must reduce both Poisson and NB2 discrimination."""
    first = np.asarray([[1.0]])
    second = np.asarray([[0.0]])
    background = np.asarray([[100.0]])

    poisson_source_only = _symmetric_spectral_separation(first, second)
    poisson_complete = _symmetric_spectral_separation(
        first,
        second,
        common_counts=background,
    )
    nb2_source_only = _symmetric_spectral_separation(
        first,
        second,
        np.asarray([0.1]),
    )
    nb2_complete = _symmetric_spectral_separation(
        first,
        second,
        np.asarray([0.1]),
        background,
    )

    np.testing.assert_allclose(poisson_complete, [0.002484470770129418])
    np.testing.assert_allclose(nb2_complete, [0.00022508834622582234])
    assert poisson_complete[0] < poisson_source_only[0]
    assert nb2_complete[0] < nb2_source_only[0]


def test_ambiguity_metrics_include_fitted_nuisance_counts() -> None:
    """Exact ambiguity must carry fitted common nuisance into NB2 means."""
    patches = (_floor_patch(0, 0.0), _ceiling_patch(1, 0.0))
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=np.asarray([[1.0, 1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0, 1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.asarray([1.0]),
        nuisance_parameters=np.zeros(0),
        objective_value=0.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={"nuisance_names": ["background_rate_cps"]},
    )
    history = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.5, 0.5, 1.0]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
        live_times_s=np.asarray([1.0]),
        spectrum_counts=np.ones((1, 2)),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        isotope_counts=None,
        isotope_covariances=None,
        station_ids=np.asarray([0]),
        isotope_names=("Cs-137",),
    )
    spatial = np.zeros((1, 2, 2), dtype=np.float64)
    spatial[0, 0, 0] = 1.0
    spatial[0, 1, 1] = 1.0
    design = _LineSpectralDesign(
        spatial_factors=spatial,
        pulse_shapes=np.eye(2, dtype=np.float64),
        line_isotope_indices=np.zeros(2, dtype=np.int64),
        nuisance_response=np.full((1, 2, 1), 100.0),
        nuisance_names=("background_rate_cps",),
        energy_chunk_size=1,
        overdispersion_alpha_by_bin=np.full(2, 0.1),
    )
    basis = np.zeros((2, 1, 2), dtype=np.float64)
    basis[0, 0, 0] = 1.0
    basis[1, 0, 1] = 1.0
    information = np.eye(3, dtype=np.float64)[None, :, :]

    source_only = _ambiguity_metrics(
        design,
        information,
        np.asarray([[0.5, 0.5, 1.0]]),
        estimate,
        history,
        basis,
        (),
        nuisance_coefficients=np.zeros(1),
    )
    complete = _ambiguity_metrics(
        design,
        information,
        np.asarray([[0.5, 0.5, 1.0]]),
        estimate,
        history,
        basis,
        (),
        nuisance_coefficients=np.ones(1),
    )
    expected = _symmetric_spectral_separation(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[0.0, 1.0]]),
        np.full(2, 0.1),
        np.full((1, 2), 100.0),
    )

    np.testing.assert_allclose(complete["floor_ceiling"], expected)
    assert complete["floor_ceiling"][0] < source_only["floor_ceiling"][0]


def test_support_ambiguity_projects_alternative_patch_grids() -> None:
    """Alternative support utility must survive coarse-to-fine patch changes."""

    def patch(patch_id: int, x0: float, x1: float) -> SurfacePatch:
        """Return one rectangle on a shared physical floor surface."""
        area = x1 - x0
        return SurfacePatch(
            patch_id=patch_id,
            centroid_xyz=np.asarray([0.5 * (x0 + x1), 0.5, 0.0]),
            normal_xyz=np.asarray([0.0, 0.0, 1.0]),
            area_m2=area,
            surface_kind="floor",
            object_id="shared-floor",
            vertices_xyz=np.asarray(
                [
                    [x0, 0.0, 0.0],
                    [x1, 0.0, 0.0],
                    [x1, 1.0, 0.0],
                    [x0, 1.0, 0.0],
                ]
            ),
            quadrature_points_xyz=np.asarray([[0.5 * (x0 + x1), 0.5, 0.0]]),
            quadrature_weights=np.asarray([1.0]),
        )

    base = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(patch(10, 0.0, 0.5), patch(11, 0.5, 1.0)),
        density_by_isotope=np.asarray([[2.0, 0.0]]),
        patch_strength_by_isotope=np.asarray([[1.0, 0.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=0.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )
    alternative = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=(patch(99, 0.0, 1.0),),
        density_by_isotope=np.asarray([[1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=0.0,
        poisson_deviance=0.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )
    history = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.5, 0.5, 1.0]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
        live_times_s=np.asarray([1.0]),
        spectrum_counts=np.ones((1, 2)),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        isotope_counts=np.ones((1, 1)),
        isotope_covariances=np.ones((1, 1, 1)),
        station_ids=np.asarray([0]),
        isotope_names=("Cs-137",),
    )
    response = np.zeros((1, 2, 2, 1), dtype=np.float64)
    response[0, 0, 0, 0] = 1.0
    response[0, 1, 1, 0] = 1.0
    basis = np.zeros((2, 1, 2), dtype=np.float64)
    basis[0, 0, 0] = 1.0
    basis[1, 0, 1] = 1.0

    metrics = _ambiguity_metrics(
        response,
        np.eye(2, dtype=np.float64)[None, :, :],
        np.asarray([[0.5, 0.5, 1.0]]),
        base,
        history,
        basis,
        (alternative,),
    )
    scaled_alternative = replace(
        alternative,
        density_by_isotope=np.asarray([[9.0]]),
        patch_strength_by_isotope=np.asarray([[9.0]]),
    )
    scaled_metrics = _ambiguity_metrics(
        response,
        np.eye(2, dtype=np.float64)[None, :, :],
        np.asarray([[0.5, 0.5, 1.0]]),
        base,
        history,
        basis,
        (scaled_alternative,),
    )
    same_support_different_total = replace(
        base,
        density_by_isotope=10.0 * base.density_by_isotope,
        patch_strength_by_isotope=10.0 * base.patch_strength_by_isotope,
    )
    same_support_metrics = _ambiguity_metrics(
        response,
        np.eye(2, dtype=np.float64)[None, :, :],
        np.asarray([[0.5, 0.5, 1.0]]),
        base,
        history,
        basis,
        (same_support_different_total,),
    )

    assert metrics["support"][0] > 0.0
    np.testing.assert_allclose(scaled_metrics["support"], metrics["support"])
    np.testing.assert_allclose(same_support_metrics["support"], 0.0, atol=1.0e-15)


def test_vertical_fisher_uses_cross_terms_and_marginalizes_nuisance() -> None:
    """Vertical utility must use the source Schur quadratic, not its diagonal."""
    patches = (_floor_patch(0, 0.0), _ceiling_patch(1, 0.0))
    estimate = MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=np.asarray([[1.0, 1.0]]),
        patch_strength_by_isotope=np.asarray([[1.0, 1.0]]),
        predicted_spectra=None,
        predicted_isotope_counts=None,
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=1.0,
        iterations=1,
        converged=True,
        diagnostics={},
    )
    history = ObservationBatch(
        detector_positions_xyz=np.asarray([[0.5, 0.5, 0.5]]),
        detector_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
        live_times_s=np.asarray([1.0]),
        spectrum_counts=np.ones((1, 2)),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        isotope_counts=np.ones((1, 1)),
        isotope_covariances=np.ones((1, 1, 1)),
        station_ids=np.asarray([0]),
        isotope_names=("Cs-137",),
    )
    response = np.ones((2, 2, 2, 1), dtype=np.float64)
    source_basis = np.zeros((2, 1, 2), dtype=np.float64)
    source_basis[0, 0, 0] = 1.0
    source_basis[1, 0, 1] = 1.0
    source_information = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    information = np.zeros((2, 3, 3), dtype=np.float64)
    information[:, :2, :2] = source_information
    information[:, 2, 2] = 2.0
    information[1, :2, 2] = [-1.0, 1.0]
    information[1, 2, :2] = [-1.0, 1.0]

    metrics = _ambiguity_metrics(
        response,
        information,
        np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]]),
        estimate,
        history,
        source_basis,
        (),
    )

    np.testing.assert_allclose(
        metrics["z_fisher"],
        np.log1p([1.0, 0.5]),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
