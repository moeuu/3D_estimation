"""Regression tests for support-restricted surface-MLE debiasing."""

from __future__ import annotations

import numpy as np

from measurement.model import EnvironmentConfig
from three_d_estimation.config import MLEConfig
from three_d_estimation.estimator import (
    _FitState,
    _bounded_debias_support,
    _debias_state,
    _full_prediction,
    _refinement_patch_ids,
)
from three_d_estimation.solver import SurfaceMapResult
from three_d_estimation.surface_patches import build_surface_patches


def _seed_result(
    densities: np.ndarray,
    areas_m2: np.ndarray,
    *,
    converged: bool = True,
) -> SurfaceMapResult:
    """Return a minimal regularized-fit result used to seed debiasing."""
    integrated = densities * areas_m2[:, None]
    return SurfaceMapResult(
        densities_cps_1m_m2=densities,
        integrated_strengths_cps_1m=integrated,
        nuisance_coefficients=np.zeros(0, dtype=float),
        expected_counts=np.asarray([[float(np.sum(integrated))]], dtype=float),
        objective=0.0,
        poisson_nll=0.0,
        l1_penalty=0.0,
        tv_penalty=0.0,
        group_penalty=0.0,
        nuisance_penalty=0.0,
        deviance=0.0,
        converged=converged,
        iterations=1,
        relative_change=0.0,
        relative_objective_change=0.0,
        kkt_residual=0.0,
        objective_history=(0.0,),
    )


def test_debias_cannot_preserve_initialized_density_outside_selected_support() -> None:
    """A rejected positive warm value must be zero in the debiased map."""
    patches = build_surface_patches(
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        spacing=2.0,
        quadrature_points_per_patch=1,
    )
    densities = np.zeros((patches.patch_count, 1), dtype=float)
    densities[0, 0] = 10.0
    densities[1, 0] = 1.0
    response = np.ones((1, 1, patches.patch_count, 1), dtype=float)
    state = _FitState(
        patches=patches,
        response=response,
        nuisance_response=np.zeros((1, 1, 0), dtype=float),
        nuisance_names=(),
        nuisance_l2_weights=np.zeros(0, dtype=float),
        overdispersion_alpha_by_bin=np.zeros(0, dtype=float),
        result=_seed_result(densities, patches.areas_m2),
        fit_indices=np.asarray([0], dtype=np.int64),
        held_out_indices=np.zeros(0, dtype=np.int64),
        spectral_details=None,
        likelihood_diagnostics={"family": "poisson"},
    )
    config = MLEConfig(
        mode="count",
        isotope_names=("Cs-137",),
        support_threshold_fraction=0.5,
        max_iterations=100,
        check_interval=1,
        tolerance=1.0e-10,
        objective_tolerance=1.0e-10,
    )

    debiased = _debias_state(state, np.asarray([[10.0]], dtype=float), config)

    assert densities[1, 0] == 1.0
    np.testing.assert_array_equal(
        debiased.result.densities_cps_1m_m2[1:],
        np.zeros((patches.patch_count - 1, 1), dtype=float),
    )
    np.testing.assert_array_equal(
        debiased.result.integrated_strengths_cps_1m[1:],
        np.zeros((patches.patch_count - 1, 1), dtype=float),
    )
    np.testing.assert_array_equal(debiased.response[:, :, 1:, :], 0.0)
    prediction = _full_prediction(debiased)
    np.testing.assert_allclose(
        prediction,
        debiased.result.integrated_strengths_cps_1m[0, 0],
    )


def test_debias_skips_unconverged_regularized_support() -> None:
    """An unstable regularized support must not be refit without shrinkage."""
    patches = build_surface_patches(
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        spacing=2.0,
        quadrature_points_per_patch=1,
    )
    densities = np.ones((patches.patch_count, 1), dtype=float)
    state = _FitState(
        patches=patches,
        response=np.ones((1, 1, patches.patch_count, 1), dtype=float),
        nuisance_response=np.zeros((1, 1, 0), dtype=float),
        nuisance_names=(),
        nuisance_l2_weights=np.zeros(0, dtype=float),
        overdispersion_alpha_by_bin=np.zeros(0, dtype=float),
        result=_seed_result(densities, patches.areas_m2, converged=False),
        fit_indices=np.asarray([0], dtype=np.int64),
        held_out_indices=np.zeros(0, dtype=np.int64),
        spectral_details=None,
        likelihood_diagnostics={"family": "poisson"},
    )

    result = _debias_state(
        state,
        np.asarray([[10.0]], dtype=float),
        MLEConfig(mode="count", isotope_names=("Cs-137",)),
    )

    assert result.result is state.result
    assert result.likelihood_diagnostics["debias_applied"] is False
    assert (
        result.likelihood_diagnostics["debias_skip_reason"]
        == "regularized_fit_not_converged"
    )


def test_debias_skips_structurally_mismatched_likelihood() -> None:
    """Large Pearson dispersion must retain the stabilizing regularization."""
    patches = build_surface_patches(
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        spacing=2.0,
        quadrature_points_per_patch=1,
    )
    densities = np.ones((patches.patch_count, 1), dtype=float)
    state = _FitState(
        patches=patches,
        response=np.ones((1, 1, patches.patch_count, 1), dtype=float),
        nuisance_response=np.zeros((1, 1, 0), dtype=float),
        nuisance_names=(),
        nuisance_l2_weights=np.zeros(0, dtype=float),
        overdispersion_alpha_by_bin=np.zeros(0, dtype=float),
        result=_seed_result(densities, patches.areas_m2),
        fit_indices=np.asarray([0], dtype=np.int64),
        held_out_indices=np.zeros(0, dtype=np.int64),
        spectral_details=None,
        likelihood_diagnostics={"family": "poisson"},
    )

    result = _debias_state(
        state,
        np.asarray([[1000.0]], dtype=float),
        MLEConfig(
            mode="count",
            isotope_names=("Cs-137",),
            debias_max_pearson_dispersion=5.0,
        ),
    )

    assert result.result is state.result
    assert (
        result.likelihood_diagnostics["debias_skip_reason"]
        == "pearson_dispersion_exceeds_limit"
    )


def test_surface_support_is_bounded_and_refinement_is_isotope_balanced() -> None:
    """Weak isotopes retain local resolution without allowing diffuse support growth."""
    patches = build_surface_patches(
        EnvironmentConfig(size_x=1.0, size_y=1.0, size_z=1.0),
        None,
        spacing=2.0,
        quadrature_points_per_patch=1,
    )
    densities = np.zeros((patches.patch_count, 2), dtype=float)
    densities[:, 0] = np.linspace(10.0, 5.0, patches.patch_count)
    densities[-1, 1] = 1.0
    state = _FitState(
        patches=patches,
        response=np.ones((1, 1, patches.patch_count, 2), dtype=float),
        nuisance_response=np.zeros((1, 1, 0), dtype=float),
        nuisance_names=(),
        nuisance_l2_weights=np.zeros(0, dtype=float),
        overdispersion_alpha_by_bin=np.zeros(0, dtype=float),
        result=_seed_result(densities, patches.areas_m2),
        fit_indices=np.asarray([0], dtype=np.int64),
        held_out_indices=np.zeros(0, dtype=np.int64),
        spectral_details=None,
        likelihood_diagnostics={"family": "poisson"},
    )
    config = MLEConfig(
        mode="count",
        isotope_names=("Cs-137", "Co-60"),
        support_threshold_fraction=0.0,
        debias_max_active_parameters=3,
    )

    support = _bounded_debias_support(state, config)
    refined = _refinement_patch_ids(state, 0.2, 3)

    assert np.count_nonzero(support) == 3
    assert support[-1, 1]
    assert patches.patches[-1].patch_id in refined
    assert len(refined) <= 3
