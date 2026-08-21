"""Tests for conditional Laplace and station-block bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from three_d_estimation.config import MLEConfig
from three_d_estimation.estimator import (
    _bootstrap_worker_count,
    _uncertainty_invalid_reasons,
)
from three_d_estimation.types import MLEEstimate, ObservationBatch, SurfacePatch
from three_d_estimation.response_operator import BlockResponseOperator, ResponseBlock
from three_d_estimation.uncertainty import (
    active_support_laplace,
    augment_clusters_with_laplace,
    bootstrap_uncertainty_summary,
    station_bootstrap_batch,
)


def _patch(patch_id: int, x: float, kind: str = "floor") -> SurfacePatch:
    """Return one oriented unit patch for uncertainty tests."""
    z = 0.0 if kind == "floor" else 1.0
    normal = np.asarray([0.0, 0.0, 1.0])
    return SurfacePatch(
        patch_id=patch_id,
        centroid_xyz=np.asarray([x + 0.5, 0.5, z]),
        normal_xyz=normal,
        area_m2=1.0,
        surface_kind=kind,  # type: ignore[arg-type]
        object_id=f"{kind}:{patch_id}",
        vertices_xyz=np.asarray(
            [[x, 0.0, z], [x + 1.0, 0.0, z], [x + 1.0, 1.0, z], [x, 1.0, z]]
        ),
        quadrature_points_xyz=np.asarray([[x + 0.5, 0.5, z]]),
        quadrature_weights=np.asarray([1.0]),
    )


def _batch() -> ObservationBatch:
    """Return four rows arranged into two related stations."""
    return ObservationBatch(
        detector_positions_xyz=np.asarray(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        ),
        detector_quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        fe_indices=np.asarray([0, 1, 0, 1]),
        pb_indices=np.asarray([0, 0, 0, 0]),
        live_times_s=np.ones(4),
        spectrum_counts=np.ones((4, 2)),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        isotope_counts=np.ones((4, 1)),
        isotope_covariances=np.tile(np.eye(1), (4, 1, 1)),
        station_ids=np.asarray([0, 0, 1, 1]),
        isotope_names=("Cs-137",),
    )


def _estimate(offset: float) -> MLEEstimate:
    """Return one two-patch estimate with a movable hotspot cluster."""
    patches = (_patch(0, 0.0), _patch(1, 1.0, "ceiling"))
    density = np.asarray([[2.0, 1.0]])
    return MLEEstimate(
        isotope_names=("Cs-137",),
        patches=patches,
        density_by_isotope=density,
        patch_strength_by_isotope=density,
        predicted_spectra=np.ones((2, 2)),
        predicted_isotope_counts=np.ones((2, 1)),
        background_parameters=np.zeros(0),
        nuisance_parameters=np.zeros(0),
        objective_value=1.0,
        poisson_deviance=1.0,
        iterations=1,
        converged=True,
        diagnostics={
            "hotspot_clusters": [
                {
                    "isotope": "Cs-137",
                    "centroid_xyz": [0.5 + offset, 0.5, 0.0],
                    "integrated_strength_cps_1m": 2.0 + offset,
                    "patch_ids": [0],
                }
            ]
        },
    )


def test_active_support_laplace_returns_finite_covariance() -> None:
    """The active conditional Fisher matrix should yield finite uncertainty."""
    response = np.zeros((2, 1, 2, 1), dtype=float)
    response[0, 0, 0, 0] = 1.0
    response[1, 0, 1, 0] = 2.0
    predicted = np.asarray([[2.0], [4.0]])

    result = active_support_laplace(
        response,
        predicted,
        predicted,
        np.asarray([[2.0], [2.0]]),
        np.ones(2),
        np.asarray([0, 1]),
        support_threshold_fraction=0.01,
        maximum_active_parameters=8,
        ridge=1.0e-8,
    )

    assert result.covariance.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(result.covariance) > 0.0)


def test_matrix_free_laplace_preserves_cross_chunk_covariance() -> None:
    """Source chunks must retain off-diagonal Fisher information exactly."""
    response = np.asarray([[[[1.0], [2.0]]], [[[2.0], [1.0]]]])

    def blocks():
        """Yield separate source chunks over the same observation rows."""
        rows = np.asarray([0, 1], dtype=np.int64)
        yield ResponseBlock(rows, np.asarray([0]), np.asarray([[1.0], [2.0]]))
        yield ResponseBlock(rows, np.asarray([1]), np.asarray([[2.0], [1.0]]))

    operator = BlockResponseOperator((2, 1), 2, 1, blocks)
    arguments = dict(
        observed=np.asarray([[3.0], [3.0]]),
        predicted=np.asarray([[3.0], [3.0]]),
        densities=np.asarray([[1.0], [1.0]]),
        patch_areas_m2=np.ones(2),
        fit_indices=np.asarray([0, 1]),
        support_threshold_fraction=0.01,
        maximum_active_parameters=8,
        ridge=1.0e-8,
    )

    materialized = active_support_laplace(response, **arguments)
    streamed = active_support_laplace(operator, **arguments)

    np.testing.assert_allclose(streamed.covariance, materialized.covariance)


def test_station_bootstrap_preserves_complete_station_blocks() -> None:
    """Bootstrap rows must repeat whole station programs, never single views."""
    result = station_bootstrap_batch(_batch(), np.random.default_rng(4))

    assert result.measurement_count == 4
    assert all(
        np.count_nonzero(result.station_ids == station) == 2 for station in (0, 1)
    )
    assert np.array_equal(result.step_ids, np.arange(4))


def test_gpu_bootstrap_serializes_sparse_solver_replicates() -> None:
    """CUDA bootstrap must not run sparse TV kernels on concurrent streams."""
    assert _bootstrap_worker_count(
        MLEConfig(use_gpu=True, bootstrap_batch_size=4),
        32,
    ) == 1
    assert _bootstrap_worker_count(
        MLEConfig(use_gpu=False, bootstrap_batch_size=4),
        32,
    ) == 4


def test_uncertainty_quality_gates_fail_closed() -> None:
    """Nonconvergence, KKT failure, and mismatch invalidate uncertainty."""
    config = MLEConfig(
        kkt_tolerance=1.0e-3,
        debias_max_pearson_dispersion=5.0,
    )
    valid = replace(
        _estimate(0.0),
        diagnostics={
            **_estimate(0.0).diagnostics,
            "kkt_residual": 1.0e-4,
            "pearson_dispersion": 1.2,
        },
    )
    invalid = replace(
        valid,
        converged=False,
        diagnostics={
            **valid.diagnostics,
            "kkt_residual": 0.1,
            "pearson_dispersion": 20.0,
        },
    )
    nonfinite = replace(
        valid,
        diagnostics={
            **valid.diagnostics,
            "kkt_residual": float("nan"),
            "pearson_dispersion": float("nan"),
        },
    )

    assert _uncertainty_invalid_reasons(valid, config) == ()
    assert _uncertainty_invalid_reasons(invalid, config) == (
        "optimizer_not_converged",
        "kkt_tolerance_not_met",
        "pearson_dispersion_exceeds_model_adequacy_limit",
    )
    assert _uncertainty_invalid_reasons(nonfinite, config) == (
        "kkt_tolerance_not_met",
        "pearson_dispersion_exceeds_model_adequacy_limit",
    )


def test_bootstrap_summary_adds_cluster_and_surface_intervals() -> None:
    """Bootstrap reports should expose position, strength, z, and surface mass."""
    summary, clusters = bootstrap_uncertainty_summary(
        _estimate(0.0),
        (_estimate(-0.1), _estimate(0.1)),
        confidence_level=0.95,
    )

    isotope = summary["isotopes"]["Cs-137"]
    assert isotope["z_interval_m"]
    assert "ceiling_source_probability" in isotope
    assert "surface_mass_probability" in isotope
    assert clusters[0]["bootstrap_selection_frequency"] == 1.0
    assert np.asarray(clusters[0]["centroid_covariance_xyz_m2"]).shape == (3, 3)


def test_bootstrap_cluster_matching_is_one_to_one_per_replicate() -> None:
    """One replicate cluster cannot make two base clusters look selected."""
    base = _estimate(0.0)
    base = replace(
        base,
        diagnostics={
            **base.diagnostics,
            "hotspot_clusters": [
                *base.diagnostics["hotspot_clusters"],
                {
                    "isotope": "Cs-137",
                    "centroid_xyz": [1.5, 0.5, 0.0],
                    "integrated_strength_cps_1m": 1.0,
                },
            ],
        },
    )

    _summary, clusters = bootstrap_uncertainty_summary(
        base,
        (_estimate(0.05),),
        confidence_level=0.95,
    )

    assert sorted(
        float(cluster["bootstrap_selection_frequency"]) for cluster in clusters
    ) == [0.0, 1.0]


def test_laplace_covariance_is_attached_to_cluster_source_modes() -> None:
    """A cluster should expose delta-method XYZ covariance without bootstrap."""
    laplace = active_support_laplace(
        np.asarray([[[[1.0], [0.5]]], [[[0.5], [1.0]]]]),
        np.asarray([[2.0], [2.0]]),
        np.asarray([[2.0], [2.0]]),
        np.asarray([[2.0], [1.0]]),
        np.ones(2),
        np.asarray([0, 1]),
        support_threshold_fraction=0.01,
        maximum_active_parameters=8,
        ridge=1.0e-6,
    )

    clusters = augment_clusters_with_laplace(
        _estimate(0.0),
        laplace,
        confidence_level=0.95,
    )

    assert np.asarray(clusters[0]["centroid_covariance_xyz_m2"]).shape == (3, 3)
    assert clusters[0]["uncertainty_method"] == "active_support_laplace_delta"


def test_laplace_skips_cluster_interval_when_support_is_partial() -> None:
    """A capped Fisher support must not yield an overconfident cluster CI."""
    base = _estimate(0.0)
    base = replace(
        base,
        diagnostics={
            **base.diagnostics,
            "hotspot_clusters": [
                {
                    **base.diagnostics["hotspot_clusters"][0],
                    "patch_ids": [0, 1],
                }
            ],
        },
    )
    laplace = active_support_laplace(
        np.asarray([[[[1.0], [0.5]]], [[[0.5], [1.0]]]]),
        np.asarray([[2.0], [2.0]]),
        np.asarray([[2.0], [2.0]]),
        np.asarray([[2.0], [1.0]]),
        np.ones(2),
        np.asarray([0, 1]),
        support_threshold_fraction=0.01,
        maximum_active_parameters=1,
        ridge=1.0e-6,
    )

    clusters = augment_clusters_with_laplace(
        base,
        laplace,
        confidence_level=0.95,
    )

    assert clusters[0]["laplace_interval_status"] == "skipped_partial_support"
    assert "integrated_strength_interval_cps_1m" not in clusters[0]
