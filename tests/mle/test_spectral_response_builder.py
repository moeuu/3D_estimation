"""Physics and batching tests for line-resolved spectral surface responses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import pickle
import tracemalloc

import numpy as np
import pytest
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import OctantShield, generate_octant_orientations
from runtime.discrepancy_calibration import DiscrepancyCalibration
from spectrum.additive_scatter import (
    AdditiveNoncollidedTransportResponse,
    PhysicsOnlyNoncollidedTransportResponse,
)
from spectrum.response_matrix import (
    BACKSCATTER_FRACTION,
    COMPTON_CONTINUUM_TO_PEAK,
    cebr3_efficiency,
    default_resolution,
    detector_response_kernel_for_incident_gamma,
)
import three_d_estimation.spectral_response_builder as spectral_builder
from three_d_estimation.config import MLEConfig
from three_d_estimation.replay import prepare_replay
from three_d_estimation.spectral_response_builder import (
    build_spectral_nuisance_response,
    build_spectral_response,
    build_spectral_response_operator,
    build_structured_spectral_nuisance_response,
)


@dataclass(frozen=True)
class _Observations:
    """Minimal spectrum observation contract used by the builder."""

    detector_positions_xyz: NDArray[np.float64]
    fe_indices: NDArray[np.int64]
    pb_indices: NDArray[np.int64]
    live_times_s: NDArray[np.float64]
    energy_bin_edges_keV: NDArray[np.float64]
    spectrum_counts: NDArray[np.float64]


@dataclass(frozen=True)
class _Patches:
    """Minimal aggregate surface quadrature contract used by the builder."""

    quadrature_points_xyz: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]
    areas_m2: NDArray[np.float64]


_CS_LINE = (
    {
        "energy_keV": 661.7,
        "weight": 1.0,
        "fe": 0.0,
        "pb": 0.0,
    },
)
_CO_LINES_UNATTENUATED = (
    {
        "energy_keV": 1173.2,
        "weight": 0.5,
        "fe": 0.0,
        "pb": 0.0,
    },
    {
        "energy_keV": 1332.5,
        "weight": 0.5,
        "fe": 0.0,
        "pb": 0.0,
    },
)


def _observations(
    detector_positions_xyz: NDArray[np.float64],
    energy_bin_edges_keV: NDArray[np.float64],
    *,
    live_times_s: NDArray[np.float64] | None = None,
    fe_indices: NDArray[np.int64] | None = None,
    pb_indices: NDArray[np.int64] | None = None,
) -> _Observations:
    """Return a valid minimal observation batch."""
    positions = np.asarray(detector_positions_xyz, dtype=float).reshape(-1, 3)
    edges = np.asarray(energy_bin_edges_keV, dtype=float)
    measurement_count = int(positions.shape[0])
    return _Observations(
        detector_positions_xyz=positions,
        fe_indices=(
            np.zeros(measurement_count, dtype=np.int64)
            if fe_indices is None
            else np.asarray(fe_indices, dtype=np.int64)
        ),
        pb_indices=(
            np.zeros(measurement_count, dtype=np.int64)
            if pb_indices is None
            else np.asarray(pb_indices, dtype=np.int64)
        ),
        live_times_s=(
            np.ones(measurement_count, dtype=float)
            if live_times_s is None
            else np.asarray(live_times_s, dtype=float)
        ),
        energy_bin_edges_keV=edges,
        spectrum_counts=np.zeros((measurement_count, edges.size - 1), dtype=float),
    )


def _patches(
    quadrature_points_xyz: NDArray[np.float64],
    *,
    quadrature_weights: NDArray[np.float64] | None = None,
    areas_m2: NDArray[np.float64] | None = None,
) -> _Patches:
    """Return a valid aggregate patch quadrature batch."""
    points = np.asarray(quadrature_points_xyz, dtype=float)
    patch_count, quadrature_count = points.shape[:2]
    return _Patches(
        quadrature_points_xyz=points,
        quadrature_weights=(
            np.full(
                (patch_count, quadrature_count),
                1.0 / quadrature_count,
                dtype=float,
            )
            if quadrature_weights is None
            else np.asarray(quadrature_weights, dtype=float)
        ),
        areas_m2=(
            np.ones(patch_count, dtype=float)
            if areas_m2 is None
            else np.asarray(areas_m2, dtype=float)
        ),
    )


def _shield_params(*, thickness_cm: float = 0.0) -> ShieldParams:
    """Return simple nested spherical shields for exact attenuation tests."""
    return ShieldParams(
        mu_fe=0.0,
        mu_pb=0.0,
        thickness_fe_cm=thickness_cm,
        thickness_pb_cm=thickness_cm,
        inner_radius_fe_cm=1.0,
        inner_radius_pb_cm=1.0 + thickness_cm,
    )


def _kernel(
    line_mu_by_isotope: dict[str, tuple[dict[str, float], ...]],
    *,
    thickness_cm: float = 0.0,
    obstacle_grid: ObstacleGrid | None = None,
    use_gpu: bool = False,
) -> ContinuousKernel:
    """Return a local shared kernel with controlled line physics."""
    return ContinuousKernel(
        mu_by_isotope={
            isotope: {"fe": 0.0, "pb": 0.0} for isotope in line_mu_by_isotope
        },
        shield_params=_shield_params(thickness_cm=thickness_cm),
        line_mu_by_isotope=line_mu_by_isotope,
        obstacle_grid=obstacle_grid,
        detector_radius_m=0.0,
        detector_aperture_radius_m=0.0,
        use_gpu=use_gpu,
        gpu_device="cuda" if use_gpu else "cpu",
        gpu_dtype="float64",
    )


def _process_context(
    kernel: ContinuousKernel,
    isotopes: tuple[str, ...],
) -> spectral_builder._SpectralProcessContext:
    """Return a minimal process context sharing one kernel across isotopes."""
    prepared = tuple(
        spectral_builder._PreparedSpectralIsotope(
            isotope_index=index,
            isotope=isotope,
            line_start=index,
            weights=np.ones(1, dtype=np.float64),
            positive_line_indices=np.zeros(1, dtype=np.int64),
            kernel=kernel,
            pulses=np.ones((1, 2), dtype=np.float64),
        )
        for index, isotope in enumerate(isotopes)
    )
    return spectral_builder._SpectralProcessContext(
        detector_positions=np.zeros((1, 3), dtype=np.float64),
        fe_indices=np.zeros(1, dtype=np.int64),
        pb_indices=np.zeros(1, dtype=np.int64),
        live_times=np.ones(1, dtype=np.float64),
        areas=np.ones(1, dtype=np.float64),
        quadrature_points=np.zeros((1, 1, 3), dtype=np.float64),
        quadrature_weights=np.ones((1, 1), dtype=np.float64),
        line_count=len(prepared),
        prepared_isotopes=prepared,
        kernel_chunk_size=1,
    )


def _line_window_sum(
    spectrum: NDArray[np.float64],
    edges: NDArray[np.float64],
    energy_keV: float,
    *,
    half_width_keV: float = 40.0,
) -> float:
    """Return response mass in a narrow, non-overlapping photopeak window."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = np.abs(centers - float(energy_keV)) <= float(half_width_keV)
    return float(np.sum(spectrum[mask]))


def test_shape_and_total_unshielded_detector_cps_semantics() -> None:
    """The tensor is MxBxGxI and sums to live-time-scaled cps@1m geometry."""
    edges = np.arange(0.0, 1505.0, 5.0)
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        edges,
        live_times_s=np.asarray([2.0, 3.0]),
        fe_indices=np.asarray([0, 7]),
        pb_indices=np.asarray([7, 0]),
    )
    patches = _patches(
        np.asarray([[[1.0, 0.0, 0.0]], [[0.0, 2.0, 1.0]]]),
    )
    line_table = {
        "Cs-137": _CS_LINE,
        "Co-60": _CO_LINES_UNATTENUATED,
    }
    kernel = _kernel(line_table)

    result = build_spectral_response(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        kernel,
    )

    assert result.response_per_integrated_strength.shape == (2, 300, 2, 2)
    expected = np.empty((2, 2, 2), dtype=float)
    flat_sources = patches.quadrature_points_xyz[:, 0, :]
    for isotope_index, isotope in enumerate(("Cs-137", "Co-60")):
        values = kernel.kernel_values_selected_pairs_for_detectors(
            isotope=isotope,
            detector_positions=observations.detector_positions_xyz,
            sources=flat_sources,
            fe_indices=observations.fe_indices,
            pb_indices=observations.pb_indices,
        )
        expected[:, :, isotope_index] = observations.live_times_s[:, None] * values
    np.testing.assert_allclose(
        np.sum(result.response_per_integrated_strength, axis=1),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_detector_response_matches_shared_pulse_height_kernel() -> None:
    """A one-line source is folded through the shared CeBr3 detector response."""
    edges = np.arange(0.0, 1005.0, 5.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))

    result = build_spectral_response(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    expected = detector_response_kernel_for_incident_gamma(
        centers,
        661.7,
        default_resolution(),
        cebr3_efficiency,
        5.0,
        continuum_to_peak=COMPTON_CONTINUUM_TO_PEAK,
        backscatter_fraction=BACKSCATTER_FRACTION,
    )
    actual = result.response_per_integrated_strength[0, :, 0, 0]
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
    assert np.count_nonzero(actual) > 20
    assert np.sum(actual[(centers > 50.0) & (centers < 400.0)]) > 0.0


def test_co60_lines_receive_distinct_shield_attenuation() -> None:
    """Co-60 line peaks must not share one aggregate attenuation multiplier."""
    edges = np.arange(1000.0, 1452.0, 2.0)
    orientation = generate_octant_orientations()[0]
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.0]]),
        edges,
        fe_indices=np.asarray([0]),
        pb_indices=np.asarray([0]),
    )
    patches = _patches(
        np.asarray([[[-orientation[0], -orientation[1], -orientation[2]]]])
    )
    differential_lines = (
        dict(_CO_LINES_UNATTENUATED[0]),
        {
            **_CO_LINES_UNATTENUATED[1],
            "fe": 2.0,
            "pb": 2.0,
        },
    )
    build_kwargs = {
        "observations": observations,
        "patches": patches,
        "isotopes": ("Co-60",),
        "continuum_to_peak": 0.0,
        "backscatter_fraction": 0.0,
    }
    unshielded = build_spectral_response(
        kernel=_kernel({"Co-60": _CO_LINES_UNATTENUATED}, thickness_cm=1.0),
        **build_kwargs,
    ).response_per_integrated_strength[0, :, 0, 0]
    shielded = build_spectral_response(
        kernel=_kernel({"Co-60": differential_lines}, thickness_cm=1.0),
        **build_kwargs,
    ).response_per_integrated_strength[0, :, 0, 0]

    ratios = []
    for energy in (1173.2, 1332.5):
        baseline = _line_window_sum(unshielded, edges, energy)
        ratios.append(_line_window_sum(shielded, edges, energy) / baseline)

    assert ratios[0] == pytest.approx(1.0, rel=1.0e-12, abs=1.0e-12)
    assert ratios[1] < 0.05
    assert ratios[1] < 0.1 * ratios[0]


def test_obstacle_uses_the_matching_line_mu_row() -> None:
    """Per-line obstacle tables attenuate only their corresponding gamma line."""
    edges = np.arange(1000.0, 1452.0, 2.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, -0.1, 0.6, 0.1, 0.1),),
        transport_mu_by_isotope={"Co-60": (0.0,)},
        transport_line_mu_by_isotope={"Co-60": ((0.0,), (1.0,))},
    )
    build_kwargs = {
        "observations": observations,
        "patches": patches,
        "isotopes": ("Co-60",),
        "continuum_to_peak": 0.0,
        "backscatter_fraction": 0.0,
    }
    baseline = build_spectral_response(
        kernel=_kernel({"Co-60": _CO_LINES_UNATTENUATED}),
        **build_kwargs,
    ).response_per_integrated_strength[0, :, 0, 0]
    attenuated = build_spectral_response(
        kernel=_kernel(
            {"Co-60": _CO_LINES_UNATTENUATED},
            obstacle_grid=obstacle_grid,
        ),
        **build_kwargs,
    ).response_per_integrated_strength[0, :, 0, 0]

    first_ratio = _line_window_sum(attenuated, edges, 1173.2) / _line_window_sum(
        baseline,
        edges,
        1173.2,
    )
    second_ratio = _line_window_sum(attenuated, edges, 1332.5) / _line_window_sum(
        baseline,
        edges,
        1332.5,
    )
    assert first_ratio == pytest.approx(1.0, rel=1.0e-12, abs=1.0e-12)
    assert second_ratio < 1.0e-6


def test_obstacle_line_slice_preserves_matching_compton_row() -> None:
    """Line kernels must select aligned total and Compton obstacle tables."""
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, -0.1, 0.6, 0.1, 0.1),),
        transport_mu_by_isotope={"Co-60": (0.4,)},
        transport_line_mu_by_isotope={"Co-60": ((0.2,), (0.4,))},
        transport_line_compton_mu_by_isotope={"Co-60": ((0.05,), (0.1,))},
    )

    selected = spectral_builder._obstacle_grid_for_line(
        obstacle_grid,
        "Co-60",
        1,
        require_line_resolved=True,
    )

    assert selected is not None
    assert selected.transport_line_mu_values("Co-60") == ((0.4,),)
    assert selected.transport_line_compton_mu_values("Co-60") == ((0.1,),)


def test_incomplete_obstacle_line_table_is_rejected() -> None:
    """A partial line table cannot silently fall back to aggregate attenuation."""
    edges = np.arange(1000.0, 1452.0, 2.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))
    incomplete_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, -0.1, 0.6, 0.1, 0.1),),
        transport_mu_by_isotope={"Co-60": (0.2,)},
        transport_line_mu_by_isotope={"Co-60": ((0.1,),)},
    )

    with pytest.raises(ValueError, match="has 1 rows but line index 1"):
        build_spectral_response(
            observations,
            patches,
            ("Co-60",),
            _kernel(
                {"Co-60": _CO_LINES_UNATTENUATED},
                obstacle_grid=incomplete_grid,
            ),
        )


def test_missing_obstacle_line_table_is_rejected_for_production_fit() -> None:
    """Spectral fitting cannot reuse one aggregate obstacle attenuation scalar."""
    edges = np.arange(1000.0, 1452.0, 2.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))
    aggregate_only_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, -0.1, 0.6, 0.1, 0.1),),
        transport_mu_by_isotope={"Co-60": (0.2,)},
    )

    with pytest.raises(ValueError, match="line-resolved obstacle attenuation"):
        build_spectral_response(
            observations,
            patches,
            ("Co-60",),
            _kernel(
                {"Co-60": _CO_LINES_UNATTENUATED},
                obstacle_grid=aggregate_only_grid,
            ),
        )


def test_density_response_applies_exact_patch_area_scaling() -> None:
    """Density columns equal integrated-strength columns times exact area."""
    edges = np.arange(0.0, 1005.0, 5.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.0]]), edges)
    patches = _patches(
        np.asarray([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        areas_m2=np.asarray([0.5, 2.0]),
    )

    result = build_spectral_response(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
    )

    np.testing.assert_allclose(
        result.response_per_integrated_strength[:, :, 0, :],
        result.response_per_integrated_strength[:, :, 1, :],
    )
    np.testing.assert_allclose(
        result.response_per_density,
        result.response_per_integrated_strength
        * np.asarray([0.5, 2.0])[None, None, :, None],
    )


def test_nuisance_rate_bases_are_nonnegative_and_live_time_scaled() -> None:
    """Background and scatter coefficients retain cps rate semantics."""
    edges = np.arange(0.0, 1005.0, 5.0)
    live_times = np.asarray([2.0, 5.0])

    response, names = build_spectral_nuisance_response(live_times, edges)
    empty, empty_names = build_spectral_nuisance_response(
        live_times,
        edges,
        include_background=False,
        include_scatter=False,
    )

    assert response.shape == (2, 200, 2)
    assert names == ("background_rate_cps", "scatter_rate_cps")
    assert np.all(response >= 0.0)
    np.testing.assert_allclose(
        np.sum(response, axis=1),
        np.repeat(live_times[:, None], 2, axis=1),
    )
    assert empty.shape == (2, 200, 0)
    assert empty_names == ()


def test_structured_nuisance_uses_shared_pair_and_station_bases() -> None:
    """Calibrated discrepancy must share coefficients rather than fit each row."""
    calibration = DiscrepancyCalibration.from_mapping(
        {
            "schema_version": 1,
            "calibration_id": "test-all-pairs",
            "independent_environment_ids": ["a", "b"],
            "shield_pair_ids": list(range(64)),
            "energy_bin_edges_keV": [0.0, 100.0, 200.0],
            "background_basis": [[0.7, 0.3]],
            "scatter_basis": [[0.4, 0.6]],
            "shield_pair_feature_basis": [[1.0, pair / 63.0] for pair in range(64)],
            "shield_leakage_basis": [[0.2, 0.8]],
            "low_rank_spectral_residual_basis": [[0.5, 0.5]],
            "gain_derivative_basis": [[-0.1, 0.1]],
            "resolution_derivative_basis": [[0.2, -0.2]],
            "overdispersion": {
                "family": "negative_binomial",
                "alpha_by_bin": [0.02, 0.03],
            },
            "shrinkage_l2_by_family": {
                "background": 1.0,
                "scatter": 2.0,
                "shield_leakage": 3.0,
                "station_rate": 4.0,
                "low_rank_residual": 5.0,
                "gain_drift": 6.0,
                "resolution_drift": 7.0,
            },
        }
    )

    response, names, shrinkage, alpha = build_structured_spectral_nuisance_response(
        np.asarray([1.0, 2.0, 1.5]),
        np.asarray([0.0, 100.0, 200.0]),
        np.asarray([0, 0, 7]),
        np.asarray([0, 1, 7]),
        np.asarray([0, 0, 1]),
        calibration,
        include_gain_resolution_drift=True,
    )

    assert response.shape[:2] == (3, 2)
    assert len(names) == response.shape[2] == shrinkage.size
    assert sum(name.startswith("station_rate:") for name in names) == 2
    assert sum(name.startswith("shield_leakage:") for name in names) == 2
    assert np.all(response >= 0.0)
    np.testing.assert_array_equal(alpha, [0.02, 0.03])


def test_kernel_chunk_size_does_not_change_cpu_result() -> None:
    """Kernel chunk boundaries alter memory use but not spectral responses."""
    edges = np.arange(0.0, 1505.0, 5.0)
    observations = _observations(
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.25, 0.0, 1.5],
                [0.5, 0.25, 2.0],
            ]
        ),
        edges,
        live_times_s=np.asarray([1.0, 1.5, 2.0]),
        fe_indices=np.asarray([0, 3, 7]),
        pb_indices=np.asarray([7, 4, 0]),
    )
    patches = _patches(
        np.asarray(
            [
                [[1.0, 0.0, 0.5], [1.25, 0.0, 0.5]],
                [[0.0, 1.0, 1.5], [0.0, 1.25, 1.5]],
            ]
        ),
        quadrature_weights=np.asarray([[0.25, 0.75], [0.6, 0.4]]),
    )
    line_table = {
        "Cs-137": _CS_LINE,
        "Co-60": _CO_LINES_UNATTENUATED,
    }

    small = build_spectral_response(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        _kernel(line_table),
        chunk_size=1,
    )
    large = build_spectral_response(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        _kernel(line_table),
        chunk_size=1_000_000,
    )

    np.testing.assert_array_equal(
        small.response_per_integrated_strength,
        large.response_per_integrated_strength,
    )
    np.testing.assert_array_equal(
        small.response_per_density,
        large.response_per_density,
    )
    np.testing.assert_array_equal(small.nuisance_response, large.nuisance_response)


def test_matrix_free_operator_matches_materialized_density_response(
    tmp_path: Path,
) -> None:
    """Forward, transpose, and materialized views must share exact physics."""
    edges = np.arange(0.0, 1505.0, 10.0)
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.5], [0.25, 0.1, 1.5]]),
        edges,
        live_times_s=np.asarray([1.0, 2.0]),
        fe_indices=np.asarray([0, 7]),
        pb_indices=np.asarray([7, 0]),
    )
    patches = _patches(
        np.asarray(
            [
                [[1.0, 0.0, 0.5], [1.2, 0.0, 0.5]],
                [[0.0, 1.0, 1.5], [0.0, 1.2, 1.5]],
            ]
        ),
        quadrature_weights=np.asarray([[0.4, 0.6], [0.25, 0.75]]),
        areas_m2=np.asarray([0.75, 1.5]),
    )
    isotopes = ("Cs-137", "Co-60")
    kernel = _kernel({"Cs-137": _CS_LINE, "Co-60": _CO_LINES_UNATTENUATED})
    materialized = build_spectral_response(
        observations,
        patches,
        isotopes,
        kernel,
        chunk_size=2,
    )
    streamed = build_spectral_response_operator(
        observations,
        patches,
        isotopes,
        kernel,
        chunk_size=2,
        energy_chunk_size=17,
        patch_chunk_size=1,
        cache_directory=tmp_path / "cache",
    )
    expected = materialized.response_per_density
    actual = streamed.operator.materialize()

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)
    vector = np.arange(1, streamed.operator.source_count + 1, dtype=float)
    residual = np.linspace(0.1, 1.0, streamed.operator.observation_count)
    np.testing.assert_allclose(
        streamed.operator.matvec(vector),
        expected.reshape(streamed.operator.observation_count, -1) @ vector,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        streamed.operator.rmatvec(residual),
        expected.reshape(streamed.operator.observation_count, -1).T @ residual,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_matrix_free_cache_appends_only_new_measurement_rows(tmp_path: Path) -> None:
    """Extending an online prefix must reuse every existing immutable row block."""
    edges = np.arange(0.0, 1005.0, 10.0)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.5]]]))
    first_observations = _observations(np.asarray([[0.0, 0.0, 0.5]]), edges)
    cache = tmp_path / "cache"
    first = build_spectral_response_operator(
        first_observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
        energy_chunk_size=25,
        patch_chunk_size=1,
        cache_directory=cache,
    )
    first.operator.row_sums()
    original_files = tuple(sorted(cache.rglob("*.npy")))
    original_bytes = {
        path.relative_to(cache): path.read_bytes() for path in original_files
    }

    extended_observations = _observations(
        np.asarray([[0.0, 0.0, 0.5], [0.5, 0.0, 1.0]]),
        edges,
    )
    extended = build_spectral_response_operator(
        extended_observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
        energy_chunk_size=25,
        patch_chunk_size=1,
        cache_directory=cache,
    )
    extended.operator.row_sums()
    final_files = tuple(sorted(cache.rglob("*.npy")))

    assert len(final_files) == 2 * len(original_files)
    assert all(
        (cache / relative).read_bytes() == content
        for relative, content in original_bytes.items()
    )


def test_factor_cache_namespace_is_stable_across_physical_object_instances(
    tmp_path: Path,
) -> None:
    """Equivalent shield objects must not create address-derived namespaces."""
    edges = np.arange(0.0, 1005.0, 10.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.5]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.5]]]))
    first_kernel = _kernel({"Cs-137": _CS_LINE})
    second_kernel = _kernel({"Cs-137": _CS_LINE})
    first_kernel.octant_shield = OctantShield()
    second_kernel.octant_shield = OctantShield()

    first = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137",),
        first_kernel,
        cache_directory=tmp_path,
    )
    second = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137",),
        second_kernel,
        cache_directory=tmp_path,
    )

    assert first.cache_directory == second.cache_directory
    assert second.operator.diagnostics["cache_stats"] == {
        "hits": 1,
        "misses": 0,
        "files": 1,
        "blocks": 1,
    }


def test_factor_cache_namespace_tracks_requested_isotope_order(
    tmp_path: Path,
) -> None:
    """Cached line columns must never be reused under another isotope order."""
    edges = np.arange(0.0, 1505.0, 10.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.5]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.5]], [[2.0, 0.0, 0.5]]]))
    kernel = _kernel({"Cs-137": _CS_LINE, "Co-60": _CO_LINES_UNATTENUATED})
    cache = tmp_path / "cache"

    original = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137", "Co-60"),
        kernel,
        cache_directory=cache,
    )
    reordered = build_spectral_response_operator(
        observations,
        patches,
        ("Co-60", "Cs-137"),
        kernel,
        cache_directory=cache,
    )
    repeated = build_spectral_response_operator(
        observations,
        patches,
        ("Co-60", "Cs-137"),
        kernel,
        cache_directory=cache,
    )
    fresh = build_spectral_response_operator(
        observations,
        patches,
        ("Co-60", "Cs-137"),
        kernel,
    )

    assert original.cache_directory != reordered.cache_directory
    assert repeated.cache_directory == reordered.cache_directory
    assert (
        original.operator.diagnostics["device_cache_key"]
        != reordered.operator.diagnostics["device_cache_key"]
    )
    assert reordered.operator.diagnostics["cache_stats"]["misses"] == 1
    assert repeated.operator.diagnostics["cache_stats"]["hits"] == 1
    np.testing.assert_array_equal(
        reordered.operator.materialize(),
        fresh.operator.materialize(),
    )


def test_spatial_factor_cache_separates_backend_and_numeric_precision() -> None:
    """Float32 GPU factors must never be reused as float64 CPU truth."""
    points = np.asarray([[[1.0, 0.0, 0.5]]], dtype=np.float64)
    patches = _patches(points)
    cpu = _kernel({"Cs-137": _CS_LINE})
    gpu_float32 = replace(
        cpu,
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float32",
    )
    gpu_float64 = replace(gpu_float32, gpu_dtype="float64")

    def cache_key(kernel: ContinuousKernel) -> str:
        """Return the private spatial key for one execution configuration."""
        return spectral_builder._spectral_spatial_cache_key(
            kernel=kernel,
            areas=patches.areas_m2,
            quadrature_points=patches.quadrature_points_xyz,
            quadrature_weights=patches.quadrature_weights,
            isotope_lines={"Cs-137": _CS_LINE},
        )

    assert cache_key(cpu) != cache_key(gpu_float32)
    assert cache_key(gpu_float32) != cache_key(gpu_float64)


def test_device_factor_key_is_stable_without_disk_cache_and_tracks_pulses() -> None:
    """In-memory CPU/CUDA reuse must not depend on enabling the disk cache."""
    edges = np.arange(0.0, 1005.0, 10.0)
    observations = _observations(np.asarray([[0.0, 0.0, 0.5]]), edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.5]]]))
    common = {
        "observations": observations,
        "patches": patches,
        "isotopes": ("Cs-137",),
    }
    first = build_spectral_response_operator(
        **common,
        kernel=_kernel({"Cs-137": _CS_LINE}),
    )
    equivalent = build_spectral_response_operator(
        **common,
        kernel=_kernel({"Cs-137": _CS_LINE}),
    )
    changed_pulse = build_spectral_response_operator(
        **common,
        kernel=_kernel({"Cs-137": _CS_LINE}),
        continuum_to_peak=0.5 * COMPTON_CONTINUUM_TO_PEAK,
    )

    first_key = first.operator.diagnostics["device_cache_key"]
    assert first.cache_directory is None
    assert isinstance(first_key, str) and first_key
    assert equivalent.operator.diagnostics["device_cache_key"] == first_key
    assert changed_pulse.operator.diagnostics["device_cache_key"] != first_key


def test_factor_builder_deduplicates_identical_measurement_rows(
    tmp_path: Path,
) -> None:
    """Bootstrap row repeats must share one transport calculation and file."""
    edges = np.arange(0.0, 1005.0, 10.0)
    positions = np.repeat(np.asarray([[0.0, 0.0, 0.5]]), 3, axis=0)
    observations = _observations(positions, edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.5]]]))

    result = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
        cache_directory=tmp_path,
    )
    construction = result.operator.diagnostics["performance"]["response_construction"]

    assert len(tuple(tmp_path.rglob("factors.npy"))) == 1
    assert construction["kernel_batched_measurements"] == 1
    np.testing.assert_array_equal(
        result.operator.spatial_factors,
        np.repeat(result.operator.spatial_factors[:1], 3, axis=0),
    )


def test_matrix_free_cache_io_is_one_file_per_unique_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factor-cache loading performs one file read per unique measurement."""
    edges = np.arange(0.0, 805.0, 10.0)
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.5], [0.25, 0.0, 1.0]]),
        edges,
    )
    points = np.zeros((3, 1, 3), dtype=float)
    points[:, 0, 0] = np.asarray([1.0, 1.5, 2.0])
    points[:, 0, 2] = 0.5
    patches = _patches(points)
    cache = tmp_path / "cache"
    options = {
        "energy_chunk_size": 7,
        "patch_chunk_size": 2,
        "cache_directory": cache,
    }
    first = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
        **options,
    )
    first.operator.row_sums()

    expected_file_count = observations.detector_positions_xyz.shape[0]
    assert len(tuple(cache.rglob("*.npy"))) == expected_file_count
    assert (
        first.operator.diagnostics["cache_file_layout"]
        == "measurement_full_patch_line_factors_v4"
    )

    load_count = 0
    original_load = spectral_builder.np.load

    def counted_load(*args: object, **kwargs: object) -> object:
        """Count cache loads while preserving NumPy behavior."""
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(spectral_builder.np, "load", counted_load)
    cached = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
        **options,
    )
    cached.operator.row_sums()

    assert load_count == expected_file_count
    assert cached.operator.diagnostics["cache_stats"] == {
        "hits": expected_file_count,
        "misses": 0,
        "files": expected_file_count,
        "blocks": expected_file_count * 2 * 12,
    }


def test_matrix_free_cpu_workers_preserve_exact_response() -> None:
    """Parallel CPU patch tasks must preserve deterministic float64 blocks."""
    edges = np.arange(0.0, 805.0, 10.0)
    observations = _observations(
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.25, 0.0, 0.5],
                [0.0, 0.25, 1.5],
                [0.25, 0.25, 1.5],
            ]
        ),
        edges,
    )
    points = np.zeros((8, 1, 3), dtype=float)
    points[:, 0, 0] = np.linspace(1.0, 2.0, 8)
    points[:, 0, 2] = 0.5
    patches = _patches(points)
    serial = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
        energy_chunk_size=16,
        patch_chunk_size=2,
        worker_count=1,
    )
    parallel = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
        energy_chunk_size=16,
        patch_chunk_size=2,
        worker_count=4,
    )

    np.testing.assert_array_equal(
        parallel.operator.materialize(),
        serial.operator.materialize(),
    )
    construction = parallel.operator.diagnostics["performance"]["response_construction"]
    assert construction["worker_count"] == 4
    assert construction["iterations"] == 1


def test_real_runtime_kernel_is_spawn_safe_and_bitwise_exact() -> None:
    """Spawn workers must rebuild the authenticated runtime scatter response."""
    fixture = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "shared_measurement_log"
        / "measurement_log"
    )
    replay = prepare_replay(
        fixture,
        config=MLEConfig(
            mode="spectral",
            isotope_names=("Co-60", "Cs-137", "Eu-154"),
            patch_spacing_m=(6.0, 6.0, 3.0),
            max_iterations=2,
            debias_refit=False,
            use_gpu=False,
        ),
    )
    assert (
        type(replay.kernel.additive_scatter_response)
        is AdditiveNoncollidedTransportResponse
    )
    kernel = replace(replay.kernel, gpu_dtype="float64")
    observations = _observations(
        replay.batch.detector_positions_xyz[:1],
        replay.batch.energy_bin_edges_keV,
        live_times_s=replay.batch.live_times_s[:1],
        fe_indices=replay.batch.fe_indices[:1],
        pb_indices=replay.batch.pb_indices[:1],
    )
    patches = _patches(
        np.asarray(
            [
                [[1.0, 0.25, 0.0]],
                [[1.25, 0.25, 0.0]],
                [[1.0, 0.75, 0.0]],
                [[1.25, 0.75, 0.0]],
            ],
            dtype=np.float64,
        )
    )
    options = {
        "observations": observations,
        "patches": patches,
        "isotopes": ("Cs-137",),
        "kernel": kernel,
        "measurement_chunk_size": 1,
        "patch_chunk_size": 1,
    }

    serial = build_spectral_response_operator(**options, worker_count=1)
    parallel = build_spectral_response_operator(**options, worker_count=4)

    serial_response = serial.operator.materialize()
    parallel_response = parallel.operator.materialize()
    assert serial_response.dtype == np.float64
    assert parallel_response.dtype == np.float64
    np.testing.assert_array_equal(parallel_response, serial_response)
    construction = parallel.operator.diagnostics["performance"]["response_construction"]
    assert construction["worker_count"] == 4


def test_physics_only_process_payload_preserves_shared_kernel_identity() -> None:
    """Physics-only payloads must round-trip once per shared runtime kernel."""
    response = PhysicsOnlyNoncollidedTransportResponse(
        detector_radius_m=0.0254,
        fe_scatter_distance_m=0.1,
        pb_scatter_distance_m=0.1,
    )
    kernel = ContinuousKernel(
        additive_scatter_response=response,
        use_gpu=False,
    )
    context = _process_context(kernel, ("Cs-137", "Co-60"))

    spawn_context = spectral_builder._fresh_spectral_process_context(context)
    spawn_context = pickle.loads(pickle.dumps(spawn_context))
    first_recipe, second_recipe = (
        prepared.kernel for prepared in spawn_context.prepared_isotopes
    )
    assert first_recipe is second_recipe

    restored = spectral_builder._materialize_spectral_process_context(spawn_context)
    first_kernel, second_kernel = (
        prepared.kernel for prepared in restored.prepared_isotopes
    )
    assert first_kernel is second_kernel
    assert isinstance(first_kernel, ContinuousKernel)
    restored_response = first_kernel.additive_scatter_response
    assert type(restored_response) is PhysicsOnlyNoncollidedTransportResponse
    assert restored_response.to_payload() == response.to_payload()


def test_process_payload_rejects_response_subclasses() -> None:
    """Worker serialization must fail fast for unsupported response subtypes."""

    class _DerivedPhysicsResponse(PhysicsOnlyNoncollidedTransportResponse):
        """Represent an unsupported behavior-bearing runtime response subtype."""

    response = _DerivedPhysicsResponse(
        detector_radius_m=0.0254,
        fe_scatter_distance_m=0.1,
        pb_scatter_distance_m=0.1,
    )
    context = _process_context(
        ContinuousKernel(additive_scatter_response=response, use_gpu=False),
        ("Cs-137",),
    )

    with pytest.raises(TypeError, match="Unsupported additive scatter response type"):
        spectral_builder._fresh_spectral_process_context(context)


def test_matrix_free_batches_eight_measurements_and_precomputes_line_pulses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight rows and both gamma lines must share one joint transport call."""
    edges = np.arange(0.0, 1505.0, 10.0)
    positions = np.zeros((8, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, 0.7, 8)
    positions[:, 2] = 0.5
    observations = _observations(positions, edges)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.5]], [[2.0, 0.0, 0.5]]]))
    scalar = build_spectral_response_operator(
        observations,
        patches,
        ("Co-60",),
        _kernel({"Co-60": _CO_LINES_UNATTENUATED}),
        measurement_chunk_size=1,
        patch_chunk_size=2,
        worker_count=1,
    )
    scalar_values = scalar.operator.materialize()
    original_pulse = spectral_builder.detector_response_kernel_for_incident_gamma
    original_joint = ContinuousKernel.kernel_values_selected_pairs_for_detectors_by_line
    original_scalar = ContinuousKernel.kernel_values_selected_pairs_for_detectors
    pulse_energies: list[float] = []
    joint_calls: list[tuple[str, int]] = []
    scalar_calls: list[str] = []

    def counted_pulse(*args: object, **kwargs: object) -> NDArray[np.float64]:
        """Record one position-independent gamma-line pulse construction."""
        energy = kwargs.get("incident_energy_keV", args[1] if len(args) > 1 else None)
        assert energy is not None
        pulse_energies.append(float(energy))
        return original_pulse(*args, **kwargs)

    def counted_joint(
        self: ContinuousKernel,
        *args: object,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        """Record one geometry-sharing line transport call."""
        isotope = str(kwargs.get("isotope", args[0] if args else ""))
        indices = np.asarray(kwargs["positive_line_indices"])
        joint_calls.append((isotope, int(indices.size)))
        return original_joint(self, *args, **kwargs)

    def counted_scalar(
        self: ContinuousKernel,
        *args: object,
        **kwargs: object,
    ) -> NDArray[np.float64]:
        """Record an obsolete scalar-line transport call."""
        isotope = str(kwargs.get("isotope", args[0] if args else ""))
        scalar_calls.append(isotope)
        return original_scalar(self, *args, **kwargs)

    monkeypatch.setattr(
        spectral_builder,
        "detector_response_kernel_for_incident_gamma",
        counted_pulse,
    )
    monkeypatch.setattr(
        ContinuousKernel,
        "kernel_values_selected_pairs_for_detectors_by_line",
        counted_joint,
    )
    monkeypatch.setattr(
        ContinuousKernel,
        "kernel_values_selected_pairs_for_detectors",
        counted_scalar,
    )
    batched = build_spectral_response_operator(
        observations,
        patches,
        ("Co-60",),
        _kernel({"Co-60": _CO_LINES_UNATTENUATED}),
        measurement_chunk_size=8,
        patch_chunk_size=2,
        worker_count=1,
    )
    batched_values = batched.operator.materialize()
    construction = batched.operator.diagnostics["performance"]["response_construction"]

    np.testing.assert_array_equal(batched_values, scalar_values)
    assert pulse_energies == [1173.2, 1332.5]
    assert joint_calls == [("Co-60", 2)]
    assert scalar_calls == []
    assert construction["kernel_batch_calls"] == 1
    assert construction["kernel_batched_measurements"] == 8


def test_operator_peak_memory_is_bounded_by_response_blocks() -> None:
    """Operator construction must not allocate the declared full response tensor."""
    edges = np.arange(0.0, 1501.0, 1.0)
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.5], [0.5, 0.0, 1.5]]),
        edges,
    )
    points = np.zeros((128, 1, 3), dtype=float)
    points[:, 0, 0] = np.linspace(1.0, 4.0, 128)
    points[:, 0, 2] = 0.5
    patches = _patches(points)
    full_response_bytes = 2 * 1500 * 128 * 8

    tracemalloc.start()
    streamed = build_spectral_response_operator(
        observations,
        patches,
        ("Cs-137",),
        _kernel({"Cs-137": _CS_LINE}),
        energy_chunk_size=32,
        patch_chunk_size=16,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < full_response_bytes
    with pytest.raises(MemoryError):
        streamed.operator.materialize(maximum_bytes=full_response_bytes - 1)


def test_invalid_pair_index_and_chunk_are_rejected_before_kernel_work() -> None:
    """Invalid pairs cannot wrap modulo the orientation count on GPU."""
    edges = np.arange(0.0, 1005.0, 5.0)
    patches = _patches(np.asarray([[[1.0, 0.0, 0.0]]]))
    invalid_pair = _observations(
        np.asarray([[0.0, 0.0, 0.0]]),
        edges,
        fe_indices=np.asarray([8]),
    )

    with pytest.raises(ValueError, match="must lie in"):
        build_spectral_response(
            invalid_pair,
            patches,
            ("Cs-137",),
            _kernel({"Cs-137": _CS_LINE}),
        )
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        build_spectral_response(
            _observations(np.asarray([[0.0, 0.0, 0.0]]), edges),
            patches,
            ("Cs-137",),
            _kernel({"Cs-137": _CS_LINE}),
            chunk_size=0,
        )


def test_cpu_gpu_equivalence_when_cuda_is_available() -> None:
    """Line-resolved spectral responses agree across CPU and CUDA kernels."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    edges = np.arange(1000.0, 1452.0, 2.0)
    observations = _observations(
        np.asarray([[0.0, 0.0, 0.5], [0.1, 0.0, 1.5]]),
        edges,
        live_times_s=np.asarray([1.0, 2.0]),
        fe_indices=np.asarray([0, 7]),
        pb_indices=np.asarray([0, 7]),
    )
    orientation = generate_octant_orientations()[0]
    patches = _patches(
        np.asarray(
            [
                [
                    [-orientation[0], -orientation[1], -orientation[2] + 0.5],
                    [
                        -1.1 * orientation[0],
                        -1.1 * orientation[1],
                        -1.1 * orientation[2] + 0.5,
                    ],
                ],
                [[1.0, 0.0, 1.5], [1.25, 0.0, 1.5]],
            ]
        ),
        quadrature_weights=np.asarray([[0.4, 0.6], [0.5, 0.5]]),
        areas_m2=np.asarray([0.75, 1.5]),
    )
    differential_lines = (
        dict(_CO_LINES_UNATTENUATED[0]),
        {
            **_CO_LINES_UNATTENUATED[1],
            "fe": 0.4,
            "pb": 0.6,
        },
    )
    obstacle_grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(1, 1),
        blocked_cells=(),
        transport_boxes_m=((0.4, -0.1, 0.0, 0.6, 0.1, 2.0),),
        transport_mu_by_isotope={"Co-60": (0.05,)},
        transport_line_mu_by_isotope={"Co-60": ((0.02,), (0.08,))},
    )

    cpu = build_spectral_response(
        observations,
        patches,
        ("Co-60",),
        _kernel(
            {"Co-60": differential_lines},
            thickness_cm=1.0,
            obstacle_grid=obstacle_grid,
        ),
        chunk_size=2,
    )
    gpu = build_spectral_response(
        observations,
        patches,
        ("Co-60",),
        _kernel(
            {"Co-60": differential_lines},
            thickness_cm=1.0,
            obstacle_grid=obstacle_grid,
            use_gpu=True,
        ),
        chunk_size=2,
    )

    np.testing.assert_allclose(
        gpu.response_per_integrated_strength,
        cpu.response_per_integrated_strength,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        gpu.response_per_density,
        cpu.response_per_density,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
