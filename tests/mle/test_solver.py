"""Tests for PF-independent regularized Poisson surface-map reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from three_d_estimation.solver import (
    _enforce_cuda_response_cache_requirement,
    _prepare_dense_torch_response,
    _torch_duplicate_grouping,
    _torch_response_product,
    _torch_stable_incidence_transpose,
    SurfaceMapConfig,
    evaluate_surface_map_objective,
    fit_surface_map_poisson,
    fit_surface_map_poisson_operator,
)
from three_d_estimation.response_operator import (
    BlockResponseOperator,
    LineFactorizedResponseOperator,
    ResponseBlock,
)


def _dense_density_operator(
    response: np.ndarray,
    areas: np.ndarray,
    isotope_count: int,
    *,
    diagnostics: dict[str, object] | None = None,
    traversal_count: list[int] | None = None,
) -> BlockResponseOperator:
    """Return a multi-block operator equivalent to one materialized response."""
    observation_shape = response.shape[:-2]
    patch_count = response.shape[-2]
    density_matrix = (
        response * areas.reshape((1,) * len(observation_shape) + (-1, 1))
    ).reshape(-1, patch_count * isotope_count)

    def factory():
        """Yield four blocks to exercise both streamed dimensions."""
        if traversal_count is not None:
            traversal_count[0] += 1
        row_split = max(1, density_matrix.shape[0] // 2)
        column_split = max(1, density_matrix.shape[1] // 2)
        for row_start, row_stop in (
            (0, row_split),
            (row_split, density_matrix.shape[0]),
        ):
            for column_start, column_stop in (
                (0, column_split),
                (column_split, density_matrix.shape[1]),
            ):
                if row_start == row_stop or column_start == column_stop:
                    continue
                yield ResponseBlock(
                    np.arange(row_start, row_stop),
                    np.arange(column_start, column_stop),
                    density_matrix[row_start:row_stop, column_start:column_stop],
                )

    return BlockResponseOperator(
        observation_shape,
        patch_count,
        isotope_count,
        factory,
        diagnostics=diagnostics,
    )


def _line_factorized_density_operator(
    *,
    spatial_factors: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> LineFactorizedResponseOperator:
    """Return a deterministic two-isotope factorized density response."""
    spatial = (
        np.asarray(
            [
                [[1.0, 0.2, 0.4], [0.3, 0.8, 0.1]],
                [[0.4, 0.7, 0.2], [1.1, 0.2, 0.5]],
                [[0.8, 0.1, 0.6], [0.2, 1.0, 0.3]],
            ],
            dtype=np.float64,
        )
        if spatial_factors is None
        else np.asarray(spatial_factors, dtype=np.float64)
    )
    pulses = np.asarray(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.0, 0.3, 0.5, 0.2],
            [0.1, 0.2, 0.4, 0.3],
        ],
        dtype=np.float64,
    )
    return LineFactorizedResponseOperator(
        spatial,
        pulses,
        np.asarray([0, 1, 0], dtype=np.int64),
        2,
        diagnostics=diagnostics,
    )


def test_response_sums_share_one_operator_traversal() -> None:
    """Row and column scaling sums must be accumulated in one block pass."""
    response = np.arange(1.0, 13.0).reshape(3, 2, 2, 1)
    traversals = [0]
    operator = _dense_density_operator(
        response,
        np.ones(2),
        isotope_count=1,
        traversal_count=traversals,
    )

    row_sums = operator.row_sums()
    column_sums = operator.column_sums()

    assert traversals == [1]
    matrix = response.reshape(6, 2)
    np.testing.assert_array_equal(row_sums, np.sum(matrix, axis=1))
    np.testing.assert_array_equal(column_sums, np.sum(matrix, axis=0))


def test_cuda_cache_budget_releases_unused_allocator_blocks_first() -> None:
    """CUDA fit checks must measure free memory after allocator cleanup."""

    class _Scalar:
        """Expose the float64 element size used by the cache estimator."""

        @staticmethod
        def element_size() -> int:
            """Return the test scalar size in bytes."""
            return 8

    class _Cuda:
        """Record allocator cleanup and free-memory query ordering."""

        def __init__(self) -> None:
            """Initialize an empty call log."""
            self.calls: list[str] = []

        def empty_cache(self) -> None:
            """Record release of unused allocator blocks."""
            self.calls.append("empty_cache")

        def mem_get_info(self, device: object) -> tuple[int, int]:
            """Return a deliberately insufficient post-cleanup budget."""
            del device
            self.calls.append("mem_get_info")
            return 100, 1_000

    class _Torch:
        """Provide the minimal torch surface needed before cache fallback."""

        def __init__(self) -> None:
            """Install the fake CUDA allocator."""
            self.cuda = _Cuda()

        @staticmethod
        def empty(shape: object, *, dtype: object) -> _Scalar:
            """Return one scalar carrying an eight-byte element size."""
            del shape, dtype
            return _Scalar()

    class _Device:
        """Represent one CUDA device for the cache-preparation test."""

        type = "cuda"

    response = np.ones((10, 10, 1), dtype=np.float64)
    operator = _dense_density_operator(
        response,
        np.ones(10, dtype=np.float64),
        isotope_count=1,
    )
    torch = _Torch()

    matrix, diagnostics, _, _ = _prepare_dense_torch_response(
        operator,
        device=_Device(),
        dtype=object(),
        cache_fraction=0.5,
        torch_module=torch,
    )

    assert matrix is None
    assert diagnostics["fallback_reason"] == "response_exceeds_device_cache_budget"
    assert torch.cuda.calls == ["empty_cache", "mem_get_info"]


def test_surface_map_recovers_piecewise_smooth_density() -> None:
    """Batched Poisson L1+TV fitting should recover a synthetic surface map."""
    response = np.asarray(
        [
            [1.0, 0.1, 0.05],
            [0.8, 0.2, 0.1],
            [0.1, 0.9, 0.2],
            [0.2, 0.8, 0.1],
            [0.1, 0.2, 1.0],
            [0.05, 0.1, 0.8],
        ],
        dtype=float,
    )
    areas = np.asarray([1.0, 1.0, 0.5], dtype=float)
    truth_density = np.asarray([40.0, 40.0, 8.0], dtype=float)
    background = np.full(response.shape[0], 2.0, dtype=float)
    observed = background + response @ (areas * truth_density)

    result = fit_surface_map_poisson(
        observed,
        response,
        areas,
        adjacency_edges=np.asarray([[0, 1], [1, 2]], dtype=int),
        adjacency_weights=np.ones(2, dtype=float),
        background=background,
        config=SurfaceMapConfig(
            l1_weight=1.0e-3,
            tv_weight=2.0e-3,
            max_iterations=5000,
            tolerance=2.0e-7,
            objective_tolerance=1.0e-8,
        ),
    )

    assert result.converged is True
    assert result.densities_cps_1m_m2[:, 0] == pytest.approx(
        truth_density,
        rel=0.02,
        abs=0.15,
    )
    assert result.integrated_strengths_cps_1m[:, 0] == pytest.approx(
        areas * truth_density,
        rel=0.02,
        abs=0.15,
    )
    assert result.deviance < 1.0e-2
    assert result.kkt_residual < 1.0e-4


def test_surface_map_zero_signal_stays_zero() -> None:
    """A zero-count observation should not create a regularized surface source."""
    result = fit_surface_map_poisson(
        np.zeros(3, dtype=float),
        np.eye(3, dtype=float),
        np.ones(3, dtype=float),
        adjacency_edges=np.asarray([[0, 1], [1, 2]], dtype=int),
        config=SurfaceMapConfig(l1_weight=0.1, tv_weight=0.2),
    )

    assert result.converged is True
    assert np.array_equal(result.densities_cps_1m_m2, np.zeros((3, 1)))
    assert np.array_equal(result.integrated_strengths_cps_1m, np.zeros((3, 1)))
    assert result.deviance == pytest.approx(0.0, abs=1.0e-10)


def test_solver_reports_iteration_progress_and_eta() -> None:
    """Long solver runs should expose checked iterations without changing results."""
    events: list[dict[str, object]] = []

    fit_surface_map_poisson(
        np.asarray([10.0, 5.0], dtype=float),
        np.eye(2, dtype=float),
        np.ones(2, dtype=float),
        config=SurfaceMapConfig(
            max_iterations=6,
            check_interval=2,
            tolerance=0.0,
            objective_tolerance=0.0,
        ),
        progress_hook=lambda event: events.append(dict(event)),
        progress_phase="test_solver",
    )

    assert events[0] == {
        "phase": "test_solver",
        "completed": 0,
        "total": 6,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
    }
    checked = [int(event["completed"]) for event in events[1:]]
    assert checked
    assert checked == sorted(checked)
    assert all(event["phase"] == "test_solver" for event in events)
    assert all(float(event["eta_seconds"]) >= 0.0 for event in events[1:])


def test_required_cuda_response_cache_fails_before_streaming() -> None:
    """A required exact GPU cache must reject a potentially multi-day fallback."""
    diagnostics = {
        "fallback_reason": "response_exceeds_device_cache_budget",
        "required_bytes": 6_394_509_312,
        "free_device_bytes_at_prepare": 4_000_000_000,
        "budget_bytes": 2_400_000_000,
    }

    with pytest.raises(RuntimeError, match="streamed host blocks were not started"):
        _enforce_cuda_response_cache_requirement(
            None,
            diagnostics,
            required=True,
        )

    _enforce_cuda_response_cache_requirement(None, diagnostics, required=False)
    _enforce_cuda_response_cache_requirement(object(), diagnostics, required=True)


def test_surface_map_area_semantics_separate_density_and_strength() -> None:
    """Equal integrated sources on unequal patches should have inverse-area density."""
    result = fit_surface_map_poisson(
        np.asarray([41.0, 41.0], dtype=float),
        np.eye(2, dtype=float),
        np.asarray([2.0, 0.5], dtype=float),
        background=1.0,
        config=SurfaceMapConfig(
            max_iterations=4000,
            tolerance=1.0e-7,
            objective_tolerance=1.0e-8,
        ),
    )

    assert result.converged is True
    assert result.integrated_strengths_cps_1m[:, 0] == pytest.approx(
        [40.0, 40.0],
        rel=1.0e-5,
    )
    assert result.densities_cps_1m_m2[:, 0] == pytest.approx(
        [20.0, 80.0],
        rel=1.0e-5,
    )


def test_surface_map_profiles_non_negative_nuisance_without_fake_source() -> None:
    """An unpenalized nuisance basis should absorb common leakage instead of a source."""
    source_response = np.asarray([[1.0], [0.8], [0.4], [0.2]], dtype=float)
    nuisance_response = source_response.copy()
    observed = 1.0 + nuisance_response[:, 0] * 100.0

    result = fit_surface_map_poisson(
        observed,
        source_response,
        np.ones(1, dtype=float),
        background=1.0,
        nuisance_response=nuisance_response,
        config=SurfaceMapConfig(
            l1_weight=1.0,
            max_iterations=4000,
            tolerance=1.0e-7,
            objective_tolerance=1.0e-8,
        ),
    )

    assert result.converged is True
    assert result.densities_cps_1m_m2[0, 0] == pytest.approx(0.0, abs=1.0e-5)
    assert result.nuisance_coefficients == pytest.approx([100.0], rel=1.0e-5)
    assert result.deviance < 1.0e-8


def test_l1_reduces_redundant_surface_support() -> None:
    """Integrated-strength L1 should suppress redundant response columns."""
    response = np.asarray(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 0.8],
        ]
    )
    observed = response @ np.asarray([30.0, 0.0, 0.0])
    unregularized = fit_surface_map_poisson(
        observed,
        response,
        np.ones(3),
        config=SurfaceMapConfig(max_iterations=5000),
    )
    sparse = fit_surface_map_poisson(
        observed,
        response,
        np.ones(3),
        config=SurfaceMapConfig(l1_weight=1.0, max_iterations=5000),
    )

    unregularized_support = np.count_nonzero(
        unregularized.integrated_strengths_cps_1m[:, 0] > 1.0e-3
    )
    sparse_support = np.count_nonzero(sparse.integrated_strengths_cps_1m[:, 0] > 1.0e-3)
    assert sparse_support < unregularized_support


def test_graph_tv_reduces_patch_to_patch_fragmentation() -> None:
    """Physical graph TV should reduce artificial neighboring density jumps."""
    observed = np.asarray([50.0, 10.0, 50.0])
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    unregularized = fit_surface_map_poisson(
        observed,
        np.eye(3),
        np.ones(3),
        adjacency_edges=edges,
        config=SurfaceMapConfig(max_iterations=4000),
    )
    smoothed = fit_surface_map_poisson(
        observed,
        np.eye(3),
        np.ones(3),
        adjacency_edges=edges,
        config=SurfaceMapConfig(tv_weight=1.0, max_iterations=4000),
    )

    raw_fragmentation = np.sum(np.abs(np.diff(unregularized.densities_cps_1m_m2[:, 0])))
    tv_fragmentation = np.sum(np.abs(np.diff(smoothed.densities_cps_1m_m2[:, 0])))
    assert tv_fragmentation < 0.05 * raw_fragmentation


def test_surface_map_tensor_batch_matches_flattened_batch() -> None:
    """Spectrum-tensor fitting should equal the same batched flattened problem."""
    response = np.asarray(
        [
            [
                [[1.0, 0.1], [0.2, 0.0]],
                [[0.8, 0.2], [0.1, 0.1]],
                [[0.2, 0.7], [0.0, 0.2]],
            ],
            [
                [[0.4, 0.1], [0.8, 0.0]],
                [[0.1, 0.2], [0.7, 0.1]],
                [[0.0, 0.4], [0.2, 0.9]],
            ],
        ],
        dtype=float,
    )
    areas = np.asarray([2.0, 0.5], dtype=float)
    density = np.asarray([[12.0, 4.0], [3.0, 18.0]], dtype=float)
    nuisance_response = np.asarray(
        [[0.1, 0.2, 0.3], [0.2, 0.1, 0.2]],
        dtype=float,
    )
    nuisance_coefficient = 7.0
    background = np.full((2, 3), 1.5, dtype=float)
    observed = (
        background
        + np.einsum("mbci,ci->mb", response, density * areas[:, None])
        + nuisance_response * nuisance_coefficient
    )
    config = SurfaceMapConfig(
        l1_weight=1.0e-3,
        tv_weight=2.0e-3,
        nuisance_l2_weight=1.0e-4,
        max_iterations=2500,
        tolerance=1.0e-7,
        objective_tolerance=1.0e-8,
    )
    common_kwargs = {
        "patch_areas_m2": areas,
        "adjacency_edges": np.asarray([[0, 1], [1, 0]], dtype=int),
        "adjacency_weights": np.asarray([0.75, 0.75], dtype=float),
        "background": background,
        "nuisance_response": nuisance_response[..., None],
        "config": config,
    }

    tensor_result = fit_surface_map_poisson(
        observed,
        response,
        **common_kwargs,
    )
    flat_result = fit_surface_map_poisson(
        observed.reshape(-1),
        response.reshape(observed.size, 2, 2),
        patch_areas_m2=areas,
        adjacency_edges=common_kwargs["adjacency_edges"],
        adjacency_weights=common_kwargs["adjacency_weights"],
        background=background.reshape(-1),
        nuisance_response=nuisance_response.reshape(-1, 1),
        config=config,
    )

    assert tensor_result.densities_cps_1m_m2 == pytest.approx(
        flat_result.densities_cps_1m_m2,
        rel=1.0e-11,
        abs=1.0e-11,
    )
    assert tensor_result.nuisance_coefficients == pytest.approx(
        flat_result.nuisance_coefficients,
        rel=1.0e-11,
        abs=1.0e-11,
    )
    assert tensor_result.objective == pytest.approx(flat_result.objective, rel=1.0e-12)


def test_matrix_free_solver_matches_materialized_tensor() -> None:
    """Streamed CPU updates must reproduce the materialized solver result."""
    response = np.asarray(
        [
            [
                [[1.0, 0.1], [0.2, 0.0]],
                [[0.8, 0.2], [0.1, 0.1]],
                [[0.2, 0.7], [0.0, 0.2]],
            ],
            [
                [[0.4, 0.1], [0.8, 0.0]],
                [[0.1, 0.2], [0.7, 0.1]],
                [[0.0, 0.4], [0.2, 0.9]],
            ],
        ],
        dtype=float,
    )
    areas = np.asarray([2.0, 0.5])
    truth = np.asarray([[12.0, 4.0], [3.0, 18.0]])
    observed = np.einsum("mbgi,gi->mb", response, truth * areas[:, None])
    config = SurfaceMapConfig(
        l1_weight=1.0e-3,
        tv_weight=2.0e-3,
        max_iterations=3000,
        tolerance=1.0e-7,
        objective_tolerance=1.0e-8,
    )
    edges = np.asarray([[0, 1]], dtype=np.int64)
    materialized = fit_surface_map_poisson(
        observed,
        response,
        areas,
        adjacency_edges=edges,
        config=config,
    )
    operator = _dense_density_operator(response, areas, isotope_count=2)
    streamed = fit_surface_map_poisson_operator(
        observed,
        operator,
        areas,
        adjacency_edges=edges,
        config=config,
    )

    np.testing.assert_allclose(
        streamed.densities_cps_1m_m2,
        materialized.densities_cps_1m_m2,
        rtol=2.0e-8,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        streamed.expected_counts,
        materialized.expected_counts,
        rtol=2.0e-8,
        atol=2.0e-8,
    )
    assert streamed.kkt_residual == pytest.approx(
        materialized.kkt_residual,
        rel=2.0e-7,
        abs=2.0e-9,
    )


def test_line_factorized_cpu_solver_matches_materialized_tensor() -> None:
    """CPU iterations must retain exact factors instead of expanding blocks."""
    operator = _line_factorized_density_operator()
    response = operator.materialize()
    areas = np.ones(operator.patch_count, dtype=np.float64)
    truth = np.asarray([[8.0, 3.0], [2.0, 11.0]], dtype=np.float64)
    observed = np.einsum("mbgi,gi->mb", response, truth)
    config = SurfaceMapConfig(
        max_iterations=800,
        check_interval=20,
        tolerance=1.0e-8,
        objective_tolerance=1.0e-9,
    )

    materialized = fit_surface_map_poisson(
        observed,
        response,
        areas,
        config=config,
    )
    factorized = fit_surface_map_poisson_operator(
        observed,
        operator,
        areas,
        config=config,
        gpu_dtype="float64",
    )

    np.testing.assert_allclose(
        factorized.densities_cps_1m_m2,
        materialized.densities_cps_1m_m2,
        rtol=2.0e-8,
        atol=2.0e-8,
    )
    cache = operator.diagnostics["performance"]["solver"]["response_cache"]
    assert cache["mode"] == "line_factorized_cpu_cache"
    assert cache["cached_bytes"] < cache["dense_equivalent_bytes"]


def test_line_factorized_cpu_cache_reuses_exact_and_selected_rows() -> None:
    """CPU factor caches must reuse immutable factors across repeated fits."""
    torch = pytest.importorskip("torch")
    cache: dict[str, object] = {}
    operator = _line_factorized_density_operator(
        diagnostics={
            "device_cache_key": "cpu-line-factor-test",
            "measurement_row_keys": ["a", "b", "c"],
        }
    )
    first_response, first_diagnostics, _, _ = _prepare_dense_torch_response(
        operator,
        device=torch.device("cpu"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    repeated_response, repeated_diagnostics, _, _ = _prepare_dense_torch_response(
        operator,
        device=torch.device("cpu"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    selected = operator.select_measurements([2, 0])
    selected_response, selected_diagnostics, _, _ = _prepare_dense_torch_response(
        selected,
        device=torch.device("cpu"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )

    assert first_response is not None
    assert repeated_response is first_response
    assert selected_response is not None
    assert first_diagnostics["mode"] == "line_factorized_cpu_cache"
    assert repeated_diagnostics["mode"] == "persistent_line_factor_cache_hit"
    assert selected_diagnostics["mode"] == "persistent_line_factor_row_gather"


def test_line_factorized_cpu_cache_reuses_factors_for_source_masks() -> None:
    """Debias masks must reuse cached factors without changing products."""
    torch = pytest.importorskip("torch")
    cache: dict[str, object] = {}
    operator = _line_factorized_density_operator(
        diagnostics={
            "device_cache_key": "cpu-line-factor-mask-test",
            "measurement_row_keys": ["a", "b", "c"],
        }
    )
    base_response, _, _, _ = _prepare_dense_torch_response(
        operator,
        device=torch.device("cpu"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    mask = np.asarray([[True, False], [False, True]])
    masked = operator.masked_sources(mask)
    masked_response, diagnostics, row_sums, column_sums = _prepare_dense_torch_response(
        masked,
        device=torch.device("cpu"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )

    assert base_response is not None
    assert masked_response is not None
    assert masked_response.spatial_factors is base_response.spatial_factors
    assert masked_response.pulse_shapes is base_response.pulse_shapes
    assert diagnostics["mode"] == "persistent_line_factor_mask_view"
    assert diagnostics["host_to_device_bytes"] == 0
    source = torch.arange(1, masked.source_count + 1, dtype=torch.float64)
    actual = _torch_response_product(
        masked,
        source,
        transpose=False,
        torch_module=torch,
        dense_response=masked_response,
    )
    np.testing.assert_allclose(actual.numpy(), masked.matvec(source.numpy()))
    expected_rows, expected_columns = masked.response_sums()
    np.testing.assert_array_equal(row_sums, expected_rows)
    np.testing.assert_array_equal(column_sums, expected_columns)


def test_line_factorized_cpu_cache_reuses_masked_selected_rows() -> None:
    """Bootstrap row gathers and debias masks must share resident factors."""
    torch = pytest.importorskip("torch")
    cache: dict[str, object] = {}
    operator = _line_factorized_density_operator(
        diagnostics={
            "device_cache_key": "cpu-line-factor-mask-gather-test",
            "measurement_row_keys": ["a", "b", "c"],
        }
    )
    base_response, _, _, _ = _prepare_dense_torch_response(
        operator,
        device=torch.device("cpu"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    masked = operator.masked_sources(
        np.asarray([[True, False], [True, False]])
    ).select_measurements([2, 0, 2])
    masked_response, diagnostics, _, _ = _prepare_dense_torch_response(
        masked,
        device=torch.device("cpu"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )

    assert base_response is not None
    assert masked_response is not None
    assert masked_response.spatial_factors is base_response.spatial_factors
    assert diagnostics["mode"] == "persistent_line_factor_masked_row_gather"
    source = torch.arange(1, masked.source_count + 1, dtype=torch.float64)
    actual = _torch_response_product(
        masked,
        source,
        transpose=False,
        torch_module=torch,
        dense_response=masked_response,
    )
    np.testing.assert_allclose(actual.numpy(), masked.matvec(source.numpy()))


def test_gpu_request_rejects_cpu_device_instead_of_silent_fallback() -> None:
    """An explicit GPU solve must never execute silently on a CPU device."""
    operator = _line_factorized_density_operator()
    observed = np.ones(operator.observation_shape, dtype=np.float64)

    with pytest.raises(ValueError, match="requires a CUDA"):
        fit_surface_map_poisson_operator(
            observed,
            operator,
            np.ones(operator.patch_count),
            use_gpu=True,
            gpu_device="cpu",
            config=SurfaceMapConfig(max_iterations=1),
        )


def test_poisson_em_warm_start_accelerates_large_zero_initialized_fit() -> None:
    """EM initialization should remove the slow scale-up from the zero boundary."""
    response = np.asarray(
        [
            [[1.0], [0.05]],
            [[0.8], [0.1]],
            [[0.1], [0.9]],
            [[0.05], [1.0]],
        ],
        dtype=float,
    )
    areas = np.ones(2, dtype=float)
    truth = np.asarray([[1_000_000.0], [500_000.0]])
    observed = np.einsum("mgi,gi->m", response, truth)
    cold = fit_surface_map_poisson(
        observed,
        response,
        areas,
        config=SurfaceMapConfig(max_iterations=40, check_interval=10),
    )
    config = SurfaceMapConfig(
        max_iterations=40,
        check_interval=10,
        poisson_em_warm_start_iterations=20,
    )
    materialized = fit_surface_map_poisson(
        observed,
        response,
        areas,
        config=config,
    )
    operator = _dense_density_operator(response, areas, isotope_count=1)
    streamed = fit_surface_map_poisson_operator(
        observed,
        operator,
        areas,
        config=config,
    )

    assert materialized.deviance < cold.deviance * 1.0e-3
    np.testing.assert_allclose(
        streamed.densities_cps_1m_m2,
        materialized.densities_cps_1m_m2,
        rtol=1.0e-9,
        atol=1.0e-6,
    )
    assert (
        operator.diagnostics["performance"]["solver"][
            "poisson_em_warm_start_iterations"
        ]
        == 20
    )


def test_poisson_em_polishes_nonzero_warm_start() -> None:
    """EM initialization must remain active after refinement or explicit resume."""
    observed = np.asarray([900.0, 650.0, 400.0], dtype=float)
    response = np.asarray(
        [
            [[1.0], [0.2]],
            [[0.4], [0.8]],
            [[0.1], [1.1]],
        ],
        dtype=float,
    )
    initial = np.asarray([[1.0], [1.0]], dtype=float)
    config = SurfaceMapConfig(
        max_iterations=1,
        check_interval=1,
        poisson_em_warm_start_iterations=20,
    )

    result = fit_surface_map_poisson(
        observed,
        response,
        np.ones(2, dtype=float),
        config=config,
        initial_densities_cps_1m_m2=initial,
    )

    initial_expected = np.einsum("mgi,gi->m", response, initial)
    initial_deviance = 2.0 * np.sum(
        observed * np.log(observed / initial_expected) - (observed - initial_expected)
    )
    assert result.deviance < initial_deviance * 0.1


def test_kkt_gate_prevents_relative_change_false_convergence() -> None:
    """Small state changes cannot declare convergence while KKT still fails."""
    observed = np.asarray([100.0, 30.0], dtype=float)
    response = np.asarray([[[1.0]], [[0.1]]], dtype=float)
    common = {
        "max_iterations": 2,
        "check_interval": 1,
        "tolerance": 1.0,
        "objective_tolerance": 1.0,
    }

    relative_only = fit_surface_map_poisson(
        observed,
        response,
        np.ones(1, dtype=float),
        config=SurfaceMapConfig(**common),
    )
    kkt_gated = fit_surface_map_poisson(
        observed,
        response,
        np.ones(1, dtype=float),
        config=SurfaceMapConfig(**common, kkt_tolerance=0.0),
    )

    assert relative_only.converged is True
    assert kkt_gated.converged is False
    assert kkt_gated.kkt_residual > 0.0


def test_matrix_free_cpu_gpu_solver_equivalence_when_available() -> None:
    """CUDA and CPU must execute equivalent streamed primal-dual updates."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    response = np.asarray(
        [[[1.0], [0.2]], [[0.3], [1.0]], [[0.8], [0.4]]],
        dtype=float,
    )
    areas = np.asarray([1.0, 2.0])
    truth = np.asarray([[10.0], [3.0]])
    observed = np.einsum("mgi,gi->m", response, truth * areas[:, None])
    operator = _dense_density_operator(response, areas, isotope_count=1)
    config = SurfaceMapConfig(
        max_iterations=1500,
        poisson_em_warm_start_iterations=20,
        tolerance=1.0e-8,
        kkt_tolerance=0.0,
    )

    cpu = fit_surface_map_poisson_operator(
        observed,
        operator,
        areas,
        config=config,
        gpu_dtype="float64",
    )
    progress_events: list[dict[str, object]] = []
    gpu = fit_surface_map_poisson_operator(
        observed,
        operator,
        areas,
        config=config,
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
        progress_hook=lambda event: progress_events.append(dict(event)),
    )

    np.testing.assert_allclose(
        gpu.densities_cps_1m_m2,
        cpu.densities_cps_1m_m2,
        rtol=1.0e-8,
        atol=1.0e-9,
    )
    solver_performance = operator.diagnostics["performance"]["solver"]
    assert solver_performance["response_cache"]["mode"] == "dense_cuda_cache"
    assert solver_performance["response_product_calls"] > 0
    cache_events = [
        event
        for event in progress_events
        if str(event["phase"]).endswith(":cuda_response_cache")
    ]
    assert cache_events
    assert cache_events[-1]["completed"] == cache_events[-1]["total"]


def test_line_factorized_cpu_gpu_products_are_equivalent_when_available() -> None:
    """CUDA line-factor forward and transpose products must equal CPU products."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    operator = _line_factorized_density_operator()
    cpu_response, cpu_diagnostics, _, _ = _prepare_dense_torch_response(
        operator,
        device=torch.device("cpu"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
    )
    gpu_response, gpu_diagnostics, _, _ = _prepare_dense_torch_response(
        operator,
        device=torch.device("cuda"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
    )
    assert cpu_response is not None
    assert gpu_response is not None
    source = np.linspace(0.2, 1.1, operator.source_count)
    residual = np.linspace(0.1, 0.9, operator.observation_count)

    cpu_forward = _torch_response_product(
        operator,
        torch.as_tensor(source, dtype=torch.float64),
        transpose=False,
        torch_module=torch,
        dense_response=cpu_response,
    )
    gpu_forward = _torch_response_product(
        operator,
        torch.as_tensor(source, dtype=torch.float64, device="cuda"),
        transpose=False,
        torch_module=torch,
        dense_response=gpu_response,
    )
    cpu_transpose = _torch_response_product(
        operator,
        torch.as_tensor(residual, dtype=torch.float64),
        transpose=True,
        torch_module=torch,
        dense_response=cpu_response,
    )
    gpu_transpose = _torch_response_product(
        operator,
        torch.as_tensor(residual, dtype=torch.float64, device="cuda"),
        transpose=True,
        torch_module=torch,
        dense_response=gpu_response,
    )

    np.testing.assert_allclose(
        gpu_forward.cpu().numpy(),
        cpu_forward.numpy(),
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        gpu_transpose.cpu().numpy(),
        cpu_transpose.numpy(),
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    assert cpu_diagnostics["mode"] == "line_factorized_cpu_cache"
    assert gpu_diagnostics["mode"] == "line_factorized_cuda_cache"
    assert (
        gpu_diagnostics["host_to_device_bytes"]
        < gpu_diagnostics["dense_equivalent_bytes"]
    )


@pytest.mark.parametrize(
    ("gpu_dtype", "rtol", "atol"),
    (("float64", 2.0e-12, 2.0e-13), ("float32", 3.0e-5, 3.0e-6)),
)
def test_line_factorized_cuda_transpose_is_bitwise_deterministic(
    gpu_dtype: str,
    rtol: float,
    atol: float,
) -> None:
    """Duplicate rows and isotope lines must reduce deterministically on CUDA."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(20260822)
    measurement_count = 64
    selected_count = 256
    energy_count = 851
    patch_count = 1292
    line_isotopes = np.asarray([0, 0, 1, 2, 2, 2, 2, 2, 2], dtype=np.int64)
    spatial = rng.uniform(
        1.0e-5,
        1.0e-2,
        (measurement_count, patch_count, line_isotopes.size),
    )
    pulses = rng.uniform(0.1, 1.0, (line_isotopes.size, energy_count))
    pulses /= np.sum(pulses, axis=1, keepdims=True)
    row_keys = [f"row-{index}" for index in range(measurement_count)]
    operator = LineFactorizedResponseOperator(
        spatial,
        pulses,
        line_isotopes,
        3,
        diagnostics={
            "device_cache_key": f"deterministic-{gpu_dtype}",
            "measurement_row_keys": row_keys,
        },
    )
    dtype = torch.float64 if gpu_dtype == "float64" else torch.float32
    cache: dict[str, object] = {}
    base_response, _, _, _ = _prepare_dense_torch_response(
        operator,
        device=torch.device("cuda"),
        dtype=dtype,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    selection = np.arange(selected_count, dtype=np.int64) % measurement_count
    rng.shuffle(selection)
    selected = operator.select_measurements(selection.tolist())
    selected_response, diagnostics, _, _ = _prepare_dense_torch_response(
        selected,
        device=torch.device("cuda"),
        dtype=dtype,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    assert base_response is not None
    assert selected_response is not None
    assert diagnostics["mode"] == "persistent_line_factor_row_gather"
    assert base_response.line_group_order is not None
    assert selected_response.measurement_group_order is not None

    for current_operator, prepared_response in (
        (operator, base_response),
        (selected, selected_response),
    ):
        residual = rng.normal(size=current_operator.observation_count).astype(
            np.float64 if gpu_dtype == "float64" else np.float32
        )
        residual_t = torch.as_tensor(residual, dtype=dtype, device="cuda")
        outputs = tuple(
            _torch_response_product(
                current_operator,
                residual_t,
                transpose=True,
                torch_module=torch,
                dense_response=prepared_response,
            )
            for _ in range(30)
        )
        torch.cuda.synchronize()
        assert all(torch.equal(outputs[0], output) for output in outputs[1:])
        np.testing.assert_allclose(
            outputs[0].cpu().numpy(),
            current_operator.rmatvec(residual),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    ("gpu_dtype", "rtol", "atol"),
    (("float64", 2.0e-12, 2.0e-13), ("float32", 3.0e-5, 3.0e-6)),
)
def test_dense_cuda_row_gather_transpose_is_bitwise_deterministic(
    gpu_dtype: str,
    rtol: float,
    atol: float,
) -> None:
    """Dense cached bootstrap rows must sum repeats without CUDA atomics."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(20260823)
    measurement_count = 64
    selected_count = 1024
    energy_count = 32
    patch_count = 16
    response = rng.uniform(
        1.0e-5,
        1.0e-2,
        (measurement_count, energy_count, patch_count, 1),
    )
    areas = rng.uniform(0.5, 2.0, patch_count)
    row_keys = [f"dense-row-{index}" for index in range(measurement_count)]
    operator = _dense_density_operator(
        response,
        areas,
        isotope_count=1,
        diagnostics={
            "device_cache_key": f"dense-deterministic-{gpu_dtype}",
            "measurement_row_keys": row_keys,
        },
    )
    dtype = torch.float64 if gpu_dtype == "float64" else torch.float32
    cache: dict[str, object] = {}
    base_response, _, _, _ = _prepare_dense_torch_response(
        operator,
        device=torch.device("cuda"),
        dtype=dtype,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    selection = np.arange(selected_count, dtype=np.int64) % measurement_count
    rng.shuffle(selection)
    selected_operator = _dense_density_operator(
        response[selection],
        areas,
        isotope_count=1,
        diagnostics={
            "device_cache_key": f"dense-deterministic-{gpu_dtype}",
            "measurement_row_keys": [row_keys[index] for index in selection],
        },
    )
    selected_response, diagnostics, _, _ = _prepare_dense_torch_response(
        selected_operator,
        device=torch.device("cuda"),
        dtype=dtype,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    assert base_response is not None
    assert selected_response is not None
    assert diagnostics["mode"] == "persistent_cuda_row_gather"
    residual = rng.normal(size=selected_operator.observation_count).astype(
        np.float64 if gpu_dtype == "float64" else np.float32
    )
    residual_t = torch.as_tensor(residual, dtype=dtype, device="cuda")
    outputs = tuple(
        _torch_response_product(
            selected_operator,
            residual_t,
            transpose=True,
            torch_module=torch,
            dense_response=selected_response,
        )
        for _ in range(30)
    )
    torch.cuda.synchronize()

    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    np.testing.assert_allclose(
        outputs[0].cpu().numpy(),
        selected_operator.rmatvec(residual),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("gpu_dtype", ("float64", "float32"))
def test_cuda_tv_incidence_transpose_is_bitwise_deterministic(
    gpu_dtype: str,
) -> None:
    """TV edge contributions must reduce stably into repeated patch columns."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(20260824)
    patch_count = 1024
    edge_count = 16384
    isotope_count = 3
    edges = rng.integers(0, patch_count, size=(edge_count, 2), dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edge_count = int(edges.shape[0])
    incidence_rows = np.repeat(np.arange(edge_count, dtype=np.int64), 2)
    incidence_columns = edges.reshape(-1)
    incidence_coefficients = np.tile(
        np.asarray([-1.0, 1.0], dtype=np.float64),
        edge_count,
    )
    dtype = torch.float64 if gpu_dtype == "float64" else torch.float32
    device = torch.device("cuda")
    group_order, group_indices, group_lengths = _torch_duplicate_grouping(
        incidence_columns,
        device=device,
        torch_module=torch,
    )
    assert group_order is not None
    edge_values = rng.normal(size=(edge_count, isotope_count)).astype(
        np.float64 if gpu_dtype == "float64" else np.float32
    )
    edge_values_t = torch.as_tensor(edge_values, dtype=dtype, device=device)
    rows_t = torch.as_tensor(incidence_rows, dtype=torch.long, device=device)
    columns_t = torch.as_tensor(incidence_columns, dtype=torch.long, device=device)
    coefficients_t = torch.as_tensor(
        incidence_coefficients,
        dtype=dtype,
        device=device,
    )
    outputs = tuple(
        _torch_stable_incidence_transpose(
            edge_values_t,
            incidence_rows=rows_t,
            incidence_columns=columns_t,
            incidence_coefficients=coefficients_t,
            patch_count=patch_count,
            group_order=group_order,
            group_indices=group_indices,
            group_lengths=group_lengths,
            torch_module=torch,
        )
        for _ in range(30)
    )
    torch.cuda.synchronize()

    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    expected = np.zeros((patch_count, isotope_count), dtype=edge_values.dtype)
    np.add.at(
        expected,
        incidence_columns,
        incidence_coefficients.astype(edge_values.dtype, copy=False)[:, None]
        * edge_values[incidence_rows],
    )
    np.testing.assert_allclose(
        outputs[0].cpu().numpy(),
        expected,
        rtol=2.0e-12 if gpu_dtype == "float64" else 3.0e-5,
        atol=2.0e-13 if gpu_dtype == "float64" else 3.0e-6,
    )


def test_cuda_tv_active_fit_is_bitwise_deterministic() -> None:
    """Repeated CUDA fits with active graph TV must be bitwise identical."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(20260827)
    patch_count = 1024
    edge_count = 16384
    measurement_count = 4
    energy_count = 4
    isotope_count = 3
    spatial = rng.lognormal(
        -4.0,
        1.0,
        (measurement_count, patch_count, isotope_count),
    )
    pulses = rng.uniform(0.1, 1.0, (isotope_count, energy_count))
    pulses /= np.sum(pulses, axis=1, keepdims=True)
    operator = LineFactorizedResponseOperator(
        spatial,
        pulses,
        np.arange(isotope_count, dtype=np.int64),
        isotope_count,
    )
    edges = rng.integers(
        0,
        patch_count,
        size=(edge_count, 2),
        dtype=np.int64,
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    observed = rng.uniform(1.0, 100.0, (measurement_count, energy_count))
    initial = rng.lognormal(1.0, 2.0, (patch_count, isotope_count))
    config = SurfaceMapConfig(
        tv_weight=1.0e6,
        max_iterations=3,
        check_interval=3,
        tolerance=0.0,
        objective_tolerance=0.0,
        kkt_tolerance=0.0,
    )
    cache: dict[str, object] = {}

    results = tuple(
        fit_surface_map_poisson_operator(
            observed,
            operator,
            np.ones(patch_count, dtype=np.float64),
            adjacency_edges=edges,
            initial_densities_cps_1m_m2=initial,
            config=config,
            use_gpu=True,
            gpu_dtype="float64",
            persistent_response_cache=cache,
        )
        for _ in range(6)
    )

    reference = results[0]
    assert reference.iterations == config.max_iterations
    assert reference.tv_penalty > 0.0
    assert np.isfinite(reference.kkt_residual)
    for result in results[1:]:
        np.testing.assert_array_equal(
            result.densities_cps_1m_m2,
            reference.densities_cps_1m_m2,
        )
        np.testing.assert_array_equal(
            result.expected_counts,
            reference.expected_counts,
        )
        assert result.objective == reference.objective
        assert result.objective_history == reference.objective_history
        assert result.kkt_residual == reference.kkt_residual


def test_line_factorized_cuda_cache_appends_and_gathers_rows() -> None:
    """CUDA factor caches must append prefixes and gather bootstrap rows."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    spatial = _line_factorized_density_operator().spatial_factors
    cache: dict[str, object] = {}
    first = _line_factorized_density_operator(
        spatial_factors=spatial[:1],
        diagnostics={
            "device_cache_key": "line-factor-test",
            "measurement_row_keys": ["a"],
        },
    )
    extended = _line_factorized_density_operator(
        spatial_factors=spatial[:2],
        diagnostics={
            "device_cache_key": "line-factor-test",
            "measurement_row_keys": ["a", "b"],
        },
    )
    first_response, first_diagnostics, _, _ = _prepare_dense_torch_response(
        first,
        device=torch.device("cuda"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    extended_response, extended_diagnostics, _, _ = _prepare_dense_torch_response(
        extended,
        device=torch.device("cuda"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    gathered = extended.select_measurements([1, 0])
    gathered_response, gathered_diagnostics, _, _ = _prepare_dense_torch_response(
        gathered,
        device=torch.device("cuda"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )

    assert first_response is not None
    assert extended_response is not None
    assert gathered_response is not None
    assert first_diagnostics["mode"] == "line_factorized_cuda_cache"
    assert extended_diagnostics["mode"] == "line_factorized_cuda_prefix_append"
    assert (
        extended_diagnostics["host_to_device_bytes"]
        < extended_diagnostics["required_bytes"]
    )
    assert gathered_diagnostics["mode"] == "persistent_line_factor_row_gather"
    assert gathered_response.measurement_group_order is None
    source = torch.arange(
        1,
        gathered.source_count + 1,
        dtype=torch.float64,
        device="cuda",
    )
    actual = _torch_response_product(
        gathered,
        source,
        transpose=False,
        torch_module=torch,
        dense_response=gathered_response,
    )
    np.testing.assert_allclose(actual.cpu().numpy(), gathered.matvec(source.cpu()))
    residual = torch.linspace(
        0.1,
        0.9,
        gathered.observation_count,
        dtype=torch.float64,
        device="cuda",
    )
    transpose = _torch_response_product(
        gathered,
        residual,
        transpose=True,
        torch_module=torch,
        dense_response=gathered_response,
    )
    np.testing.assert_allclose(
        transpose.cpu().numpy(),
        gathered.rmatvec(residual.cpu().numpy()),
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    masked = extended.masked_sources(np.asarray([[True, False], [False, True]]))
    masked_response, masked_diagnostics, _, _ = _prepare_dense_torch_response(
        masked,
        device=torch.device("cuda"),
        dtype=torch.float64,
        cache_fraction=0.6,
        torch_module=torch,
        persistent_cache=cache,
    )
    assert masked_response is not None
    assert masked_response.spatial_factors is extended_response.spatial_factors
    assert masked_diagnostics["mode"] == "persistent_line_factor_mask_view"
    masked_source = torch.arange(
        1,
        masked.source_count + 1,
        dtype=torch.float64,
        device="cuda",
    )
    masked_actual = _torch_response_product(
        masked,
        masked_source,
        transpose=False,
        torch_module=torch,
        dense_response=masked_response,
    )
    np.testing.assert_allclose(
        masked_actual.cpu().numpy(),
        masked.matvec(masked_source.cpu()),
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_cuda_response_cache_appends_and_gathers_measurement_rows() -> None:
    """Prefixes, resamples, and solver dtypes must reuse resident CUDA rows."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    areas = np.asarray([1.0, 1.5])
    response = np.asarray(
        [
            [[[1.0], [0.2]], [[0.7], [0.4]]],
            [[[0.3], [1.1]], [[0.8], [0.5]]],
        ],
        dtype=np.float64,
    )
    cache: dict[str, object] = {}
    common = {
        "device_cache_key": "test-response-geometry",
    }
    first = _dense_density_operator(
        response[:1],
        areas,
        isotope_count=1,
        diagnostics={**common, "measurement_row_keys": ["a"]},
    )
    extended = _dense_density_operator(
        response,
        areas,
        isotope_count=1,
        diagnostics={**common, "measurement_row_keys": ["a", "b"]},
    )
    config = SurfaceMapConfig(max_iterations=40, check_interval=10)
    first_observed = np.asarray([[7.0, 5.0]])
    fit_surface_map_poisson_operator(
        first_observed,
        first,
        areas,
        config=config,
        use_gpu=True,
        persistent_response_cache=cache,
    )
    fit_surface_map_poisson_operator(
        np.asarray([[7.0, 5.0], [4.0, 6.0]]),
        extended,
        areas,
        config=config,
        use_gpu=True,
        persistent_response_cache=cache,
    )
    prefix_cache = extended.diagnostics["performance"]["solver"]["response_cache"]

    assert prefix_cache["mode"] == "persistent_cuda_prefix_append"
    assert prefix_cache["persistent_prefix_measurements"] == 1
    assert prefix_cache["host_to_device_bytes"] == response[1:].nbytes

    resampled_indices = np.asarray([1, 0, 1])
    resampled = _dense_density_operator(
        response[resampled_indices],
        areas,
        isotope_count=1,
        diagnostics={
            **common,
            "measurement_row_keys": ["b", "a", "b"],
        },
    )
    gpu = fit_surface_map_poisson_operator(
        np.asarray([[4.0, 6.0], [7.0, 5.0], [4.0, 6.0]]),
        resampled,
        areas,
        config=config,
        use_gpu=True,
        persistent_response_cache=cache,
    )
    cpu = fit_surface_map_poisson_operator(
        np.asarray([[4.0, 6.0], [7.0, 5.0], [4.0, 6.0]]),
        resampled,
        areas,
        config=config,
    )
    gathered_cache = resampled.diagnostics["performance"]["solver_calls"][0][
        "response_cache"
    ]

    assert gathered_cache["mode"] == "persistent_cuda_row_gather"
    assert gathered_cache["host_to_device_bytes"] == 0
    assert gathered_cache["materialized_row_gather_bytes"] == 0
    assert gathered_cache["allocator_cache_cleared_before_reuse"] is True
    np.testing.assert_allclose(
        gpu.densities_cps_1m_m2,
        cpu.densities_cps_1m_m2,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    converted = fit_surface_map_poisson_operator(
        np.asarray([[4.0, 6.0], [7.0, 5.0], [4.0, 6.0]]),
        resampled,
        areas,
        config=config,
        use_gpu=True,
        gpu_dtype="float32",
        persistent_response_cache=cache,
    )
    uncached = fit_surface_map_poisson_operator(
        np.asarray([[4.0, 6.0], [7.0, 5.0], [4.0, 6.0]]),
        resampled,
        areas,
        config=config,
        use_gpu=True,
        gpu_dtype="float32",
        persistent_response_cache={},
    )
    converted_cache = resampled.diagnostics["performance"]["solver_calls"][-2][
        "response_cache"
    ]

    assert converted_cache["mode"] == "persistent_cuda_row_gather"
    assert converted_cache["cross_dtype_cache_reused"] is True
    assert converted_cache["host_to_device_bytes"] == 0
    np.testing.assert_allclose(
        converted.densities_cps_1m_m2,
        uncached.densities_cps_1m_m2,
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_cuda_response_cache_evicts_stale_patch_layouts() -> None:
    """Online refinement caches must stay bounded as patch layouts change."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    areas = np.asarray([1.0, 1.5])
    response = np.asarray(
        [[[[1.0], [0.2]], [[0.7], [0.4]]]],
        dtype=np.float64,
    )
    observed = np.asarray([[7.0, 5.0]])
    cache: dict[str, object] = {}
    config = SurfaceMapConfig(max_iterations=20, check_interval=10)

    operators = []
    for layout_index in range(3):
        operator = _dense_density_operator(
            response,
            areas,
            isotope_count=1,
            diagnostics={
                "device_cache_key": f"patch-layout-{layout_index}",
                "measurement_row_keys": ["a"],
            },
        )
        operators.append(operator)
        fit_surface_map_poisson_operator(
            observed,
            operator,
            areas,
            config=config,
            use_gpu=True,
            persistent_response_cache=cache,
        )

    entries = cache.get("entries")
    assert isinstance(entries, dict)
    assert len(entries) == 2
    keys = {identity[2] for identity in entries}
    assert keys == {"patch-layout-1", "patch-layout-2"}
    diagnostics = operators[-1].diagnostics["performance"]["solver"]["response_cache"]
    assert diagnostics["persistent_cache_entry_limit"] == 2
    assert diagnostics["persistent_cache_evicted_entries"] == 1


def test_calibrated_negative_binomial_operator_fit_is_finite() -> None:
    """Calibrated overdispersion must enter fitting rather than diagnostics only."""
    response = np.asarray(
        [[[1.0], [0.2]], [[0.3], [1.0]], [[0.8], [0.4]], [[0.2], [0.9]]],
        dtype=float,
    )
    areas = np.asarray([1.0, 1.0])
    truth = np.asarray([[20.0], [5.0]])
    observed = np.einsum("mgi,gi->m", response, truth)
    operator = _dense_density_operator(response, areas, isotope_count=1)

    result = fit_surface_map_poisson_operator(
        observed,
        operator,
        areas,
        config=SurfaceMapConfig(
            likelihood_family="negative_binomial",
            overdispersion_alpha=(0.03, 0.03, 0.03, 0.03),
            max_iterations=4000,
            tolerance=1.0e-6,
            objective_tolerance=1.0e-7,
        ),
    )

    assert np.all(np.isfinite(result.densities_cps_1m_m2))
    assert result.densities_cps_1m_m2[:, 0] == pytest.approx(
        truth[:, 0],
        rel=0.08,
        abs=0.5,
    )
    assert result.deviance >= -1.0e-8


def test_group_penalty_shrinks_patchwise_isotope_support() -> None:
    """The optional isotope-group proximal term should remove weak patch groups."""
    response = np.zeros((4, 2, 2), dtype=float)
    response[:, 0, 0] = [1.0, 0.8, 0.1, 0.0]
    response[:, 0, 1] = [0.8, 1.0, 0.0, 0.1]
    response[:, 1, 0] = [0.0, 0.1, 0.8, 1.0]
    response[:, 1, 1] = [0.1, 0.0, 1.0, 0.8]
    observed = np.asarray([40.0, 35.0, 1.0, 1.0], dtype=float)

    result = fit_surface_map_poisson(
        observed,
        response,
        np.ones(2, dtype=float),
        config=SurfaceMapConfig(
            isotope_group_weight=1.0,
            max_iterations=5000,
            tolerance=1.0e-7,
            objective_tolerance=1.0e-8,
        ),
    )

    assert np.linalg.norm(result.densities_cps_1m_m2[0]) > 1.0
    assert np.linalg.norm(result.densities_cps_1m_m2[1]) < 0.1
    assert result.group_penalty >= 0.0
    assert result.objective_history


def test_surface_map_objective_matches_manual_oracle() -> None:
    """The public objective should match a direct Poisson, L1, TV, and nuisance oracle."""
    observed = np.asarray([7.0, 11.0], dtype=float)
    response = np.asarray(
        [
            [[1.0, 0.5], [0.2, 0.1]],
            [[0.1, 0.3], [0.8, 0.4]],
        ],
        dtype=float,
    )
    areas = np.asarray([2.0, 0.5], dtype=float)
    density = np.asarray([[3.0, 1.0], [2.0, 4.0]], dtype=float)
    nuisance_response = np.asarray([[0.2], [0.4]], dtype=float)
    nuisance = np.asarray([2.5], dtype=float)
    background = np.asarray([1.0, 1.5], dtype=float)
    config = SurfaceMapConfig(
        l1_weight=0.3,
        tv_weight=0.7,
        nuisance_l1_weight=0.2,
        nuisance_l2_weight=0.1,
    )

    objective = evaluate_surface_map_objective(
        observed,
        response,
        areas,
        density,
        adjacency_edges=np.asarray([[0, 1]], dtype=int),
        adjacency_weights=np.asarray([1.5], dtype=float),
        background=background,
        nuisance_response=nuisance_response,
        nuisance_coefficients=nuisance,
        config=config,
    )

    expected = (
        background
        + response.reshape(2, -1) @ (density * areas[:, None]).reshape(-1)
        + nuisance_response[:, 0] * nuisance[0]
    )
    poisson_nll = float(np.sum(expected - observed * np.log(expected)))
    l1_penalty = 0.3 * float(np.sum(density * areas[:, None]))
    tv_penalty = 0.7 * 1.5 * float(np.sum(np.abs(density[1] - density[0])))
    nuisance_penalty = 0.2 * nuisance[0] + 0.5 * 0.1 * nuisance[0] ** 2

    assert objective.poisson_nll == pytest.approx(poisson_nll)
    assert objective.l1_penalty == pytest.approx(l1_penalty)
    assert objective.tv_penalty == pytest.approx(tv_penalty)
    assert objective.nuisance_penalty == pytest.approx(nuisance_penalty)
    assert objective.total == pytest.approx(
        poisson_nll + l1_penalty + tv_penalty + nuisance_penalty
    )
