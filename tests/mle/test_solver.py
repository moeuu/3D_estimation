"""Tests for PF-independent regularized Poisson surface-map reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from three_d_estimation.solver import (
    _prepare_dense_torch_response,
    SurfaceMapConfig,
    evaluate_surface_map_objective,
    fit_surface_map_poisson,
    fit_surface_map_poisson_operator,
)
from three_d_estimation.response_operator import BlockResponseOperator, ResponseBlock


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
    config = SurfaceMapConfig(max_iterations=1500, tolerance=1.0e-8)

    cpu = fit_surface_map_poisson_operator(
        observed,
        operator,
        areas,
        config=config,
        gpu_dtype="float64",
    )
    gpu = fit_surface_map_poisson_operator(
        observed,
        operator,
        areas,
        config=config,
        use_gpu=True,
        gpu_device="cuda",
        gpu_dtype="float64",
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


def test_cuda_response_cache_appends_and_gathers_measurement_rows() -> None:
    """Online prefixes and bootstrap resamples must reuse resident float64 rows."""
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
    diagnostics = operators[-1].diagnostics["performance"]["solver"][
        "response_cache"
    ]
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
