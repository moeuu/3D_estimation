"""Contract and concurrency tests for streamed response operators."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import BinaryIO

import numpy as np
import pytest

from three_d_estimation import response_operator
from three_d_estimation.response_operator import (
    BlockResponseOperator,
    LineFactorizedResponseOperator,
    ResponseBlock,
    ResponseOperator,
    atomic_save_npy,
    weighted_response_gram,
)


def _operator() -> BlockResponseOperator:
    """Return a small restartable operator with two observation blocks."""
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=np.float64,
    )

    def blocks() -> Iterator[ResponseBlock]:
        """Yield the matrix in two row blocks."""
        for start in (0, 2):
            rows = np.arange(start, start + 2, dtype=np.int64)
            yield ResponseBlock(rows, np.arange(2, dtype=np.int64), matrix[rows])

    return BlockResponseOperator((2, 2), 2, 1, blocks)


def _line_factorized_operator() -> LineFactorizedResponseOperator:
    """Return a small exact response with three lines and two isotopes."""
    spatial_factors = np.asarray(
        [
            [[1.0, 0.5, 0.2], [0.3, 1.2, 0.4], [0.7, 0.1, 0.9]],
            [[0.8, 0.4, 0.6], [1.1, 0.2, 0.5], [0.2, 0.9, 0.3]],
        ],
        dtype=np.float64,
    )
    pulse_shapes = np.asarray(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.0, 0.2, 0.5, 0.3],
            [0.1, 0.3, 0.4, 0.2],
        ],
        dtype=np.float64,
    )
    return LineFactorizedResponseOperator(
        spatial_factors,
        pulse_shapes,
        np.asarray([0, 1, 0], dtype=np.int64),
        2,
        energy_chunk_size=2,
        patch_chunk_size=2,
    )


def _dense_line_response(
    operator: LineFactorizedResponseOperator,
) -> np.ndarray:
    """Expand line factors independently as a deterministic test oracle."""
    measurement_count, energy_count = operator.observation_shape
    dense = np.zeros(
        (
            measurement_count,
            energy_count,
            operator.patch_count,
            operator.isotope_count,
        ),
        dtype=np.float64,
    )
    for line_index, isotope_index in enumerate(operator.line_isotope_indices.tolist()):
        dense[:, :, :, isotope_index] += np.einsum(
            "mg,b->mbg",
            operator.spatial_factors[:, :, line_index],
            operator.pulse_shapes[line_index],
        )
    dense *= operator.source_mask[None, None, :, :]
    return dense


@pytest.mark.parametrize(
    ("indices", "error_type"),
    [
        (np.asarray([0.0]), TypeError),
        (np.asarray([True]), TypeError),
        (np.asarray([[0]]), ValueError),
        (np.asarray([0, 0]), ValueError),
        (np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64), ValueError),
    ],
)
def test_response_block_rejects_ambiguous_indices(
    indices: np.ndarray,
    error_type: type[Exception],
) -> None:
    """Index validation must never reshape, truncate, or wrap input silently."""
    with pytest.raises(error_type):
        ResponseBlock(indices, np.asarray([0]), np.ones((indices.size, 1)))


def test_response_block_owns_immutable_inputs() -> None:
    """A block must not change when arrays owned by its caller are mutated."""
    rows = np.asarray([0, 1], dtype=np.int64)
    values = np.asarray([[1.0], [-1.0e-13]], dtype=np.float64)
    block = ResponseBlock(rows, np.asarray([0]), values)

    rows[0] = 7
    values[:] = 9.0

    np.testing.assert_array_equal(block.observation_indices, [0, 1])
    np.testing.assert_array_equal(block.values, [[1.0], [0.0]])
    assert not block.observation_indices.flags.writeable
    assert not block.source_indices.flags.writeable
    assert not block.values.flags.writeable


@pytest.mark.parametrize(
    ("shape", "patch_count", "isotope_count", "error_type"),
    [
        ((2.5, 2), 2, 1, TypeError),
        ((True, 2), 2, 1, TypeError),
        ((2, 2), 2.0, 1, TypeError),
        ((2, 2), 2, False, TypeError),
        ((2, 0), 2, 1, ValueError),
    ],
)
def test_operator_rejects_lossy_dimensions(
    shape: tuple[object, ...],
    patch_count: object,
    isotope_count: object,
    error_type: type[Exception],
) -> None:
    """Operator dimensions must be exact positive integers."""
    with pytest.raises(error_type):
        BlockResponseOperator(shape, patch_count, isotope_count, iter)


def test_vector_products_require_one_dimensional_inputs() -> None:
    """Matrix products must reject arrays whose shape only flattens correctly."""
    operator = _operator()

    with pytest.raises(ValueError, match="one finite value per source"):
        operator.matvec(np.ones((2, 1)))
    with pytest.raises(ValueError, match="one finite value per observation"):
        operator.rmatvec(np.ones((2, 2)))


def test_selection_and_masking_are_stable_views() -> None:
    """Selected order and source masks must be copied into deterministic views."""
    operator = _operator()
    selected = operator.select_measurements([1, 0])
    mask = np.asarray([True, False])
    masked = operator.masked_sources(mask)
    mask[:] = True

    expected = operator.materialize()
    np.testing.assert_array_equal(selected.materialize(), expected[[1, 0]])
    np.testing.assert_array_equal(
        masked.materialize(),
        expected * np.asarray([True, False]).reshape(1, 1, 2, 1),
    )

    with pytest.raises(TypeError, match="integer indices"):
        operator.select_measurements([0.0])
    with pytest.raises(ValueError, match="duplicate"):
        operator.select_measurements([0, 0])
    with pytest.raises(TypeError, match="boolean"):
        operator.masked_sources([1, 0])


def test_line_factorization_matches_dense_products_and_sums() -> None:
    """Factorized products, sums, and blocks must equal dense line expansion."""
    operator = _line_factorized_operator()
    dense = _dense_line_response(operator)
    matrix = dense.reshape(operator.observation_count, operator.source_count)
    source = np.linspace(0.2, 1.3, operator.source_count)
    observations = np.linspace(0.1, 0.8, operator.observation_count)

    np.testing.assert_allclose(operator.materialize(), dense, rtol=1.0e-14)
    np.testing.assert_allclose(operator.matvec(source), matrix @ source)
    np.testing.assert_allclose(operator.rmatvec(observations), matrix.T @ observations)
    np.testing.assert_allclose(operator.row_sums(), np.sum(matrix, axis=1))
    np.testing.assert_allclose(operator.column_sums(), np.sum(matrix, axis=0))
    np.testing.assert_allclose(
        np.dot(operator.matvec(source), observations),
        np.dot(source, operator.rmatvec(observations)),
    )
    assert operator.factor_storage_bytes < operator.dense_storage_bytes


def test_line_factorization_accumulates_weighted_active_gram_exactly() -> None:
    """Active Gram reduction must match dense weighted source correlations."""
    operator = _line_factorized_operator().select_measurements([1, 0])
    matrix = operator.materialize().reshape(
        operator.observation_count,
        operator.source_count,
    )
    active = np.asarray([0, 3, 5], dtype=np.int64)
    weights = np.linspace(0.1, 1.2, operator.observation_count)
    expected = matrix[:, active].T @ (weights[:, None] * matrix[:, active])

    np.testing.assert_allclose(
        operator.weighted_gram(active, weights),
        expected,
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_weighted_gram_keeps_legacy_structural_operators_compatible() -> None:
    """The optional Gram accelerator must not expand the base protocol."""
    delegate = _operator()
    legacy = SimpleNamespace(
        observation_shape=delegate.observation_shape,
        patch_count=delegate.patch_count,
        isotope_count=delegate.isotope_count,
        observation_count=delegate.observation_count,
        source_count=delegate.source_count,
        iter_blocks=delegate.iter_blocks,
        matvec=delegate.matvec,
        rmatvec=delegate.rmatvec,
        row_sums=delegate.row_sums,
        column_sums=delegate.column_sums,
        response_sums=delegate.response_sums,
        select_measurements=delegate.select_measurements,
        masked_sources=delegate.masked_sources,
    )
    active = np.asarray([0, 1], dtype=np.int64)
    weights = np.linspace(0.25, 1.0, delegate.observation_count)
    matrix = delegate.materialize().reshape(
        delegate.observation_count,
        delegate.source_count,
    )

    assert isinstance(legacy, ResponseOperator)
    np.testing.assert_allclose(
        weighted_response_gram(legacy, active, weights),
        matrix.T @ (weights[:, None] * matrix),
    )


def test_line_factorization_preserves_selection_and_source_masks() -> None:
    """Factorized views must preserve order and combine immutable masks."""
    operator = _line_factorized_operator()
    dense = _dense_line_response(operator)
    selected = operator.select_measurements([1, 0])
    mask = np.asarray(
        [[True, False], [False, True], [True, True]],
        dtype=bool,
    )
    masked = operator.masked_sources(mask)
    mask[:] = True

    assert selected.backing_spatial_factors is operator.backing_spatial_factors
    assert masked.backing_spatial_factors is operator.backing_spatial_factors
    np.testing.assert_allclose(selected.materialize(), dense[[1, 0]])
    np.testing.assert_allclose(
        masked.materialize(),
        dense
        * np.asarray(
            [[True, False], [False, True], [True, True]],
            dtype=bool,
        )[None, None, :, :],
    )
    with pytest.raises(MemoryError, match="above the limit"):
        operator.materialize(maximum_bytes=operator.dense_storage_bytes - 1)


def test_line_factorized_selection_supports_bootstrap_duplicates() -> None:
    """Replacement sampling must preserve duplicated measurement rows."""
    operator = _line_factorized_operator()
    selected = operator.select_measurements([1, 0, 1])
    source = np.arange(1.0, operator.source_count + 1.0)
    expected = operator.matvec(source).reshape(operator.observation_shape)[[1, 0, 1]]

    np.testing.assert_allclose(
        selected.matvec(source).reshape(selected.observation_shape),
        expected,
    )


@pytest.mark.parametrize("maximum_bytes", [True, 64.0])
def test_materialize_rejects_non_integer_memory_limits(
    maximum_bytes: object,
) -> None:
    """A malformed memory guard must not be silently coerced."""
    with pytest.raises(TypeError, match="maximum_bytes must be an integer"):
        _operator().materialize(maximum_bytes=maximum_bytes)


def test_atomic_save_does_not_replace_existing_cache(tmp_path: Path) -> None:
    """Publishing a later result must preserve an already complete cache file."""
    target = tmp_path / "block.npy"
    atomic_save_npy(target, np.asarray([1.0, 2.0]))
    atomic_save_npy(target, np.asarray([9.0, 9.0]))

    np.testing.assert_array_equal(np.load(target, allow_pickle=False), [1.0, 2.0])
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_save_is_safe_for_concurrent_publishers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent writers must publish one whole file and remove all temporaries."""
    writer_count = 8
    target = tmp_path / "shared.npy"
    barrier = Barrier(writer_count)
    original_temporary_file = response_operator.NamedTemporaryFile

    def synchronized_temporary_file(*args: object, **kwargs: object) -> BinaryIO:
        """Hold writers after the existence check so their link calls race."""
        handle = original_temporary_file(*args, **kwargs)
        barrier.wait()
        return handle

    monkeypatch.setattr(
        response_operator,
        "NamedTemporaryFile",
        synchronized_temporary_file,
    )
    candidates = tuple(
        np.full((32, 8), float(index), dtype=np.float64)
        for index in range(writer_count)
    )
    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        futures = [
            executor.submit(atomic_save_npy, target, candidate)
            for candidate in candidates
        ]
        for future in futures:
            future.result()

    published = np.load(target, allow_pickle=False)
    assert any(np.array_equal(published, candidate) for candidate in candidates)
    assert list(tmp_path.iterdir()) == [target]
