"""Contract and concurrency tests for streamed response operators."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import BinaryIO

import numpy as np
import pytest

from three_d_estimation import response_operator
from three_d_estimation.response_operator import (
    BlockResponseOperator,
    ResponseBlock,
    atomic_save_npy,
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
