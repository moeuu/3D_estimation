"""Streaming linear operators for memory-bounded surface-MLE responses."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _integer_vector(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    """Return an owned, unique, non-empty one-dimensional index vector."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise TypeError(f"{name} must contain integer indices.")
    if np.issubdtype(raw.dtype, np.unsignedinteger) and np.any(
        raw > np.iinfo(np.int64).max
    ):
        raise ValueError(f"{name} entries exceed the supported integer range.")
    result = np.array(raw, dtype=np.int64, copy=True)
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} must not contain duplicate indices.")
    result.setflags(write=False)
    return result


def _integer(value: object, *, name: str, minimum: int) -> int:
    """Return an exact built-in integer no smaller than ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


@dataclass(frozen=True, slots=True)
class ResponseBlock:
    """Store one observations-by-source response block and global indices."""

    observation_indices: NDArray[np.int64]
    source_indices: NDArray[np.int64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate an immutable finite non-negative block."""
        rows = _integer_vector(
            self.observation_indices,
            name="observation_indices",
        )
        columns = _integer_vector(self.source_indices, name="source_indices")
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (rows.size, columns.size):
            raise ValueError(
                "ResponseBlock values must align with row and column indices."
            )
        if np.any(rows < 0) or np.any(columns < 0):
            raise ValueError("ResponseBlock indices must be non-negative.")
        if np.any(~np.isfinite(values)) or np.any(values < -1.0e-12):
            raise ValueError("ResponseBlock values must be finite and non-negative.")
        values = np.maximum(
            np.array(values, dtype=np.float64, order="C", copy=True),
            0.0,
        )
        values.setflags(write=False)
        object.__setattr__(self, "observation_indices", rows)
        object.__setattr__(self, "source_indices", columns)
        object.__setattr__(self, "values", values)


@runtime_checkable
class ResponseOperator(Protocol):
    """Define the matrix-free response operations consumed by the solver."""

    observation_shape: tuple[int, ...]
    patch_count: int
    isotope_count: int

    @property
    def observation_count(self) -> int:
        """Return the flattened observation count."""

    @property
    def source_count(self) -> int:
        """Return the flattened patch-isotope count."""

    def iter_blocks(self) -> Iterator[ResponseBlock]:
        """Yield deterministic non-overlapping response blocks."""

    def matvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return ``A @ values`` without materializing ``A``."""

    def rmatvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return ``A.T @ values`` without materializing ``A``."""

    def row_sums(self) -> NDArray[np.float64]:
        """Return non-negative absolute row sums."""

    def column_sums(self) -> NDArray[np.float64]:
        """Return non-negative absolute column sums."""

    def response_sums(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return row and column sums from one shared traversal."""

    def select_measurements(self, indices: Sequence[int]) -> ResponseOperator:
        """Return a view containing complete selected measurement rows."""

    def masked_sources(self, mask: ArrayLike) -> ResponseOperator:
        """Return an operator with excluded source columns set to zero."""


class BlockResponseOperator:
    """Implement a response operator from a restartable block iterator."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        patch_count: int,
        isotope_count: int,
        block_factory: Callable[[], Iterator[ResponseBlock]],
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        """Store validated dimensions and a deterministic block factory."""
        shape_values = tuple(observation_shape)
        if not shape_values:
            raise ValueError("observation_shape must contain positive dimensions.")
        shape = tuple(
            _integer(value, name=f"observation_shape[{index}]", minimum=1)
            for index, value in enumerate(shape_values)
        )
        validated_patch_count = _integer(
            patch_count,
            name="patch_count",
            minimum=1,
        )
        validated_isotope_count = _integer(
            isotope_count,
            name="isotope_count",
            minimum=1,
        )
        if not callable(block_factory):
            raise TypeError("block_factory must be callable.")
        self.observation_shape = shape
        self.patch_count = validated_patch_count
        self.isotope_count = validated_isotope_count
        self._block_factory = block_factory
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)
        self._row_sums: NDArray[np.float64] | None = None
        self._column_sums: NDArray[np.float64] | None = None

    @property
    def observation_count(self) -> int:
        """Return the flattened observation count."""
        return prod(self.observation_shape)

    @property
    def source_count(self) -> int:
        """Return the flattened patch-isotope count."""
        return self.patch_count * self.isotope_count

    def iter_blocks(self) -> Iterator[ResponseBlock]:
        """Yield validated blocks inside the declared global dimensions."""
        for block in self._block_factory():
            if not isinstance(block, ResponseBlock):
                raise TypeError("block_factory must yield ResponseBlock instances.")
            if np.any(block.observation_indices >= self.observation_count) or np.any(
                block.source_indices >= self.source_count
            ):
                raise ValueError("ResponseBlock index exceeds operator dimensions.")
            yield block

    def matvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return a streamed forward product."""
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.source_count,) or np.any(~np.isfinite(vector)):
            raise ValueError("matvec values must be one finite value per source.")
        result = np.zeros(self.observation_count, dtype=np.float64)
        for block in self.iter_blocks():
            result[block.observation_indices] += (
                block.values @ vector[block.source_indices]
            )
        return result

    def rmatvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return a streamed transpose product."""
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.observation_count,) or np.any(~np.isfinite(vector)):
            raise ValueError("rmatvec values must be one finite value per observation.")
        result = np.zeros(self.source_count, dtype=np.float64)
        for block in self.iter_blocks():
            result[block.source_indices] += (
                block.values.T @ vector[block.observation_indices]
            )
        return result

    def row_sums(self) -> NDArray[np.float64]:
        """Return cached streamed absolute row sums."""
        self.response_sums()
        assert self._row_sums is not None
        return self._row_sums

    def column_sums(self) -> NDArray[np.float64]:
        """Return cached streamed absolute column sums."""
        self.response_sums()
        assert self._column_sums is not None
        return self._column_sums

    def response_sums(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return cached row and column sums using at most one traversal."""
        if self._row_sums is None or self._column_sums is None:
            row_values = np.zeros(self.observation_count, dtype=np.float64)
            column_values = np.zeros(self.source_count, dtype=np.float64)
            for block in self.iter_blocks():
                row_values[block.observation_indices] += np.sum(block.values, axis=1)
                column_values[block.source_indices] += np.sum(block.values, axis=0)
            row_values.setflags(write=False)
            column_values.setflags(write=False)
            self._row_sums = row_values
            self._column_sums = column_values
        return self._row_sums, self._column_sums

    def materialize(self, *, maximum_bytes: int | None = None) -> NDArray[np.float64]:
        """Materialize the operator for tests and bounded diagnostics only."""
        required = self.observation_count * self.source_count * 8
        if maximum_bytes is not None:
            limit = _integer(maximum_bytes, name="maximum_bytes", minimum=0)
            if required > limit:
                raise MemoryError(
                    f"Materialized response requires {required} bytes, above the limit."
                )
        matrix = np.zeros(
            (self.observation_count, self.source_count),
            dtype=np.float64,
        )
        for block in self.iter_blocks():
            matrix[np.ix_(block.observation_indices, block.source_indices)] += (
                block.values
            )
        return matrix.reshape(
            *self.observation_shape, self.patch_count, self.isotope_count
        )

    def select_measurements(self, indices: Sequence[int]) -> BlockResponseOperator:
        """Return a compact operator over complete selected measurement rows."""
        if len(self.observation_shape) < 2:
            raise ValueError(
                "Measurement selection requires measurement-first responses."
            )
        selected = _integer_vector(
            np.asarray(tuple(indices)),
            name="measurement indices",
        )
        measurement_count = self.observation_shape[0]
        if np.any(selected < 0) or np.any(selected >= measurement_count):
            raise ValueError("Measurement indices must be unique and in range.")
        trailing = prod(self.observation_shape[1:])
        global_rows = np.concatenate(
            [np.arange(index * trailing, (index + 1) * trailing) for index in selected]
        ).astype(np.int64, copy=False)
        inverse = np.full(self.observation_count, -1, dtype=np.int64)
        inverse[global_rows] = np.arange(global_rows.size, dtype=np.int64)

        def factory() -> Iterator[ResponseBlock]:
            """Yield only selected observation rows with compact indices."""
            for block in self.iter_blocks():
                keep = inverse[block.observation_indices] >= 0
                if not np.any(keep):
                    continue
                yield ResponseBlock(
                    observation_indices=inverse[block.observation_indices[keep]],
                    source_indices=block.source_indices,
                    values=block.values[keep],
                )

        return BlockResponseOperator(
            (selected.size, *self.observation_shape[1:]),
            self.patch_count,
            self.isotope_count,
            factory,
            diagnostics={
                **self.diagnostics,
                "selected_measurements": selected.tolist(),
            },
        )

    def masked_sources(self, mask: ArrayLike) -> BlockResponseOperator:
        """Return an operator whose excluded columns are exactly zero."""
        values = np.asarray(mask)
        if not np.issubdtype(values.dtype, np.bool_):
            raise TypeError("Source mask must contain boolean values.")
        if values.shape == (self.patch_count, self.isotope_count):
            vector = values.reshape(-1)
        elif values.shape == (self.source_count,):
            vector = values
        else:
            raise ValueError("Source mask must match patches by isotopes.")
        vector = np.array(vector, dtype=bool, copy=True)
        vector.setflags(write=False)

        def factory() -> Iterator[ResponseBlock]:
            """Yield source-masked response blocks."""
            for block in self.iter_blocks():
                selected = vector[block.source_indices]
                if not np.any(selected):
                    continue
                yield ResponseBlock(
                    observation_indices=block.observation_indices,
                    source_indices=block.source_indices[selected],
                    values=block.values[:, selected],
                )

        return BlockResponseOperator(
            self.observation_shape,
            self.patch_count,
            self.isotope_count,
            factory,
            diagnostics={**self.diagnostics, "source_masked": True},
        )


def atomic_save_npy(path: str | Path, values: ArrayLike) -> None:
    """Atomically publish one durable NumPy cache block without replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.save(
                handle,
                np.asarray(values, dtype=np.float64),
                allow_pickle=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "BlockResponseOperator",
    "ResponseBlock",
    "ResponseOperator",
    "atomic_save_npy",
]
