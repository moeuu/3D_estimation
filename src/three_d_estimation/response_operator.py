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


def _integer_vector(
    values: ArrayLike,
    *,
    name: str,
    allow_duplicates: bool = False,
) -> NDArray[np.int64]:
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
    if not allow_duplicates and np.unique(result).size != result.size:
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


def _weighted_gram_inputs(
    source_count: int,
    observation_count: int,
    source_indices: ArrayLike,
    observation_weights: ArrayLike | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return validated active columns and non-negative observation weights."""
    selected = _integer_vector(source_indices, name="source_indices")
    if np.any(selected < 0) or np.any(selected >= source_count):
        raise ValueError("source_indices entries must be in range.")
    if observation_weights is None:
        weights = np.ones(observation_count, dtype=np.float64)
    else:
        weights = np.asarray(observation_weights, dtype=np.float64)
        if weights.shape != (observation_count,):
            raise ValueError("observation_weights must match flattened observations.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("observation_weights must be finite and non-negative.")
    return selected, weights


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


@runtime_checkable
class WeightedGramResponseOperator(Protocol):
    """Define the optional exact weighted-Gram acceleration capability."""

    def weighted_gram(
        self,
        source_indices: ArrayLike,
        observation_weights: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return an exact active-column weighted response Gram matrix."""


def _streamed_weighted_gram(
    operator: ResponseOperator,
    selected: NDArray[np.int64],
    weights: NDArray[np.float64],
    *,
    maximum_workspace_bytes: int = 64 * 1024 * 1024,
) -> NDArray[np.float64]:
    """Accumulate a cross-block exact Gram with bounded row workspace."""
    active_count = int(selected.size)
    bytes_per_row = max(1, active_count * np.dtype(np.float64).itemsize)
    row_step = max(1, int(maximum_workspace_bytes) // bytes_per_row)
    lookup = np.full(operator.source_count, -1, dtype=np.int64)
    lookup[selected] = np.arange(active_count, dtype=np.int64)
    gram = np.zeros((active_count, active_count), dtype=np.float64)
    for row_start in range(0, operator.observation_count, row_step):
        row_stop = min(row_start + row_step, operator.observation_count)
        design = np.zeros((row_stop - row_start, active_count), dtype=np.float64)
        for block in operator.iter_blocks():
            local_columns = lookup[block.source_indices]
            keep_columns = local_columns >= 0
            keep_rows = (block.observation_indices >= row_start) & (
                block.observation_indices < row_stop
            )
            if not np.any(keep_columns) or not np.any(keep_rows):
                continue
            local_rows = block.observation_indices[keep_rows] - row_start
            design[np.ix_(local_rows, local_columns[keep_columns])] += block.values[
                np.ix_(keep_rows, keep_columns)
            ]
        design *= np.sqrt(weights[row_start:row_stop])[:, None]
        gram += design.T @ design
    return 0.5 * (gram + gram.T)


def weighted_response_gram(
    operator: ResponseOperator,
    source_indices: ArrayLike,
    observation_weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return an exact Gram while preserving legacy operator compatibility."""
    selected, weights = _weighted_gram_inputs(
        operator.source_count,
        operator.observation_count,
        source_indices,
        observation_weights,
    )
    if isinstance(operator, WeightedGramResponseOperator):
        gram = np.asarray(
            operator.weighted_gram(selected, observation_weights),
            dtype=np.float64,
        )
        if gram.shape != (selected.size, selected.size) or np.any(~np.isfinite(gram)):
            raise ValueError("weighted_gram returned an invalid active Gram matrix.")
        return 0.5 * (gram + gram.T)
    return _streamed_weighted_gram(operator, selected, weights)


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

    def weighted_gram(
        self,
        source_indices: ArrayLike,
        observation_weights: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return an exact active-column Gram matrix across all source blocks."""
        selected, weights = _weighted_gram_inputs(
            self.source_count,
            self.observation_count,
            source_indices,
            observation_weights,
        )
        return _streamed_weighted_gram(self, selected, weights)


class LineFactorizedResponseOperator:
    """Represent an exact line-spectrum response without dense energy expansion.

    ``spatial_factors[m, g, l]`` stores the live-time and area-scaled response
    for one measurement, patch, and gamma line. ``pulse_shapes[l, b]`` maps
    each transported line into detector energy bins. Every line belongs to one
    isotope, so their product is exactly the dense spectral response while the
    resident state scales with line count instead of energy-bin count.
    """

    def __init__(
        self,
        spatial_factors: ArrayLike,
        pulse_shapes: ArrayLike,
        line_isotope_indices: ArrayLike,
        isotope_count: int,
        *,
        energy_chunk_size: int = 128,
        patch_chunk_size: int = 128,
        source_mask: ArrayLike | None = None,
        diagnostics: dict[str, object] | None = None,
        _copy_factors: bool = True,
    ) -> None:
        """Validate and own one exact factorized spectral response."""
        if _copy_factors:
            spatial = np.array(
                spatial_factors,
                dtype=np.float64,
                order="C",
                copy=True,
            )
            pulses = np.array(
                pulse_shapes,
                dtype=np.float64,
                order="C",
                copy=True,
            )
        else:
            spatial = np.ascontiguousarray(spatial_factors, dtype=np.float64)
            pulses = np.ascontiguousarray(pulse_shapes, dtype=np.float64)
        if spatial.ndim != 3 or any(size < 1 for size in spatial.shape):
            raise ValueError(
                "spatial_factors must have non-empty shape "
                "(measurements, patches, lines)."
            )
        if pulses.ndim != 2 or pulses.shape[0] != spatial.shape[2] or not pulses.size:
            raise ValueError(
                "pulse_shapes must have shape (lines, non-empty energy bins)."
            )
        if (
            np.any(~np.isfinite(spatial))
            or np.any(spatial < -1.0e-12)
            or np.any(~np.isfinite(pulses))
            or np.any(pulses < -1.0e-12)
        ):
            raise ValueError("Line response factors must be finite and non-negative.")
        if np.any(spatial < 0.0):
            if not spatial.flags.writeable:
                spatial = spatial.copy(order="C")
            spatial[spatial < 0.0] = 0.0
        if np.any(pulses < 0.0):
            if not pulses.flags.writeable:
                pulses = pulses.copy(order="C")
            pulses[pulses < 0.0] = 0.0
        validated_isotope_count = _integer(
            isotope_count,
            name="isotope_count",
            minimum=1,
        )
        raw_line_isotopes = np.asarray(line_isotope_indices)
        if (
            raw_line_isotopes.shape != (spatial.shape[2],)
            or not np.issubdtype(raw_line_isotopes.dtype, np.integer)
            or np.issubdtype(raw_line_isotopes.dtype, np.bool_)
        ):
            raise TypeError(
                "line_isotope_indices must contain one integer per gamma line."
            )
        line_isotopes = np.array(raw_line_isotopes, dtype=np.int64, copy=True)
        if np.any(line_isotopes < 0) or np.any(
            line_isotopes >= validated_isotope_count
        ):
            raise ValueError("Line isotope indices exceed the isotope dimensions.")
        patch_count = int(spatial.shape[1])
        if source_mask is None:
            mask = np.ones(
                (patch_count, validated_isotope_count),
                dtype=bool,
            )
        else:
            raw_mask = np.asarray(source_mask)
            if not np.issubdtype(raw_mask.dtype, np.bool_):
                raise TypeError("Source mask must contain boolean values.")
            if raw_mask.shape == (patch_count * validated_isotope_count,):
                raw_mask = raw_mask.reshape(patch_count, validated_isotope_count)
            if raw_mask.shape != (patch_count, validated_isotope_count):
                raise ValueError("Source mask must match patches by isotopes.")
            mask = np.array(raw_mask, dtype=bool, copy=True)
        spatial.setflags(write=False)
        pulses.setflags(write=False)
        line_isotopes.setflags(write=False)
        mask.setflags(write=False)
        self._backing_spatial_factors = spatial
        self._measurement_indices: NDArray[np.int64] | None = None
        self.pulse_shapes = pulses
        self.line_isotope_indices = line_isotopes
        self.source_mask = mask
        self.patch_count = patch_count
        self.isotope_count = validated_isotope_count
        self.observation_shape = (int(spatial.shape[0]), int(pulses.shape[1]))
        self.energy_chunk_size = _integer(
            energy_chunk_size,
            name="energy_chunk_size",
            minimum=1,
        )
        self.patch_chunk_size = _integer(
            patch_chunk_size,
            name="patch_chunk_size",
            minimum=1,
        )
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)
        self._row_sums: NDArray[np.float64] | None = None
        self._column_sums: NDArray[np.float64] | None = None

    @property
    def backing_spatial_factors(self) -> NDArray[np.float64]:
        """Return the immutable shared measurement-by-patch-by-line backing."""
        return self._backing_spatial_factors

    @property
    def measurement_indices(self) -> NDArray[np.int64] | None:
        """Return selected backing-row indices, or ``None`` for every row."""
        return self._measurement_indices

    @property
    def spatial_factors(self) -> NDArray[np.float64]:
        """Return factors in this operator's measurement order."""
        if self._measurement_indices is None:
            return self._backing_spatial_factors
        selected = np.ascontiguousarray(
            self._backing_spatial_factors[self._measurement_indices],
            dtype=np.float64,
        )
        selected.setflags(write=False)
        return selected

    @property
    def observation_count(self) -> int:
        """Return the flattened measurement-by-energy size."""
        return prod(self.observation_shape)

    @property
    def source_count(self) -> int:
        """Return the flattened patch-by-isotope size."""
        return self.patch_count * self.isotope_count

    @property
    def factor_storage_bytes(self) -> int:
        """Return bytes needed by the exact factors and source metadata."""
        return int(
            self._backing_spatial_factors.nbytes
            + self.pulse_shapes.nbytes
            + self.line_isotope_indices.nbytes
            + self.source_mask.nbytes
        )

    @property
    def dense_storage_bytes(self) -> int:
        """Return bytes required by the equivalent dense float64 response."""
        return self.observation_count * self.source_count * 8

    def matvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return the exact factorized forward response product."""
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.source_count,) or np.any(~np.isfinite(vector)):
            raise ValueError("matvec values must be one finite value per source.")
        densities = vector.reshape(self.patch_count, self.isotope_count)
        densities = densities * self.source_mask
        density_by_line = densities[:, self.line_isotope_indices]
        amplitudes = np.einsum(
            "mgl,gl->ml",
            self._backing_spatial_factors,
            density_by_line,
            optimize=True,
        )
        if self._measurement_indices is not None:
            amplitudes = amplitudes[self._measurement_indices]
        spectra = amplitudes @ self.pulse_shapes
        return np.asarray(spectra, dtype=np.float64).reshape(-1)

    def rmatvec(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return the exact factorized transpose response product."""
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.observation_count,) or np.any(~np.isfinite(vector)):
            raise ValueError("rmatvec values must be one finite value per observation.")
        residual = vector.reshape(self.observation_shape)
        residual_by_line = residual @ self.pulse_shapes.T
        if self._measurement_indices is not None:
            backing_residual = np.zeros(
                (
                    self._backing_spatial_factors.shape[0],
                    self.line_isotope_indices.size,
                ),
                dtype=np.float64,
            )
            np.add.at(
                backing_residual,
                self._measurement_indices,
                residual_by_line,
            )
            residual_by_line = backing_residual
        gradient_by_line = np.einsum(
            "mgl,ml->gl",
            self._backing_spatial_factors,
            residual_by_line,
            optimize=True,
        )
        gradient = np.zeros(
            (self.patch_count, self.isotope_count),
            dtype=np.float64,
        )
        for isotope_index in range(self.isotope_count):
            selected = self.line_isotope_indices == isotope_index
            if np.any(selected):
                gradient[:, isotope_index] = np.sum(
                    gradient_by_line[:, selected],
                    axis=1,
                    dtype=np.float64,
                )
        gradient *= self.source_mask
        return gradient.reshape(-1)

    def response_sums(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return cached exact row and column sums from factorized products."""
        if self._row_sums is None or self._column_sums is None:
            rows = self.matvec(np.ones(self.source_count, dtype=np.float64))
            columns = self.rmatvec(np.ones(self.observation_count, dtype=np.float64))
            rows.setflags(write=False)
            columns.setflags(write=False)
            self._row_sums = rows
            self._column_sums = columns
        return self._row_sums, self._column_sums

    def row_sums(self) -> NDArray[np.float64]:
        """Return cached exact response row sums."""
        return self.response_sums()[0]

    def column_sums(self) -> NDArray[np.float64]:
        """Return cached exact response column sums."""
        return self.response_sums()[1]

    def weighted_gram(
        self,
        source_indices: ArrayLike,
        observation_weights: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Accumulate an exact active Gram matrix without dense expansion."""
        unit_weights = observation_weights is None
        selected, weights = _weighted_gram_inputs(
            self.source_count,
            self.observation_count,
            source_indices,
            observation_weights,
        )
        patch_indices = selected // self.isotope_count
        isotope_indices = selected % self.isotope_count
        line_membership = isotope_indices[:, None] == self.line_isotope_indices[None, :]
        source_enabled = self.source_mask[patch_indices, isotope_indices]
        measurement_count, energy_count = self.observation_shape
        measurement_indices = (
            np.arange(measurement_count, dtype=np.int64)
            if self._measurement_indices is None
            else self._measurement_indices
        )
        amplitudes = (
            self._backing_spatial_factors[
                measurement_indices[:, None],
                patch_indices[None, :],
            ]
            * line_membership[None, :, :]
            * source_enabled[None, :, None]
        )
        if unit_weights:
            pulse_gram = self.pulse_shapes @ self.pulse_shapes.T
            gram = np.einsum(
                "mkl,lr,mjr->kj",
                amplitudes,
                pulse_gram,
                amplitudes,
                optimize=True,
            )
        else:
            pulse_grams = np.einsum(
                "lb,mb,rb->mlr",
                self.pulse_shapes,
                weights.reshape(measurement_count, energy_count),
                self.pulse_shapes,
                optimize=True,
            )
            gram = np.einsum(
                "mkl,mlr,mjr->kj",
                amplitudes,
                pulse_grams,
                amplitudes,
                optimize=True,
            )
        return 0.5 * (gram + gram.T)

    def iter_blocks(self) -> Iterator[ResponseBlock]:
        """Expand bounded blocks only for generic diagnostics and test oracles."""
        measurement_count, energy_count = self.observation_shape
        for measurement_index in range(measurement_count):
            backing_measurement_index = (
                measurement_index
                if self._measurement_indices is None
                else int(self._measurement_indices[measurement_index])
            )
            for patch_start in range(0, self.patch_count, self.patch_chunk_size):
                patch_stop = min(
                    patch_start + self.patch_chunk_size,
                    self.patch_count,
                )
                selected_spatial = self._backing_spatial_factors[
                    backing_measurement_index,
                    patch_start:patch_stop,
                ]
                selected_mask = self.source_mask[patch_start:patch_stop]
                source_indices = np.arange(
                    patch_start * self.isotope_count,
                    patch_stop * self.isotope_count,
                    dtype=np.int64,
                )
                for energy_start in range(0, energy_count, self.energy_chunk_size):
                    energy_stop = min(
                        energy_start + self.energy_chunk_size,
                        energy_count,
                    )
                    values = np.zeros(
                        (
                            energy_stop - energy_start,
                            patch_stop - patch_start,
                            self.isotope_count,
                        ),
                        dtype=np.float64,
                    )
                    for isotope_index in range(self.isotope_count):
                        selected_lines = self.line_isotope_indices == isotope_index
                        if not np.any(selected_lines):
                            continue
                        values[:, :, isotope_index] = (
                            self.pulse_shapes[
                                selected_lines,
                                energy_start:energy_stop,
                            ].T
                            @ selected_spatial[:, selected_lines].T
                        )
                    values *= selected_mask[None, :, :]
                    observation_indices = measurement_index * energy_count + np.arange(
                        energy_start, energy_stop, dtype=np.int64
                    )
                    yield ResponseBlock(
                        observation_indices=observation_indices,
                        source_indices=source_indices,
                        values=values.reshape(energy_stop - energy_start, -1),
                    )

    def materialize(self, *, maximum_bytes: int | None = None) -> NDArray[np.float64]:
        """Materialize the dense tensor for bounded equivalence diagnostics."""
        if maximum_bytes is not None:
            limit = _integer(maximum_bytes, name="maximum_bytes", minimum=0)
            if self.dense_storage_bytes > limit:
                raise MemoryError(
                    "Materialized response requires "
                    f"{self.dense_storage_bytes} bytes, above the limit."
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
            *self.observation_shape,
            self.patch_count,
            self.isotope_count,
        )

    def _shared_view(
        self,
        *,
        measurement_indices: NDArray[np.int64] | None,
        source_mask: NDArray[np.bool_],
        diagnostics: dict[str, object],
    ) -> LineFactorizedResponseOperator:
        """Return a validated immutable view sharing the large factor arrays."""
        view = object.__new__(LineFactorizedResponseOperator)
        view._backing_spatial_factors = self._backing_spatial_factors
        view._measurement_indices = measurement_indices
        view.pulse_shapes = self.pulse_shapes
        view.line_isotope_indices = self.line_isotope_indices
        view.source_mask = source_mask
        view.patch_count = self.patch_count
        view.isotope_count = self.isotope_count
        measurement_count = (
            self._backing_spatial_factors.shape[0]
            if measurement_indices is None
            else measurement_indices.size
        )
        view.observation_shape = (
            int(measurement_count),
            self.observation_shape[1],
        )
        view.energy_chunk_size = self.energy_chunk_size
        view.patch_chunk_size = self.patch_chunk_size
        view.diagnostics = diagnostics
        view._row_sums = None
        view._column_sums = None
        return view

    def select_measurements(
        self,
        indices: Sequence[int],
    ) -> LineFactorizedResponseOperator:
        """Return a zero-copy factorized view over selected measurement rows."""
        selected = _integer_vector(
            np.asarray(tuple(indices)),
            name="measurement indices",
            allow_duplicates=True,
        )
        if np.any(selected < 0) or np.any(selected >= self.observation_shape[0]):
            raise ValueError("Measurement indices must be in range.")
        current_indices = (
            np.arange(
                self._backing_spatial_factors.shape[0],
                dtype=np.int64,
            )
            if self._measurement_indices is None
            else self._measurement_indices
        )
        backing_indices = np.array(current_indices[selected], dtype=np.int64, copy=True)
        backing_indices.setflags(write=False)
        identity_selection = np.array_equal(
            selected,
            np.arange(self.observation_shape[0], dtype=np.int64),
        )
        view_indices = (
            self._measurement_indices if identity_selection else backing_indices
        )
        return self._shared_view(
            measurement_indices=view_indices,
            source_mask=self.source_mask,
            diagnostics={
                **self.diagnostics,
                "selected_measurements": backing_indices.tolist(),
            },
        )

    def masked_sources(self, mask: ArrayLike) -> LineFactorizedResponseOperator:
        """Return a factor-sharing operator with additional source masking."""
        values = np.asarray(mask)
        if not np.issubdtype(values.dtype, np.bool_):
            raise TypeError("Source mask must contain boolean values.")
        if values.shape == (self.source_count,):
            values = values.reshape(self.patch_count, self.isotope_count)
        if values.shape != (self.patch_count, self.isotope_count):
            raise ValueError("Source mask must match patches by isotopes.")
        combined = np.ascontiguousarray(self.source_mask & values, dtype=bool)
        combined.setflags(write=False)
        return self._shared_view(
            measurement_indices=self._measurement_indices,
            source_mask=combined,
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
    "LineFactorizedResponseOperator",
    "ResponseBlock",
    "ResponseOperator",
    "WeightedGramResponseOperator",
    "atomic_save_npy",
    "weighted_response_gram",
]
