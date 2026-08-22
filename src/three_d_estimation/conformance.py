"""Provider-neutral forward-response conformance generation for pure MLE."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from numpy.typing import NDArray

from measurement.observation_model import (
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from runtime import ForwardConformanceFixture
from runtime.forward_conformance import (
    FORWARD_CONFORMANCE_CASE_ORDER,
    FORWARD_CONFORMANCE_SCHEMA_VERSION,
    FORWARD_CONFORMANCE_UNITS,
)


EXPECTED_CASE_ORDER = FORWARD_CONFORMANCE_CASE_ORDER
EXPECTED_UNITS = FORWARD_CONFORMANCE_UNITS
_RUNTIME_CONFIG = {
    "source_rate_model": "detector_cps_1m",
    "line_resolved_shield_attenuation": True,
}


@dataclass(frozen=True, slots=True)
class ForwardConformanceResult:
    """Store canonical case IDs and unit-strength expected responses."""

    case_ids: NDArray[np.str_]
    unit_response: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate aligned, finite, one-dimensional conformance arrays."""
        case_ids = np.asarray(self.case_ids, dtype=np.str_)
        responses = np.asarray(self.unit_response, dtype=np.float64)
        if case_ids.ndim != 1 or responses.shape != case_ids.shape:
            raise ValueError("case_ids and unit_response must be aligned vectors.")
        if case_ids.size == 0 or any(not str(value) for value in case_ids):
            raise ValueError("case_ids must contain non-empty IDs.")
        if len(set(str(value) for value in case_ids)) != case_ids.size:
            raise ValueError("case_ids must be unique.")
        if np.any(~np.isfinite(responses)) or np.any(responses < 0.0):
            raise ValueError("unit_response must contain finite non-negative values.")
        case_ids = np.array(case_ids, dtype=np.str_, copy=True)
        responses = np.array(responses, dtype=np.float64, copy=True)
        case_ids.setflags(write=False)
        responses.setflags(write=False)
        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "unit_response", responses)


def load_forward_conformance_axes(
    path: str | Path,
) -> ForwardConformanceFixture:
    """Load provider-neutral axes through the shared runtime contract."""
    return ForwardConformanceFixture.from_path(path)


def compute_forward_conformance(
    axes_or_path: ForwardConformanceFixture | Mapping[str, object] | str | Path,
) -> ForwardConformanceResult:
    """Compute every canonical unit-strength case with the local runtime model."""
    fixture = (
        axes_or_path
        if isinstance(axes_or_path, ForwardConformanceFixture)
        else ForwardConformanceFixture.from_payload(axes_or_path)
        if isinstance(axes_or_path, Mapping)
        else ForwardConformanceFixture.from_path(axes_or_path)
    )
    observation_model = build_runtime_observation_model(
        _RUNTIME_CONFIG,
        isotopes=fixture.isotopes,
    )
    kernels = tuple(
        continuous_kernel_from_observation_model(
            observation_model,
            obstacle_grid=fixture.obstacle_grid(obstacle.obstacle_id),
            use_gpu=False,
        )
        for obstacle in fixture.obstacles
    )

    case_ids: list[str] = []
    responses: list[float] = []
    for isotope in fixture.isotopes:
        for pose in fixture.detector_poses:
            detector_position = np.asarray(pose.xyz, dtype=np.float64)
            for fe_index in fixture.fe_orientation_indices:
                for pb_index in fixture.pb_orientation_indices:
                    for source in fixture.source_points:
                        source_position = np.asarray(source.xyz, dtype=np.float64)
                        for obstacle, kernel in zip(
                            fixture.obstacles,
                            kernels,
                            strict=True,
                        ):
                            case_ids.append(
                                f"{isotope}|pose={pose.pose_id}|fe={fe_index:02d}"
                                f"|pb={pb_index:02d}|source={source.source_id}"
                                f"|obstacle={obstacle.obstacle_id}"
                            )
                            responses.append(
                                kernel.expected_counts_pair(
                                    isotope=isotope,
                                    detector_pos=detector_position,
                                    sources=source_position.reshape(1, 3),
                                    strengths=np.ones(1, dtype=np.float64),
                                    fe_index=fe_index,
                                    pb_index=pb_index,
                                    live_time_s=pose.live_time_s,
                                    background=0.0,
                                )
                            )
    return ForwardConformanceResult(
        case_ids=np.asarray(case_ids, dtype=np.str_),
        unit_response=np.asarray(responses, dtype=np.float64),
    )


def _npy_bytes(array: NDArray[np.generic]) -> bytes:
    """Return deterministic, non-pickle NPY bytes."""
    buffer = BytesIO()
    np.lib.format.write_array(
        buffer,
        np.asarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def save_forward_conformance(
    output_path: str | Path,
    result: ForwardConformanceResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save exact case_ids and unit_response NPZ members."""
    if not isinstance(result, ForwardConformanceResult):
        raise TypeError("result must be a ForwardConformanceResult.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Forward conformance output exists: {target}")
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in (
            ("case_ids", result.case_ids),
            ("unit_response", result.unit_response),
        ):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(array))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "EXPECTED_CASE_ORDER",
    "EXPECTED_UNITS",
    "FORWARD_CONFORMANCE_SCHEMA_VERSION",
    "ForwardConformanceResult",
    "compute_forward_conformance",
    "load_forward_conformance_axes",
    "save_forward_conformance",
]
