"""Tests for shared-runtime array views at the MLE observation boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogArrayView,
    MeasurementLogView,
    load_measurement_log,
)

from three_d_estimation.observation_batch import (
    observation_batch_from_log,
    observation_batch_from_records,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "shared_measurement_log" / "measurement_log"


def test_log_conversion_delegates_canonical_arrays_to_runtime(
    monkeypatch: Any,
) -> None:
    """Replay conversion should obtain every core array from MeasurementLog."""
    log = load_measurement_log(FIXTURE)
    original = MeasurementLog.array_view
    calls: list[MeasurementLog] = []

    def capture(instance: MeasurementLog) -> MeasurementLogArrayView:
        """Capture the log used for canonical array packing."""
        calls.append(instance)
        return original(instance)

    monkeypatch.setattr(MeasurementLog, "array_view", capture)

    batch = observation_batch_from_log(log)

    assert calls == [log]
    arrays = original(log)
    np.testing.assert_array_equal(batch.step_ids, arrays.step_id)
    np.testing.assert_array_equal(batch.action_ids, arrays.action_id)
    np.testing.assert_array_equal(batch.station_ids, arrays.station_id)
    np.testing.assert_array_equal(batch.spectrum_counts, arrays.spectrum_counts)


def test_live_records_use_a_truth_free_transient_runtime_view(
    monkeypatch: Any,
) -> None:
    """Live history should reuse runtime packing with its actual RunContext."""
    source = load_measurement_log(FIXTURE)
    original = MeasurementLogView.array_view
    calls: list[MeasurementLogView] = []

    def capture(instance: MeasurementLogView) -> MeasurementLogArrayView:
        """Capture the pathless runtime view used for live history packing."""
        calls.append(instance)
        return original(instance)

    monkeypatch.setattr(MeasurementLogView, "array_view", capture)

    batch = observation_batch_from_records(
        source.records[:2],
        source.context.isotopes,
        context=source.context,
    )

    assert len(calls) == 1
    assert calls[0].source_log_sha256 is None
    assert calls[0].context.run_id == source.run_id
    np.testing.assert_array_equal(batch.step_ids, [0, 1])
    np.testing.assert_array_equal(batch.station_ids, [0, 0])
