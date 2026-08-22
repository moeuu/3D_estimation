"""Tests for live shared-runtime array views at the MLE observation boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from runtime.measurement_log import (
    MeasurementLogArrayView,
    MeasurementLogView,
    load_measurement_log,
)

from three_d_estimation.observation_batch import observation_batch_from_records


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "shared_measurement_log" / "measurement_log"


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
