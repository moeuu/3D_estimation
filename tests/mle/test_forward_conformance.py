"""Forward-response conformance tests for the MLE provider."""

import json
from pathlib import Path

import numpy as np
from runtime import ForwardConformanceFixture

from three_d_estimation.conformance import (
    compute_forward_conformance,
    load_forward_conformance_axes,
    save_forward_conformance,
)


ROOT = Path(__file__).resolve().parents[2]
AXES = ROOT / "fixtures/forward_response_conformance.json"


def test_mle_forward_conformance_is_complete_and_deterministic(tmp_path) -> None:
    """Exercise every declared MLE forward-response conformance case."""
    axes = load_forward_conformance_axes(AXES)
    assert isinstance(axes, ForwardConformanceFixture)
    result = compute_forward_conformance(axes)

    assert result.case_ids.shape == (3 * 3 * 8 * 8 * 4 * 2,)
    assert result.unit_response.shape == result.case_ids.shape
    assert np.all(np.isfinite(result.unit_response))
    assert np.all(result.unit_response >= 0.0)
    first = save_forward_conformance(tmp_path / "first.npz", result)
    second = save_forward_conformance(tmp_path / "second.npz", result)
    assert first.read_bytes() == second.read_bytes()


def test_mle_conformance_uses_fixture_orientation_subsets() -> None:
    """The MLE evaluator must preserve declared Fe/Pb orientation subsets."""
    payload = json.loads(AXES.read_text(encoding="utf-8"))
    payload["shield_program"]["fe_orientation_indices"] = [1, 3]
    payload["shield_program"]["pb_orientation_indices"] = [2]

    result = compute_forward_conformance(payload)

    assert result.case_ids.shape == (3 * 3 * 2 * 1 * 4 * 2,)
    assert all("|fe=01|" in value or "|fe=03|" in value for value in result.case_ids)
    assert all("|pb=02|" in value for value in result.case_ids)
