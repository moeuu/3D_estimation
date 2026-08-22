"""Tests for validated MLE context construction from shared runtime data."""

from pathlib import Path

from runtime.measurement_log import MEASUREMENT_LOG_SCHEMA_VERSION
from three_d_estimation.config import MLEConfig
from three_d_estimation.estimator_context import prepare_estimator_context


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures/shared_measurement_log/measurement_log"


def test_estimator_context_consumes_shared_log_without_local_simulator() -> None:
    """MLE must build its context from the installed shared runtime contract."""
    config = MLEConfig(
        mode="spectral",
        isotope_names=("Co-60", "Cs-137", "Eu-154"),
        patch_spacing_m=(6.0, 6.0, 3.0),
        max_iterations=2,
        debias_refit=False,
        use_gpu=False,
    )

    context = prepare_estimator_context(FIXTURE, config=config)

    assert context.log.schema_version == MEASUREMENT_LOG_SCHEMA_VERSION == 2
    assert context.batch.measurement_count == 12
    assert context.batch.isotope_counts is None
    assert context.batch.spectrum_counts.shape == (12, 851)
    assert not (ROOT / "src/measurement").exists()
    assert not (ROOT / "src/sim").exists()
