"""End-to-end purity and cross-repository contract tests for standalone MLE."""

from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from runtime.measurement_log import load_measurement_log, measurement_log_sha256
from runtime.records import canonical_json_bytes
from three_d_estimation.estimator import (
    SurfaceMLEEstimator,
    _split_fit_indices,
    _union_group_labels,
)
from three_d_estimation.observation_batch import observation_batch_from_records
from three_d_estimation.types import ObservationBatch


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "shared_measurement_log" / "measurement_log"


def _grouping_batch() -> ObservationBatch:
    """Return rows that exercise station, XY-height, and shield-block relations."""
    return ObservationBatch(
        detector_positions_xyz=np.asarray(
            [
                [0.2, 0.3, 0.4],
                [0.2, 0.3, 1.4],
                [0.2, 0.3, 0.8],
                [0.2, 0.3, 1.8],
                [1.2, 1.3, 0.4],
                [1.2, 1.3, 1.4],
            ],
            dtype=float,
        ),
        detector_quaternions_wxyz=np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
            (6, 1),
        ),
        fe_indices=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
        pb_indices=np.asarray([7, 6, 5, 4, 3, 2], dtype=np.int64),
        live_times_s=np.ones(6, dtype=float),
        spectrum_counts=np.ones((6, 2), dtype=float),
        spectrum_variances=None,
        energy_bin_edges_keV=np.asarray([0.0, 500.0, 1000.0], dtype=float),
        isotope_counts=np.ones((6, 1), dtype=float),
        isotope_covariances=np.ones((6, 1, 1), dtype=float),
        station_ids=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        isotope_names=("Cs-137",),
        step_ids=np.arange(6, dtype=np.int64),
        action_ids=np.arange(6, dtype=np.int64),
        travel_times_s=np.zeros(6, dtype=float),
        shield_actuation_times_s=np.zeros(6, dtype=float),
        shield_program_block_ids=("a", "a", "b", "b", "b", "b"),
    )


def test_active_package_has_no_pf_import_or_pf_candidate_api() -> None:
    """Pure MLE source must not import PF modules or accept PF support state."""
    source_root = ROOT / "src" / "three_d_estimation"
    forbidden_import_roots = {"pf", "particle_filter", "planning"}
    for path in sorted(source_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".", 1)[0] for alias in node.names}
                assert imported.isdisjoint(forbidden_import_roots), path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden_import_roots, path

    parameters = inspect.signature(SurfaceMLEEstimator.fit).parameters
    assert "pf_state" not in parameters
    assert "pf_particles" not in parameters
    assert "pf_candidates" not in parameters
    assert "candidate_positions" not in parameters


def test_group_labels_keep_station_views_together_for_every_default_mode() -> None:
    """Related station rows never cross fit/held-out except explicit row mode."""
    batch = _grouping_batch()
    station = _union_group_labels(batch, "station_id", 1.0e-6)
    assert station[0] == station[1]
    assert station[2] == station[3]

    same_xy = _union_group_labels(batch, "same_xy_height", 1.0e-6)
    assert len({same_xy[index] for index in (0, 1, 2, 3)}) == 1
    assert same_xy[4] == same_xy[5]

    shield_block = _union_group_labels(batch, "shield_program_block", 1.0e-6)
    assert shield_block[0] == shield_block[1]
    assert len({shield_block[index] for index in (2, 3, 4, 5)}) == 1

    for grouping in ("station_id", "same_xy_height", "shield_program_block"):
        fit, held_out, labels = _split_fit_indices(
            batch,
            0.4,
            7,
            grouping=grouping,
            xy_tolerance_m=1.0e-6,
        )
        fit_labels = {labels[int(index)] for index in fit}
        held_labels = {labels[int(index)] for index in held_out}
        assert fit_labels.isdisjoint(held_labels)

    row_labels = _union_group_labels(batch, "row", 1.0e-6)
    assert row_labels[0] != row_labels[1]


def test_shared_fixture_preserves_pose_timing_and_shield_blocks() -> None:
    """Live record conversion retains every raw record and timing field."""
    log = load_measurement_log(FIXTURE)
    batch = observation_batch_from_records(
        log.records,
        log.context.isotopes,
        context=log.context,
    )
    assert batch.detector_positions_xyz.shape == (12, 3)
    assert set(batch.detector_positions_xyz[:, 2]) == {0.4}
    np.testing.assert_array_equal(batch.step_ids, np.arange(12))
    np.testing.assert_array_equal(batch.action_ids, np.arange(12))
    assert np.all(batch.travel_times_s >= 0.0)
    assert np.all(batch.shield_actuation_times_s >= 0.0)
    assert batch.shield_program_block_ids == tuple(
        f"station:{station_id}" for station_id in np.repeat(np.arange(6), 2)
    )


def test_measurement_log_digest_covers_full_raw_inventory_and_rejects_truth(
    tmp_path: Path,
) -> None:
    """Hash every regular input including the manifest, never in-log truth."""
    copied = tmp_path / "measurement_log"
    shutil.copytree(FIXTURE, copied)
    inventory = {
        path.relative_to(copied).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(copied.rglob("*"))
        if path.is_file()
    }
    assert "run_manifest.json" in inventory
    expected = sha256(canonical_json_bytes(inventory)).hexdigest()
    assert measurement_log_sha256(copied) == expected

    for index, relative_name in enumerate(
        (
            "evaluation/ground-truth-holdout.json",
            "assets/source-layout-backup.json",
        )
    ):
        candidate = tmp_path / f"forbidden-{index}"
        shutil.copytree(FIXTURE, candidate)
        forbidden = candidate / relative_name
        forbidden.parent.mkdir(parents=True, exist_ok=True)
        forbidden.write_text("{}\n", encoding="utf-8")
        with pytest.raises(
            ValueError,
            match="Truth/source-layout artifacts must be stored outside",
        ):
            measurement_log_sha256(candidate)


def test_forward_model_manifest_hash_or_identifier_mismatch_fails_clearly(
    tmp_path: Path,
) -> None:
    """The runtime loader refuses an inventory with changed model physics."""
    corrupted = tmp_path / "measurement_log"
    shutil.copytree(FIXTURE, corrupted)
    forward_path = corrupted / "forward_model_manifest.json"
    forward = json.loads(forward_path.read_text(encoding="utf-8"))
    forward["model_identifiers"]["detector"]["sha256"] = "0" * 64
    forward_path.write_bytes(canonical_json_bytes(forward))

    run_manifest_path = corrupted / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    digest = sha256(forward_path.read_bytes()).hexdigest()
    run_manifest["forward_model_manifest_sha256"] = digest
    run_manifest["artifact_hashes"]["forward_model_manifest.json"] = digest
    run_manifest["model_identifiers"] = forward["model_identifiers"]
    run_manifest_path.write_bytes(canonical_json_bytes(run_manifest))

    with pytest.raises(ValueError, match="compatibility error for detector"):
        load_measurement_log(corrupted)
