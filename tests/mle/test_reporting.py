from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from measurement.model import EnvironmentConfig
from runtime import AtomicBundlePublisher
from three_d_estimation.config import MLEConfig
from three_d_estimation.reporting import (
    DIAGNOSTICS_FILENAME,
    ESTIMATE_FILENAME,
    HOTSPOT_CLUSTERS_FILENAME,
    load_mle_config_payload,
    load_mle_estimate,
    save_mle_estimate,
)
from three_d_estimation.surface_patches import (
    build_surface_patches,
    refine_surface_patches,
)
from three_d_estimation.types import MLEEstimate


def _estimate(*, include_predictions: bool = True) -> MLEEstimate:
    base = build_surface_patches(
        EnvironmentConfig(size_x=2.0, size_y=2.0, size_z=2.0),
        None,
        spacing=3.0,
        quadrature_points_per_patch=4,
    )
    floor_id = next(
        patch.patch_id for patch in base.patches if patch.surface_kind == "floor"
    )
    patches = refine_surface_patches(
        base,
        [floor_id],
        quadrature_points_per_patch=1,
    )
    patch_count = patches.patch_count
    density = (
        np.arange(1, 2 * patch_count + 1, dtype=float).reshape(2, patch_count)
        / 10.0
    )
    strength = patches.integrated_strengths_cps_1m(density)
    strongest = int(np.argmax(strength[0]))
    hotspot = {
        "isotope": "Cs-137",
        "cluster_id": 0,
        "patch_ids": [patches.patches[strongest].patch_id],
        "centroid_xyz": patches.patches[strongest].centroid_xyz.tolist(),
        "integrated_strength_cps_1m": float(strength[0, strongest]),
    }
    return MLEEstimate(
        isotope_names=("Cs-137", "Co-60"),
        patches=patches.patches,
        density_by_isotope=density,
        patch_strength_by_isotope=strength,
        predicted_spectra=(
            np.arange(12, dtype=float).reshape(3, 4) if include_predictions else None
        ),
        predicted_isotope_counts=(
            np.arange(6, dtype=float).reshape(3, 2) if include_predictions else None
        ),
        background_parameters=np.asarray([0.25, 0.5]),
        nuisance_parameters=np.asarray([-0.1, 0.2]),
        objective_value=12.25,
        poisson_deviance=3.5,
        iterations=37,
        converged=True,
        diagnostics={
            "mode": "spectral",
            "density_unit": "detector_cps_1m_per_m2",
            "objective_history": [20.0, 14.0, 12.25],
            "hotspot_clusters": [hotspot],
            "nested": {"enabled": True, "count": 3},
        },
    )


def _assert_estimates_equal(expected: MLEEstimate, actual: MLEEstimate) -> None:
    assert actual.isotope_names == expected.isotope_names
    np.testing.assert_array_equal(actual.density_by_isotope, expected.density_by_isotope)
    np.testing.assert_array_equal(
        actual.patch_strength_by_isotope,
        expected.patch_strength_by_isotope,
    )
    if expected.predicted_spectra is None:
        assert actual.predicted_spectra is None
    else:
        np.testing.assert_array_equal(actual.predicted_spectra, expected.predicted_spectra)
    if expected.predicted_isotope_counts is None:
        assert actual.predicted_isotope_counts is None
    else:
        np.testing.assert_array_equal(
            actual.predicted_isotope_counts,
            expected.predicted_isotope_counts,
        )
    np.testing.assert_array_equal(
        actual.background_parameters,
        expected.background_parameters,
    )
    np.testing.assert_array_equal(actual.nuisance_parameters, expected.nuisance_parameters)
    assert actual.objective_value == expected.objective_value
    assert actual.poisson_deviance == expected.poisson_deviance
    assert actual.iterations == expected.iterations
    assert actual.converged is expected.converged
    assert actual.diagnostics == expected.diagnostics
    assert len(actual.patches) == len(expected.patches)
    for expected_patch, actual_patch in zip(expected.patches, actual.patches):
        assert actual_patch.patch_id == expected_patch.patch_id
        assert actual_patch.surface_kind == expected_patch.surface_kind
        assert actual_patch.object_id == expected_patch.object_id
        assert actual_patch.area_m2 == expected_patch.area_m2
        np.testing.assert_array_equal(actual_patch.centroid_xyz, expected_patch.centroid_xyz)
        np.testing.assert_array_equal(actual_patch.normal_xyz, expected_patch.normal_xyz)
        np.testing.assert_array_equal(actual_patch.vertices_xyz, expected_patch.vertices_xyz)
        np.testing.assert_array_equal(
            actual_patch.quadrature_points_xyz,
            expected_patch.quadrature_points_xyz,
        )
        np.testing.assert_array_equal(
            actual_patch.quadrature_weights,
            expected_patch.quadrature_weights,
        )
        assert actual_patch.neighbor_patch_ids == expected_patch.neighbor_patch_ids
        assert (
            actual_patch.neighbor_shared_edge_lengths_m
            == expected_patch.neighbor_shared_edge_lengths_m
        )
        assert actual_patch.parent_patch_id == expected_patch.parent_patch_id
        assert actual_patch.refinement_level == expected_patch.refinement_level


def _config() -> MLEConfig:
    return MLEConfig(
        mode="spectral",
        isotope_names=("Cs-137", "Co-60"),
        patch_spacing_m=(2.0, 2.0, 2.0),
        quadrature_order=4,
        l1_weight=0.1,
        tv_weight=0.2,
        random_seed=9,
    )


def test_report_round_trip_preserves_complete_estimate_and_config(tmp_path: Path) -> None:
    estimate = _estimate()
    output = tmp_path / "report"

    paths = save_mle_estimate(estimate, output, _config())

    assert paths.estimate_npz == output / ESTIMATE_FILENAME
    assert paths.diagnostics_json == output / DIAGNOSTICS_FILENAME
    assert paths.hotspot_clusters_json == output / HOTSPOT_CLUSTERS_FILENAME
    assert all(path.is_file() for path in (paths.estimate_npz, paths.diagnostics_json))
    assert paths.hotspot_clusters_json is not None
    assert paths.hotspot_clusters_json.is_file()
    loaded = load_mle_estimate(output)
    _assert_estimates_equal(estimate, loaded)
    _assert_estimates_equal(estimate, load_mle_estimate(paths.estimate_npz))

    config_payload = load_mle_config_payload(output)
    assert config_payload is not None
    assert MLEConfig.from_dict(config_payload) == _config()
    diagnostics_payload = json.loads(paths.diagnostics_json.read_text(encoding="utf-8"))
    assert diagnostics_payload["config"] == _config().to_dict() | {
        "isotope_names": ["Cs-137", "Co-60"],
        "patch_spacing_m": [2.0, 2.0, 2.0],
    }
    hotspot_payload = json.loads(
        paths.hotspot_clusters_json.read_text(encoding="utf-8")
    )
    assert hotspot_payload["hotspot_clusters"] == estimate.diagnostics[
        "hotspot_clusters"
    ]


def test_identical_reports_have_byte_identical_json_and_npz(tmp_path: Path) -> None:
    estimate = _estimate()
    first = save_mle_estimate(estimate, tmp_path / "first", _config())
    # Output-first order is supported for the standalone CLI persistence hook.
    second = save_mle_estimate(tmp_path / "second", estimate, _config())

    assert first.estimate_npz.read_bytes() == second.estimate_npz.read_bytes()
    assert first.diagnostics_json.read_bytes() == second.diagnostics_json.read_bytes()
    assert first.hotspot_clusters_json is not None
    assert second.hotspot_clusters_json is not None
    assert (
        first.hotspot_clusters_json.read_bytes()
        == second.hotspot_clusters_json.read_bytes()
    )
    assert first.diagnostics_json.read_bytes().endswith(b"\n")


def test_existing_report_requires_explicit_overwrite_and_removes_stale_clusters(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report"
    original = _estimate()
    paths = save_mle_estimate(original, output, _config())
    original_npz = paths.estimate_npz.read_bytes()
    unrelated = output / "operator-notes.txt"
    unrelated.write_bytes(b"retain this file exactly\n")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_mle_estimate(original, output, _config())
    assert paths.estimate_npz.read_bytes() == original_npz

    replacement = replace(
        original,
        predicted_spectra=None,
        predicted_isotope_counts=None,
        diagnostics={"mode": "count", "status": "replaced"},
    )
    replacement_paths = save_mle_estimate(
        replacement,
        output,
        {"mode": "count"},
        overwrite=True,
    )
    assert replacement_paths.hotspot_clusters_json is None
    assert not (output / HOTSPOT_CLUSTERS_FILENAME).exists()
    assert unrelated.read_bytes() == b"retain this file exactly\n"
    _assert_estimates_equal(replacement, load_mle_estimate(output))


def test_new_report_refuses_an_existing_unrelated_directory(tmp_path: Path) -> None:
    """Create publication must never merge into a pre-existing directory."""
    output = tmp_path / "report"
    output.mkdir()
    unrelated = output / "operator-notes.txt"
    unrelated.write_bytes(b"do not replace\n")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_mle_estimate(_estimate(), output, _config())

    assert unrelated.read_bytes() == b"do not replace\n"
    assert not (output / ESTIMATE_FILENAME).exists()


def test_failed_overwrite_preserves_the_complete_previous_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staging failure must leave every published report member unchanged."""
    output = tmp_path / "report"
    paths = save_mle_estimate(_estimate(), output, _config())
    original = {
        path.name: path.read_bytes()
        for path in (
            paths.estimate_npz,
            paths.diagnostics_json,
            paths.hotspot_clusters_json,
        )
        if path is not None
    }
    write_bytes = AtomicBundlePublisher.write_bytes
    calls = 0

    def fail_second_write(
        publisher: AtomicBundlePublisher,
        relative_path: str | Path,
        payload: bytes,
    ) -> Path:
        """Fail after one private member write but before generation publication."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected report staging failure")
        return write_bytes(publisher, relative_path, payload)

    monkeypatch.setattr(AtomicBundlePublisher, "write_bytes", fail_second_write)

    with pytest.raises(RuntimeError, match="injected report staging failure"):
        save_mle_estimate(
            replace(_estimate(), diagnostics={"mode": "count"}),
            output,
            {"mode": "count"},
            overwrite=True,
        )

    assert {
        name: (output / name).read_bytes()
        for name in original
    } == original
    assert not tuple(tmp_path.glob(".report.bundle-*"))


def test_optional_predictions_and_hotspot_file_presence_round_trip(tmp_path: Path) -> None:
    estimate = replace(
        _estimate(include_predictions=False),
        diagnostics={"mode": "count", "hotspot_clusters": []},
    )
    paths = save_mle_estimate(estimate, tmp_path / "with-empty-clusters")
    assert paths.hotspot_clusters_json is not None
    assert load_mle_estimate(paths.output_dir).predicted_spectra is None
    assert load_mle_estimate(paths.output_dir).predicted_isotope_counts is None

    no_clusters = replace(estimate, diagnostics={"mode": "count"})
    no_cluster_paths = save_mle_estimate(no_clusters, tmp_path / "without-clusters")
    assert no_cluster_paths.hotspot_clusters_json is None
    assert not (no_cluster_paths.output_dir / HOTSPOT_CLUSTERS_FILENAME).exists()
    _assert_estimates_equal(no_clusters, load_mle_estimate(no_cluster_paths.output_dir))


def test_nonfinite_diagnostics_and_config_are_rejected_before_writing(
    tmp_path: Path,
) -> None:
    bad_diagnostics = replace(_estimate(), diagnostics={"bad": np.nan})
    output = tmp_path / "bad-diagnostics"
    with pytest.raises(ValueError, match="Non-finite JSON value"):
        save_mle_estimate(bad_diagnostics, output)
    assert not output.exists()

    with pytest.raises(ValueError, match="Non-finite JSON value"):
        save_mle_estimate(
            _estimate(),
            tmp_path / "bad-config",
            {"threshold": float("inf")},
        )
    assert not (tmp_path / "bad-config").exists()


def test_diagnostics_are_digest_bound_to_the_npz(tmp_path: Path) -> None:
    paths = save_mle_estimate(_estimate(), tmp_path / "report", _config())
    payload = json.loads(paths.diagnostics_json.read_text(encoding="utf-8"))
    payload["iterations"] += 1
    paths.diagnostics_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        load_mle_estimate(paths.output_dir)
