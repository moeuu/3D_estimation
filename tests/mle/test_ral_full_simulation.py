"""Tests for strict RA-L runtime acquisition and MLE launch integration."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from three_d_estimation.cli import (
    RAL_MLE_CONFIG,
    RAL_PLANNING_CONFIG,
    RAL_STOP_CONFIG,
    build_argument_parser,
)
from three_d_estimation.config import MLEConfig
from three_d_estimation.ral import (
    _runtime_config_errors,
    _validated_mle_output_path,
    preflight_ral_full_simulation,
)


def _physical_config(registry_digest: str) -> dict[str, object]:
    """Return the minimum standard RA-L runtime configuration for preflight."""
    return {
        "backend": "geant4",
        "engine_mode": "external",
        "isotope_experiment_profile": "ral_eu154",
        "energy_bin_count": 851,
        "energy_min_keV": 0.0,
        "energy_max_keV": 1700.0,
        "bin_width_keV": 2.0,
        "thread_count": 32,
        "primary_sampling_fraction": 1.0,
        "secondary_transport_mode": "full_transport",
        "source_rate_model": "detector_cps_1m",
        "detector_scoring_mode": "incident_gamma_energy",
        "sample_detector_response": True,
        "line_resolved_shield_attenuation": True,
        "accelerated_weighted_transport_enable": False,
        "target_sampled_primaries": None,
        "executable_path": "build/geant4_sidecar",
        "full_spectrum_model_registry_path": "configs/models/registry.json",
        "full_spectrum_model_registry_file_sha256": registry_digest,
    }


def test_ral_preflight_binds_runtime_assets_and_mle_profiles(
    tmp_path: Path,
) -> None:
    """Preflight should prove all authoritative runtime and local config assets."""
    registry = tmp_path / "configs" / "models" / "registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}\n", encoding="utf-8")
    sidecar = tmp_path / "build" / "geant4_sidecar"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("test sidecar\n", encoding="utf-8")
    sidecar.chmod(0o755)
    adaptive_runtime = tmp_path / "src" / "runtime" / "adaptive.py"
    adaptive_runtime.parent.mkdir(parents=True)
    adaptive_runtime.write_text("# test adaptive runtime\n", encoding="utf-8")
    config_path = (
        tmp_path
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(_physical_config(sha256(registry.read_bytes()).hexdigest())),
        encoding="utf-8",
    )

    result = preflight_ral_full_simulation(
        mle_config_path=RAL_MLE_CONFIG,
        planning_config_path=RAL_PLANNING_CONFIG,
        stop_config_path=RAL_STOP_CONFIG,
        runtime_root=tmp_path,
    )

    assert result.ready is True
    assert result.runtime_config_path == config_path
    assert result.geant4_sidecar_path == sidecar
    assert result.to_dict()["control_contract"] == {
        "mode": "mle_closed_loop",
        "precomputed_actions": False,
        "fit_scope": "station_complete",
        "stop_policy": "compound_mle_convergence_with_safety_bound",
    }


def test_ral_runtime_contract_rejects_shortcuts() -> None:
    """Analytic, thinned, or weighted acquisition cannot be called RA-L full."""
    config = _physical_config("0" * 64)
    config.update(
        {
            "backend": "analytic",
            "primary_sampling_fraction": 0.1,
            "weighted_transport": True,
        }
    )

    errors = _runtime_config_errors(config, require_named_profile=True)

    assert any("backend" in error for error in errors)
    assert any("primary_sampling_fraction" in error for error in errors)
    assert any("weighted_transport" in error for error in errors)


def test_ral_mle_output_cannot_mutate_the_measurement_log(
    tmp_path: Path,
) -> None:
    """MLE artifacts must stay outside the runtime's immutable log tree."""
    run_dir = tmp_path / "measurement-log"

    with pytest.raises(ValueError, match="outside the immutable"):
        _validated_mle_output_path(run_dir / "mle", run_dir)

    assert _validated_mle_output_path(tmp_path / "mle", run_dir) == (tmp_path / "mle")


def test_ral_full_simulation_cli_supports_scenario_and_existing_log() -> None:
    """One command should cover live closed-loop acquisition and replay."""
    parser = build_argument_parser()
    preflight = parser.parse_args(["ral-full-simulation", "--preflight-only", "--json"])
    adaptive = parser.parse_args(
        [
            "ral-full-simulation",
            "--scenario",
            "/private/ral-scenario.json",
            "--private-scene-profile",
            "ral-cs4-co3-eu0",
            "--output-dir",
            "/tmp/ral-mle",
        ]
    )
    replay = parser.parse_args(
        [
            "ral-full-simulation",
            "--run-dir",
            "/runtime/measurement-log",
            "--output-dir",
            "/tmp/ral-mle",
            "--final-only",
        ]
    )

    assert preflight.preflight_only is True
    assert adaptive.scenario == Path("/private/ral-scenario.json")
    assert adaptive.private_scene_profile == "ral-cs4-co3-eu0"
    assert not hasattr(adaptive, "plan")
    assert adaptive.max_measurements == 256
    assert adaptive.minimum_information_gain_nats is None
    assert adaptive.low_information_patience is None
    assert replay.run_dir == Path("/runtime/measurement-log")
    assert replay.final_only is True


def test_mle_config_rejects_invalid_online_and_laplace_controls() -> None:
    """Online refinement levels and Laplace support fractions are strictly typed."""
    with pytest.raises(ValueError, match="online_fit_scope"):
        MLEConfig(online_fit_scope="observation")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="online_coarse_to_fine_levels"):
        MLEConfig(online_coarse_to_fine_levels=True)
    with pytest.raises(ValueError, match="laplace_support_threshold_fraction"):
        MLEConfig(laplace_support_threshold_fraction=1.1)
    with pytest.raises(ValueError, match="bootstrap_batch_size"):
        MLEConfig(bootstrap_batch_size=True)
    with pytest.raises(ValueError, match="debias_max_active_parameters"):
        MLEConfig(debias_max_active_parameters=0)
    with pytest.raises(TypeError, match="debias_requires_convergence"):
        MLEConfig(debias_requires_convergence=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refinement_max_patches"):
        MLEConfig(refinement_max_patches=0)
    with pytest.raises(ValueError, match="debias_max_pearson_dispersion"):
        MLEConfig(debias_max_pearson_dispersion=0.0)
    with pytest.raises(ValueError, match="Structured spectral nuisance"):
        MLEConfig(mode="spectral", fit_station_rate_nuisance=True)


def test_mle_config_resolves_calibration_relative_to_config_file(
    tmp_path: Path,
) -> None:
    """A calibration asset path must not depend on the caller's working directory."""
    config_path = tmp_path / "config" / "mle.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps({"discrepancy_calibration_path": "assets/calibration.json"}),
        encoding="utf-8",
    )

    config = MLEConfig.load(config_path)

    assert (
        config.discrepancy_calibration_path
        == (config_path.parent / "assets" / "calibration.json").resolve().as_posix()
    )
