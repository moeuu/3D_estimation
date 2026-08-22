"""Strict RA-L full-simulation launch and replay integration."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from runtime.assets import simulation_runtime_root, standard_geant4_config_path
from runtime.defaults import (
    DEFAULT_CUI_SPLIT_VIEW_HOST,
    DEFAULT_CUI_SPLIT_VIEW_PORT,
)
from runtime.measurement_log import MeasurementLog, load_measurement_log
from sim.runtime import load_runtime_config

from .config import MLEConfig
from .information_planner import MLEPlanningConfig
from .online import run_online_replay
from .replay import run_replay
from .reporting import save_mle_estimate

RAL_ISOTOPES = ("Co-60", "Cs-137", "Eu-154")


@dataclass(frozen=True, slots=True)
class RALPreflightResult:
    """Describe whether the authoritative shared runtime is RA-L ready."""

    runtime_root: Path
    runtime_config_path: Path
    geant4_sidecar_path: Path
    mle_config_path: Path
    planning_config_path: Path
    stop_config_path: Path
    errors: tuple[str, ...]

    @property
    def ready(self) -> bool:
        """Return whether every runtime and MLE requirement is satisfied."""
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """Return strict JSON preflight data."""
        return {
            "schema_version": 1,
            "profile": "ral_mix9_surface_mle_v1",
            "ready": self.ready,
            "runtime_root": self.runtime_root.as_posix(),
            "runtime_config_path": self.runtime_config_path.as_posix(),
            "geant4_sidecar_path": self.geant4_sidecar_path.as_posix(),
            "mle_config_path": self.mle_config_path.as_posix(),
            "planning_config_path": self.planning_config_path.as_posix(),
            "stop_config_path": self.stop_config_path.as_posix(),
            "errors": list(self.errors),
            "physical_contract": {
                "backend": "geant4",
                "engine_mode": "external",
                "isotope_experiment_profile": "ral_eu154",
                "isotopes": list(RAL_ISOTOPES),
                "energy_bin_count": 851,
                "energy_range_keV": [0.0, 1700.0],
                "thread_count": 32,
                "primary_sampling_fraction": 1.0,
                "transport_history_mode": "full_unit_weight",
            },
            "control_contract": {
                "mode": "mle_closed_loop",
                "precomputed_actions": False,
                "fit_scope": "station_complete",
                "stop_policy": "compound_mle_convergence_with_safety_bound",
            },
        }


@dataclass(frozen=True, slots=True)
class RALFullSimulationResult:
    """Identify one validated RA-L log and its completed MLE report."""

    measurement_log_path: Path
    mle_output_dir: Path
    run_id: str
    record_count: int
    execution_mode: str
    dashboard_url: str | None

    def to_dict(self) -> dict[str, object]:
        """Return strict JSON pipeline result data."""
        return {
            "schema_version": 1,
            "status": "complete",
            "profile": "ral_mix9_surface_mle_v1",
            "measurement_log_path": self.measurement_log_path.as_posix(),
            "mle_output_dir": self.mle_output_dir.as_posix(),
            "run_id": self.run_id,
            "record_count": self.record_count,
            "execution_mode": self.execution_mode,
            "dashboard_url": self.dashboard_url,
        }


def _runtime_config_errors(
    config: Mapping[str, object],
    *,
    require_named_profile: bool,
) -> list[str]:
    """Return violations of the standard RA-L physical acquisition contract."""
    expected = {
        "backend": "geant4",
        "engine_mode": "external",
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
    }
    errors = [
        f"runtime field {name} must equal {value!r}; got {config.get(name)!r}"
        for name, value in expected.items()
        if config.get(name) != value
    ]
    false_fields = (
        "accelerated_weighted_transport_enable",
        "history_thinning_enabled",
        "theory_tvl_attenuation",
        "weighted_transport",
    )
    for name in false_fields:
        if config.get(name, False) is not False:
            errors.append(f"runtime field {name} must be false")
    if config.get("target_sampled_primaries") not in (None, 0):
        errors.append("runtime field target_sampled_primaries must be null or zero")
    if require_named_profile and config.get("isotope_experiment_profile") != (
        "ral_eu154"
    ):
        errors.append("runtime isotope_experiment_profile must be 'ral_eu154'")
    return errors


def preflight_ral_full_simulation(
    *,
    mle_config_path: str | Path,
    planning_config_path: str | Path,
    stop_config_path: str | Path,
    runtime_root: str | Path | None = None,
    runtime_config_path: str | Path | None = None,
) -> RALPreflightResult:
    """Verify runtime assets and MLE configs without starting Geant4."""
    root = (
        simulation_runtime_root()
        if runtime_root is None
        else Path(runtime_root).expanduser().resolve()
    )
    if runtime_config_path is None:
        physical_path = (
            standard_geant4_config_path()
            if runtime_root is None
            else root
            / "configs"
            / "geant4"
            / "variance_reduction_external_no_isaac_32threads.json"
        )
    else:
        physical_path = Path(runtime_config_path).expanduser().resolve()
    mle_path = Path(mle_config_path).expanduser().resolve()
    planning_path = Path(planning_config_path).expanduser().resolve()
    stop_path = Path(stop_config_path).expanduser().resolve()
    errors: list[str] = []
    config: dict[str, object] = {}
    adaptive_runtime = root / "src" / "runtime" / "adaptive.py"
    if not adaptive_runtime.is_file():
        errors.append(
            f"shared runtime lacks adaptive-session support: {adaptive_runtime}"
        )
    if not physical_path.is_file():
        errors.append(f"shared runtime config is missing: {physical_path}")
    else:
        try:
            config = load_runtime_config(physical_path)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"shared runtime config is invalid: {exc}")
        else:
            errors.extend(_runtime_config_errors(config, require_named_profile=True))
    executable = Path(str(config.get("executable_path", "build/geant4_sidecar")))
    if not executable.is_absolute():
        executable = (root / executable).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        errors.append(f"Geant4 sidecar is not built and executable: {executable}")
    registry_value = config.get("full_spectrum_model_registry_path")
    if isinstance(registry_value, str) and registry_value:
        registry = Path(registry_value)
        if not registry.is_absolute():
            registry = (root / registry).resolve()
        if not registry.is_file():
            errors.append(f"full-spectrum model registry is missing: {registry}")
        else:
            expected_digest = config.get("full_spectrum_model_registry_file_sha256")
            observed_digest = sha256(registry.read_bytes()).hexdigest()
            if expected_digest != observed_digest:
                errors.append("full-spectrum model registry SHA-256 is incompatible")
    else:
        errors.append("shared runtime config lacks a model registry path")
    try:
        mle_config = MLEConfig.load(mle_path)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"RAL MLE config is invalid: {exc}")
    else:
        if mle_config.mode != "spectral":
            errors.append("RAL MLE config must use spectral mode")
        if tuple(mle_config.isotope_names) != RAL_ISOTOPES:
            errors.append(f"RAL MLE isotope order must equal {RAL_ISOTOPES!r}")
        if mle_config.spectral_response_mode != "matrix_free":
            errors.append("RAL MLE config must use matrix_free spectral response")
        if mle_config.online_fit_scope != "station_complete":
            errors.append("RAL MLE config must fit only at station completion")
        if mle_config.online_patch_spacing_m is None:
            errors.append("RAL MLE config must define a coarse online patch spacing")
        if not mle_config.uncertainty_enable:
            errors.append("RAL MLE config must enable final uncertainty")
        if not mle_config.use_gpu:
            errors.append("RAL MLE config must enable the production GPU path")
        if mle_config.discrepancy_calibration_path is not None:
            calibration_path = Path(mle_config.discrepancy_calibration_path)
            if not calibration_path.is_absolute():
                calibration_path = (mle_path.parent / calibration_path).resolve()
            if not calibration_path.is_file():
                errors.append(
                    f"RAL discrepancy calibration is missing: {calibration_path}"
                )
    try:
        planning_config = MLEPlanningConfig.load(planning_path)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"RAL MLE planning config is invalid: {exc}")
    else:
        if int(planning_config.shield_program_length) < 1:
            errors.append("RAL shield_program_length must be positive")
    try:
        from .closed_loop import MLEStopConfig

        MLEStopConfig.load(stop_path)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"RAL MLE stop config is invalid: {exc}")
    return RALPreflightResult(
        runtime_root=root,
        runtime_config_path=physical_path,
        geant4_sidecar_path=executable,
        mle_config_path=mle_path,
        planning_config_path=planning_path,
        stop_config_path=stop_path,
        errors=tuple(errors),
    )


def validate_ral_measurement_log(run_dir: str | Path) -> MeasurementLog:
    """Load and strictly validate one completed RA-L full-simulation log."""
    log = load_measurement_log(Path(run_dir).expanduser().resolve())
    errors = _runtime_config_errors(
        log.context.runtime_config,
        require_named_profile=False,
    )
    if tuple(log.context.isotopes) != RAL_ISOTOPES:
        errors.append(f"MeasurementLog isotopes must equal {RAL_ISOTOPES!r}")
    if not log.records:
        errors.append("RAL closed-loop acquisition requires at least one record")
    for record_index, record in enumerate(log.records):
        expected_complete = record_index + 1 == len(log.records) or (
            log.records[record_index + 1].station_id != record.station_id
        )
        if (record.metadata.get("station_complete") is True) != expected_complete:
            errors.append(f"step {record.step_id} station_complete boundary is invalid")
        metadata = record.metadata
        required_metadata = {
            "engine_mode": "external",
            "primary_sampling_fraction": 1,
            "history_thinning_enabled": False,
            "secondary_transport_mode": "full_transport",
            "weighted_transport": False,
            "theory_tvl_attenuation": False,
            "transport_history_mode": "full_unit_weight",
        }
        for name, expected in required_metadata.items():
            if metadata.get(name) != expected:
                errors.append(
                    f"step {record.step_id} metadata {name} must equal {expected!r}"
                )
    if errors:
        joined = "\n- ".join(errors[:32])
        suffix = "" if len(errors) <= 32 else f"\n- ... {len(errors) - 32} more"
        raise ValueError(
            f"Not a completed RA-L full-simulation log:\n- {joined}{suffix}"
        )
    return log


def _validated_mle_output_path(
    output_dir: str | Path,
    measurement_log_path: str | Path,
) -> Path:
    """Resolve an MLE target and keep it outside the immutable runtime log."""
    target = Path(output_dir).expanduser().resolve()
    log_path = Path(measurement_log_path).expanduser().resolve()
    if target == log_path or log_path in target.parents:
        raise ValueError(
            "MLE output must be outside the immutable MeasurementLog directory."
        )
    return target


def run_ral_full_simulation(
    run_dir: str | Path,
    *,
    mle_config_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    final_only: bool = False,
    enable_dashboard: bool = True,
    serve_dashboard: bool = True,
    dashboard_host: str = DEFAULT_CUI_SPLIT_VIEW_HOST,
    dashboard_port: int = DEFAULT_CUI_SPLIT_VIEW_PORT,
    dashboard_public_host: str | None = None,
    dashboard_url_hook: Callable[[str], None] | None = None,
) -> RALFullSimulationResult:
    """Validate a runtime RA-L log and execute its authoritative MLE replay."""
    log = validate_ral_measurement_log(run_dir)
    target = _validated_mle_output_path(output_dir, log.path)
    config_path = Path(mle_config_path).expanduser().resolve()
    if final_only:
        replay = run_replay(log.path, config=config_path)
        save_mle_estimate(
            target,
            replay.estimate,
            config=replay.context.config,
            overwrite=overwrite,
        )
        execution_mode = "final_cold_spectral_mle"
        dashboard_url = None
    else:
        replay = run_online_replay(
            log.path,
            config=config_path,
            output_dir=target,
            overwrite=overwrite,
            enable_dashboard=enable_dashboard,
            serve_dashboard=serve_dashboard,
            dashboard_host=dashboard_host,
            dashboard_port=dashboard_port,
            dashboard_public_host=dashboard_public_host,
            dashboard_url_hook=dashboard_url_hook,
        )
        execution_mode = "online_station_complete_spectral_mle"
        dashboard_url = replay.dashboard_url
    return RALFullSimulationResult(
        measurement_log_path=log.path.resolve(),
        mle_output_dir=target,
        run_id=log.run_id,
        record_count=len(log.records),
        execution_mode=execution_mode,
        dashboard_url=dashboard_url,
    )


__all__ = [
    "RALFullSimulationResult",
    "RALPreflightResult",
    "preflight_ral_full_simulation",
    "run_ral_full_simulation",
    "validate_ral_measurement_log",
]
