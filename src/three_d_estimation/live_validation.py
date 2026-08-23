"""Strict generic live-acquisition preflight and publication validation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

from runtime.assets import simulation_runtime_root
from runtime.experiment_profiles import (
    DEFAULT_EXPERIMENT_PROFILE_ID,
    AcquisitionContract,
    experiment_profile_from_environment,
    require_experiment_profile,
)
from runtime.measurement_log import MeasurementLog, load_measurement_log
from sim.runtime import load_runtime_config

from .config import MLEConfig
from .information_planner import MLEPlanningConfig

@dataclass(frozen=True, slots=True)
class LivePreflightResult:
    """Describe whether the authoritative shared runtime is live-MLE ready."""

    runtime_root: Path
    runtime_config_path: Path
    geant4_sidecar_path: Path
    mle_config_path: Path
    planning_config_path: Path
    stop_config_path: Path
    experiment_profile_id: str
    acquisition_contract: AcquisitionContract
    candidate_isotopes: tuple[str, ...]
    isotope_experiment_profile: str
    errors: tuple[str, ...]

    @property
    def ready(self) -> bool:
        """Return whether every runtime and MLE requirement is satisfied."""
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """Return strict JSON preflight data."""
        return {
            "schema_version": 1,
            "profile": "live_surface_mle_v1",
            "experiment_profile_id": self.experiment_profile_id,
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
                "isotope_experiment_profile": self.isotope_experiment_profile,
                "isotopes": list(self.candidate_isotopes),
                "energy_bin_count": 851,
                "energy_range_keV": [0.0, 1700.0],
                "thread_count": 32,
                "primary_sampling_fraction": 1.0,
                "transport_history_mode": "full_unit_weight",
            },
            "acquisition_contract": self.acquisition_contract.to_payload(),
            "control_contract": {
                "mode": "mle_closed_loop",
                "precomputed_actions": False,
                "fit_scope": "station_complete",
                "stop_policy": "compound_mle_convergence_with_safety_bound",
            },
        }


def _full_fidelity_runtime_config_errors(
    config: Mapping[str, object],
    *,
    isotope_experiment_profile: str | None,
) -> list[str]:
    """Return violations of the full-fidelity physical acquisition contract."""
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
    if (
        isotope_experiment_profile is not None
        and config.get("isotope_experiment_profile") != isotope_experiment_profile
    ):
        errors.append(
            "runtime isotope_experiment_profile must equal "
            f"{isotope_experiment_profile!r}"
        )
    return errors


def load_live_mle_config(
    path: str | Path,
    isotope_names: tuple[str, ...],
) -> MLEConfig:
    """Load estimator settings while binding isotope axes from the runtime."""
    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Live MLE configuration root must be an object.")
    if "isotope_names" in payload or "isotopes" in payload:
        raise ValueError(
            "Live MLE config must not pin isotopes; the runtime profile owns them."
        )
    return MLEConfig.from_dict({**dict(payload), "isotope_names": isotope_names})


def load_live_planning_config(
    path: str | Path,
    acquisition_contract: AcquisitionContract,
) -> MLEPlanningConfig:
    """Load planning policy while binding runtime-owned station measurements."""
    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Live MLE planning configuration root must be an object.")
    duplicated = sorted(
        field for field in ("live_time_s", "shield_program_length") if field in payload
    )
    if duplicated:
        raise ValueError(
            "Live MLE planning config duplicates runtime acquisition fields: "
            + ", ".join(duplicated)
        )
    loaded = MLEPlanningConfig.from_dict(payload)
    return replace(
        loaded,
        live_time_s=acquisition_contract.live_time_s,
        shield_program_length=acquisition_contract.views_per_station,
    )


def preflight_live_simulation(
    *,
    mle_config_path: str | Path,
    planning_config_path: str | Path,
    stop_config_path: str | Path,
    runtime_root: str | Path | None = None,
    runtime_config_path: str | Path | None = None,
    experiment_profile_id: str = DEFAULT_EXPERIMENT_PROFILE_ID,
) -> LivePreflightResult:
    """Verify runtime assets and MLE configs without starting Geant4."""
    experiment = require_experiment_profile(experiment_profile_id)
    root = (
        simulation_runtime_root()
        if runtime_root is None
        else Path(runtime_root).expanduser().resolve()
    )
    if runtime_config_path is None:
        physical_path = (root / experiment.runtime_config_relative_path).resolve()
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
            errors.extend(
                _full_fidelity_runtime_config_errors(
                    config,
                    isotope_experiment_profile=(
                        experiment.isotope_experiment_profile
                    ),
                )
            )
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
        mle_config = load_live_mle_config(
            mle_path,
            experiment.candidate_isotopes,
        )
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Live MLE config is invalid: {exc}")
    else:
        if mle_config.mode != "spectral":
            errors.append("Live MLE config must use spectral mode")
        if mle_config.spectral_response_mode != "matrix_free":
            errors.append("Live MLE config must use matrix_free spectral response")
        if mle_config.online_fit_scope != "station_complete":
            errors.append("Live MLE config must fit only at station completion")
        if mle_config.online_patch_spacing_m is None:
            errors.append("Live MLE config must define a coarse online patch spacing")
        if not mle_config.uncertainty_enable:
            errors.append("Live MLE config must enable final uncertainty")
        if not mle_config.use_gpu:
            errors.append("Live MLE config must enable the production GPU path")
        if mle_config.discrepancy_calibration_path is not None:
            calibration_path = Path(mle_config.discrepancy_calibration_path)
            if not calibration_path.is_absolute():
                calibration_path = (mle_path.parent / calibration_path).resolve()
            if not calibration_path.is_file():
                errors.append(
                    f"Live discrepancy calibration is missing: {calibration_path}"
                )
    try:
        load_live_planning_config(planning_path, experiment.acquisition)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Live MLE planning config is invalid: {exc}")
    try:
        from .closed_loop import MLEStopConfig

        MLEStopConfig.load(stop_path)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Live MLE stop config is invalid: {exc}")
    return LivePreflightResult(
        runtime_root=root,
        runtime_config_path=physical_path,
        geant4_sidecar_path=executable,
        mle_config_path=mle_path,
        planning_config_path=planning_path,
        stop_config_path=stop_path,
        experiment_profile_id=experiment.profile_id,
        acquisition_contract=experiment.acquisition,
        candidate_isotopes=experiment.candidate_isotopes,
        isotope_experiment_profile=experiment.isotope_experiment_profile,
        errors=tuple(errors),
    )


def validate_live_measurement_log(run_dir: str | Path) -> MeasurementLog:
    """Load and strictly validate one completed live full-simulation log."""
    log = load_measurement_log(Path(run_dir).expanduser().resolve())
    experiment = experiment_profile_from_environment(log.context.environment)
    acquisition = experiment.acquisition
    errors = _full_fidelity_runtime_config_errors(
        log.context.runtime_config,
        isotope_experiment_profile=experiment.isotope_experiment_profile,
    )
    if tuple(log.context.isotopes) != tuple(experiment.candidate_isotopes):
        errors.append("MeasurementLog isotopes differ from its experiment profile")
    if not log.records:
        errors.append("Live closed-loop acquisition requires at least one record")
    if len(log.records) > acquisition.max_measurements:
        errors.append("MeasurementLog exceeds runtime max_measurements")
    if len({record.station_id for record in log.records}) > acquisition.max_stations:
        errors.append("MeasurementLog exceeds runtime max_stations")
    for record_index, record in enumerate(log.records):
        if abs(float(record.live_time_s) - acquisition.live_time_s) > 1.0e-9:
            errors.append(
                f"step {record.step_id} live_time_s differs from runtime contract"
            )
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
            f"Not a completed live full-simulation log:\n- {joined}{suffix}"
        )
    return log


__all__ = [
    "LivePreflightResult",
    "load_live_mle_config",
    "load_live_planning_config",
    "preflight_live_simulation",
    "validate_live_measurement_log",
]
