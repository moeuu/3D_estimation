"""Configuration objects for standalone count and spectral surface MLE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MLEConfig:
    """Configure surface patches, response construction, fitting, and reporting."""

    mode: Literal["count", "spectral"] = "count"
    isotope_names: tuple[str, ...] = ("Cs-137", "Co-60", "Eu-154")
    patch_spacing_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quadrature_order: int = 1
    obstacle_height_m: float = 2.0
    l1_weight: float = 0.0
    tv_weight: float = 0.0
    isotope_group_weight: float = 0.0
    nuisance_l1_weight: float = 0.0
    nuisance_l2_weight: float = 0.0
    regularization_selection: Literal["fixed", "grouped_cv"] = "fixed"
    cv_l1_weights: tuple[float, ...] = (0.0,)
    cv_tv_weights: tuple[float, ...] = (0.0,)
    cv_fold_count: int = 5
    cv_grouping: Literal["station_id", "same_xy_height"] = "station_id"
    cv_one_standard_error: bool = True
    tuning_environment_id: str | None = None
    final_holdout_environment_id: str | None = None
    uncertainty_enable: bool = False
    laplace_support_threshold_fraction: float = 1.0e-3
    laplace_max_active_parameters: int = 256
    laplace_ridge: float = 1.0e-8
    station_bootstrap_replicates: int = 0
    bootstrap_batch_size: int = 1
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 173
    bootstrap_refit_mode: Literal["full", "fixed_final_grid"] = "full"
    bootstrap_max_iterations: int | None = None
    bootstrap_gpu_dtype: Literal["inherit", "float32", "float64"] = "inherit"
    fit_background_nuisance: bool = True
    fit_scatter_nuisance: bool = True
    discrepancy_calibration_path: str | None = None
    fit_shield_leakage_nuisance: bool = False
    fit_station_rate_nuisance: bool = False
    fit_low_rank_residual_nuisance: bool = False
    fit_gain_resolution_drift: bool = False
    spectral_likelihood: Literal["poisson", "calibrated_overdispersed"] = "poisson"
    count_likelihood: Literal[
        "poisson",
        "covariance_gaussian",
        "multivariate_student_t",
    ] = "poisson"
    count_student_t_degrees_of_freedom: float = 4.0
    count_covariance_regularization: float = 1.0e-6
    count_covariance_max_condition_number: float = 1.0e10
    max_iterations: int = 4000
    poisson_em_warm_start_iterations: int = 0
    tolerance: float = 1.0e-6
    objective_tolerance: float = 1.0e-7
    kkt_tolerance: float | None = None
    check_interval: int = 20
    step_safety: float = 0.95
    over_relaxation: float = 1.0
    min_mean: float = 1.0e-12
    response_chunk_size: int = 262144
    spectral_response_mode: Literal["materialized", "matrix_free"] = "materialized"
    response_measurement_chunk_size: int = 8
    response_energy_chunk_size: int = 128
    response_patch_chunk_size: int = 128
    response_worker_count: int = 0
    response_cache_dir: str | None = None
    response_device_cache_fraction: float = 0.6
    require_gpu_response_cache: bool = False
    online_fit_scope: Literal["station_complete"] = "station_complete"
    online_patch_spacing_m: tuple[float, float, float] | None = None
    online_coarse_to_fine_levels: int = 0
    use_gpu: bool = False
    gpu_device: str = "cuda"
    gpu_dtype: Literal["float32", "float64"] = "float64"
    continuum_to_peak: float = 2.0
    backscatter_fraction: float = 0.03
    support_threshold_fraction: float = 1.0e-3
    debias_refit: bool = True
    debias_requires_convergence: bool = True
    debias_max_active_parameters: int = 256
    debias_max_pearson_dispersion: float = 10.0
    coarse_to_fine_levels: int = 0
    refinement_fraction: float = 0.1
    refinement_max_patches: int = 64
    response_correlation_threshold: float = 0.995
    cluster_threshold_fraction: float = 0.1
    cluster_min_strength_cps_1m: float = 0.0
    held_out_fraction: float = 0.0
    held_out_grouping: Literal[
        "station_id",
        "same_xy_height",
        "shield_program_block",
        "row",
    ] = "station_id"
    held_out_xy_tolerance_m: float = 1.0e-6
    random_seed: int = 0

    def __post_init__(self) -> None:
        """Validate values that affect physical or optimization semantics."""
        if self.mode not in {"count", "spectral"}:
            raise ValueError("mode must be 'count' or 'spectral'.")
        if self.online_fit_scope != "station_complete":
            raise ValueError("online_fit_scope must be 'station_complete'.")
        if self.spectral_likelihood not in {
            "poisson",
            "calibrated_overdispersed",
        }:
            raise ValueError(
                "spectral_likelihood must be poisson or calibrated_overdispersed."
            )
        if self.count_likelihood not in {
            "poisson",
            "covariance_gaussian",
            "multivariate_student_t",
        }:
            raise ValueError("Unsupported count_likelihood.")
        if self.regularization_selection not in {"fixed", "grouped_cv"}:
            raise ValueError("regularization_selection must be fixed or grouped_cv.")
        if self.cv_grouping not in {"station_id", "same_xy_height"}:
            raise ValueError("cv_grouping must be station_id or same_xy_height.")
        if isinstance(self.cv_fold_count, bool) or int(self.cv_fold_count) < 2:
            raise ValueError("cv_fold_count must be an integer of at least two.")
        cv_l1 = tuple(float(value) for value in self.cv_l1_weights)
        cv_tv = tuple(float(value) for value in self.cv_tv_weights)
        if (
            not cv_l1
            or not cv_tv
            or any(not np.isfinite(value) or value < 0.0 for value in (*cv_l1, *cv_tv))
        ):
            raise ValueError(
                "CV regularization grids must be nonempty and nonnegative."
            )
        if self.tuning_environment_id is not None and (
            not isinstance(self.tuning_environment_id, str)
            or not self.tuning_environment_id.strip()
        ):
            raise ValueError("tuning_environment_id must be null or nonempty.")
        if self.final_holdout_environment_id is not None and (
            not isinstance(self.final_holdout_environment_id, str)
            or not self.final_holdout_environment_id.strip()
        ):
            raise ValueError("final_holdout_environment_id must be null or nonempty.")
        if (
            self.tuning_environment_id is not None
            and self.final_holdout_environment_id is not None
            and self.tuning_environment_id == self.final_holdout_environment_id
        ):
            raise ValueError("Tuning and final holdout environments must differ.")
        if (
            not np.isfinite(self.count_student_t_degrees_of_freedom)
            or float(self.count_student_t_degrees_of_freedom) <= 2.0
        ):
            raise ValueError("count_student_t_degrees_of_freedom must exceed two.")
        if (
            not np.isfinite(self.count_covariance_regularization)
            or float(self.count_covariance_regularization) < 0.0
        ):
            raise ValueError("count_covariance_regularization must be non-negative.")
        if (
            not np.isfinite(self.count_covariance_max_condition_number)
            or float(self.count_covariance_max_condition_number) <= 1.0
        ):
            raise ValueError("count_covariance_max_condition_number must exceed one.")
        if (
            self.spectral_likelihood == "calibrated_overdispersed"
            and self.discrepancy_calibration_path is None
        ):
            raise ValueError(
                "Calibrated overdispersion requires discrepancy_calibration_path."
            )
        calibrated_nuisance_requested = any(
            (
                bool(self.fit_shield_leakage_nuisance),
                bool(self.fit_station_rate_nuisance),
                bool(self.fit_low_rank_residual_nuisance),
                bool(self.fit_gain_resolution_drift),
            )
        )
        if (
            self.mode == "spectral"
            and calibrated_nuisance_requested
            and self.discrepancy_calibration_path is None
        ):
            raise ValueError(
                "Structured spectral nuisance terms require "
                "discrepancy_calibration_path."
            )
        if (
            self.spectral_likelihood == "calibrated_overdispersed"
            and self.spectral_response_mode != "matrix_free"
        ):
            raise ValueError(
                "Calibrated overdispersion requires spectral_response_mode=matrix_free."
            )
        names = tuple(str(value).strip() for value in self.isotope_names)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("isotope_names must contain unique non-empty names.")
        spacing = tuple(float(value) for value in self.patch_spacing_m)
        if len(spacing) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in spacing
        ):
            raise ValueError(
                "patch_spacing_m must contain three finite positive values."
            )
        online_spacing = None
        if self.online_patch_spacing_m is not None:
            online_spacing = tuple(
                float(value) for value in self.online_patch_spacing_m
            )
            if len(online_spacing) != 3 or any(
                not np.isfinite(value) or value <= 0.0 for value in online_spacing
            ):
                raise ValueError(
                    "online_patch_spacing_m must be null or three positive values."
                )
        if int(self.quadrature_order) not in {1, 4}:
            raise ValueError("quadrature_order must be 1 or 4.")
        nonnegative = {
            "obstacle_height_m": self.obstacle_height_m,
            "l1_weight": self.l1_weight,
            "tv_weight": self.tv_weight,
            "isotope_group_weight": self.isotope_group_weight,
            "nuisance_l1_weight": self.nuisance_l1_weight,
            "nuisance_l2_weight": self.nuisance_l2_weight,
            "tolerance": self.tolerance,
            "objective_tolerance": self.objective_tolerance,
            "continuum_to_peak": self.continuum_to_peak,
            "backscatter_fraction": self.backscatter_fraction,
            "support_threshold_fraction": self.support_threshold_fraction,
            "cluster_threshold_fraction": self.cluster_threshold_fraction,
            "cluster_min_strength_cps_1m": self.cluster_min_strength_cps_1m,
            "held_out_fraction": self.held_out_fraction,
            "held_out_xy_tolerance_m": self.held_out_xy_tolerance_m,
            "laplace_support_threshold_fraction": (
                self.laplace_support_threshold_fraction
            ),
            "laplace_ridge": self.laplace_ridge,
            "debias_max_pearson_dispersion": (self.debias_max_pearson_dispersion),
        }
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative.values()):
            raise ValueError(
                "MLE weights, tolerances, and fractions must be finite and non-negative."
            )
        if int(self.max_iterations) < 1 or int(self.check_interval) < 1:
            raise ValueError("Iteration counts must be positive.")
        if (
            isinstance(self.poisson_em_warm_start_iterations, (bool, np.bool_))
            or int(self.poisson_em_warm_start_iterations)
            != self.poisson_em_warm_start_iterations
            or int(self.poisson_em_warm_start_iterations) < 0
        ):
            raise ValueError(
                "poisson_em_warm_start_iterations must be a nonnegative integer."
            )
        if self.kkt_tolerance is not None and (
            not np.isfinite(self.kkt_tolerance)
            or float(self.kkt_tolerance) < 0.0
        ):
            raise ValueError("kkt_tolerance must be null or finite and nonnegative.")
        if int(self.response_chunk_size) < 1:
            raise ValueError("response_chunk_size must be positive.")
        if self.spectral_response_mode not in {"materialized", "matrix_free"}:
            raise ValueError(
                "spectral_response_mode must be materialized or matrix_free."
            )
        if (
            int(self.response_measurement_chunk_size) < 1
            or int(self.response_energy_chunk_size) < 1
            or int(self.response_patch_chunk_size) < 1
        ):
            raise ValueError(
                "Response measurement, energy, and patch chunk sizes must be positive."
            )
        if (
            isinstance(self.response_worker_count, bool)
            or int(self.response_worker_count) < 0
        ):
            raise ValueError("response_worker_count must be a nonnegative integer.")
        if (
            not np.isfinite(self.response_device_cache_fraction)
            or not 0.0 <= float(self.response_device_cache_fraction) < 1.0
        ):
            raise ValueError("response_device_cache_fraction must lie in [0, 1).")
        if self.response_cache_dir is not None and (
            not isinstance(self.response_cache_dir, str)
            or not self.response_cache_dir.strip()
        ):
            raise ValueError("response_cache_dir must be null or a non-empty path.")
        if self.discrepancy_calibration_path is not None and (
            not isinstance(self.discrepancy_calibration_path, str)
            or not self.discrepancy_calibration_path.strip()
        ):
            raise ValueError(
                "discrepancy_calibration_path must be null or a non-empty path."
            )
        if not 0.0 < float(self.step_safety) < 1.0:
            raise ValueError("step_safety must lie strictly between zero and one.")
        if not 0.0 <= float(self.over_relaxation) <= 1.0:
            raise ValueError("over_relaxation must lie between zero and one.")
        if not np.isfinite(self.min_mean) or self.min_mean <= 0.0:
            raise ValueError("min_mean must be finite and positive.")
        if not 0.0 <= float(self.refinement_fraction) <= 1.0:
            raise ValueError("refinement_fraction must lie between zero and one.")
        if not 0.0 <= float(self.response_correlation_threshold) <= 1.0:
            raise ValueError(
                "response_correlation_threshold must lie between zero and one."
            )
        if not 0.0 <= float(self.held_out_fraction) < 1.0:
            raise ValueError("held_out_fraction must lie in [0, 1).")
        if self.held_out_grouping not in {
            "station_id",
            "same_xy_height",
            "shield_program_block",
            "row",
        }:
            raise ValueError(
                "held_out_grouping must be station_id, same_xy_height, "
                "shield_program_block, or row."
            )
        if (
            not np.isfinite(self.held_out_xy_tolerance_m)
            or float(self.held_out_xy_tolerance_m) <= 0.0
        ):
            raise ValueError("held_out_xy_tolerance_m must be finite and positive.")
        if int(self.coarse_to_fine_levels) < 0:
            raise ValueError("coarse_to_fine_levels must be non-negative.")
        if (
            isinstance(self.online_coarse_to_fine_levels, bool)
            or int(self.online_coarse_to_fine_levels) < 0
        ):
            raise ValueError("online_coarse_to_fine_levels must be non-negative.")
        if not isinstance(self.debias_requires_convergence, (bool, np.bool_)):
            raise TypeError("debias_requires_convergence must be boolean.")
        if not isinstance(self.require_gpu_response_cache, (bool, np.bool_)):
            raise TypeError("require_gpu_response_cache must be boolean.")
        if (
            isinstance(self.debias_max_active_parameters, bool)
            or int(self.debias_max_active_parameters) < 1
        ):
            raise ValueError("debias_max_active_parameters must be positive.")
        if (
            not np.isfinite(self.debias_max_pearson_dispersion)
            or float(self.debias_max_pearson_dispersion) <= 0.0
        ):
            raise ValueError(
                "debias_max_pearson_dispersion must be finite and positive."
            )
        if (
            isinstance(self.refinement_max_patches, bool)
            or int(self.refinement_max_patches) < 1
        ):
            raise ValueError("refinement_max_patches must be positive.")
        if (
            isinstance(self.laplace_max_active_parameters, bool)
            or int(self.laplace_max_active_parameters) < 1
        ):
            raise ValueError("laplace_max_active_parameters must be positive.")
        if (
            isinstance(self.station_bootstrap_replicates, bool)
            or int(self.station_bootstrap_replicates) < 0
        ):
            raise ValueError("station_bootstrap_replicates must be nonnegative.")
        if (
            isinstance(self.bootstrap_batch_size, bool)
            or int(self.bootstrap_batch_size) < 1
        ):
            raise ValueError("bootstrap_batch_size must be a positive integer.")
        if self.bootstrap_refit_mode not in {"full", "fixed_final_grid"}:
            raise ValueError(
                "bootstrap_refit_mode must be full or fixed_final_grid."
            )
        if self.bootstrap_max_iterations is not None and (
            isinstance(self.bootstrap_max_iterations, bool)
            or int(self.bootstrap_max_iterations) < 1
        ):
            raise ValueError(
                "bootstrap_max_iterations must be null or a positive integer."
            )
        if self.bootstrap_gpu_dtype not in {"inherit", "float32", "float64"}:
            raise ValueError(
                "bootstrap_gpu_dtype must be inherit, float32, or float64."
            )
        if not 0.0 < float(self.bootstrap_confidence_level) < 1.0:
            raise ValueError("bootstrap_confidence_level must lie in (0, 1).")
        if not 0.0 <= float(self.laplace_support_threshold_fraction) <= 1.0:
            raise ValueError("laplace_support_threshold_fraction must lie in [0, 1].")
        if self.gpu_dtype not in {"float32", "float64"}:
            raise ValueError("gpu_dtype must be float32 or float64.")
        object.__setattr__(self, "isotope_names", names)
        object.__setattr__(self, "patch_spacing_m", spacing)
        object.__setattr__(self, "online_patch_spacing_m", online_spacing)
        object.__setattr__(self, "cv_l1_weights", cv_l1)
        object.__setattr__(self, "cv_tv_weights", cv_tv)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MLEConfig":
        """Build a configuration from a JSON-like mapping."""
        values = dict(payload)
        if "isotopes" in values and "isotope_names" not in values:
            values["isotope_names"] = values.pop("isotopes")
        if "patch_spacing_m" in values:
            raw_spacing = values["patch_spacing_m"]
            if isinstance(raw_spacing, (int, float)):
                raw_spacing = (float(raw_spacing),) * 3
            values["patch_spacing_m"] = tuple(float(value) for value in raw_spacing)
        if values.get("online_patch_spacing_m") is not None:
            raw_online_spacing = values["online_patch_spacing_m"]
            if isinstance(raw_online_spacing, (int, float)):
                raw_online_spacing = (float(raw_online_spacing),) * 3
            values["online_patch_spacing_m"] = tuple(
                float(value) for value in raw_online_spacing
            )
        if "isotope_names" in values:
            values["isotope_names"] = tuple(
                str(value) for value in values["isotope_names"]
            )
        for key in ("cv_l1_weights", "cv_tv_weights"):
            if key in values:
                values[key] = tuple(float(value) for value in values[key])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "MLEConfig":
        """Load a JSON configuration file."""
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("MLE configuration root must be a JSON object.")
        values = dict(payload)
        calibration_path = values.get("discrepancy_calibration_path")
        if isinstance(calibration_path, str) and calibration_path.strip():
            calibration = Path(calibration_path).expanduser()
            if not calibration.is_absolute():
                calibration = (source.parent / calibration).resolve()
            values["discrepancy_calibration_path"] = calibration.as_posix()
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        payload = asdict(self)
        payload["isotope_names"] = list(self.isotope_names)
        payload["patch_spacing_m"] = list(self.patch_spacing_m)
        payload["online_patch_spacing_m"] = (
            None
            if self.online_patch_spacing_m is None
            else list(self.online_patch_spacing_m)
        )
        payload["cv_l1_weights"] = list(self.cv_l1_weights)
        payload["cv_tv_weights"] = list(self.cv_tv_weights)
        return payload

    def save(self, path: str | Path) -> None:
        """Write deterministic JSON configuration output."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def build_default_config(
    *,
    mode: Literal["count", "spectral"] = "count",
    isotopes: Sequence[str] | None = None,
) -> MLEConfig:
    """Return a default configuration for programmatic callers."""
    names = (
        MLEConfig().isotope_names
        if isotopes is None
        else tuple(str(value) for value in isotopes)
    )
    return MLEConfig(mode=mode, isotope_names=names)
