"""Command-line interface for standalone surface MLE replay and reporting."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from runtime.measurement_log import load_measurement_log

from .closed_loop import (
    MLEStopConfig,
    RAL_PRIVATE_SCENE_PROFILES,
    run_ral_closed_loop,
)
from .config import MLEConfig
from .conformance import compute_forward_conformance, save_forward_conformance
from .future_scoring import (
    save_future_candidate_scores,
    score_future_count_candidates,
)
from .holdout import run_ral_holdout
from .information_planner import (
    MLEPlanningConfig,
    plan_next_measurement,
    save_mle_planning_result,
)
from .observation_batch import subset_observation_batch
from .online import run_online_replay
from .ral import (
    preflight_ral_full_simulation,
    run_ral_full_simulation,
)
from .replay import prepare_replay, run_replay
from .reporting import load_mle_estimate, save_mle_estimate

ROOT = Path(__file__).resolve().parents[2]
RAL_MLE_CONFIG = ROOT / "configs" / "mle" / "ral_full_spectral.json"
RAL_PLANNING_CONFIG = ROOT / "configs" / "mle" / "ral_full_planning.json"
RAL_STOP_CONFIG = ROOT / "configs" / "mle" / "ral_full_stop.json"


def _print_cui_dashboard_url(url: str, *, json_output: bool) -> None:
    """Print one immediately flushable CUI URL without corrupting JSON stdout."""
    stream = sys.stderr if json_output else sys.stdout
    print(f"CUI dashboard URL: {url}", file=stream, flush=True)


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by count and spectral replay commands."""
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Versioned measurement-log directory.",
    )
    parser.add_argument(
        "--mle-config", type=Path, default=None, help="MLE JSON configuration file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory (default: RUN_DIR/mle_count or mle_spectral).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing result directory."
    )
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        "--gpu", action="store_true", help="Use the local CUDA kernel when available."
    )
    device.add_argument(
        "--cpu", action="store_true", help="Force CPU response construction."
    )
    parser.add_argument(
        "--no-debias",
        action="store_true",
        help="Disable the support-selected unregularized refit.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the fit summary as JSON."
    )
    parser.add_argument(
        "--initial-estimate",
        type=Path,
        default=None,
        help="Prior station-complete MLE report used only as a warm initialization.",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level replay/report command parser."""
    parser = argparse.ArgumentParser(
        prog="estimate-radiation-mle",
        description="Standalone rotating-shield surface maximum-likelihood estimation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay_parser = subparsers.add_parser(
        "replay",
        help="Fit count-domain MLE from response_poisson isotope counts.",
    )
    _add_fit_arguments(replay_parser)
    spectral_parser = subparsers.add_parser(
        "fit-spectrum",
        help="Fit line-resolved MLE directly from raw spectra.",
    )
    _add_fit_arguments(spectral_parser)
    online_parser = subparsers.add_parser(
        "online-replay",
        aliases=["online"],
        help=(
            "Causally replay a runtime log and publish an all-history MLE "
            "at every station boundary."
        ),
    )
    online_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Finalized shared-runtime MeasurementLog v2 directory.",
    )
    online_parser.add_argument(
        "--mle-config",
        type=Path,
        default=None,
        help="Spectral MLE JSON configuration file.",
    )
    online_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Online report directory (default: RUN_DIR/mle_online).",
    )
    online_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing online report directory.",
    )
    online_device = online_parser.add_mutually_exclusive_group()
    online_device.add_argument(
        "--gpu",
        action="store_true",
        help="Use the shared runtime CUDA kernel when available.",
    )
    online_device.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU response construction.",
    )
    online_parser.add_argument(
        "--no-debias",
        action="store_true",
        help="Disable the support-selected unregularized refit.",
    )
    online_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final online summary as JSON.",
    )
    online_parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not write the self-refreshing browser dashboard.",
    )
    online_parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Write dashboard files without starting the URL server.",
    )
    online_parser.add_argument(
        "--dashboard-host",
        default="0.0.0.0",
        help="Dashboard server bind host (default: 0.0.0.0).",
    )
    online_parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8878,
        help="Dashboard server TCP port (default: 8878).",
    )
    online_parser.add_argument(
        "--dashboard-public-host",
        default=None,
        help="Browser-visible host printed in the dashboard URL.",
    )
    ral_parser = subparsers.add_parser(
        "ral-full-simulation",
        help=(
            "Run a private RA-L scenario through a live MLE-controlled shared "
            "runtime session."
        ),
    )
    ral_source = ral_parser.add_mutually_exclusive_group()
    ral_source.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help=(
            "Private runtime scenario authored explicitly for this run, containing "
            "truth/environment/config/output but no acquisition actions. The MLE "
            "does not discover or generate this file."
        ),
    )
    ral_parser.add_argument(
        "--private-scene-profile",
        choices=RAL_PRIVATE_SCENE_PROFILES,
        default="ral-mix9",
        help=(
            "Runtime-private source-cardinality contract used only for live "
            "scenario validation."
        ),
    )
    ral_parser.add_argument(
        "--resume-stage",
        type=Path,
        default=None,
        help=(
            "Resume live adaptive acquisition after the last verified completed "
            "station in a shared-runtime stream stage."
        ),
    )
    ral_parser.add_argument(
        "--resume-compatibility",
        type=Path,
        default=None,
        help=(
            "Runtime compatibility provenance required for a cross-commit "
            "adaptive resume."
        ),
    )
    ral_source.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing completed RA-L MeasurementLog to validate and replay.",
    )
    ral_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="MLE output outside the immutable MeasurementLog directory.",
    )
    ral_parser.add_argument(
        "--mle-config",
        type=Path,
        default=RAL_MLE_CONFIG,
        help="RAL spectral-MLE configuration.",
    )
    ral_parser.add_argument(
        "--planning-config",
        type=Path,
        default=RAL_PLANNING_CONFIG,
        help="RAL MLE planning profile checked during preflight.",
    )
    ral_parser.add_argument(
        "--stop-config",
        type=Path,
        default=RAL_STOP_CONFIG,
        help="Compound MLE convergence and coverage stop configuration.",
    )
    ral_parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="Optional shared-runtime checkout override.",
    )
    ral_parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify Geant4/runtime/MLE readiness without starting acquisition.",
    )
    ral_parser.add_argument(
        "--final-only",
        action="store_true",
        help="For --run-dir only, run one cold final MLE instead of causal replay.",
    )
    ral_parser.add_argument(
        "--max-measurements",
        type=int,
        default=256,
        help="Emergency safety bound; MLE information convergence normally stops first.",
    )
    ral_parser.add_argument(
        "--minimum-information-gain-nats",
        type=float,
        default=None,
        help="Optional override of the compound stop profile's EIG threshold.",
    )
    ral_parser.add_argument(
        "--low-information-patience",
        type=int,
        default=None,
        help="Optional override of the compound stop profile's patience window.",
    )
    ral_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing MLE output directory.",
    )
    ral_parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the MLE browser dashboard.",
    )
    ral_parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Write dashboard files without starting its URL server.",
    )
    ral_parser.add_argument(
        "--dashboard-host",
        default="0.0.0.0",
        help="Dashboard bind host (default: 0.0.0.0).",
    )
    ral_parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8878,
        help="Dashboard TCP port (default: 8878).",
    )
    ral_parser.add_argument(
        "--dashboard-public-host",
        default=None,
        help="Browser-visible dashboard host.",
    )
    ral_parser.add_argument(
        "--json",
        action="store_true",
        help="Print preflight or completed pipeline data as JSON.",
    )
    planning_parser = subparsers.add_parser(
        "plan-next",
        help=(
            "Select a runtime-supplied detector pose and Fe/Pb program by "
            "MLE Fisher information."
        ),
    )
    planning_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Shared-runtime MeasurementLog containing the estimate history.",
    )
    planning_parser.add_argument(
        "--estimate",
        type=Path,
        required=True,
        help="Station-complete or final MLE report directory.",
    )
    planning_parser.add_argument(
        "--mle-config",
        type=Path,
        required=True,
        help="Exact spectral MLE configuration used for the estimate.",
    )
    planning_parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Truth-free runtime candidate-pose JSON.",
    )
    planning_parser.add_argument(
        "--planning-config",
        type=Path,
        default=None,
        help="Optional MLE Fisher-planning JSON configuration.",
    )
    planning_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination next-action JSON artifact.",
    )
    planning_device = planning_parser.add_mutually_exclusive_group()
    planning_device.add_argument(
        "--gpu",
        action="store_true",
        help="Use the shared runtime CUDA kernel when available.",
    )
    planning_device.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU response construction.",
    )
    planning_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing next-action JSON artifact.",
    )
    planning_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete planning result as JSON.",
    )
    report_parser = subparsers.add_parser(
        "report",
        help="Read a saved MLE estimate and print its summary.",
    )
    report_parser.add_argument(
        "--estimate",
        type=Path,
        required=True,
        help="Result directory or mle_estimate.npz.",
    )
    report_parser.add_argument("--json", action="store_true", help="Print JSON output.")
    conformance_parser = subparsers.add_parser(
        "forward-conformance",
        help="Generate canonical unit-strength forward-response cases.",
    )
    conformance_parser.add_argument(
        "--axes",
        type=Path,
        default=ROOT / "fixtures" / "forward_response_conformance.json",
        help="Provider-neutral forward-response axes JSON.",
    )
    conformance_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination NPZ containing case_ids and unit_response.",
    )
    conformance_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing conformance NPZ.",
    )
    score_parser = subparsers.add_parser(
        "score-future",
        help="Score frozen count-MLE candidates on post-cutoff observations only.",
    )
    score_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Current station-complete MeasurementLog prefix.",
    )
    score_parser.add_argument(
        "--mle-config",
        type=Path,
        required=True,
        help="Exact count-MLE JSON configuration used by the snapshot report.",
    )
    score_parser.add_argument(
        "--snapshot-estimate",
        type=Path,
        required=True,
        help="Earlier count-MLE report directory or mle_estimate.npz.",
    )
    score_parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Earlier MLESnapshot v2 JSON artifact.",
    )
    holdout_parser = subparsers.add_parser(
        "ral-holdout",
        help="Run final spectral MLE on a proven unseen Geant4 environment.",
    )
    holdout_parser.add_argument("--tuning-run-dir", type=Path, required=True)
    holdout_parser.add_argument("--holdout-run-dir", type=Path, required=True)
    holdout_parser.add_argument("--mle-config", type=Path, required=True)
    holdout_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination future-only score JSON.",
    )
    score_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing score JSON artifact.",
    )
    score_parser.add_argument(
        "--json", action="store_true", help="Print the complete score as JSON."
    )
    return parser


def _config_for_command(args: argparse.Namespace, mode: str) -> MLEConfig:
    """Load or derive a configuration and apply explicit CLI overrides."""
    if args.mle_config is None:
        log = load_measurement_log(args.run_dir)
        config = MLEConfig(mode=mode, isotope_names=log.context.isotopes)
    else:
        config = MLEConfig.load(args.mle_config)
        config = replace(config, mode=mode)
    if args.gpu:
        config = replace(config, use_gpu=True)
    elif args.cpu:
        config = replace(config, use_gpu=False)
    if args.no_debias:
        config = replace(config, debias_refit=False)
    return config


def _estimate_summary(
    estimate: object, output_dir: Path | None = None
) -> dict[str, object]:
    """Return a compact JSON-safe estimate summary."""
    diagnostics = dict(estimate.diagnostics)
    clusters = diagnostics.get("hotspot_clusters", [])
    provenance = diagnostics.get("provenance", {})
    return {
        "schema_version": 1,
        "estimator_family": diagnostics.get("estimator_family"),
        "estimator_variant": diagnostics.get("estimator_variant"),
        "candidate_domain": diagnostics.get("candidate_domain"),
        "uses_pf_state": diagnostics.get("uses_pf_state"),
        "uses_pf_candidates": diagnostics.get("uses_pf_candidates"),
        "provenance": provenance if isinstance(provenance, dict) else {},
        "mode": diagnostics.get("mode"),
        "isotopes": list(estimate.isotope_names),
        "patch_count": len(estimate.patches),
        "objective": float(estimate.objective_value),
        "poisson_deviance": float(estimate.poisson_deviance),
        "iterations": int(estimate.iterations),
        "converged": bool(estimate.converged),
        "cluster_count": len(clusters) if isinstance(clusters, list) else 0,
        "output_dir": None if output_dir is None else str(output_dir),
    }


def _run_fit(args: argparse.Namespace, mode: str) -> int:
    """Run one replay command, save all outputs, and print a summary."""
    config = _config_for_command(args, mode)
    config_source_sha256 = (
        None
        if args.mle_config is None
        else sha256(args.mle_config.read_bytes()).hexdigest()
    )
    replay_result = run_replay(
        args.run_dir,
        config=config,
        config_source_sha256=config_source_sha256,
        initial_estimate_path=args.initial_estimate,
    )
    estimate = replay_result.estimate
    output_dir = args.output_dir or args.run_dir / f"mle_{mode}"
    save_mle_estimate(
        output_dir,
        estimate,
        config=config,
        overwrite=bool(args.overwrite),
    )
    summary = _estimate_summary(estimate, output_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"mode: {summary['mode']}")
        print(f"patches: {summary['patch_count']}")
        print(f"objective: {summary['objective']:.8g}")
        print(f"poisson_deviance: {summary['poisson_deviance']:.8g}")
        print(f"iterations: {summary['iterations']}")
        print(f"converged: {summary['converged']}")
        print(f"output_dir: {summary['output_dir']}")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Load a saved estimate and print its summary without refitting."""
    estimate = load_mle_estimate(args.estimate)
    estimate_path = Path(args.estimate)
    output_dir = estimate_path if estimate_path.is_dir() else estimate_path.parent
    summary = _estimate_summary(estimate, output_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


def _run_online(args: argparse.Namespace) -> int:
    """Run station-causal runtime-log replay and print its final summary."""
    config = _config_for_command(args, "spectral")
    output_dir = args.output_dir or args.run_dir / "mle_online"

    def announce_dashboard(url: str) -> None:
        """Print the live URL without corrupting JSON standard output."""
        _print_cui_dashboard_url(url, json_output=bool(args.json))

    online_result = run_online_replay(
        args.run_dir,
        config=config,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
        enable_dashboard=not args.no_dashboard,
        serve_dashboard=not args.no_dashboard and not args.no_serve,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        dashboard_public_host=args.dashboard_public_host,
        dashboard_url_hook=announce_dashboard,
    )
    summary = _estimate_summary(online_result.final_estimate, output_dir)
    summary.update(
        {
            "execution_mode": "online_station_complete",
            "station_report_count": len(online_result.station_reports),
            "station_cutoff_steps": [
                report.data_cutoff_step for report in online_result.station_reports
            ],
            "state_path": str(online_result.state_path),
            "dashboard_url": online_result.dashboard_url,
        }
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"mode: {summary['mode']}")
        print(f"stations: {summary['station_report_count']}")
        print(f"patches: {summary['patch_count']}")
        print(f"objective: {summary['objective']:.8g}")
        print(f"converged: {summary['converged']}")
        print(f"output_dir: {summary['output_dir']}")
    return 0


def _run_ral_full_simulation(args: argparse.Namespace) -> int:
    """Preflight and run the strict runtime-acquisition plus MLE pipeline."""
    preflight = preflight_ral_full_simulation(
        mle_config_path=args.mle_config,
        planning_config_path=args.planning_config,
        stop_config_path=args.stop_config,
        runtime_root=args.runtime_root,
    )
    if args.preflight_only:
        payload = preflight.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"ready: {payload['ready']}")
            print(f"runtime_config: {payload['runtime_config_path']}")
            print(f"geant4_sidecar: {payload['geant4_sidecar_path']}")
            for error in payload["errors"]:
                print(f"error: {error}")
        return 0 if preflight.ready else 1
    if not preflight.ready:
        raise RuntimeError(
            "RA-L full-simulation preflight failed:\n- " + "\n- ".join(preflight.errors)
        )
    if args.output_dir is None:
        raise ValueError("ral-full-simulation requires --output-dir.")
    if args.scenario is None and args.run_dir is None:
        raise ValueError(
            "ral-full-simulation requires a private --scenario or completed --run-dir."
        )
    if args.scenario is not None and args.final_only:
        raise ValueError("Live MLE closed-loop acquisition cannot use --final-only.")
    if args.resume_stage is not None and args.scenario is None:
        raise ValueError("--resume-stage requires a private --scenario.")
    if args.resume_compatibility is not None and args.resume_stage is None:
        raise ValueError("--resume-compatibility requires --resume-stage.")

    def announce_dashboard(url: str) -> None:
        """Relay the MLE dashboard URL as soon as its server starts."""
        _print_cui_dashboard_url(url, json_output=bool(args.json))

    def relay_runtime(line: str) -> None:
        """Relay non-protocol runtime output without corrupting JSON results."""
        stream = sys.stderr if args.json else sys.stdout
        print(line, file=stream, flush=True)

    if args.scenario is not None:
        stop_config = MLEStopConfig.load(args.stop_config)
        if args.minimum_information_gain_nats is not None:
            stop_config = replace(
                stop_config,
                maximum_expected_information_gain_nats=(
                    args.minimum_information_gain_nats
                ),
            )
        if args.low_information_patience is not None:
            stop_config = replace(
                stop_config,
                low_information_patience=args.low_information_patience,
            )
        result = run_ral_closed_loop(
            args.scenario,
            runtime_root=preflight.runtime_root,
            private_scene_profile=args.private_scene_profile,
            resume_stage_path=args.resume_stage,
            resume_compatibility_path=args.resume_compatibility,
            mle_config_path=args.mle_config,
            planning_config_path=args.planning_config,
            output_dir=args.output_dir,
            max_measurements=args.max_measurements,
            minimum_information_gain_nats=(
                stop_config.maximum_expected_information_gain_nats
            ),
            low_information_patience=stop_config.low_information_patience,
            stop_config=stop_config,
            overwrite=bool(args.overwrite),
            enable_dashboard=not args.no_dashboard,
            serve_dashboard=not args.no_dashboard and not args.no_serve,
            dashboard_host=args.dashboard_host,
            dashboard_port=args.dashboard_port,
            dashboard_public_host=args.dashboard_public_host,
            dashboard_url_hook=announce_dashboard,
            output_hook=relay_runtime,
        )
    else:
        result = run_ral_full_simulation(
            args.run_dir.expanduser().resolve(),
            mle_config_path=args.mle_config,
            output_dir=args.output_dir,
            overwrite=bool(args.overwrite),
            final_only=bool(args.final_only),
            enable_dashboard=not args.no_dashboard,
            serve_dashboard=not args.no_dashboard and not args.no_serve,
            dashboard_host=args.dashboard_host,
            dashboard_port=args.dashboard_port,
            dashboard_public_host=args.dashboard_public_host,
            dashboard_url_hook=announce_dashboard,
        )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"run_id: {result.run_id}")
        print(f"records: {result.record_count}")
        print(f"measurement_log: {result.measurement_log_path}")
        print(f"mle_output: {result.mle_output_dir}")
        if hasattr(result, "stop_reason"):
            print(f"stop_reason: {result.stop_reason}")
        if result.dashboard_url is not None:
            _print_cui_dashboard_url(result.dashboard_url, json_output=False)
    return 0


def _candidate_payload(path: Path) -> dict[str, object]:
    """Load and validate the runtime-owned truth-free candidate JSON shell."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Candidate-pose JSON root must be an object.")
    allowed = {
        "candidate_poses_xyz",
        "travel_costs",
        "allowed_pair_ids",
        "current_pair_id",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Candidate-pose JSON has unknown fields: {unknown}")
    if "candidate_poses_xyz" not in payload:
        raise ValueError("Candidate-pose JSON requires candidate_poses_xyz.")
    return payload


def _estimate_history_indices(
    estimate: object,
    available_step_ids: object,
) -> np.ndarray:
    """Resolve and verify the exact causal history covered by an MLE report."""
    diagnostics = dict(estimate.diagnostics)
    lineage = diagnostics.get("online_lineage")
    raw_steps = (
        lineage.get("covered_step_ids")
        if isinstance(lineage, dict)
        else diagnostics.get("observation_step_ids")
    )
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("MLE report must declare its covered observation step IDs.")
    steps = np.asarray(raw_steps)
    if not np.issubdtype(steps.dtype, np.integer) or np.issubdtype(
        steps.dtype,
        np.bool_,
    ):
        raise ValueError("MLE report covered step IDs must be integers.")
    available = np.asarray(available_step_ids, dtype=np.int64)
    expected = available[: steps.size]
    if expected.shape != steps.shape or not np.array_equal(
        expected,
        steps.astype(np.int64),
    ):
        raise ValueError(
            "MLE report history must be an exact causal prefix of the runtime log."
        )
    return np.arange(steps.size, dtype=np.int64)


def _run_plan_next(args: argparse.Namespace) -> int:
    """Run MLE-local Fisher planning over runtime-supplied candidate poses."""
    mle_config = MLEConfig.load(args.mle_config)
    mle_config = replace(mle_config, mode="spectral")
    if args.gpu:
        mle_config = replace(mle_config, use_gpu=True)
    elif args.cpu:
        mle_config = replace(mle_config, use_gpu=False)
    context = prepare_replay(args.run_dir, config=mle_config)
    estimate = load_mle_estimate(args.estimate)
    provenance = estimate.diagnostics.get("provenance", {})
    if isinstance(provenance, dict):
        estimate_run_id = provenance.get("measurement_run_id")
        if estimate_run_id is not None and estimate_run_id != context.log.run_id:
            raise ValueError("MLE estimate belongs to a different runtime run_id.")
        estimate_config_digest = provenance.get("resolved_estimator_config_sha256")
        if estimate_config_digest is not None and (
            estimate_config_digest != context.resolved_estimator_config_sha256
        ):
            raise ValueError(
                "MLE estimate was fitted with a different resolved MLE configuration."
            )
    indices = _estimate_history_indices(estimate, context.batch.step_ids)
    history = subset_observation_batch(context.batch, indices)
    candidates = _candidate_payload(args.candidates)
    planning_config = (
        MLEPlanningConfig()
        if args.planning_config is None
        else MLEPlanningConfig.load(args.planning_config)
    )
    current_pair = candidates.get("current_pair_id")
    if current_pair is None:
        orientation_count = len(context.kernel.orientations)
        current_pair = int(history.fe_indices[-1]) * orientation_count + int(
            history.pb_indices[-1]
        )
    result = plan_next_measurement(
        estimate,
        history,
        context.kernel,
        mle_config,
        candidates["candidate_poses_xyz"],
        planning_config=planning_config,
        allowed_pair_ids=candidates.get("allowed_pair_ids"),
        travel_costs=candidates.get("travel_costs"),
        current_pair_id=current_pair,
    )
    result = replace(
        result,
        diagnostics={
            **result.diagnostics,
            "measurement_run_id": context.log.run_id,
            "data_cutoff_step": int(history.step_ids[-1]),
            "covered_step_ids": history.step_ids.astype(int).tolist(),
            "resolved_estimator_config_sha256": (
                context.resolved_estimator_config_sha256
            ),
        },
    )
    output = save_mle_planning_result(
        result,
        args.output,
        overwrite=bool(args.overwrite),
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        action = result.selected_action
        print(f"candidate_index: {action.candidate_index}")
        print(f"detector_pose_xyz: {list(action.detector_pose_xyz)}")
        print(f"shield_pair_ids: {list(action.shield_pair_ids)}")
        print(f"information_gain_nats: {action.information_gain_nats:.8g}")
        print(f"score: {action.score:.8g}")
        print(f"output: {output}")
    return 0


def _run_forward_conformance(args: argparse.Namespace) -> int:
    """Generate all canonical local forward responses and report their count."""
    result = compute_forward_conformance(args.axes)
    output = save_forward_conformance(
        args.output,
        result,
        overwrite=bool(args.overwrite),
    )
    print(f"cases: {result.case_ids.size}")
    print(f"output: {output}")
    return 0


def _run_score_future(args: argparse.Namespace) -> int:
    """Run frozen future-only candidate verification and save its artifact."""
    payload = score_future_count_candidates(
        args.run_dir,
        config=args.mle_config,
        snapshot_estimate=args.snapshot_estimate,
        snapshot=args.snapshot,
    )
    output = save_future_candidate_scores(
        args.output,
        payload,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"snapshot_id: {payload['snapshot_id']}")
        print(f"future_steps: {len(payload['future_step_ids'])}")
        print(f"candidates: {len(payload['candidates'])}")
        print(f"output: {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested standalone operation."""
    parser = build_argument_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    if args.command == "replay":
        return _run_fit(args, "count")
    if args.command == "fit-spectrum":
        return _run_fit(args, "spectral")
    if args.command in {"online-replay", "online"}:
        return _run_online(args)
    if args.command == "ral-full-simulation":
        return _run_ral_full_simulation(args)
    if args.command == "plan-next":
        return _run_plan_next(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "forward-conformance":
        return _run_forward_conformance(args)
    if args.command == "score-future":
        return _run_score_future(args)
    if args.command == "ral-holdout":
        result = run_ral_holdout(
            args.tuning_run_dir,
            args.holdout_run_dir,
            config_path=args.mle_config,
            output_dir=args.output_dir,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
