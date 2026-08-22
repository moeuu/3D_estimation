"""Command-line interface for live surface MLE control and reporting."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from runtime.cui import CUI_URL_MESSAGE_PREFIX
from runtime.defaults import (
    DEFAULT_CUI_SPLIT_VIEW_HOST,
    DEFAULT_CUI_SPLIT_VIEW_PORT,
)

from .closed_loop import (
    MLEStopConfig,
    RAL_PRIVATE_SCENE_PROFILES,
    run_ral_closed_loop,
)
from .conformance import compute_forward_conformance, save_forward_conformance
from .ral import preflight_ral_full_simulation
from .reporting import load_mle_estimate

ROOT = Path(__file__).resolve().parents[2]
RAL_MLE_CONFIG = ROOT / "configs" / "mle" / "ral_full_spectral.json"
RAL_PLANNING_CONFIG = ROOT / "configs" / "mle" / "ral_full_planning.json"
RAL_STOP_CONFIG = ROOT / "configs" / "mle" / "ral_full_stop.json"


def _print_cui_dashboard_url(url: str, *, json_output: bool) -> None:
    """Print one immediately flushable CUI URL without corrupting JSON stdout."""
    stream = sys.stderr if json_output else sys.stdout
    print(f"{CUI_URL_MESSAGE_PREFIX} {url}", file=stream, flush=True)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level live-control and report command parser."""
    parser = argparse.ArgumentParser(
        prog="estimate-radiation-mle",
        description="Standalone rotating-shield surface maximum-likelihood estimation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ral_parser = subparsers.add_parser(
        "ral-full-simulation",
        help=(
            "Run a private RA-L scenario through a live MLE-controlled shared "
            "runtime session."
        ),
    )
    ral_parser.add_argument(
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
        default=DEFAULT_CUI_SPLIT_VIEW_HOST,
        help=f"Dashboard bind host (default: {DEFAULT_CUI_SPLIT_VIEW_HOST}).",
    )
    ral_parser.add_argument(
        "--dashboard-port",
        type=int,
        default=DEFAULT_CUI_SPLIT_VIEW_PORT,
        help=f"Dashboard TCP port (default: {DEFAULT_CUI_SPLIT_VIEW_PORT}).",
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
    return parser


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


def _run_ral_live_acquisition(args: argparse.Namespace) -> int:
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
    if args.scenario is None:
        raise ValueError("ral-full-simulation requires a private --scenario.")
    if args.resume_compatibility is not None and args.resume_stage is None:
        raise ValueError("--resume-compatibility requires --resume-stage.")

    def announce_dashboard(url: str) -> None:
        """Relay the MLE dashboard URL as soon as its server starts."""
        _print_cui_dashboard_url(url, json_output=bool(args.json))

    def relay_runtime(line: str) -> None:
        """Relay non-protocol runtime output without corrupting JSON results."""
        stream = sys.stderr if args.json else sys.stdout
        print(line, file=stream, flush=True)

    stop_config = MLEStopConfig.load(args.stop_config)
    if args.minimum_information_gain_nats is not None:
        stop_config = replace(
            stop_config,
            maximum_expected_information_gain_nats=(args.minimum_information_gain_nats),
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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested standalone operation."""
    parser = build_argument_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    if args.command == "ral-full-simulation":
        return _run_ral_live_acquisition(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "forward-conformance":
        return _run_forward_conformance(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
