"""Verify that the MLE repository contains estimator-specific code only."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATHS = (
    "native",
    "obstacle_layouts",
    "source_layouts",
    "src/measurement",
    "src/pf",
    "src/planning",
    "src/runtime",
    "src/sim",
    "src/spectrum",
    "src/realtime_demo.py",
    "src/three_d_estimation/service.py",
)


def _check_forbidden_paths() -> list[str]:
    """Report copied simulator, PF, planner, or log-owner paths."""
    return [
        f"forbidden duplicated implementation: {relative}"
        for relative in FORBIDDEN_PATHS
        if (ROOT / relative).exists()
    ]


def _check_package_boundary() -> list[str]:
    """Require only the MLE package in wheel discovery."""
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = payload["tool"]["setuptools"]["packages"]["find"]["include"]
    if included != ["three_d_estimation*"]:
        return [f"unexpected package discovery include: {included!r}"]
    dependencies = payload["project"]["dependencies"]
    runtime_dependencies = [
        dependency
        for dependency in dependencies
        if dependency.startswith("rotating-shield-simulation-runtime")
    ]
    if not runtime_dependencies:
        return ["shared simulation runtime dependency is missing"]
    if len(runtime_dependencies) != 1 or "==" not in runtime_dependencies[0]:
        return ["shared simulation runtime dependency must use an exact version"]
    runtime_source = (
        payload.get("tool", {})
        .get("uv", {})
        .get("sources", {})
        .get("rotating-shield-simulation-runtime")
    )
    if not isinstance(runtime_source, dict):
        return ["shared simulation runtime source pin is missing"]
    revision = runtime_source.get("rev")
    if (
        not isinstance(runtime_source.get("git"), str)
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        return ["shared simulation runtime source must pin one Git commit"]
    research_dependencies = [
        dependency
        for dependency in dependencies
        if dependency.startswith(("radiation-", "rotating-shield-"))
    ]
    if research_dependencies != runtime_dependencies:
        return ["shared runtime must be the only research-package dependency"]
    sources = payload.get("tool", {}).get("uv", {}).get("sources", {})
    if set(sources) != {"rotating-shield-simulation-runtime"}:
        return ["shared runtime must be the only pinned research source"]
    scripts = payload.get("project", {}).get("scripts", {})
    if any(
        "service" in name.lower() or ".service:" in target
        for name, target in scripts.items()
    ):
        return ["out-of-process estimator service entry points must be absent"]
    extras = payload.get("project", {}).get("optional-dependencies", {})
    if any("service" in name.lower() for name in extras):
        return ["out-of-process estimator service extras must be absent"]
    return []


def _check_python_syntax() -> list[str]:
    """Parse all retained Python files without importing optional GPU code."""
    errors: list[str] = []
    for root_name in ("src/three_d_estimation", "scripts"):
        root = ROOT / root_name
        for path in root.rglob("*.py"):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")
    return errors


def run_checks() -> list[str]:
    """Return every repository-boundary error."""
    return [
        *_check_forbidden_paths(),
        *_check_package_boundary(),
        *_check_python_syntax(),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the MLE repository-boundary audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)
    errors = run_checks()
    payload = {"boundary_ok": not errors, "errors": errors}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print("MLE repository boundary passed.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
