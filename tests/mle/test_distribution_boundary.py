"""Distribution-boundary checks for the estimator-only MLE wheel."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

from setuptools.discovery import PackageFinder


ROOT = Path(__file__).resolve().parents[2]


def test_build_metadata_separates_runtime_and_development_tools() -> None:
    """Wheel metadata must use PEP 639 and exclude test or lint dependencies."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    project = configuration["project"]
    runtime_dependencies = tuple(project["dependencies"])
    development_dependencies = tuple(configuration["dependency-groups"]["dev"])

    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert not any(
        dependency.startswith(("pytest", "ruff")) for dependency in runtime_dependencies
    )
    assert any(
        dependency.startswith("pytest") for dependency in development_dependencies
    )
    assert any(dependency.startswith("ruff") for dependency in development_dependencies)
    runtime_dependency = next(
        dependency
        for dependency in runtime_dependencies
        if dependency.startswith("rotating-shield-simulation-runtime")
    )
    runtime_source = configuration["tool"]["uv"]["sources"][
        "rotating-shield-simulation-runtime"
    ]
    assert "==" in runtime_dependency
    assert set(runtime_source) == {"git", "rev"}
    assert len(runtime_source["rev"]) == 40
    research_dependencies = tuple(
        dependency
        for dependency in runtime_dependencies
        if dependency.startswith(("radiation-", "rotating-shield-"))
    )
    assert research_dependencies == (runtime_dependency,)
    assert set(configuration["tool"]["uv"].get("sources", {})) == {
        "rotating-shield-simulation-runtime"
    }
    assert not any(
        "service" in name.lower() or ".service:" in target
        for name, target in project.get("scripts", {}).items()
    )
    assert not any(
        "service" in name.lower() for name in project.get("optional-dependencies", {})
    )
    assert not (ROOT / "src" / "three_d_estimation" / "service.py").exists()
    assert not (ROOT / "src" / "three_d_estimation" / "estimator_context.py").exists()
    assert not (ROOT / "src" / "three_d_estimation" / "holdout.py").exists()
    assert not (ROOT / "src" / "three_d_estimation" / "future_scoring.py").exists()
    assert not (ROOT / "src" / "three_d_estimation" / "replay.py").exists()
    assert not (ROOT / "scripts" / "run_mle_replay.py").exists()


def test_completed_log_estimator_api_is_absent() -> None:
    """The installed package must not expose completed-log fit launchers."""
    import three_d_estimation
    import three_d_estimation.observation_batch as observation_batch
    import three_d_estimation.online as online
    import three_d_estimation.ral as ral

    assert importlib.util.find_spec("three_d_estimation.replay") is None
    assert importlib.util.find_spec("three_d_estimation.holdout") is None
    assert importlib.util.find_spec("three_d_estimation.future_scoring") is None
    assert importlib.util.find_spec("three_d_estimation.estimator_context") is None
    for name in ("ReplayContext", "ReplayResult", "prepare_replay", "run_replay"):
        assert not hasattr(three_d_estimation, name)
    for name in (
        "EstimatorContext",
        "WarmStartArtifact",
        "covered_station_boundaries_sha256",
        "observation_batch_from_log",
        "prepare_estimator_context",
        "save_future_candidate_scores",
        "score_future_count_candidates",
        "validate_warm_start_artifact",
    ):
        assert not hasattr(three_d_estimation, name)
    assert not hasattr(observation_batch, "observation_batch_from_log")
    assert not hasattr(online, "run_online_replay")
    assert not hasattr(ral, "RALFullSimulationResult")
    assert not hasattr(ral, "run_ral_full_simulation")


def test_package_discovery_contains_only_mle_code() -> None:
    """The wheel must not vendor shared simulation or another estimator."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    setuptools_config = configuration["tool"]["setuptools"]
    find_config = setuptools_config["packages"]["find"]
    packages = set(
        PackageFinder.find(
            str(ROOT / find_config["where"][0]),
            include=tuple(find_config["include"]),
            exclude=tuple(find_config.get("exclude", ())),
        )
    )

    assert not any(name == "pf" or name.startswith("pf.") for name in packages)
    assert not any(
        name == "planning" or name.startswith("planning.") for name in packages
    )
    assert "realtime_demo" not in setuptools_config.get("py-modules", ())
    assert packages
    assert all(
        name == "three_d_estimation" or name.startswith("three_d_estimation.")
        for name in packages
    )


def test_source_distribution_contains_only_mle_assets() -> None:
    """The sdist must include only estimator configuration and fixtures."""
    directives = {
        line.strip()
        for line in (ROOT / "MANIFEST.in").read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "recursive-include configs/mle *.json" in directives
    assert "prune tests" in directives
    assert not (ROOT / "src/measurement").exists()
    assert not (ROOT / "src/sim").exists()
    assert not (ROOT / "src/spectrum").exists()
