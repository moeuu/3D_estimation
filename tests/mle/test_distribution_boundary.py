"""Distribution-boundary checks for the estimator-only MLE wheel."""

from __future__ import annotations

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
    contract_dependency = next(
        dependency
        for dependency in runtime_dependencies
        if dependency.startswith("radiation-estimator-service-contracts")
    )
    contract_source = configuration["tool"]["uv"]["sources"][
        "radiation-estimator-service-contracts"
    ]
    assert contract_dependency == "radiation-estimator-service-contracts==0.1.0"
    assert set(contract_source) == {"git", "rev"}
    assert contract_source["git"] == (
        "https://github.com/moeuu/radiation-estimator-service-contracts.git"
    )
    assert len(contract_source["rev"]) == 40
    assert project["scripts"]["radiation-surface-mle-service"] == (
        "three_d_estimation.service:main"
    )


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
