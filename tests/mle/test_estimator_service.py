"""Subprocess conformance tests for the fixed surface-MLE service boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from radiation_estimator_service_contracts import (
    FILE_SHA256_DIGEST_ALGORITHM,
    ArtifactRef,
    ArtifactTarget,
    Capabilities,
    ContractRef,
    DigestRef,
    ExecuteRequest,
    ExecuteResponse,
    MeasurementLogRef,
    NamedArtifactRef,
    canonical_json_bytes,
    digest_artifact_directory,
    file_uri_from_path,
    sha256_bytes,
    validate_artifact_ref,
)
from runtime.measurement_log import load_measurement_log
from runtime.prefix import measurement_records_digest

from three_d_estimation.config import MLEConfig
from three_d_estimation.provenance import repository_commit
from three_d_estimation.reporting import load_mle_estimate
from three_d_estimation.service import (
    ESTIMATE_OPERATION,
    ESTIMATOR_FAMILY,
    MEASUREMENT_LOG_CONTRACT,
    MLE_CONFIG_CONTRACT,
    MLE_RESULT_CONTRACT,
    service_capabilities,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "shared_measurement_log" / "measurement_log"
SERVICE = Path(sys.executable).parent / "radiation-surface-mle-service"


def _run_service(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the installed dedicated service entry point without a shell."""
    return subprocess.run(
        (SERVICE.as_posix(), *arguments),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )


def _capabilities(tmp_path: Path) -> Capabilities:
    """Probe and parse one capability response through the real executable."""
    response = tmp_path / "capabilities.json"
    completed = _run_service("capabilities", "--response", response.as_posix())
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    return Capabilities.from_json_bytes(response.read_bytes())


def _file_reference(path: Path, *, contract: ContractRef) -> ArtifactRef:
    """Return one authenticated raw-SHA256 file reference."""
    payload = path.read_bytes()
    return ArtifactRef(
        uri=file_uri_from_path(path),
        kind="file",
        digest=DigestRef(
            algorithm=FILE_SHA256_DIGEST_ALGORITHM,
            value=sha256_bytes(payload),
        ),
        media_type="application/json",
        contract=contract,
        size_bytes=len(payload),
    )


def _measurement_reference(path: Path = FIXTURE) -> MeasurementLogRef:
    """Return a fully authenticated reference to the shared fixture log."""
    log = load_measurement_log(path)
    artifact_digest, artifact_size = digest_artifact_directory(path)
    records_digest = measurement_records_digest(log.records)
    return MeasurementLogRef(
        artifact=ArtifactRef(
            uri=file_uri_from_path(path),
            kind="directory",
            digest=artifact_digest,
            media_type="application/vnd.radiation.measurement-log",
            contract=MEASUREMENT_LOG_CONTRACT,
            size_bytes=artifact_size,
        ),
        schema_version=log.schema_version,
        run_id=log.run_id,
        record_count=len(log.records),
        terminal_step_id=log.records[-1].step_id,
        records_digest=DigestRef(
            algorithm=records_digest.algorithm,
            value=records_digest.sha256,
        ),
    )


def _config_reference(tmp_path: Path) -> ArtifactRef:
    """Create and reference one fast service-safe spectral MLE config."""
    path = tmp_path / "mle-config.json"
    MLEConfig(
        mode="spectral",
        isotope_names=("Co-60", "Cs-137", "Eu-154"),
        patch_spacing_m=(6.0, 6.0, 3.0),
        max_iterations=2,
        check_interval=1,
        debias_refit=False,
        fit_background_nuisance=False,
        fit_scatter_nuisance=False,
        use_gpu=False,
        random_seed=91,
    ).save(path)
    return _file_reference(path, contract=MLE_CONFIG_CONTRACT)


def _request(tmp_path: Path, capabilities: Capabilities) -> ExecuteRequest:
    """Build one request bound to exact MLE capabilities and fixture artifacts."""
    return ExecuteRequest(
        request_id="surface-mle-service:test-1",
        estimator_family=ESTIMATOR_FAMILY,
        operation=ESTIMATE_OPERATION,
        measurement_log=_measurement_reference(),
        config=_config_reference(tmp_path),
        random_seed=17,
        input_artifacts=(),
        output=ArtifactTarget(file_uri_from_path(tmp_path / "mle-result")),
        expected_capabilities_digest=capabilities.digest,
        requested_result_contract=MLE_RESULT_CONTRACT,
    )


def _execute_request(
    tmp_path: Path,
    request: ExecuteRequest,
    *,
    name: str = "execute",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Persist and execute one request through the real service process."""
    request_path = tmp_path / f"{name}-request.json"
    response_path = tmp_path / f"{name}-response.json"
    request_path.write_bytes(request.to_json_bytes())
    completed = _run_service(
        "execute",
        "--request",
        request_path.as_posix(),
        "--response",
        response_path.as_posix(),
    )
    return completed, response_path


def test_capabilities_advertise_one_truth_free_mle_contract(tmp_path: Path) -> None:
    """The dedicated executable must expose exactly one explicit MLE operation."""
    capabilities = _capabilities(tmp_path)

    assert capabilities.estimator_family == ESTIMATOR_FAMILY
    assert capabilities.implementation.revision == repository_commit()
    assert capabilities.accepts_truth is False
    assert capabilities.measurement_log_schema_versions == (2,)
    assert len(capabilities.operations) == 1
    operation = capabilities.operations[0]
    assert operation.operation == ESTIMATE_OPERATION
    assert operation.config_contracts == (MLE_CONFIG_CONTRACT,)
    assert operation.result_contracts == (MLE_RESULT_CONTRACT,)
    assert operation.required_input_roles == ()
    assert operation.optional_input_roles == ()


def test_execute_publishes_authenticated_existing_mle_report(tmp_path: Path) -> None:
    """A valid request must run replay and return only an opaque result directory."""
    capabilities = _capabilities(tmp_path)
    request = _request(tmp_path, capabilities)

    completed, response_path = _execute_request(tmp_path, request)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    response = ExecuteResponse.from_json_bytes(response_path.read_bytes())
    assert response.status == "succeeded"
    assert response.error is None
    assert response.artifacts == ()
    assert response.result_artifact is not None
    assert response.result_artifact.contract == MLE_RESULT_CONTRACT
    result_path = validate_artifact_ref(response.result_artifact)
    assert result_path == tmp_path / "mle-result"
    estimate = load_mle_estimate(result_path)
    assert estimate.diagnostics["estimator_family"] == "surface_mle"
    assert not any("truth" in path.name.lower() for path in result_path.rglob("*"))


@pytest.mark.parametrize(
    "mismatch",
    ["capabilities", "config", "log-identity", "result", "truth"],
)
def test_execute_rejects_contract_mismatches_and_truth(
    tmp_path: Path,
    mismatch: str,
) -> None:
    """Unsupported contracts and truth roles must fail before MLE execution."""
    capabilities = _capabilities(tmp_path)
    request = _request(tmp_path, capabilities)
    if mismatch == "capabilities":
        request = replace(
            request,
            expected_capabilities_digest=DigestRef(
                algorithm=request.expected_capabilities_digest.algorithm,
                value="0" * 64,
            ),
        )
    elif mismatch == "config":
        request = replace(
            request,
            config=replace(
                request.config,
                contract=ContractRef("radiation.other-config", 1),
            ),
        )
    elif mismatch == "log-identity":
        request = replace(
            request,
            measurement_log=replace(
                request.measurement_log,
                record_count=request.measurement_log.record_count + 1,
            ),
        )
    elif mismatch == "result":
        request = replace(
            request,
            requested_result_contract=ContractRef("radiation.other-result", 1),
        )
    else:
        request = replace(
            request,
            input_artifacts=(
                NamedArtifactRef(role="source-truth", artifact=request.config),
            ),
        )

    completed, response_path = _execute_request(tmp_path, request, name=mismatch)

    assert completed.returncode == 1
    response = ExecuteResponse.from_json_bytes(response_path.read_bytes())
    assert response.status == "failed"
    assert response.capabilities_digest == capabilities.digest
    assert response.result_artifact is None
    assert response.error is not None
    assert not (tmp_path / "mle-result").exists()


def test_capabilities_omit_an_unavailable_repository_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unpackaged builds may omit revision but must not publish a placeholder."""
    monkeypatch.setattr(
        "three_d_estimation.service.repository_commit",
        lambda: "unknown-standalone-build",
    )

    assert service_capabilities().implementation.revision is None


def test_service_rejects_symlink_and_traversal_inputs(tmp_path: Path) -> None:
    """Neither request files nor artifact URIs may use links or traversal."""
    capabilities = _capabilities(tmp_path)
    request = _request(tmp_path, capabilities)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(request.to_json_bytes())
    request_link = tmp_path / "request-link.json"
    request_link.symlink_to(request_path)
    response_path = tmp_path / "symlink-response.json"

    symlinked = _run_service(
        "execute",
        "--request",
        request_link.as_posix(),
        "--response",
        response_path.as_posix(),
    )

    assert symlinked.returncode == 65
    assert not response_path.exists()

    config_path = validate_artifact_ref(request.config)
    config_link = tmp_path / "config-link.json"
    config_link.symlink_to(config_path)
    linked_config_request = replace(
        request,
        config=replace(request.config, uri=file_uri_from_path(config_link)),
    )
    linked_completed, linked_response_path = _execute_request(
        tmp_path,
        linked_config_request,
        name="linked-config",
    )
    assert linked_completed.returncode == 1
    linked_response = ExecuteResponse.from_json_bytes(
        linked_response_path.read_bytes()
    )
    assert linked_response.status == "failed"

    traversing = request.to_dict()
    config = traversing["config"]
    assert isinstance(config, dict)
    config["uri"] = file_uri_from_path(tmp_path / "nested" / "config.json").replace(
        "/nested/config.json",
        "/nested/../mle-config.json",
    )
    traversal_request = tmp_path / "traversal-request.json"
    traversal_request.write_bytes(canonical_json_bytes(traversing))
    traversal_response = tmp_path / "traversal-response.json"

    traversed = _run_service(
        "execute",
        "--request",
        traversal_request.as_posix(),
        "--response",
        traversal_response.as_posix(),
    )

    assert traversed.returncode == 65
    assert not traversal_response.exists()
    assert not (tmp_path / "mle-result").exists()


def test_service_rejects_shell_templates_relative_paths_and_response_links(
    tmp_path: Path,
) -> None:
    """The service CLI must offer no shell or unsafe path escape hatch."""
    relative = _run_service("capabilities", "--response", "capabilities.json")
    assert relative.returncode == 64

    response_path = tmp_path / "capabilities.json"
    extra = _run_service(
        "capabilities",
        "--response",
        response_path.as_posix(),
        "--command-template",
        "touch SHOULD_NOT_EXIST",
    )
    assert extra.returncode == 64
    assert not response_path.exists()
    assert not (ROOT / "SHOULD_NOT_EXIST").exists()

    capabilities = _capabilities(tmp_path)
    request = _request(tmp_path, capabilities)
    payload = request.to_dict()
    payload["command_template"] = "touch SHOULD_NOT_EXIST"
    request_path = tmp_path / "template-request.json"
    request_path.write_bytes(canonical_json_bytes(payload))
    template_response = tmp_path / "template-response.json"
    templated = _run_service(
        "execute",
        "--request",
        request_path.as_posix(),
        "--response",
        template_response.as_posix(),
    )
    assert templated.returncode == 65
    assert not template_response.exists()
    assert not (ROOT / "SHOULD_NOT_EXIST").exists()

    protected = tmp_path / "protected.json"
    protected.write_text("unchanged\n", encoding="utf-8")
    response_link = tmp_path / "response-link.json"
    response_link.symlink_to(protected)
    linked = _run_service(
        "capabilities",
        "--response",
        response_link.as_posix(),
    )
    assert linked.returncode == 64
    assert protected.read_text(encoding="utf-8") == "unchanged\n"


def test_service_preserves_measurement_log_and_separates_response_target(
    tmp_path: Path,
) -> None:
    """Service paths must not mutate the log or collide with estimator output."""
    capabilities = _capabilities(tmp_path)
    log_path = tmp_path / "measurement-log"
    shutil.copytree(FIXTURE, log_path)
    measurement = _measurement_reference(log_path)
    base = replace(_request(tmp_path, capabilities), measurement_log=measurement)
    before_digest = digest_artifact_directory(log_path)

    nested_output = replace(
        base,
        output=ArtifactTarget(file_uri_from_path(log_path / "mle-result")),
    )
    nested_completed, nested_response = _execute_request(
        tmp_path,
        nested_output,
        name="nested-output",
    )
    assert nested_completed.returncode == 1
    assert ExecuteResponse.from_json_bytes(nested_response.read_bytes()).status == (
        "failed"
    )
    assert digest_artifact_directory(log_path) == before_digest

    request_path = tmp_path / "inside-log-request.json"
    request_path.write_bytes(base.to_json_bytes())
    forbidden_response = log_path / "service-response.json"
    inside_completed = _run_service(
        "execute",
        "--request",
        request_path.as_posix(),
        "--response",
        forbidden_response.as_posix(),
    )
    assert inside_completed.returncode == 65
    assert not forbidden_response.exists()
    assert digest_artifact_directory(log_path) == before_digest

    collision_path = tmp_path / "colliding-result-and-response"
    collision = replace(
        base,
        output=ArtifactTarget(file_uri_from_path(collision_path)),
    )
    collision_request = tmp_path / "collision-request.json"
    collision_request.write_bytes(collision.to_json_bytes())
    collision_completed = _run_service(
        "execute",
        "--request",
        collision_request.as_posix(),
        "--response",
        collision_path.as_posix(),
    )
    assert collision_completed.returncode == 65
    assert not collision_path.exists()
