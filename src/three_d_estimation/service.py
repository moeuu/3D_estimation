"""Fixed two-verb service adapter for the local surface-MLE implementation.

The adapter authenticates transport-neutral references, delegates MeasurementLog
parsing and artifact publication mechanics to the runtime, and delegates all
estimation and report construction to the existing MLE replay implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from radiation_estimator_service_contracts import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    Capabilities,
    ContractRef,
    DigestRef,
    EstimatorServiceContractError,
    ExecuteRequest,
    ExecuteResponse,
    ImplementationRef,
    MeasurementLogRef,
    OperationCapability,
    OperationRef,
    ServiceError,
    artifact_ref_from_path,
    canonical_json_bytes,
    parse_service_argv,
    path_from_file_uri,
    read_bounded_regular_file,
    strict_json_object_from_bytes,
    validate_artifact_ref,
    validate_artifact_target,
    validate_new_file_path,
    validate_request_against_capabilities,
    write_new_file,
)
from runtime import AtomicBundlePublisher
from runtime.measurement_log import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementLog,
    load_measurement_log,
)
from runtime.prefix import measurement_records_digest

from .config import MLEConfig
from .provenance import repository_commit
from .replay import run_replay
from .reporting import save_mle_estimate


ESTIMATOR_FAMILY = "surface-mle"
ESTIMATE_OPERATION = OperationRef("estimate", 1)
MLE_CONFIG_CONTRACT = ContractRef("radiation.surface-mle-config", 1)
MLE_RESULT_CONTRACT = ContractRef("radiation.surface-mle-result", 1)
MEASUREMENT_LOG_CONTRACT = ContractRef(
    "runtime.measurement-log",
    MEASUREMENT_LOG_SCHEMA_VERSION,
)
MLE_RESULT_MEDIA_TYPE = "application/vnd.radiation.surface-mle-result"
_DISTRIBUTION = "three-d-estimation"


class MLEServiceError(ValueError):
    """Report one controlled adapter failure without exposing solver internals."""


def _implementation_version() -> str:
    """Return the installed MLE distribution version."""
    try:
        return version(_DISTRIBUTION)
    except PackageNotFoundError:
        return "0.2.0"


def _implementation_revision() -> str | None:
    """Return a full repository revision when local provenance provides one."""
    revision = repository_commit()
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        return None
    return revision


def service_capabilities() -> Capabilities:
    """Return the one immutable operation supported by this MLE service."""
    return Capabilities(
        estimator_family=ESTIMATOR_FAMILY,
        implementation=ImplementationRef(
            distribution=_DISTRIBUTION,
            version=_implementation_version(),
            revision=_implementation_revision(),
        ),
        operations=(
            OperationCapability(
                operation=ESTIMATE_OPERATION,
                config_contracts=(MLE_CONFIG_CONTRACT,),
                result_contracts=(MLE_RESULT_CONTRACT,),
                required_input_roles=(),
                optional_input_roles=(),
            ),
        ),
        measurement_log_schema_versions=(MEASUREMENT_LOG_SCHEMA_VERSION,),
        accepts_truth=False,
    )


def _validate_measurement_identity(
    reference: MeasurementLogRef,
    log: MeasurementLog,
) -> None:
    """Bind the wire MeasurementLog identity to runtime-validated records."""
    if reference.artifact.contract != MEASUREMENT_LOG_CONTRACT:
        raise MLEServiceError("MeasurementLog contract is unsupported.")
    if reference.schema_version != log.schema_version:
        raise MLEServiceError("MeasurementLog schema version differs from its content.")
    if reference.run_id != log.run_id:
        raise MLEServiceError("MeasurementLog run_id differs from its content.")
    if reference.record_count != len(log.records):
        raise MLEServiceError("MeasurementLog record_count differs from its content.")
    if reference.terminal_step_id != log.records[-1].step_id:
        raise MLEServiceError("MeasurementLog terminal_step_id differs from its content.")
    records_digest = measurement_records_digest(log.records)
    if (
        reference.records_digest.algorithm != records_digest.algorithm
        or reference.records_digest.value != records_digest.sha256
    ):
        raise MLEServiceError("MeasurementLog records digest differs from its content.")


def _authenticate_measurement_log(reference: MeasurementLogRef) -> tuple[Path, MeasurementLog]:
    """Authenticate and load one truth-free MeasurementLog through runtime APIs."""
    path = validate_artifact_ref(reference.artifact)
    log = load_measurement_log(path)
    _validate_measurement_identity(reference, log)
    validate_artifact_ref(reference.artifact)
    return path, log


def _authenticated_config(request: ExecuteRequest) -> tuple[MLEConfig, DigestRef]:
    """Authenticate one service-safe MLE config and bind the request seed."""
    path = validate_artifact_ref(request.config)
    payload_bytes = read_bounded_regular_file(path)
    if sha256(payload_bytes).hexdigest() != request.config.digest.value:
        raise MLEServiceError("MLE config changed after artifact authentication.")
    if request.config.size_bytes is not None and len(payload_bytes) != (
        request.config.size_bytes
    ):
        raise MLEServiceError("MLE config size differs from its reference.")
    payload = strict_json_object_from_bytes(payload_bytes)
    for field_name in ("discrepancy_calibration_path", "response_cache_dir"):
        if payload.get(field_name) is not None:
            raise MLEServiceError(
                f"Service MLE config cannot reference external path {field_name}."
            )
    config = MLEConfig.from_dict(payload)
    resolved = replace(
        config,
        random_seed=request.random_seed,
        bootstrap_seed=request.random_seed,
    )
    digest = DigestRef.from_bytes(
        canonical_json_bytes(resolved.to_dict()),
        algorithm=CANONICAL_JSON_DIGEST_ALGORITHM,
    )
    return resolved, digest


def _successful_response(
    request: ExecuteRequest,
    capabilities: Capabilities,
) -> ExecuteResponse:
    """Execute existing replay/reporting code and attest its opaque directory."""
    validate_request_against_capabilities(request, capabilities)
    output_path = validate_artifact_target(request.output)
    log_path, _ = _authenticate_measurement_log(request.measurement_log)
    if output_path.is_relative_to(log_path):
        raise MLEServiceError("MLE result cannot be written inside MeasurementLog.")
    config, resolved_config_digest = _authenticated_config(request)
    with AtomicBundlePublisher(output_path, policy="create") as publisher:
        replay = run_replay(
            log_path,
            config=config,
            output_dir=publisher.staging_path,
            save_hook=partial(save_mle_estimate, config=config, overwrite=True),
            config_source_sha256=request.config.digest.value,
        )
        if replay.saved_output is None:
            raise MLEServiceError("MLE replay did not publish its result report.")
        _validate_measurement_identity(request.measurement_log, replay.context.log)
        validate_artifact_ref(request.measurement_log.artifact)
        publisher.publish()
    result_artifact = artifact_ref_from_path(
        output_path,
        media_type=MLE_RESULT_MEDIA_TYPE,
        contract=request.requested_result_contract,
    )
    return ExecuteResponse(
        request_id=request.request_id,
        request_digest=request.digest,
        estimator_family=request.estimator_family,
        operation=request.operation,
        capabilities_digest=capabilities.digest,
        status="succeeded",
        resolved_config_digest=resolved_config_digest,
        result_artifact=result_artifact,
        artifacts=(),
        error=None,
    )


def _failed_response(
    request: ExecuteRequest,
    capabilities: Capabilities,
    error: Exception,
) -> ExecuteResponse:
    """Return one bounded failure response without any artifact attestation."""
    message = " ".join(str(error).splitlines())[:2048] or type(error).__name__
    return ExecuteResponse(
        request_id=request.request_id,
        request_digest=request.digest,
        estimator_family=request.estimator_family,
        operation=request.operation,
        capabilities_digest=capabilities.digest,
        status="failed",
        resolved_config_digest=None,
        result_artifact=None,
        artifacts=(),
        error=ServiceError(
            code="request-rejected",
            message=message,
            retryable=False,
        ),
    )


def _execute(request_path: Path, response_path: Path) -> int:
    """Decode, execute, and persist one authenticated estimator request."""
    try:
        request = ExecuteRequest.from_json_bytes(
            read_bounded_regular_file(request_path)
        )
    except Exception as exc:
        print(f"surface-mle-service: invalid request: {exc}", file=sys.stderr)
        return 65
    measurement_path = path_from_file_uri(request.measurement_log.artifact.uri)
    output_path = path_from_file_uri(request.output.uri)
    if response_path.is_relative_to(measurement_path):
        print(
            "surface-mle-service: invalid request: service response "
            "cannot be written inside MeasurementLog.",
            file=sys.stderr,
        )
        return 65
    if response_path == output_path:
        print(
            "surface-mle-service: invalid request: service response and "
            "estimator output paths must differ.",
            file=sys.stderr,
        )
        return 65
    capabilities = service_capabilities()
    try:
        response = _successful_response(request, capabilities)
    except Exception as exc:
        response = _failed_response(request, capabilities, exc)
        write_new_file(response_path, response.to_json_bytes())
        print(f"surface-mle-service: {response.error.message}", file=sys.stderr)
        return 1
    write_new_file(response_path, response.to_json_bytes())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run exactly the shared capabilities or execute service invocation."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    try:
        invocation = parse_service_argv(arguments)
        validate_new_file_path(invocation.response_path)
        if invocation.verb == "capabilities":
            write_new_file(
                invocation.response_path,
                service_capabilities().to_json_bytes(),
            )
            return 0
        assert invocation.request_path is not None
        return _execute(invocation.request_path, invocation.response_path)
    except (EstimatorServiceContractError, MLEServiceError, OSError) as exc:
        print(f"surface-mle-service: {exc}", file=sys.stderr)
        return 64


__all__ = [
    "ESTIMATE_OPERATION",
    "ESTIMATOR_FAMILY",
    "MEASUREMENT_LOG_CONTRACT",
    "MLE_CONFIG_CONTRACT",
    "MLE_RESULT_CONTRACT",
    "main",
    "service_capabilities",
]


if __name__ == "__main__":
    raise SystemExit(main())
