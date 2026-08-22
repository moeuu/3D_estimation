"""Bind estimator lineage to runtime-owned MeasurementLog record digests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from runtime.measurement_log import MeasurementLogRecord
from runtime.prefix import measurement_records_digest
from runtime.provenance import DigestIdentity


def covered_records_lineage(
    records: Sequence[MeasurementLogRecord],
) -> dict[str, object]:
    """Return the active algorithm-bound digest and its transitional alias."""
    digest = measurement_records_digest(records)
    return {
        "covered_records_digest": digest.to_payload(),
        "covered_records_sha256": digest.sha256,
    }


def validate_covered_records_lineage(
    payload: Mapping[str, object],
    records: Sequence[MeasurementLogRecord],
    *,
    location: str,
) -> DigestIdentity:
    """Validate both active digest fields without legacy self-downgrade."""
    expected = measurement_records_digest(records)
    raw_digest = payload.get("covered_records_digest")
    if not isinstance(raw_digest, Mapping):
        raise ValueError(f"{location}.covered_records_digest must be an object.")
    try:
        actual = DigestIdentity.from_payload(raw_digest)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{location}.covered_records_digest is invalid."
        ) from exc
    if actual != expected:
        raise ValueError(
            f"{location}.covered_records_digest differs from runtime records."
        )
    if payload.get("covered_records_sha256") != expected.sha256:
        raise ValueError(
            f"{location}.covered_records_sha256 must equal the active digest."
        )
    return expected


__all__ = [
    "covered_records_lineage",
    "validate_covered_records_lineage",
]
