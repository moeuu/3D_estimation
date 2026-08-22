"""Repository and estimator provenance for standalone surface MLE outputs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _commit_digest(value: str) -> str | None:
    """Return one normalized Git object digest or ``None`` when malformed."""
    candidate = value.strip().lower()
    if len(candidate) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        return None
    return candidate


def _git_reference_directories(git_directory: Path) -> tuple[Path, ...]:
    """Return private and shared reference roots for one Git directory."""
    directories = [git_directory]
    common_file = git_directory / "commondir"
    if common_file.is_file():
        common_text = common_file.read_text(encoding="utf-8").strip()
        if common_text:
            common_directory = Path(common_text)
            if not common_directory.is_absolute():
                common_directory = git_directory / common_directory
            common_directory = common_directory.resolve()
            if common_directory != git_directory:
                directories.append(common_directory)
    return tuple(directories)


def _reference_commit(
    reference: str,
    reference_directories: tuple[Path, ...],
) -> str | None:
    """Resolve one symbolic Git reference from loose or packed storage."""
    reference_path = Path(reference)
    if (
        not reference.startswith("refs/")
        or reference_path.is_absolute()
        or ".." in reference_path.parts
    ):
        return None
    for directory in reference_directories:
        loose = directory / reference_path
        if loose.is_file():
            digest = _commit_digest(loose.read_text(encoding="utf-8"))
            if digest is not None:
                return digest
    for directory in reference_directories:
        packed = directory / "packed-refs"
        if not packed.is_file():
            continue
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")) or not line.strip():
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or fields[1].strip() != reference:
                continue
            digest = _commit_digest(fields[0])
            if digest is not None:
                return digest
    return None


def repository_commit(root: str | Path = _REPOSITORY_ROOT) -> str:
    """Return the local Git commit without invoking Git or another repository."""
    repository = Path(root)
    git_entry = repository / ".git"
    if git_entry.is_file():
        text = git_entry.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            git_directory = Path(text.removeprefix("gitdir:").strip())
            if not git_directory.is_absolute():
                git_directory = repository / git_directory
            git_entry = git_directory.resolve()
    if git_entry.is_dir():
        head = (git_entry / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            reference = head.removeprefix("ref:").strip()
            commit = _reference_commit(
                reference,
                _git_reference_directories(git_entry),
            )
            if commit is not None:
                return commit
        else:
            commit = _commit_digest(head)
            if commit is not None:
                return commit
    return "unknown-standalone-build"


def resolved_mapping_sha256(payload: Mapping[str, object]) -> str:
    """Return a compact canonical SHA-256 for a resolved JSON mapping."""
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def estimator_provenance(
    *,
    variant: str,
    measurement_log_schema_version: int | None = None,
    measurement_run_id: str | None = None,
    measurement_repository_commit: str | None = None,
    resolved_config_sha256: str | None = None,
    forward_model_manifest_sha256: str | None = None,
    measurement_log_sha256: str | None = None,
    config_sha256: str | None = None,
    resolved_estimator_config_sha256: str | None = None,
) -> dict[str, object]:
    """Return mandatory pure-MLE provenance with optional replay identities."""
    normalized_variant = str(variant).strip().lower()
    if normalized_variant not in {"count", "spectral"}:
        raise ValueError("variant must be 'count' or 'spectral'.")
    commit = repository_commit()
    return {
        "estimator_family": "surface_mle",
        "estimator_variant": normalized_variant,
        "candidate_domain": "complete_surface_dictionary",
        "uses_pf_state": False,
        "uses_pf_candidates": False,
        "estimator_repository": "moeuu/radiation-surface-mle-estimator",
        "estimator_commit": commit,
        "repository_commit": commit,
        "measurement_log_schema_version": measurement_log_schema_version,
        "measurement_run_id": measurement_run_id,
        "measurement_repository_commit": measurement_repository_commit,
        "resolved_config_sha256": resolved_config_sha256,
        "forward_model_manifest_sha256": forward_model_manifest_sha256,
        "measurement_log_sha256": measurement_log_sha256,
        "config_sha256": config_sha256,
        "resolved_estimator_config_sha256": resolved_estimator_config_sha256,
    }


__all__ = ["estimator_provenance", "repository_commit", "resolved_mapping_sha256"]
