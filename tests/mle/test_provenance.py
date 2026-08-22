"""Regression tests for repository and estimator provenance."""

from __future__ import annotations

from pathlib import Path

from three_d_estimation.provenance import repository_commit


def test_repository_commit_reads_linked_worktree_common_reference(
    tmp_path: Path,
) -> None:
    """Linked worktrees must resolve branch refs stored in the common Git dir."""
    repository = tmp_path / "checkout"
    common_directory = tmp_path / "project.git"
    worktree_directory = common_directory / "worktrees" / "checkout"
    reference_path = common_directory / "refs" / "heads" / "review" / "mle"
    repository.mkdir()
    worktree_directory.mkdir(parents=True)
    reference_path.parent.mkdir(parents=True)
    repository.joinpath(".git").write_text(
        f"gitdir: {worktree_directory}\n",
        encoding="utf-8",
    )
    worktree_directory.joinpath("HEAD").write_text(
        "ref: refs/heads/review/mle\n",
        encoding="utf-8",
    )
    worktree_directory.joinpath("commondir").write_text(
        "../..\n",
        encoding="utf-8",
    )
    expected = "a" * 40
    reference_path.write_text(f"{expected}\n", encoding="utf-8")

    assert repository_commit(repository) == expected
