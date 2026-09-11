"""Narrow Git-worktree and command-profile harness for executor attempts."""
from __future__ import annotations

import hashlib
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class HarnessError(ValueError):
    """A requested harness action falls outside the bounded S0 surface."""


COMMAND_PROFILES: dict[str, tuple[str, str, tuple[str, ...], int]] = {
    "git-diff-check": ("static", "git", ("diff", "--check"), 30),
    "python-unittest": ("test", "python3", ("-m", "unittest", "discover", "-s", "tests", "-v"), 300),
}


@dataclass(frozen=True)
class CommandResult:
    kind: str
    command_profile: str
    exit_code: int
    duration_ms: int
    stdout_sha256: str
    artifact: str
    changed_files: list[str]


class BoundedHarness:
    """Creates isolated Git worktrees and executes only named profiles."""

    def __init__(self, *, worktree_root: Path, artifact_root: Path):
        self.worktree_root = worktree_root.resolve()
        self.artifact_root = artifact_root.resolve()
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _run(argv: list[str], *, cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout)

    @staticmethod
    def _uuid(value: str, label: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as error:
            raise HarnessError(f"{label} must be a UUID") from error

    def prepare_worktree(self, *, repo_path: Path, revision: str, task_id: str, branch_id: str) -> Path:
        repo = repo_path.resolve()
        if not repo.is_dir():
            raise HarnessError("repository path does not exist")
        checked_task = self._uuid(task_id, "task_id")
        checked_branch = self._uuid(branch_id, "branch_id")
        check = self._run(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"])
        if check.returncode or check.stdout.strip() != "true":
            raise HarnessError("repository path is not a Git worktree")
        revision_check = self._run(["git", "-C", str(repo), "rev-parse", "--verify", f"{revision}^{{commit}}"])
        if revision_check.returncode:
            raise HarnessError("revision is not a commit in the repository")
        target = (self.worktree_root / checked_task / checked_branch).resolve()
        if self.worktree_root not in target.parents:
            raise HarnessError("worktree target escapes configured root")
        if target.exists():
            raise HarnessError("worktree target already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        result = self._run(["git", "-C", str(repo), "worktree", "add", "--detach", str(target), revision], timeout=120)
        if result.returncode:
            raise HarnessError(f"git worktree add failed: {result.stderr.strip()}")
        return target

    def run_profile(self, *, worktree: Path, attempt_id: str, profile: str) -> CommandResult:
        if profile not in COMMAND_PROFILES:
            raise HarnessError(f"unknown command profile: {profile}")
        checked_attempt = self._uuid(attempt_id, "attempt_id")
        target = worktree.resolve()
        if self.worktree_root not in target.parents:
            raise HarnessError("worktree escapes configured root")
        kind, executable, args, timeout = COMMAND_PROFILES[profile]
        started = time.monotonic()
        try:
            result = self._run([executable, *args], cwd=target, timeout=timeout)
            output = result.stdout + result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or "") + (error.stderr or "")
            exit_code = 124
        duration_ms = round((time.monotonic() - started) * 1000)
        artifact_dir = self.artifact_root / checked_attempt
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = artifact_dir / f"{profile}.log"
        artifact.write_text(output)
        changed = self._run(["git", "diff", "--name-only"], cwd=target)
        changed_files = [line for line in changed.stdout.splitlines() if line]
        return CommandResult(
            kind=kind,
            command_profile=profile,
            exit_code=exit_code,
            duration_ms=duration_ms,
            stdout_sha256=hashlib.sha256(output.encode()).hexdigest(),
            artifact=str(artifact),
            changed_files=changed_files,
        )
