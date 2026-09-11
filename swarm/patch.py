"""Validate and apply a small executor-proposed unified patch in a worktree."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import PurePosixPath


class PatchError(ValueError):
    """A patch is outside the controlled corpus patch contract."""


MAX_PATCH_CHARS = 32 * 1024
_DIFF_HEADER = re.compile(r"^diff --git a/([^\s]+) b/([^\s]+)$")
_FILE_HEADER = re.compile(r"^(---|\+\+\+) ([ab])/([^\s]+)$")


@dataclass(frozen=True)
class PatchApplyResult:
    stdout: str
    stderr: str


def _safe_path(path: str) -> bool:
    parsed = PurePosixPath(path)
    return bool(path) and not parsed.is_absolute() and ".git" not in parsed.parts and ".." not in parsed.parts


def validate_unified_patch(patch: str) -> str:
    if not isinstance(patch, str) or not patch.strip():
        raise PatchError("patch is required")
    if len(patch) > MAX_PATCH_CHARS or "\x00" in patch:
        raise PatchError("patch exceeds the bounded text contract")
    if not patch.startswith("diff --git "):
        raise PatchError("patch must start with a unified diff header")
    if any(marker in patch for marker in (
        "GIT binary patch", "120000", "old mode ", "new mode ", "deleted file mode ", "rename from ", "rename to ",
    )):
        raise PatchError("binary, mode, symlink, and rename patches are not allowed")
    headers = 0
    for line in patch.splitlines():
        diff = _DIFF_HEADER.match(line)
        if diff:
            if not all(_safe_path(value) for value in diff.groups()):
                raise PatchError("patch path escapes the worktree")
            headers += 1
        file_header = _FILE_HEADER.match(line)
        if file_header and not _safe_path(file_header.group(3)):
            raise PatchError("patch file header escapes the worktree")
    if headers < 1:
        raise PatchError("patch has no valid file diff")
    return patch


def apply_unified_patch(*, worktree: str, patch: str) -> PatchApplyResult:
    checked_patch = validate_unified_patch(patch)
    check = subprocess.run(
        ["git", "apply", "--check", "--whitespace=error", "--"], cwd=worktree, input=checked_patch,
        text=True, capture_output=True, check=False, timeout=30,
    )
    if check.returncode:
        raise PatchError(f"git apply --check failed: {check.stderr.strip()}")
    apply = subprocess.run(
        ["git", "apply", "--whitespace=error", "--"], cwd=worktree, input=checked_patch,
        text=True, capture_output=True, check=False, timeout=30,
    )
    if apply.returncode:
        raise PatchError(f"git apply failed after check: {apply.stderr.strip()}")
    return PatchApplyResult(stdout=apply.stdout, stderr=apply.stderr)
