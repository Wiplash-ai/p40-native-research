"""Validate and apply one exact in-file replacement proposed by an executor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class EditError(ValueError):
    """A text-edit proposal violates the bounded executor contract."""


MAX_EDIT_TEXT_CHARS = 12_000


@dataclass(frozen=True)
class ExactTextEdit:
    path: str
    expected_text: str
    replacement_text: str


def safe_relative_path(path: str) -> bool:
    parsed = PurePosixPath(path)
    return bool(path) and not parsed.is_absolute() and ".git" not in parsed.parts and ".." not in parsed.parts


def validate_exact_text_edit(
    *, path: object, expected_text: object, replacement_text: object, allowed_files: dict[str, str],
) -> ExactTextEdit:
    if not isinstance(path, str) or not safe_relative_path(path) or path not in allowed_files:
        raise EditError("edit path is outside the supplied source context")
    if not isinstance(expected_text, str) or not isinstance(replacement_text, str):
        raise EditError("edit texts must be strings")
    if not expected_text or not replacement_text or expected_text == replacement_text:
        raise EditError("edit must replace non-empty, distinct text")
    if len(expected_text) > MAX_EDIT_TEXT_CHARS or len(replacement_text) > MAX_EDIT_TEXT_CHARS:
        raise EditError("edit text exceeds the bounded text contract")
    if allowed_files[path].count(expected_text) != 1:
        raise EditError("expected text must occur exactly once in supplied source")
    return ExactTextEdit(path=path, expected_text=expected_text, replacement_text=replacement_text)


def apply_exact_text_edit(*, worktree: str, edit: ExactTextEdit) -> None:
    root = Path(worktree).resolve()
    target = (root / edit.path).resolve()
    if root not in target.parents or not target.is_file():
        raise EditError("edit target escapes the disposable worktree")
    current = target.read_text()
    if current.count(edit.expected_text) != 1:
        raise EditError("worktree no longer matches the validated source context")
    target.write_text(current.replace(edit.expected_text, edit.replacement_text, 1))
