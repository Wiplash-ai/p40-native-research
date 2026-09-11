"""Schema-bounded planner requests for an executor candidate.

This module gives a small model only a planning surface. It cannot emit a
shell command or select an executable; the later controller may map its
``validation_profile`` only to a harness-owned fixed command profile.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from .edit import ExactTextEdit, validate_exact_text_edit
from .harness import COMMAND_PROFILES
from .patch import validate_unified_patch


class ExecutorPlanError(ValueError):
    """A model response does not satisfy the executor planning contract."""


PLAN_FIELDS = ("hypothesis", "proposed_change", "validation_profile", "expected_signal", "stop_condition")
PATCH_FIELDS = ("summary", "patch", "validation_profile")
TEXT_EDIT_FIELDS = ("summary", "path", "expected_text", "replacement_text", "validation_profile")
MAX_FIELD_CHARS = 1_200


@dataclass(frozen=True)
class ExecutorPlan:
    hypothesis: str
    proposed_change: str
    validation_profile: str
    expected_signal: str
    stop_condition: str


@dataclass(frozen=True)
class PlanResponse:
    plan: ExecutorPlan
    model: str
    response_sha256: str
    prompt_eval_count: int
    eval_count: int


@dataclass(frozen=True)
class PatchProposal:
    summary: str
    patch: str
    validation_profile: str


@dataclass(frozen=True)
class PatchResponse:
    proposal: PatchProposal
    model: str
    response_sha256: str
    prompt_eval_count: int
    eval_count: int


@dataclass(frozen=True)
class TextEditProposal:
    summary: str
    edit: ExactTextEdit
    validation_profile: str


@dataclass(frozen=True)
class TextEditResponse:
    proposal: TextEditProposal
    model: str
    response_sha256: str
    prompt_eval_count: int
    eval_count: int


def plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(PLAN_FIELDS),
        "properties": {
            "hypothesis": {"type": "string", "minLength": 8, "maxLength": MAX_FIELD_CHARS},
            "proposed_change": {"type": "string", "minLength": 8, "maxLength": MAX_FIELD_CHARS},
            "validation_profile": {"type": "string", "enum": sorted(COMMAND_PROFILES)},
            "expected_signal": {"type": "string", "minLength": 8, "maxLength": MAX_FIELD_CHARS},
            "stop_condition": {"type": "string", "minLength": 8, "maxLength": MAX_FIELD_CHARS},
        },
    }


def patch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(PATCH_FIELDS),
        "properties": {
            "summary": {"type": "string", "minLength": 8, "maxLength": MAX_FIELD_CHARS},
            # Ollama lowers maxLength into GBNF repetition.  Its parser rejects
            # the 32K bound before model execution.  The host-side validator
            # retains MAX_PATCH_CHARS, so omit only the transport-level upper
            # bound here.
            "patch": {"type": "string", "minLength": 40},
            "validation_profile": {"type": "string", "enum": sorted(COMMAND_PROFILES)},
        },
    }


def text_edit_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(TEXT_EDIT_FIELDS),
        "properties": {
            "summary": {"type": "string", "minLength": 8, "maxLength": MAX_FIELD_CHARS},
            "path": {"type": "string", "minLength": 1, "maxLength": 512},
            # Upper bounds remain in the host-side validator: Ollama lowers
            # JSON maxLength into GBNF repetitions that reject large values.
            "expected_text": {"type": "string", "minLength": 1},
            "replacement_text": {"type": "string", "minLength": 1},
            "validation_profile": {"type": "string", "enum": sorted(COMMAND_PROFILES)},
        },
    }


def validate_plan(payload: Any) -> ExecutorPlan:
    if not isinstance(payload, dict) or set(payload) != set(PLAN_FIELDS):
        raise ExecutorPlanError("plan must contain exactly the fixed planning fields")
    values: dict[str, str] = {}
    for field in PLAN_FIELDS:
        value = payload[field]
        if not isinstance(value, str):
            raise ExecutorPlanError(f"{field} must be a string")
        cleaned = value.strip()
        if not cleaned or len(cleaned) > MAX_FIELD_CHARS:
            raise ExecutorPlanError(f"{field} is empty or too long")
        if any(ord(char) < 32 and char not in "\n\t" for char in cleaned):
            raise ExecutorPlanError(f"{field} contains a control character")
        values[field] = cleaned
    if values["validation_profile"] not in COMMAND_PROFILES:
        raise ExecutorPlanError("plan selected a non-allowlisted validation profile")
    return ExecutorPlan(**values)


def validate_patch_proposal(
    payload: Any, *, allowed_paths: set[str] | None = None,
) -> PatchProposal:
    if not isinstance(payload, dict) or set(payload) != set(PATCH_FIELDS):
        raise ExecutorPlanError("patch proposal must contain exactly summary, patch, and validation_profile")
    summary = payload["summary"]
    profile = payload["validation_profile"]
    if not isinstance(summary, str) or not 8 <= len(summary.strip()) <= MAX_FIELD_CHARS:
        raise ExecutorPlanError("patch summary is invalid")
    if not isinstance(profile, str) or profile not in COMMAND_PROFILES:
        raise ExecutorPlanError("patch selected a non-allowlisted validation profile")
    try:
        patch = validate_unified_patch(payload["patch"], allowed_paths=allowed_paths)
    except (TypeError, ValueError) as exc:
        raise ExecutorPlanError(str(exc)) from exc
    return PatchProposal(summary=summary.strip(), patch=patch, validation_profile=profile)


def validate_text_edit_proposal(payload: Any, *, allowed_files: dict[str, str]) -> TextEditProposal:
    if not isinstance(payload, dict) or set(payload) != set(TEXT_EDIT_FIELDS):
        raise ExecutorPlanError("text edit must contain exactly the fixed edit fields")
    summary = payload["summary"]
    profile = payload["validation_profile"]
    if not isinstance(summary, str) or not 8 <= len(summary.strip()) <= MAX_FIELD_CHARS:
        raise ExecutorPlanError("text edit summary is invalid")
    if not isinstance(profile, str) or profile not in COMMAND_PROFILES:
        raise ExecutorPlanError("text edit selected a non-allowlisted validation profile")
    try:
        edit = validate_exact_text_edit(
            path=payload["path"], expected_text=payload["expected_text"],
            replacement_text=payload["replacement_text"], allowed_files=allowed_files,
        )
    except (TypeError, ValueError) as exc:
        raise ExecutorPlanError(str(exc)) from exc
    return TextEditProposal(summary=summary.strip(), edit=edit, validation_profile=profile)


def executor_messages(*, objective: str, branch_hypothesis: str, role: str) -> list[dict[str, str]]:
    if not objective.strip() or not branch_hypothesis.strip() or not role.strip():
        raise ExecutorPlanError("objective, branch_hypothesis, and role are required")
    return [
        {
            "role": "system",
            "content": (
                "You are a bounded software-engineering search executor. Return only JSON matching the supplied "
                "schema. Plan one small falsifiable change. Never include shell commands, code, credentials, or "
                "claims that work was executed. The controller owns all execution."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "objective": objective.strip(), "branch_hypothesis": branch_hypothesis.strip(), "role": role.strip(),
                "available_validation_profiles": sorted(COMMAND_PROFILES),
            }, sort_keys=True),
        },
    ]


def patch_messages(*, objective: str, branch_hypothesis: str, files: dict[str, str]) -> list[dict[str, str]]:
    if not objective.strip() or not branch_hypothesis.strip() or not files:
        raise ExecutorPlanError("objective, branch_hypothesis, and files are required")
    safe_files = []
    for path, content in sorted(files.items()):
        if not isinstance(path, str) or not isinstance(content, str) or not path or path.startswith("/") or ".." in path.split("/"):
            raise ExecutorPlanError("source context contains an unsafe path")
        if len(content) > 12_000:
            raise ExecutorPlanError("individual source context file is too large")
        safe_files.append({"path": path, "content": content})
    return [
        {
            "role": "system",
            "content": (
                "You are a bounded software-engineering executor. Return only JSON matching the supplied schema. "
                "Propose one small unified diff for the provided files. Do not emit shell commands, prose outside JSON, "
                "credentials, file deletions, renames, symlinks, or paths outside the provided worktree."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "objective": objective.strip(), "branch_hypothesis": branch_hypothesis.strip(),
                "files": safe_files, "validation_profiles": sorted(COMMAND_PROFILES),
            }, sort_keys=True),
        },
    ]


def text_edit_messages(*, objective: str, branch_hypothesis: str, files: dict[str, str]) -> list[dict[str, str]]:
    if not objective.strip() or not branch_hypothesis.strip() or not files:
        raise ExecutorPlanError("objective, branch_hypothesis, and files are required")
    safe_files = []
    for path, content in sorted(files.items()):
        if not isinstance(path, str) or not isinstance(content, str) or not path or path.startswith("/") or ".." in path.split("/"):
            raise ExecutorPlanError("source context contains an unsafe path")
        if len(content) > MAX_FIELD_CHARS * 10:
            raise ExecutorPlanError("individual source context file is too large")
        safe_files.append({"path": path, "content": content})
    return [
        {
            "role": "system",
            "content": (
                "You are a bounded software-engineering executor. Return only JSON matching the supplied schema. "
                "Propose exactly one in-file text replacement. expected_text must occur exactly once in the supplied "
                "file. Do not emit shell commands, a unified diff, credentials, file deletions, renames, symlinks, "
                "or paths outside the provided worktree."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "objective": objective.strip(), "branch_hypothesis": branch_hypothesis.strip(),
                "files": safe_files, "validation_profiles": sorted(COMMAND_PROFILES),
            }, sort_keys=True),
        },
    ]


def post_chat(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 -- configured private endpoint
        decoded = json.loads(response.read())
    if not isinstance(decoded, dict):
        raise ExecutorPlanError("Ollama returned a non-object response")
    return decoded


class OllamaPlanClient:
    """A private Ollama client with fixed context and schema-only output."""

    def __init__(self, url: str = "http://172.17.0.1:11434/api/chat", timeout_s: int = 180):
        self.url = url
        self.timeout_s = timeout_s

    def request_plan(
        self, *, model: str, objective: str, branch_hypothesis: str, role: str,
    ) -> PlanResponse:
        if not model.strip():
            raise ExecutorPlanError("model is required")
        response = post_chat(self.url, {
            "model": model.strip(),
            "messages": executor_messages(
                objective=objective, branch_hypothesis=branch_hypothesis, role=role,
            ),
            "format": plan_schema(),
            "stream": False,
            "think": False,
            "keep_alive": "0s",
            "options": {"num_ctx": 4096, "seed": 42, "temperature": 0},
        }, self.timeout_s)
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ExecutorPlanError("Ollama response lacks message content")
        try:
            plan = validate_plan(json.loads(content))
        except json.JSONDecodeError as exc:
            raise ExecutorPlanError("Ollama plan is not strict JSON") from exc
        return PlanResponse(
            plan=plan,
            model=model.strip(),
            response_sha256=hashlib.sha256(content.encode()).hexdigest(),
            prompt_eval_count=int(response.get("prompt_eval_count", 0)),
            eval_count=int(response.get("eval_count", 0)),
        )

    def request_patch(
        self, *, model: str, objective: str, branch_hypothesis: str, files: dict[str, str],
    ) -> PatchResponse:
        if not model.strip():
            raise ExecutorPlanError("model is required")
        response = post_chat(self.url, {
            "model": model.strip(),
            "messages": patch_messages(
                objective=objective, branch_hypothesis=branch_hypothesis, files=files,
            ),
            "format": patch_schema(),
            "stream": False,
            "think": False,
            "keep_alive": "0s",
            "options": {"num_ctx": 4096, "seed": 42, "temperature": 0},
        }, self.timeout_s)
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ExecutorPlanError("Ollama response lacks message content")
        try:
            proposal = validate_patch_proposal(json.loads(content), allowed_paths=set(files))
        except json.JSONDecodeError as exc:
            raise ExecutorPlanError("Ollama patch proposal is not strict JSON") from exc
        return PatchResponse(
            proposal=proposal,
            model=model.strip(),
            response_sha256=hashlib.sha256(content.encode()).hexdigest(),
            prompt_eval_count=int(response.get("prompt_eval_count", 0)),
            eval_count=int(response.get("eval_count", 0)),
        )

    def request_text_edit(
        self, *, model: str, objective: str, branch_hypothesis: str, files: dict[str, str],
    ) -> TextEditResponse:
        if not model.strip():
            raise ExecutorPlanError("model is required")
        response = post_chat(self.url, {
            "model": model.strip(),
            "messages": text_edit_messages(
                objective=objective, branch_hypothesis=branch_hypothesis, files=files,
            ),
            "format": text_edit_schema(),
            "stream": False,
            "think": False,
            "keep_alive": "0s",
            "options": {"num_ctx": 4096, "seed": 42, "temperature": 0},
        }, self.timeout_s)
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ExecutorPlanError("Ollama response lacks message content")
        try:
            proposal = validate_text_edit_proposal(json.loads(content), allowed_files=files)
        except json.JSONDecodeError as exc:
            raise ExecutorPlanError("Ollama text edit is not strict JSON") from exc
        return TextEditResponse(
            proposal=proposal,
            model=model.strip(),
            response_sha256=hashlib.sha256(content.encode()).hexdigest(),
            prompt_eval_count=int(response.get("prompt_eval_count", 0)),
            eval_count=int(response.get("eval_count", 0)),
        )
