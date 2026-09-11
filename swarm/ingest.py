"""Trusted-artifact ingestion for bounded executor attempts.

The controller does not accept arbitrary agent claims over HTTP. This module
accepts the result object emitted by the controller-owned corpus runner and
maps its baseline/model/validation facts into append-only store evidence.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .store import SwarmStateError, SwarmStore


class IngestError(ValueError):
    """A purported bounded-run artifact does not meet the ingestion contract."""


PROFILES = {"git-diff-check": "static", "python-unittest": "test"}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _result(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IngestError(f"{label} must be an object")
    return value


def ingest_corpus_attempt(
    *, store: SwarmStore, branch_id: str, result: dict[str, Any], worktree_path: str | None = None,
) -> dict[str, Any]:
    """Append a verified corpus result and close its attempt.

    ``passed`` is checked against the controller-required baseline/static/test
    conditions; it is never accepted as an independent assertion.
    """
    if result.get("kind") not in {
        "isolated-controller-rendered-text-edit", "isolated-controller-rendered-text-edit-corpus",
    }:
        raise IngestError("artifact kind is not a bounded text-edit result")
    baseline = _result(result.get("baseline"), "baseline")
    if not isinstance(baseline.get("exit_code"), int):
        raise IngestError("baseline exit_code is required")
    validation_raw = result.get("validation")
    validation = validation_raw if isinstance(validation_raw, dict) else {}
    derived_pass = baseline["exit_code"] != 0
    checked: dict[str, dict[str, Any]] = {}
    for profile, kind in PROFILES.items():
        record = validation.get(profile)
        if not isinstance(record, dict) or not isinstance(record.get("exit_code"), int):
            derived_pass = False
            continue
        checked[profile] = record
        derived_pass = derived_pass and record["exit_code"] == 0
    if result.get("passed") is not derived_pass:
        raise IngestError("artifact passed flag disagrees with required validation")
    attempt = store.create_attempt(branch_id, worktree_path=worktree_path)
    store.append_evidence(
        attempt_id=attempt["id"], kind="harness", command_profile="baseline/python-unittest", exit_code=0,
        duration_ms=int(baseline.get("duration_ms", 0)),
        stdout_sha256=str(baseline.get("stdout_sha256", digest(baseline))),
        metrics={"phase": "baseline", "observed_exit_code": baseline["exit_code"]},
    )
    proposal = result.get("proposal")
    if isinstance(proposal, dict) and isinstance(proposal.get("response_sha256"), str):
        store.append_evidence(
            attempt_id=attempt["id"], kind="model", command_profile="schema-text-edit", exit_code=0,
            duration_ms=0, stdout_sha256=proposal["response_sha256"],
            metrics={"model": result.get("model", "unknown"), "task": result.get("task", "unknown")},
        )
    else:
        store.append_evidence(
            attempt_id=attempt["id"], kind="harness", command_profile="executor-contract", exit_code=1,
            duration_ms=0, stdout_sha256=digest(result.get("error")),
            metrics={"phase": "executor-contract", "error": result.get("error")},
        )
    for profile, record in checked.items():
        sha = record.get("stdout_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise IngestError(f"{profile} lacks a SHA-256 output digest")
        store.append_evidence(
            attempt_id=attempt["id"], kind=PROFILES[profile], command_profile=profile,
            exit_code=record["exit_code"], duration_ms=int(record.get("duration_ms", 0)),
            stdout_sha256=sha, metrics={"changed_files": record.get("changed_files", [])},
        )
    return store.finish_attempt(attempt["id"], "completed" if derived_pass else "failed")
