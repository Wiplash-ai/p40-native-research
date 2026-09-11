"""SQLite state for the Stage-0 controller.

The store owns task, branch, attempt, and evidence metadata. Evidence is
append-only; an executor cannot mark its own branch promoted.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SwarmStateError(ValueError):
    """A state transition violates the bounded controller contract."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SwarmStore:
    """Small, local, transactional controller store."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # S0's loopback HTTP server owns request serialization. Disabling the
        # Python thread-affinity check permits construction and serving on
        # different threads in tests without claiming concurrent writers.
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self.db.close()

    def _migrate(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
              id TEXT PRIMARY KEY,
              objective TEXT NOT NULL,
              repo_path TEXT NOT NULL,
              revision TEXT NOT NULL,
              max_branches INTEGER NOT NULL CHECK(max_branches BETWEEN 1 AND 8),
              acceptance_json TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('created','planned','running','complete','failed')),
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS branches (
              id TEXT PRIMARY KEY,
              task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE RESTRICT,
              hypothesis TEXT NOT NULL,
              role TEXT NOT NULL,
              model_profile TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('queued','running','evaluating','promoted','pruned')),
              created_at TEXT NOT NULL,
              promoted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
              id TEXT PRIMARY KEY,
              branch_id TEXT NOT NULL REFERENCES branches(id) ON DELETE RESTRICT,
              ordinal INTEGER NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('created','completed','failed','cancelled')),
              worktree_path TEXT,
              created_at TEXT NOT NULL,
              finished_at TEXT,
              UNIQUE(branch_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS evidence (
              id TEXT PRIMARY KEY,
              attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE RESTRICT,
              kind TEXT NOT NULL CHECK(kind IN ('test','build','benchmark','static','harness','model')),
              command_profile TEXT NOT NULL,
              exit_code INTEGER NOT NULL,
              duration_ms INTEGER NOT NULL CHECK(duration_ms >= 0),
              stdout_sha256 TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              artifacts_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        for field in ("acceptance_json", "metrics_json", "artifacts_json"):
            if field in item:
                item[field.removesuffix("_json")] = json.loads(item.pop(field))
        return item

    def create_task(
        self, *, objective: str, repo_path: str, revision: str, max_branches: int,
        acceptance: list[str],
    ) -> dict[str, Any]:
        if not objective.strip() or not revision.strip():
            raise SwarmStateError("objective and revision are required")
        if not Path(repo_path).is_absolute():
            raise SwarmStateError("repo_path must be absolute")
        if not 1 <= max_branches <= 8:
            raise SwarmStateError("max_branches must be 1..8")
        task_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, 'created', ?)",
            (task_id, objective.strip(), repo_path, revision.strip(), max_branches,
             json.dumps(acceptance, sort_keys=True), now()),
        )
        self.db.commit()
        return self.task(task_id)

    def task(self, task_id: str) -> dict[str, Any]:
        task = self._row(self.db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
        if task is None:
            raise KeyError(task_id)
        task["branches"] = [self._row(row) for row in self.db.execute(
            "SELECT * FROM branches WHERE task_id = ? ORDER BY created_at, id", (task_id,)
        )]
        return task

    def create_branch(
        self, *, task_id: str, hypothesis: str, role: str, model_profile: str,
    ) -> dict[str, Any]:
        task = self.task(task_id)
        if task["status"] in {"complete", "failed"}:
            raise SwarmStateError("task is closed")
        if len(task["branches"]) >= task["max_branches"]:
            raise SwarmStateError("task branch budget exhausted")
        if not hypothesis.strip() or not role.strip() or not model_profile.strip():
            raise SwarmStateError("hypothesis, role, and model_profile are required")
        branch_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO branches VALUES (?, ?, ?, ?, ?, 'queued', ?, NULL)",
            (branch_id, task_id, hypothesis.strip(), role.strip(), model_profile.strip(), now()),
        )
        self.db.execute("UPDATE tasks SET status = 'planned' WHERE id = ?", (task_id,))
        self.db.commit()
        return self.branch(branch_id)

    def branch(self, branch_id: str) -> dict[str, Any]:
        branch = self._row(self.db.execute("SELECT * FROM branches WHERE id = ?", (branch_id,)).fetchone())
        if branch is None:
            raise KeyError(branch_id)
        return branch

    def create_attempt(self, branch_id: str, worktree_path: str | None = None) -> dict[str, Any]:
        branch = self.branch(branch_id)
        if branch["status"] not in {"queued", "running", "evaluating"}:
            raise SwarmStateError("branch cannot receive another attempt")
        ordinal = self.db.execute(
            "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM attempts WHERE branch_id = ?", (branch_id,)
        ).fetchone()[0]
        attempt_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO attempts VALUES (?, ?, ?, 'created', ?, ?, NULL)",
            (attempt_id, branch_id, ordinal, worktree_path, now()),
        )
        self.db.execute("UPDATE branches SET status = 'running' WHERE id = ?", (branch_id,))
        self.db.commit()
        return self.attempt(attempt_id)

    def attempt(self, attempt_id: str) -> dict[str, Any]:
        attempt = self._row(self.db.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone())
        if attempt is None:
            raise KeyError(attempt_id)
        return attempt

    def append_evidence(
        self, *, attempt_id: str, kind: str, command_profile: str, exit_code: int,
        duration_ms: int, stdout_sha256: str, metrics: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        attempt = self.attempt(attempt_id)
        if attempt["status"] != "created":
            raise SwarmStateError("evidence can only be appended to an open attempt")
        if len(stdout_sha256) != 64:
            raise SwarmStateError("stdout_sha256 must be a SHA-256 hex digest")
        evidence_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (evidence_id, attempt_id, kind, command_profile, exit_code, duration_ms,
             stdout_sha256, json.dumps(metrics or {}, sort_keys=True),
             json.dumps(artifacts or [], sort_keys=True), now()),
        )
        self.db.commit()
        return self.evidence(evidence_id)

    def evidence(self, evidence_id: str) -> dict[str, Any]:
        record = self._row(self.db.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone())
        if record is None:
            raise KeyError(evidence_id)
        return record

    def finish_attempt(self, attempt_id: str, status: str) -> dict[str, Any]:
        if status not in {"completed", "failed", "cancelled"}:
            raise SwarmStateError("invalid attempt final status")
        attempt = self.attempt(attempt_id)
        if attempt["status"] != "created":
            raise SwarmStateError("attempt is already closed")
        branch_status = "evaluating" if status == "completed" else "pruned"
        self.db.execute("UPDATE attempts SET status = ?, finished_at = ? WHERE id = ?", (status, now(), attempt_id))
        self.db.execute("UPDATE branches SET status = ? WHERE id = ?", (branch_status, attempt["branch_id"]))
        self.db.commit()
        return self.attempt(attempt_id)

    def branch_evidence(self, branch_id: str) -> list[dict[str, Any]]:
        return [self._row(row) for row in self.db.execute(
            """SELECT evidence.* FROM evidence JOIN attempts ON attempts.id = evidence.attempt_id
               WHERE attempts.branch_id = ? ORDER BY evidence.created_at, evidence.id""",
            (branch_id,),
        )]

    def promote(self, branch_id: str) -> dict[str, Any]:
        branch = self.branch(branch_id)
        if branch["status"] != "evaluating":
            raise SwarmStateError("only evaluated branches can be promoted")
        evidence = self.branch_evidence(branch_id)
        passed = {record["kind"] for record in evidence if record["exit_code"] == 0}
        if not {"test", "static"}.issubset(passed):
            raise SwarmStateError("promotion requires passing test and static evidence")
        self.db.execute(
            "UPDATE branches SET status = 'promoted', promoted_at = ? WHERE id = ?", (now(), branch_id)
        )
        self.db.commit()
        return self.branch(branch_id)
