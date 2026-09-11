#!/usr/bin/env python3
"""Run one named S5 text-edit corpus task through the bounded executor path."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.edit import apply_exact_text_edit  # noqa: E402
from swarm.executor import OllamaPlanClient  # noqa: E402
from swarm.fixtures import TASKS, get_task  # noqa: E402
from swarm.harness import BoundedHarness  # noqa: E402


def run(argv: list[str], *, cwd: Path) -> None:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(argv)}: {result.stderr.strip()}")


def command_evidence(result) -> dict[str, object]:
    return {
        "profile": result.command_profile,
        "exit_code": result.exit_code,
        "duration_ms": result.duration_ms,
        "stdout_sha256": result.stdout_sha256,
        "changed_files": result.changed_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--task", choices=[task.identifier for task in TASKS], required=True)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.workspace_root.resolve(), args.output.resolve()
    if root.exists() or output.exists():
        raise SystemExit("workspace root and output must not already exist")
    root.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    task = get_task(args.task)
    started = time.time()
    baseline = proposal = validation = None
    error = None
    try:
        repo = root / "fixture-repo"
        repo.mkdir()
        for relative, content in task.files.items():
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        run(["git", "init"], cwd=repo)
        run(["git", "add", "."], cwd=repo)
        run([
            "git", "-c", "user.name=Swarm Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-m", f"fixture: {task.identifier}",
        ], cwd=repo)
        harness = BoundedHarness(worktree_root=root / "worktrees", artifact_root=root / "artifacts")
        baseline_worktree = harness.prepare_worktree(
            repo_path=repo, revision="HEAD", task_id=str(uuid.uuid4()), branch_id=str(uuid.uuid4()),
        )
        baseline = harness.run_profile(
            worktree=baseline_worktree, attempt_id=str(uuid.uuid4()), profile="python-unittest",
        )
        response = OllamaPlanClient(url=args.url, timeout_s=240).request_text_edit(
            model=args.model, objective=task.objective, branch_hypothesis=task.hypothesis, files=task.files,
        )
        edit = response.proposal.edit
        proposal = {
            "summary": response.proposal.summary,
            "validation_profile": response.proposal.validation_profile,
            "path": edit.path,
            "expected_text": edit.expected_text,
            "replacement_text": edit.replacement_text,
            "edit_sha256": hashlib.sha256(
                (edit.path + "\0" + edit.expected_text + "\0" + edit.replacement_text).encode(),
            ).hexdigest(),
            "response_sha256": response.response_sha256,
            "prompt_eval_count": response.prompt_eval_count,
            "eval_count": response.eval_count,
        }
        candidate_worktree = harness.prepare_worktree(
            repo_path=repo, revision="HEAD", task_id=str(uuid.uuid4()), branch_id=str(uuid.uuid4()),
        )
        apply_exact_text_edit(worktree=str(candidate_worktree), edit=edit)
        validation = {}
        for profile in ("git-diff-check", "python-unittest"):
            validation[profile] = command_evidence(harness.run_profile(
                worktree=candidate_worktree, attempt_id=str(uuid.uuid4()), profile=profile,
            ))
    except Exception as exc:
        error = {"kind": type(exc).__name__, "message": str(exc)}
    passed = bool(
        baseline and validation and baseline.exit_code != 0
        and all(result["exit_code"] == 0 for result in validation.values())
    )
    payload = {
        "schema_version": 1,
        "stage": "S5",
        "kind": "isolated-controller-rendered-text-edit-corpus",
        "started_unix_s": started,
        "finished_unix_s": time.time(),
        "model": args.model,
        "endpoint": args.url,
        "task": task.identifier,
        "baseline": command_evidence(baseline) if baseline else None,
        "proposal": proposal,
        "validation": validation,
        "error": error,
        "passed": passed,
        "workspace_root": str(root),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "task": task.identifier, "passed": passed, "error": error}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
