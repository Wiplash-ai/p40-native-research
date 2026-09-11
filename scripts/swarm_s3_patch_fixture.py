#!/usr/bin/env python3
"""Run one schema-bound executor patch against an isolated fixture repository.

This is an S3 research driver, not a production controller endpoint.  It keeps
the model's only effect to a unified patch applied in a disposable worktree,
then records the baseline and validation evidence as JSON.
"""
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

from swarm.executor import OllamaPlanClient  # noqa: E402
from swarm.harness import BoundedHarness  # noqa: E402
from swarm.patch import apply_unified_patch  # noqa: E402


def run(argv: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(argv)}: {result.stderr.strip()}")


def create_fixture(root: Path) -> Path:
    repo = root / "fixture-repo"
    repo.mkdir()
    (repo / "calculator.py").write_text(
        "def is_even(value):\n"
        "    return value % 2 == 1\n"
    )
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        "import unittest\n\n"
        "from calculator import is_even\n\n"
        "class ParityTest(unittest.TestCase):\n"
        "    def test_even_values(self):\n"
        "        self.assertTrue(is_even(2))\n"
        "        self.assertTrue(is_even(0))\n\n"
        "    def test_odd_values(self):\n"
        "        self.assertFalse(is_even(1))\n"
        "        self.assertFalse(is_even(-3))\n"
    )
    run(["git", "init"], cwd=repo)
    run(["git", "add", "."], cwd=repo)
    run([
        "git", "-c", "user.name=Swarm Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-m", "fixture: inverted parity predicate",
    ], cwd=repo)
    return repo


def serialise_command(result) -> dict[str, object]:
    return {
        "profile": result.command_profile,
        "exit_code": result.exit_code,
        "duration_ms": result.duration_ms,
        "stdout_sha256": result.stdout_sha256,
        "changed_files": result.changed_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Private Ollama /api/chat endpoint")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.workspace_root.resolve()
    output = args.output.resolve()
    if root.exists() or output.exists():
        raise SystemExit("workspace root and output must not already exist")
    root.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repo = create_fixture(root)
    harness = BoundedHarness(
        worktree_root=root / "worktrees",
        artifact_root=root / "artifacts",
    )
    baseline_worktree = harness.prepare_worktree(
        repo_path=repo, revision="HEAD", task_id=str(uuid.uuid4()), branch_id=str(uuid.uuid4()),
    )
    baseline = harness.run_profile(
        worktree=baseline_worktree, attempt_id=str(uuid.uuid4()), profile="python-unittest",
    )
    files = {
        "calculator.py": (repo / "calculator.py").read_text(),
        "tests/test_calculator.py": (repo / "tests" / "test_calculator.py").read_text(),
    }
    response = OllamaPlanClient(url=args.url, timeout_s=240).request_patch(
        model=args.model,
        objective="Correct the parity implementation so the supplied unit tests pass.",
        branch_hypothesis="The modulo equality is inverted.",
        files=files,
    )
    proposal = response.proposal
    candidate_worktree = harness.prepare_worktree(
        repo_path=repo, revision="HEAD", task_id=str(uuid.uuid4()), branch_id=str(uuid.uuid4()),
    )
    apply_unified_patch(worktree=str(candidate_worktree), patch=proposal.patch, allowed_paths=set(files))
    validation = harness.run_profile(
        worktree=candidate_worktree,
        attempt_id=str(uuid.uuid4()), profile=proposal.validation_profile,
    )
    payload = {
        "schema_version": 1,
        "stage": "S3",
        "kind": "isolated-schema-bound-executor-patch",
        "started_unix_s": started,
        "finished_unix_s": time.time(),
        "model": response.model,
        "endpoint": args.url,
        "fixture": "inverted-parity-predicate",
        "baseline": serialise_command(baseline),
        "proposal": {
            "summary": proposal.summary,
            "validation_profile": proposal.validation_profile,
            "patch": proposal.patch,
            "patch_sha256": hashlib.sha256(proposal.patch.encode()).hexdigest(),
            "response_sha256": response.response_sha256,
            "prompt_eval_count": response.prompt_eval_count,
            "eval_count": response.eval_count,
        },
        "validation": serialise_command(validation),
        "passed": baseline.exit_code != 0 and validation.exit_code == 0,
        "workspace_root": str(root),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "passed": payload["passed"],
        "baseline_exit": baseline.exit_code, "validation_exit": validation.exit_code,
    }, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
