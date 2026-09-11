#!/usr/bin/env python3
"""Create one local evidence-only task from a trusted corpus result artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.ingest import ingest_corpus_attempt  # noqa: E402
from swarm.scoring import rank_task  # noqa: E402
from swarm.store import SwarmStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--objective", default="Evaluate one bounded executor artifact")
    parser.add_argument("--role", default="implementer")
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text())
    if not isinstance(artifact, dict):
        raise SystemExit("artifact must be a JSON object")
    repo_path = str(artifact.get("workspace_root", ""))
    if not Path(repo_path).is_absolute():
        raise SystemExit("artifact must identify an absolute disposable worktree")
    store = SwarmStore(args.db)
    try:
        task = store.create_task(
            objective=args.objective, repo_path=repo_path, revision="artifact-only", max_branches=1,
            acceptance=["git-diff-check", "python-unittest"],
        )
        branch = store.create_branch(
            task_id=task["id"], hypothesis="accept only measured evidence", role=args.role,
            model_profile=str(artifact.get("model", "unknown")),
        )
        attempt = ingest_corpus_attempt(
            store=store, branch_id=branch["id"], result=artifact, worktree_path=repo_path,
        )
        print(json.dumps({
            "task_id": task["id"], "branch_id": branch["id"], "attempt": attempt,
            "ranking": rank_task(store, task["id"]),
        }, sort_keys=True))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
