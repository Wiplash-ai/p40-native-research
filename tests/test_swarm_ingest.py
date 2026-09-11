import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.ingest import IngestError, ingest_corpus_attempt
from swarm.scoring import rank_task
from swarm.store import SwarmStore


SHA = hashlib.sha256(b"output").hexdigest()


def artifact(*, passed: bool, validation: dict | None = None, proposal: dict | None = None) -> dict:
    return {
        "kind": "isolated-controller-rendered-text-edit-corpus",
        "model": "fixture-model",
        "task": "fixture",
        "baseline": {"exit_code": 1, "duration_ms": 1, "stdout_sha256": SHA},
        "proposal": proposal if proposal is not None else {"response_sha256": SHA},
        "validation": validation if validation is not None else {
            "git-diff-check": {"exit_code": 0, "duration_ms": 1, "stdout_sha256": SHA, "changed_files": ["x.py"]},
            "python-unittest": {"exit_code": 0, "duration_ms": 2, "stdout_sha256": SHA, "changed_files": ["x.py"]},
        },
        "passed": passed,
    }


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SwarmStore(Path(self.temp.name) / "state.sqlite3")
        self.task = self.store.create_task(
            objective="Fix fixture", repo_path="/tmp/repo", revision="abc", max_branches=2,
            acceptance=["python-unittest", "git-diff-check"],
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def branch(self):
        return self.store.create_branch(
            task_id=self.task["id"], hypothesis="fixture", role="implementer", model_profile="small",
        )

    def test_passing_artifact_becomes_promotion_ready_evidence(self):
        branch = self.branch()
        attempt = ingest_corpus_attempt(store=self.store, branch_id=branch["id"], result=artifact(passed=True))
        self.assertEqual(attempt["status"], "completed")
        self.assertEqual(rank_task(self.store, self.task["id"])[0]["decision"], "promote-ready")

    def test_contract_failure_prunes_and_cannot_claim_pass(self):
        branch = self.branch()
        failed = artifact(passed=False, validation={}, proposal={})
        attempt = ingest_corpus_attempt(store=self.store, branch_id=branch["id"], result=failed)
        self.assertEqual(attempt["status"], "failed")
        self.assertEqual(rank_task(self.store, self.task["id"])[0]["decision"], "prune")
        with self.assertRaises(IngestError):
            ingest_corpus_attempt(store=self.store, branch_id=branch["id"], result=artifact(passed=False))
