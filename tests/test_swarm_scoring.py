import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.scoring import rank_task, score_branch
from swarm.store import SwarmStore


SHA = hashlib.sha256(b"evidence").hexdigest()


class ScoringTests(unittest.TestCase):
    def test_model_confidence_cannot_outscore_passing_validation(self):
        weak = score_branch(branch_id="a", evidence=[{
            "kind": "model", "exit_code": 0, "metrics": {"confidence": 1.0},
        }])
        strong = score_branch(branch_id="b", evidence=[
            {"kind": "static", "exit_code": 0, "metrics": {}},
            {"kind": "test", "exit_code": 0, "metrics": {}},
        ])
        self.assertEqual((weak.decision, weak.score), ("expand", 0))
        self.assertEqual((strong.decision, strong.score), ("promote-ready", 100))

    def test_required_failure_prunes_even_when_other_checks_pass(self):
        result = score_branch(branch_id="a", evidence=[
            {"kind": "static", "exit_code": 0, "metrics": {}},
            {"kind": "test", "exit_code": 1, "metrics": {}},
            {"kind": "benchmark", "exit_code": 0, "metrics": {"improvement_pct": 99}},
        ])
        self.assertEqual((result.decision, result.score), ("prune", -100))

    def test_task_ranking_uses_evidence_not_branch_role(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SwarmStore(Path(temp) / "state.sqlite3")
            try:
                task = store.create_task(
                    objective="Fix a bug", repo_path="/tmp/repo", revision="abc", max_branches=2,
                    acceptance=["python-unittest", "git-diff-check"],
                )
                unverified = store.create_branch(
                    task_id=task["id"], hypothesis="sounds compelling", role="planner", model_profile="large",
                )
                verified = store.create_branch(
                    task_id=task["id"], hypothesis="measured", role="critic", model_profile="small",
                )
                attempt = store.create_attempt(verified["id"])
                for kind, profile in (("static", "git-diff-check"), ("test", "python-unittest")):
                    store.append_evidence(
                        attempt_id=attempt["id"], kind=kind, command_profile=profile, exit_code=0,
                        duration_ms=1, stdout_sha256=SHA,
                    )
                ranking = rank_task(store, task["id"])
                self.assertEqual(ranking[0]["branch_id"], verified["id"])
                self.assertEqual(ranking[1]["branch_id"], unverified["id"])
            finally:
                store.close()
