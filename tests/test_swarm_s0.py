import hashlib
import http.client
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.api import server
from swarm.harness import BoundedHarness, HarnessError
from swarm.probe import CapacityProbeError, parse_snapshot, ssh_argv
from swarm.store import SwarmStateError, SwarmStore


SHA = hashlib.sha256(b"evidence").hexdigest()


class SwarmStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SwarmStore(Path(self.temp.name) / "swarm.sqlite3")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def task(self, max_branches=2):
        return self.store.create_task(
            objective="Fix a bounded bug", repo_path="/tmp/repository", revision="abc123",
            max_branches=max_branches, acceptance=["python-unittest", "git-diff-check"],
        )

    def test_evidence_is_required_for_promotion(self):
        task = self.task()
        branch = self.store.create_branch(
            task_id=task["id"], hypothesis="cache the parser", role="implementer", model_profile="fake-a",
        )
        attempt = self.store.create_attempt(branch["id"])
        self.store.append_evidence(
            attempt_id=attempt["id"], kind="test", command_profile="python-unittest", exit_code=0,
            duration_ms=12, stdout_sha256=SHA,
        )
        self.store.finish_attempt(attempt["id"], "completed")
        with self.assertRaises(SwarmStateError):
            self.store.promote(branch["id"])
        retry = self.store.create_attempt(branch["id"])
        self.store.append_evidence(
            attempt_id=retry["id"], kind="test", command_profile="python-unittest", exit_code=0,
            duration_ms=12, stdout_sha256=SHA,
        )
        self.store.append_evidence(
            attempt_id=retry["id"], kind="static", command_profile="git-diff-check", exit_code=0,
            duration_ms=5, stdout_sha256=SHA,
        )
        self.store.finish_attempt(retry["id"], "completed")
        self.assertEqual(self.store.promote(branch["id"])["status"], "promoted")

    def test_branch_budget_and_closed_evidence_are_enforced(self):
        task = self.task(max_branches=1)
        branch = self.store.create_branch(
            task_id=task["id"], hypothesis="A", role="critic", model_profile="fake-a",
        )
        with self.assertRaises(SwarmStateError):
            self.store.create_branch(
                task_id=task["id"], hypothesis="B", role="critic", model_profile="fake-b",
            )
        attempt = self.store.create_attempt(branch["id"])
        self.store.finish_attempt(attempt["id"], "failed")
        with self.assertRaises(SwarmStateError):
            self.store.append_evidence(
                attempt_id=attempt["id"], kind="test", command_profile="python-unittest", exit_code=0,
                duration_ms=1, stdout_sha256=SHA,
            )


class HarnessTests(unittest.TestCase):
    def test_worktree_and_fixed_profile_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
            (repo / "README.md").write_text("fixture\n")
            subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "-m", "fixture"], check=True, capture_output=True, text=True,
            )
            harness = BoundedHarness(worktree_root=root / "worktrees", artifact_root=root / "artifacts")
            task_id, branch_id, attempt_id = (str(uuid.uuid4()) for _ in range(3))
            worktree = harness.prepare_worktree(repo_path=repo, revision="HEAD", task_id=task_id, branch_id=branch_id)
            result = harness.run_profile(worktree=worktree, attempt_id=attempt_id, profile="git-diff-check")
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(Path(result.artifact).is_file())
            self.assertEqual(result.changed_files, [])
            with self.assertRaises(HarnessError):
                harness.run_profile(worktree=worktree, attempt_id=attempt_id, profile="shell-rm")


class ProbeTests(unittest.TestCase):
    def test_fixed_probe_requires_two_gpus_and_inventory(self):
        snapshot = parse_snapshot(json.dumps({
            "qwen_service": "inactive",
            "gpus": [{"index": "0"}, {"index": "1"}],
            "ollama": {"models": [{"name": "model-a"}]},
            "ai_ssd": "/dev/sda1 440G 233G 185G 56% /mnt/ai-ssd",
        }))
        self.assertEqual(snapshot.ollama_models[0]["name"], "model-a")
        self.assertNotIn("ollama pull", ssh_argv("host", 8)[-1])
        with self.assertRaises(CapacityProbeError):
            parse_snapshot(json.dumps({"qwen_service": "inactive", "gpus": [], "ollama": {"models": []}, "ai_ssd": "x"}))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.httpd = server(db_path=Path(self.temp.name) / "state.sqlite3", port=0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=2)
        self.httpd.RequestHandlerClass.store.close()
        self.httpd.server_close()
        self.temp.cleanup()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        encoded = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if encoded else {}
        connection.request(method, path, encoded, headers)
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return response.status, payload

    def test_loopback_api_records_plans_but_refuses_dispatch(self):
        status, health = self.request("GET", "/v1/health")
        self.assertEqual((status, health["dispatch"]), (200, "disabled"))
        status, task = self.request("POST", "/v1/tasks", {
            "objective": "Fix a test", "repo_path": "/tmp/repo", "revision": "abc", "max_branches": 2,
            "acceptance": ["python-unittest"],
        })
        self.assertEqual(status, 201)
        status, branch = self.request("POST", f"/v1/tasks/{task['id']}/branches", {
            "hypothesis": "change the parser", "role": "implementer", "model_profile": "fake-a",
        })
        self.assertEqual(status, 201)
        status, payload = self.request("POST", f"/v1/branches/{branch['id']}/run", {})
        self.assertEqual((status, payload["error"]["code"]), (409, "dispatch_disabled"))


if __name__ == "__main__":
    unittest.main()
