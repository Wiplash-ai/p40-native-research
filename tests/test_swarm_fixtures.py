import sys
import json
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.fixtures import TASKS, get_task
from swarm.edit import apply_exact_text_edit, validate_exact_text_edit
from swarm.harness import BoundedHarness


class FixtureTests(unittest.TestCase):
    def test_fixture_identifiers_and_source_paths_are_stable(self):
        self.assertEqual([task.identifier for task in TASKS], [
            "parity", "display-name-whitespace", "currency-grouping", "multifile-timeout",
            "two-file-required", "already-green",
        ])
        for task in TASKS:
            self.assertTrue(task.objective)
            self.assertTrue(task.hypothesis)
            self.assertTrue(task.files)
            self.assertTrue(all(not path.startswith("/") and ".." not in path.split("/") for path in task.files))

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(ValueError):
            get_task("not-a-task")

    def test_already_green_fixture_does_not_call_the_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "result.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "swarm_s5_text_edit_corpus.py"),
                "--url", "http://127.0.0.1:9/api/chat", "--task", "already-green",
                "--workspace-root", str(root / "worktree"), "--output", str(output),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            payload = json.loads(output.read_text())
            self.assertIsNone(payload["proposal"])
            self.assertEqual(payload["error"]["message"], "baseline gate rejected an already-passing task")

    def test_valid_partial_edit_cannot_promote_a_two_file_task(self):
        task = get_task("two-file-required")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            for relative, content in task.files.items():
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run([
                "git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-m", "fixture",
            ], check=True, capture_output=True, text=True)
            harness = BoundedHarness(worktree_root=root / "worktrees", artifact_root=root / "artifacts")
            worktree = harness.prepare_worktree(
                repo_path=repo, revision="HEAD", task_id=str(uuid.uuid4()), branch_id=str(uuid.uuid4()),
            )
            edit = validate_exact_text_edit(
                path="primary.py", expected_text="return value.strip()", replacement_text="return value.strip().lower()",
                allowed_files=task.files,
            )
            apply_exact_text_edit(worktree=str(worktree), edit=edit)
            validation = harness.run_profile(
                worktree=worktree, attempt_id=str(uuid.uuid4()), profile="python-unittest",
            )
            self.assertNotEqual(validation.exit_code, 0)
            self.assertEqual(validation.changed_files, ["primary.py"])


if __name__ == "__main__":
    unittest.main()
