import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.executor import (
    ExecutorPlanError, OllamaPlanClient, patch_schema, plan_schema, text_edit_schema,
    validate_patch_proposal, validate_plan, validate_text_edit_proposal,
)
from swarm.edit import apply_exact_text_edit
from swarm.harness import BoundedHarness
from swarm.patch import PatchError, apply_unified_patch


PLAN = {
    "hypothesis": "A cached parser avoids repeatedly reading the same schema.",
    "proposed_change": "Add one cache behind the parser boundary.",
    "validation_profile": "python-unittest",
    "expected_signal": "The parser test stays green with fewer repeated reads.",
    "stop_condition": "Stop if cache invalidation changes parsed output.",
}

PATCH = {
    "summary": "Correct the even predicate.",
    "patch": """diff --git a/calculator.py b/calculator.py
index 12e10ad..a3212f1 100644
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def is_even(value):
-    return value % 2 == 1
+    return value % 2 == 0
""",
    "validation_profile": "python-unittest",
}

TEXT_EDIT = {
    "summary": "Correct the even predicate.",
    "path": "calculator.py",
    "expected_text": "return value % 2 == 1",
    "replacement_text": "return value % 2 == 0",
    "validation_profile": "python-unittest",
}


class ExecutorPlanTests(unittest.TestCase):
    def test_fixed_schema_rejects_command_surface(self):
        self.assertEqual(validate_plan(PLAN).validation_profile, "python-unittest")
        with self.assertRaises(ExecutorPlanError):
            validate_plan({**PLAN, "command": "rm -rf /"})
        with self.assertRaises(ExecutorPlanError):
            validate_plan({**PLAN, "validation_profile": "bash"})
        self.assertFalse(plan_schema()["additionalProperties"])

    @patch("swarm.executor.post_chat")
    def test_client_requires_strict_schema_and_fixed_ollama_request(self, post):
        post.return_value = {
            "message": {"content": json.dumps(PLAN)}, "prompt_eval_count": 11, "eval_count": 22,
        }
        result = OllamaPlanClient().request_plan(
            model="qwen3:8b", objective="Fix parser", branch_hypothesis="cache reads", role="implementer",
        )
        body = post.call_args.args[1]
        self.assertEqual(result.plan, validate_plan(PLAN))
        self.assertEqual(body["keep_alive"], "0s")
        self.assertFalse(body["think"])
        self.assertEqual(body["options"], {"num_ctx": 4096, "seed": 42, "temperature": 0})
        self.assertIn("format", body)
        with self.assertRaises(ExecutorPlanError):
            OllamaPlanClient(keep_alive="forever")

    @patch("swarm.executor.post_chat")
    def test_patch_client_has_no_command_surface(self, post):
        post.return_value = {
            "message": {"content": json.dumps(PATCH)}, "prompt_eval_count": 11, "eval_count": 22,
        }
        result = OllamaPlanClient().request_patch(
            model="qwen3:8b", objective="Fix parity", branch_hypothesis="modulo result is inverted",
            files={"calculator.py": "def is_even(value):\n    return value % 2 == 1\n"},
        )
        self.assertEqual(result.proposal, validate_patch_proposal(PATCH))
        self.assertEqual(result.proposal.validation_profile, "python-unittest")
        self.assertFalse(patch_schema()["additionalProperties"])
        self.assertNotIn("maxLength", patch_schema()["properties"]["patch"])

    def test_patch_is_limited_to_a_disposable_worktree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "calculator.py").write_text("def is_even(value):\n    return value % 2 == 1\n")
            tests = repo / "tests"
            tests.mkdir()
            (tests / "test_calculator.py").write_text(
                "import unittest\nfrom calculator import is_even\n\nclass ParityTest(unittest.TestCase):\n"
                "    def test_even(self):\n        self.assertTrue(is_even(2))\n"
            )
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "-m", "fixture"], check=True, capture_output=True, text=True,
            )
            harness = BoundedHarness(worktree_root=root / "worktrees", artifact_root=root / "artifacts")
            import uuid
            worktree = harness.prepare_worktree(
                repo_path=repo, revision="HEAD", task_id=str(uuid.uuid4()), branch_id=str(uuid.uuid4()),
            )
            apply_unified_patch(worktree=str(worktree), patch=PATCH["patch"])
            self.assertEqual(harness.run_profile(
                worktree=worktree, attempt_id=str(uuid.uuid4()), profile="python-unittest",
            ).exit_code, 0)
            with self.assertRaises(PatchError):
                apply_unified_patch(worktree=str(worktree), patch=PATCH["patch"].replace("calculator.py", "../outside", 2))
            with self.assertRaises(ExecutorPlanError):
                validate_patch_proposal({**PATCH, "patch": PATCH["patch"] + "new mode 100755\n"})
            with self.assertRaises(ExecutorPlanError):
                validate_patch_proposal(PATCH, allowed_paths={"tests/test_calculator.py"})

    @patch("swarm.executor.post_chat")
    def test_text_edit_is_exact_and_context_limited(self, post):
        post.return_value = {
            "message": {"content": json.dumps(TEXT_EDIT)}, "prompt_eval_count": 11, "eval_count": 22,
        }
        source = {"calculator.py": "def is_even(value):\n    return value % 2 == 1\n"}
        result = OllamaPlanClient(keep_alive="5m").request_text_edit(
            model="qwen3:8b", objective="Fix parity", branch_hypothesis="modulo result is inverted", files=source,
            editable_paths={"calculator.py"},
        )
        self.assertEqual(result.proposal, validate_text_edit_proposal(TEXT_EDIT, allowed_files=source))
        self.assertFalse(text_edit_schema()["additionalProperties"])
        self.assertNotIn("maxLength", text_edit_schema()["properties"]["expected_text"])
        self.assertEqual(post.call_args.args[1]["keep_alive"], "5m")
        with self.assertRaises(ExecutorPlanError):
            validate_text_edit_proposal({**TEXT_EDIT, "path": "../outside"}, allowed_files=source)
        with self.assertRaises(ExecutorPlanError):
            validate_text_edit_proposal(TEXT_EDIT, allowed_files=source, editable_paths={"tests/test_calculator.py"})
        with tempfile.TemporaryDirectory() as temp:
            worktree = Path(temp)
            target = worktree / "calculator.py"
            target.write_text(source["calculator.py"])
            apply_exact_text_edit(worktree=str(worktree), edit=result.proposal.edit)
            self.assertIn("== 0", target.read_text())


if __name__ == "__main__":
    unittest.main()
