import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.executor import ExecutorPlanError, OllamaPlanClient, plan_schema, validate_plan


PLAN = {
    "hypothesis": "A cached parser avoids repeatedly reading the same schema.",
    "proposed_change": "Add one cache behind the parser boundary.",
    "validation_profile": "python-unittest",
    "expected_signal": "The parser test stays green with fewer repeated reads.",
    "stop_condition": "Stop if cache invalidation changes parsed output.",
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


if __name__ == "__main__":
    unittest.main()
