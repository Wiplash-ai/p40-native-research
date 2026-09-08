from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "codex_marathon.py"
SPEC = importlib.util.spec_from_file_location("codex_marathon", SCRIPT)
assert SPEC and SPEC.loader
MARATHON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MARATHON
SPEC.loader.exec_module(MARATHON)


class MarathonTests(unittest.TestCase):
    def test_rate_summary_uses_codex_five_hour_bucket(self) -> None:
        result = MARATHON.rate_summary(
            {
                "rateLimits": {
                    "planType": "plus",
                    "rateLimitReachedType": None,
                    "spendControlReached": False,
                    "rateLimits": [
                        {"id": "weekly", "usedPercent": 2, "windowDurationMins": 10080},
                        {"id": "codex", "usedPercent": 14, "windowDurationMins": 300, "resetsAt": 123},
                    ],
                }
            }
        )
        self.assertEqual(result["five_hour_remaining_percent"], 86.0)
        self.assertEqual(result["five_hour_resets_at"], 123)
        self.assertFalse(result["rate_limit_reached"])

    def test_rate_summary_uses_app_server_primary_window(self) -> None:
        result = MARATHON.rate_summary(
            {
                "rateLimits": {
                    "planType": "plus",
                    "primary": {"usedPercent": 24, "windowDurationMins": 300, "resetsAt": 456},
                    "rateLimitReachedType": None,
                    "spendControlReached": False,
                }
            }
        )
        self.assertEqual(result["five_hour_remaining_percent"], 76.0)
        self.assertEqual(result["five_hour_resets_at"], 456)

    def test_rate_summary_fails_closed_without_five_hour_bucket(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "five-hour"):
            MARATHON.rate_summary({"rateLimits": {"rateLimits": []}})

    def test_json_marker_requires_matching_task_and_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "marker.json"
            marker.write_text(json.dumps({"task": "T01", "status": "pass"}))
            self.assertEqual(MARATHON.marker_status(marker, "T01"), "pass")
            self.assertIsNone(MARATHON.marker_status(marker, "T03"))

    def test_markdown_marker_accepts_explicit_blocked_thermal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "marker.md"
            marker.write_text("# T00\n\nAcceptance decision: blocked_thermal\n")
            self.assertEqual(MARATHON.marker_status(marker, "T00"), "blocked_thermal")

    def test_auto_selects_first_incomplete_task(self) -> None:
        self.assertEqual(MARATHON.select_task("auto", {"completed": ["T01"]}).id, "T03")


if __name__ == "__main__":
    unittest.main()
