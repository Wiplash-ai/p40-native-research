#!/usr/bin/env python3
"""Static contracts for the guarded T44 exact-Qwen scheduling control."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUARD = (ROOT / "remote/p40-t44-qwen-qtier-reverse-issue-guard.py").read_text()


class QTierReverseIssueGuardTest(unittest.TestCase):
    def test_uses_isolated_binary_and_canonical_output(self):
        self.assertIn("colibri-t44-qtier-reverse-issue/c/qwen36", GUARD)
        self.assertIn("69d4b1f51ce998d1487daa8c2b68b65c917953616aa7b4e7f73044234dbd67ea", GUARD)
        self.assertIn("5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f", GUARD)

    def test_requires_the_one_variable_and_active_marker(self):
        self.assertIn('result.append("COLI_QTIER_REVERSE_ISSUE=1")', GUARD)
        self.assertIn("ACTIVE_MARKER", GUARD)
        self.assertIn("qtier_reverse_issue_marker_missing", GUARD)


if __name__ == "__main__":
    unittest.main()
