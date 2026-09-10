#!/usr/bin/env python3
"""Static contracts for opt-in reverse asynchronous Qwen expert issue order."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0021-qwen36-qtier-reverse-issue.patch").read_text()


class QTierReverseIssuePatchTest(unittest.TestCase):
    def test_is_opt_in_and_default_order_is_retained(self):
        self.assertIn('getenv("COLI_QTIER_REVERSE_ISSUE")', PATCH)
        self.assertIn("int di=reverse ? G.ndev-1-oi : oi;", PATCH)
        self.assertIn('"[qtier-issue-order] reverse device launch order active', PATCH)

    def test_changes_issue_order_only_not_take_or_math(self):
        self.assertIn("for(int oi=0;oi<G.ndev;oi++){", PATCH)
        self.assertNotIn("coli_cuda_expert_group_take", PATCH)
        self.assertNotIn("out[d]+=w*row[d]", PATCH)
        self.assertNotIn("W4A8", PATCH)


if __name__ == "__main__":
    unittest.main()
