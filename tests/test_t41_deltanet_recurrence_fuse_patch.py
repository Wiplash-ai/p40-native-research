#!/usr/bin/env python3
"""Static contracts for the opt-in exact DeltaNet recurrence fusion patch."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0018-qwen36-deltanet-recurrence-fuse.patch").read_text()


class DeltaNetRecurrenceFusePatchTest(unittest.TestCase):
    def test_is_opt_in_and_keeps_the_unfused_path(self):
        self.assertIn('getenv("COLI_DN_REC_FUSE")', PATCH)
        self.assertIn('if (!rec_fuse) {', PATCH)
        self.assertIn('for (int t = 0; t < kdim * vdim; t++) Sh[t] *= egh;', PATCH)
        self.assertIn('[dn-rec-fuse] active:', PATCH)

    def test_fuses_only_the_two_safe_state_pass_pairs(self):
        self.assertIn('float scaled = Sr[vv] * egh;', PATCH)
        self.assertIn('kvl[vv] += kkd * scaled;', PATCH)
        self.assertIn('float updated = Sr[vv] + kkd * dl[vv];', PATCH)
        self.assertIn('ov[vv] += qkd * updated;', PATCH)

    def test_retains_head_parallelism_and_disallows_approximate_math(self):
        self.assertIn('#pragma omp parallel for schedule(static)', PATCH)
        self.assertNotIn('W4A8', PATCH)
        self.assertNotIn('DP4A', PATCH)
        self.assertNotIn('fast-math', PATCH)


if __name__ == "__main__":
    unittest.main()
