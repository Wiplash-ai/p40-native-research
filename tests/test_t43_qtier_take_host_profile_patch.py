#!/usr/bin/env python3
"""Static contracts for opt-in Qwen expert-tier host-take attribution."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0020-qwen36-qtier-take-host-profile.patch").read_text()


class QTierTakeHostProfilePatchTest(unittest.TestCase):
    def test_is_opt_in_and_does_not_change_device_dispatch(self):
        self.assertIn('getenv("COLI_QTIER_TAKE_PROFILE")', PATCH)
        self.assertIn("const float *y=coli_cuda_expert_group_take(G.dev[di]);", PATCH)
        self.assertNotIn("cudaEvent", PATCH)
        self.assertNotIn("COLI_QTIER_PROFILE", PATCH)

    def test_splits_wait_from_weighted_cpu_accumulation(self):
        self.assertIn("double t0=profile?take_profile_now_ms():0.0;", PATCH)
        self.assertIn("double t1=profile?take_profile_now_ms():0.0;", PATCH)
        self.assertIn("G.take_wait_ms[di]+=t1-t0;", PATCH)
        self.assertIn("G.take_acc_ms[di]+=take_profile_now_ms()-t1;", PATCH)
        self.assertIn('"[qtier]   host-take dev %d:', PATCH)

    def test_keeps_exact_weighted_accumulation_order(self):
        self.assertIn("float w=val[G.is_k[di][j]];", PATCH)
        self.assertIn("for(int d=0;d<G.D;d++) out[d]+=w*row[d];", PATCH)


if __name__ == "__main__":
    unittest.main()
