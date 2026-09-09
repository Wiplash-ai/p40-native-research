#!/usr/bin/env python3
"""Static contracts for the opt-in exact DeltaNet pair issue/join patch."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0006-qwen36-deltanet-pair-issue-join.patch").read_text()


class DeltaNetPairPatchTest(unittest.TestCase):
    def test_is_opt_in_and_exact_kernel_reuse_only(self):
        self.assertIn('getenv("COLI_CUDA_DN_PAIR")', PATCH)
        self.assertIn("qwen_dn_cpuorder_q8_matvec<<<first->output, 32, 0, context.stream>>>", PATCH)
        self.assertIn("qwen_dn_cpuorder_q8_matvec<<<second->output, 32, 0, context.stream>>>", PATCH)
        self.assertNotIn("W4A8", PATCH)
        self.assertNotIn("DP4A", PATCH)

    def test_issues_before_cpu_b_and_a_then_joins_before_recurrence(self):
        issue = PATCH.index("p40_qwen_dn_pair_issue(qkv_t,z_t,xs)")
        b = PATCH.index("matmul(b,   xs, l->dn_b,   1, H, vh);")
        a = PATCH.index("matmul(a,   xs, l->dn_a,   1, H, vh);")
        take = PATCH.index("p40_qwen_dn_pair_take(qkv,z)")
        self.assertLess(issue, b)
        self.assertLess(b, a)
        self.assertLess(a, take)

    def test_fallback_recomputes_serial_exact_projections(self):
        self.assertIn("QKV/Z pair issue failed; reverting to serial exact Q8", PATCH)
        self.assertIn("QKV/Z pair join failed; reverting to serial exact Q8", PATCH)
        self.assertGreaterEqual(PATCH.count("matmul_d(qkv, xs, l->dn_qkv, 1, H, conv_dim);"), 2)
        self.assertGreaterEqual(PATCH.count("matmul_d(z,   xs, l->dn_z,   1, H, value_dim);"), 2)

    def test_one_pending_pair_and_one_host_join(self):
        self.assertIn("context.pair_first || context.pair_second", PATCH)
        self.assertIn("pair synchronization", PATCH)
        self.assertIn("context.pair_first = nullptr;", PATCH)
        self.assertIn("context.pair_second = nullptr;", PATCH)


if __name__ == "__main__":
    unittest.main()
