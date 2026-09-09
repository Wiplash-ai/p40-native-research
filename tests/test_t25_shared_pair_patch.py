#!/usr/bin/env python3
"""Static contracts for the opt-in exact shared-MLP pair patch."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0007-qwen36-shared-pair-issue-join.patch").read_text()


class SharedPairPatchTest(unittest.TestCase):
    def test_is_opt_in_and_uses_existing_exact_pair_helper(self):
        self.assertIn('getenv("COLI_CUDA_SHARED_PAIR")', PATCH)
        self.assertIn("p40_qwen_dn_pair_issue(g_t,u_t,xs)", PATCH)
        self.assertIn("p40_qwen_dn_pair_take(sh,shu)", PATCH)
        self.assertNotIn("W4A8", PATCH)
        self.assertNotIn("DP4A", PATCH)

    def test_gate_up_share_input_and_scalar_gate_runs_before_join(self):
        issue = PATCH.index("p40_qwen_dn_pair_issue(g_t,u_t,xs)")
        scalar = PATCH.index("float sgate = 1.f;")
        take = PATCH.index("p40_qwen_dn_pair_take(sh,shu)")
        product = PATCH.index("+                for (int i = 0; i < Ish; i++) { float sv = sh[i];", take)
        self.assertLess(issue, scalar)
        self.assertLess(scalar, take)
        self.assertLess(take, product)

    def test_reuses_generic_flag_checked_lookup_and_has_serial_fallback(self):
        self.assertIn("qdw_cpuorder_tensor_for", PATCH)
        self.assertIn("QDW_SHARED_CPUORDER", PATCH)
        self.assertIn("shared gate/up pair issue failed; reverting to serial exact Q8", PATCH)
        self.assertIn("shared gate/up pair join failed; reverting to serial exact Q8", PATCH)
        self.assertGreaterEqual(PATCH.count("matmul_d(sh, xs, l->sh_g, 1, D, Ish);"), 2)
        self.assertGreaterEqual(PATCH.count("matmul_d(shu, xs, l->sh_u, 1, D, Ish);"), 2)


if __name__ == "__main__":
    unittest.main()
