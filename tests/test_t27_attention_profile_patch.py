#!/usr/bin/env python3
"""Static contracts for the exact-Q8 attention profiling control."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0008-qwen36-exact-attention-subprofile.patch").read_text()


class AttentionProfilePatchTest(unittest.TestCase):
    def test_profile_is_opt_in_and_does_not_change_kernel_or_precision(self):
        self.assertIn('getenv("COLI_ATTN_PROFILE")', PATCH)
        self.assertIn('[timers]   attn-sub:', PATCH)
        self.assertNotIn('W4A8', PATCH)
        self.assertNotIn('DP4A', PATCH)
        self.assertNotIn('qwen_dn_cpuorder_q8_matvec<<<', PATCH)

    def test_projection_boundaries_preserve_q_k_v_then_cpu_middle_then_o(self):
        q = PATCH.index('matmul_d(q, x, l->q, S, D, q_out);')
        k = PATCH.index('matmul_d(k, x, l->k, S, D, kv_out);')
        v = PATCH.index('matmul_d(vv, x, l->v, S, D, kv_out);')
        middle = PATCH.index('g_attn_sub[3]+=t-_a0')
        o = PATCH.index('matmul_d(out, ag, l->o, S, H*hd, D);')
        self.assertLess(q, k)
        self.assertLess(k, v)
        self.assertLess(v, middle)
        self.assertLess(middle, o)

    def test_profile_only_accumulates_single_token_decode(self):
        self.assertIn('S==1 && tm_on() && attn_profile_on()', PATCH)
        self.assertIn('g_attn_sub[0]', PATCH)
        self.assertIn('g_attn_sub[4]', PATCH)


if __name__ == "__main__":
    unittest.main()
