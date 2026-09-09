#!/usr/bin/env python3
"""Static contracts for the actual-Qwen groupwise W4A8 shadow control."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0009-qwen36-w4a8-groupwise-shadow.patch").read_text()


class GroupwiseShadowPatchTest(unittest.TestCase):
    def test_is_shadow_only_and_pascal_dp4a_based(self):
        self.assertIn('"groupwise-shadow"', PATCH)
        self.assertIn('"w4a8-groupwise-shadow"', PATCH)
        self.assertIn("__dp4a", PATCH)
        self.assertIn("grouped_hidden_w4_dual", PATCH)
        self.assertIn("grouped_down_w4", PATCH)
        self.assertNotIn("WMMA", PATCH)

    def test_uses_groupwise_scales_for_both_expert_stages(self):
        self.assertIn("#define DP8_GROUPWISE_WIDTH 256", PATCH)
        self.assertIn("dp8_quant_rows_groupwise", PATCH)
        self.assertIn("grouped_hidden_w4a8_groupwise_shadow", PATCH)
        self.assertIn("grouped_down_w4a8_groupwise_shadow", PATCH)
        self.assertGreaterEqual(PATCH.count("dp8_quant_rows_groupwise<<<"), 2)

    def test_shadow_runs_before_unchanged_exact_w4_return_path(self):
        shadow = PATCH.index("dp8_groupwise_shadow_launch")
        exact_hidden = PATCH.index("grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>", shadow)
        exact_down = PATCH.index("grouped_down_w4<<<og,256,0,ctx->stream>>>", exact_hidden)
        self.assertLess(shadow, exact_hidden)
        self.assertLess(exact_hidden, exact_down)


if __name__ == "__main__":
    unittest.main()
