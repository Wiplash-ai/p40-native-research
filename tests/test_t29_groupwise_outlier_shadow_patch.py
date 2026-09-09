#!/usr/bin/env python3
"""Static contracts for the T29 real-Qwen sparse-residual shadow control."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0010-qwen36-w4a8-groupwise-outlier-shadow.patch").read_text()


class GroupwiseOutlierShadowPatchTest(unittest.TestCase):
    def test_is_shadow_only_and_pascal_dp4a_based(self):
        self.assertIn('"groupwise-outlier-shadow"', PATCH)
        self.assertIn('"w4a8-groupwise-outlier-shadow"', PATCH)
        self.assertIn("__dp4a", PATCH)
        self.assertIn("grouped_hidden_w4_dual", PATCH)
        self.assertIn("grouped_down_w4", PATCH)
        self.assertNotIn("WMMA", PATCH)

    def test_is_fixed_small_topk_residual_control(self):
        self.assertIn("#define DP8_OUTLIER_TOPK 8", PATCH)
        self.assertIn("dp8_select_groupwise_outliers", PATCH)
        self.assertIn("dp8_s4_at", PATCH)
        self.assertIn("dp8_outlier_residual", PATCH)
        self.assertIn("dp8_outlier_index", PATCH)
        self.assertGreaterEqual(PATCH.count("dp8_select_groupwise_outliers<<<"), 2)

    def test_corrects_both_mlp_stages_after_groupwise_dp4a(self):
        self.assertIn("grouped_hidden_w4a8_groupwise_outlier_shadow", PATCH)
        self.assertIn("grouped_down_w4a8_groupwise_outlier_shadow", PATCH)
        self.assertIn("e*dp8_s4_at(gr,i)*d.gs[o]", PATCH)
        self.assertIn("e*dp8_s4_at(ur,i)*d.us[o]", PATCH)
        self.assertIn("dp8_s4_at(w,i)*d.ds[o]", PATCH)

    def test_shadow_precedes_unchanged_exact_return_path(self):
        shadow = PATCH.index("dp8_groupwise_outlier_shadow_launch")
        exact_hidden = PATCH.index("grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>", shadow)
        exact_down = PATCH.index("grouped_down_w4<<<og,256,0,ctx->stream>>>", exact_hidden)
        self.assertLess(shadow, exact_hidden)
        self.assertLess(exact_hidden, exact_down)


if __name__ == "__main__":
    unittest.main()
