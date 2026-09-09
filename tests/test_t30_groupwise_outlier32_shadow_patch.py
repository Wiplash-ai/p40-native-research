#!/usr/bin/env python3
"""Static contracts for the one-variable T30 top-32 residual control."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0011-qwen36-w4a8-groupwise-outlier32-shadow.patch").read_text()


class GroupwiseOutlier32PatchTest(unittest.TestCase):
    def test_changes_only_topk_capacity_and_selector_label(self):
        self.assertIn("-#define DP8_OUTLIER_TOPK 8", PATCH)
        self.assertIn("+#define DP8_OUTLIER_TOPK 32", PATCH)
        self.assertIn('"groupwise-outlier32-shadow"', PATCH)
        self.assertIn('"w4a8-groupwise-outlier32-shadow"', PATCH)
        self.assertNotIn("WMMA", PATCH)

    def test_reuses_the_t29_shadow_implementation(self):
        self.assertIn("dp8_groupwise_outlier_shadow_launch", PATCH)
        self.assertIn("DP8_GROUPWISE_WIDTH", PATCH)
        self.assertNotIn("dp8_select_groupwise_outliers<<<", PATCH)
        self.assertNotIn("cudaMemcpy", PATCH)


if __name__ == "__main__":
    unittest.main()
