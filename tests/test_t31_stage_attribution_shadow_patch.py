#!/usr/bin/env python3
"""Static contracts for T31's exact-output stage-attribution shadow."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0012-qwen36-w4a8-stage-attribution-shadow.patch").read_text()


class StageAttributionShadowPatchTest(unittest.TestCase):
    def test_keeps_two_shadow_outputs_and_exact_return(self):
        self.assertIn('"groupwise-stage-shadow"', PATCH)
        self.assertIn("dp8_stage_y", PATCH)
        self.assertIn("host_dp8_stage_y", PATCH)
        self.assertIn("grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>", PATCH)
        self.assertIn("grouped_down_w4<<<og,256,0,ctx->stream>>>", PATCH)

    def test_reports_input_and_hidden_attribution_separately(self):
        self.assertIn('"w4a8-stage-input"', PATCH)
        self.assertIn('"w4a8-stage-hidden"', PATCH)
        self.assertGreaterEqual(PATCH.count("dp8_select_groupwise_outliers<<<"), 2)
        self.assertIn("grouped_down_w4a8_groupwise_outlier_shadow", PATCH)

    def test_releases_extra_shadow_buffers(self):
        self.assertIn("cudaFree(ctx->dp8_stage_y)", PATCH)
        self.assertIn("cudaFreeHost(ctx->host_dp8_stage_y)", PATCH)


if __name__ == "__main__":
    unittest.main()
