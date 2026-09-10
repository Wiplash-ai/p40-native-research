#!/usr/bin/env python3
"""Static contracts for opt-in asynchronous Qwen expert-tier profiling."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0019-qwen36-qtier-async-event-profile.patch").read_text()


class QTierAsyncEventProfilePatchTest(unittest.TestCase):
    def test_is_opt_in_and_reuses_one_event_set_per_device(self):
        self.assertIn('getenv("COLI_QTIER_PROFILE")', PATCH)
        self.assertIn("cudaEvent_t group_ev[4]; int group_ev_ok, group_profile_pending;", PATCH)
        self.assertIn("if (ctx->group_ev_ok) return 1;", PATCH)
        self.assertIn("cudaEventDestroy(ctx->group_ev[e])", PATCH)

    def test_measures_h2d_kernel_and_d2h_after_existing_take_sync(self):
        self.assertIn('qtier_profile_record(ctx->group_ev[0],ctx->stream,device,"start")', PATCH)
        self.assertIn('qtier_profile_record(ctx->group_ev[1],ctx->stream,device,"post-h2d")', PATCH)
        self.assertIn('qtier_profile_record(ctx->group_ev[2],ctx->stream,device,"post-kernel")', PATCH)
        self.assertIn('qtier_profile_record(ctx->group_ev[3],ctx->stream,device,"post-d2h")', PATCH)
        self.assertIn('cudaStreamSynchronize(ctx->stream),"expert group take"', PATCH)
        self.assertIn("cudaEventElapsedTime(&h2d,ctx->group_ev[0],ctx->group_ev[1])", PATCH)
        self.assertIn("cudaEventElapsedTime(&kernel,ctx->group_ev[1],ctx->group_ev[2])", PATCH)
        self.assertIn("cudaEventElapsedTime(&d2h,ctx->group_ev[2],ctx->group_ev[3])", PATCH)

    def test_reports_each_device_without_changing_expert_math(self):
        self.assertIn("coli_cuda_group_stats_device(G.dev[i]", PATCH)
        self.assertIn('"[qtier]   profile dev %d:', PATCH)
        self.assertNotIn("DP4A", PATCH)
        self.assertNotIn("W4A8", PATCH)
        self.assertNotIn("grouped_hidden_w4_dual<<<", PATCH)


if __name__ == "__main__":
    unittest.main()
