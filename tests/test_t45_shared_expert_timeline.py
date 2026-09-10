#!/usr/bin/env python3
"""Static contracts for the isolated T45 shared/expert timeline profiler."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUARD = (ROOT / "remote/p40-t45-qwen-shared-expert-timeline-guard.py").read_text()
SOURCE = ROOT / "checkouts/colibri-t45-shared-expert-timeline/c"


class SharedExpertTimelineTest(unittest.TestCase):
    def test_guard_isolated_and_opt_in(self):
        self.assertIn("colibri-t45-shared-expert-timeline/c/qwen36", GUARD)
        self.assertIn("COLI_SHARED_EXPERT_TIMELINE=1", GUARD)
        self.assertIn("ee46ac81fd2c32a8115801951384e75762832c85ad1abd0ae61ede69f063f4c5", GUARD)
        self.assertIn("EXPECTED_WINDOWS = 63 * 40", GUARD)
        self.assertIn("5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f", GUARD)

    def test_profile_only_source_keeps_math_calls(self):
        qwen = (SOURCE / "qwen36.c").read_text()
        tier = (SOURCE / "qwen36_tier.c").read_text()
        cuda = (SOURCE / "backend_cuda.cu").read_text()
        self.assertIn("COLI_SHARED_EXPERT_TIMELINE", qwen)
        self.assertIn("matmul_d(sh, xs, l->sh_g, 1, D, Ish);", qwen)
        self.assertIn("matmul_d(shu, xs, l->sh_u, 1, D, Ish);", qwen)
        self.assertIn("matmul_d(shd, sh, l->sh_d, 1, Ish, D);", qwen)
        self.assertIn("qt_timeline_window", qwen)
        self.assertIn("coli_cuda_expert_group_timeline_window", tier)
        self.assertIn("group_timeline_done", cuda)


if __name__ == "__main__":
    unittest.main()
