#!/usr/bin/env python3
"""Static contracts for the T46 ineligible-window timeline revision."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUARD = (ROOT / "remote/p40-t46-qwen-shared-expert-timeline-guard.py").read_text()
SOURCE = ROOT / "checkouts/colibri-t46-shared-expert-timeline/c"


class SharedExpertTimelineV2Test(unittest.TestCase):
    def test_guard_counts_windows_and_ineligible_pairs(self):
        self.assertIn("colibri-t46-shared-expert-timeline/c/qwen36", GUARD)
        self.assertIn("b930f61378bddab23a957276f1f015e8a4d53b92cf1a44bad0012579819698a2", GUARD)
        self.assertIn("int(match.group(1)) + int(match.group(2)) != EXPECTED_WINDOWS", GUARD)
        self.assertIn("int(match.group(3)) != 0", GUARD)

    def test_source_separates_skip_from_event_error(self):
        qwen = (SOURCE / "qwen36.c").read_text()
        cuda = (SOURCE / "backend_cuda.cu").read_text()
        self.assertIn("g_shared_expert_timeline_no_gpu0", qwen)
        self.assertIn("else if(window<0) g_shared_expert_timeline_no_gpu0++", qwen)
        self.assertIn("if(!ctx->group_timeline_done_ok || !ctx->group_timeline_pending) return -1;", cuda)


if __name__ == "__main__":
    unittest.main()
