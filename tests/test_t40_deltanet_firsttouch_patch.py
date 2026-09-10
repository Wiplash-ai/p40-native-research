#!/usr/bin/env python3
"""Static contracts for the opt-in exact DeltaNet first-touch patch."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "patches/0017-qwen36-deltanet-firsttouch.patch").read_text()


class DeltaNetFirstTouchPatchTest(unittest.TestCase):
    def test_is_opt_in_and_preserves_the_serial_default(self):
        self.assertIn('getenv("COLI_DN_FIRSTTOUCH")', PATCH)
        self.assertIn('if (!firsttouch) {', PATCH)
        self.assertIn('memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * head_cells * sizeof(float));', PATCH)

    def test_parallel_work_matches_the_existing_static_head_partition(self):
        self.assertIn('#pragma omp parallel for schedule(static)', PATCH)
        self.assertIn('for (int h = 0; h < c->dn_vheads; h++) {', PATCH)
        self.assertIn('float *head = m->DN_rec[i] + (size_t)h * head_cells;', PATCH)
        self.assertIn('memset(head, 0, head_cells * sizeof(float));', PATCH)

    def test_reports_observed_page_nodes_without_changing_math(self):
        self.assertIn('SYS_get_mempolicy', PATCH)
        self.assertIn('MPOL_F_ADDR | MPOL_F_NODE', PATCH)
        self.assertIn('[dn-firsttouch] active:', PATCH)
        self.assertNotIn('W4A8', PATCH)
        self.assertNotIn('DP4A', PATCH)


if __name__ == "__main__":
    unittest.main()
