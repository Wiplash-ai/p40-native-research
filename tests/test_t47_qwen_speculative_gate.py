#!/usr/bin/env python3
"""Static proof that direct Qwen has no usable speculative verifier surface."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
QWEN = (ROOT / "checkouts/colibri-t46-shared-expert-timeline/c/qwen36.c").read_text()


class QwenSpeculativeGateTests(unittest.TestCase):
    def test_generate_and_teacher_forcing_are_single_token_after_prefill(self):
        self.assertIn("logit = step(m, &one, 1, len - 1);", QWEN)
        self.assertIn("logit = step(m, &full[i], 1, i);", QWEN)

    def test_batched_step_exposes_only_the_final_row_logits(self):
        self.assertIn("x + (int64_t)(S-1)*D", QWEN)
        self.assertIn("matmul_d(logit, last, m->lm_head, 1, D, c->vocab);", QWEN)

    def test_no_draft_or_mtp_protocol_exists_in_the_direct_engine(self):
        lowered = QWEN.lower()
        self.assertNotIn("speculative", lowered)
        self.assertNotIn("draft", lowered)
        self.assertNotIn("mtp", lowered)


if __name__ == "__main__":
    unittest.main()
