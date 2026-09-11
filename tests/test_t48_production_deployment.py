#!/usr/bin/env python3
"""Static contracts for the local-only accepted-Q8 deployment files."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
UNIT = (ROOT / "deployment/colibri-qwen36.service").read_text()
ENV = (ROOT / "deployment/colibri-qwen36-t24.env").read_text()


class ProductionDeploymentTests(unittest.TestCase):
    def test_listener_is_loopback_and_has_one_kv_slot(self):
        self.assertIn("--host 127.0.0.1", UNIT)
        self.assertIn("--kv-slots 1", UNIT)
        self.assertNotIn("--gpu", UNIT)

    def test_accepted_exact_path_is_explicit(self):
        for setting in (
            "COLI_CUDA=1", "COLI_GPUS=0,1", "COLI_DENSE_I8=1",
            "COLI_CUDA_DN_CPUORDER=1", "COLI_CUDA_DN_PAIR=1",
            "COLI_CUDA_LMHEAD_CPUORDER=1", "COLI_CUDA_ATTN_CPUORDER=1",
            "COLI_CUDA_SHARED_CPUORDER=1",
        ):
            self.assertIn(setting, ENV)


if __name__ == "__main__":
    unittest.main()
