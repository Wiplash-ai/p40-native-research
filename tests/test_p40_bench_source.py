import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "benchmarks/p40_bench.cu").read_text(encoding="utf-8")


class P40BenchSourceTests(unittest.TestCase):
    def test_only_allowlisted_initial_primitives_exist(self):
        self.assertIn('"P01-copy"', SOURCE)
        self.assertIn('"P03-dp4a-gemv"', SOURCE)
        self.assertIn("__dp4a", SOURCE)

    def test_dry_run_precedes_cuda_device_selection(self):
        self.assertLess(SOURCE.index("if (options.dry_run)"), SOURCE.index("cudaSetDevice"))

    def test_p01_has_a_correctness_readback(self):
        self.assertIn("cudaMemcpy(&observed", SOURCE)
        self.assertIn("const bool correct", SOURCE)

    def test_build_is_explicitly_pascal(self):
        makefile = (ROOT / "benchmarks/Makefile").read_text(encoding="utf-8")
        self.assertIn("CUDA_ARCH ?= sm_61", makefile)
        self.assertIn("-arch=$(CUDA_ARCH)", makefile)
