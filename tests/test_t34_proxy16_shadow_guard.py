#!/usr/bin/env python3
"""Local contracts for T34's pinned exact-output proxy16 shadow."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GUARD_SPEC = importlib.util.spec_from_file_location("p40_t34_guard", ROOT / "remote/p40-t34-qwen-proxy16-shadow-guard.py")
guard = importlib.util.module_from_spec(GUARD_SPEC)
assert GUARD_SPEC and GUARD_SPEC.loader
sys.modules[GUARD_SPEC.name] = guard
GUARD_SPEC.loader.exec_module(guard)


class Proxy16GuardTests(unittest.TestCase):
    def test_pins_proxy16_and_keeps_dry_runs_cpu_only(self):
        self.assertEqual(guard.EXPECTED_ORIGINAL_COMMAND, "p40-t34-qwen-w4a8-proxy16-shadow")
        self.assertEqual(guard.EXPECTED_GROUP_RECORDS, 2397)
        self.assertIn("COLI_CUDA_W4A8_DP4A=groupwise-proxy16-shadow", guard.model_argv())
        self.assertFalse(guard.run({"dry_run": True})["cuda_initialized"])

    def test_requires_exact_output_one_finite_shadow_and_one_cost_per_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "qwen36"; engine.write_bytes(b"t34"); engine.chmod(0o755)
            stderr = root / "stderr.txt"
            stderr.write_text(
                "[w4a8-groupwise-proxy16-shadow] rows=1 width=2 rel_l2=0.1 max_abs=0.2 cosine=0.9 finite=1\n"
                "[w4a8-groupwise-proxy16-cost] rows=1 proxy_gpu_ms=2 exact_gpu_ms=1 ratio=2 transfer_excluded=1\n"
            )
            exact = hashlib.sha256(b"exact").hexdigest()
            base_result = {"status": "pass", "stdout": {"sha256": exact}, "stderr": {"path": str(stderr)}}
            with mock.patch.object(guard, "ENGINE", engine), \
                 mock.patch.object(guard, "ENGINE_SHA256", hashlib.sha256(b"t34").hexdigest()), \
                 mock.patch.object(guard, "EXPECTED_STDOUT_SHA256", exact), \
                 mock.patch.object(guard, "EXPECTED_GROUP_RECORDS", 1), \
                 mock.patch.object(guard, "_base_run", return_value=base_result):
                result = guard.run({"dry_run": False})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["shadow_metrics"][0]["finite"], 1)
        self.assertEqual(result["correction_cost"][0]["ratio"], 2.0)

    def test_rejects_missing_cost_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "qwen36"; engine.write_bytes(b"t34"); engine.chmod(0o755)
            stderr = root / "stderr.txt"
            stderr.write_text("[w4a8-groupwise-proxy16-shadow] rows=1 width=2 rel_l2=0.1 max_abs=0.2 cosine=0.9 finite=1\n")
            exact = hashlib.sha256(b"exact").hexdigest()
            base_result = {"status": "pass", "stdout": {"sha256": exact}, "stderr": {"path": str(stderr)}}
            with mock.patch.object(guard, "ENGINE", engine), \
                 mock.patch.object(guard, "ENGINE_SHA256", hashlib.sha256(b"t34").hexdigest()), \
                 mock.patch.object(guard, "EXPECTED_STDOUT_SHA256", exact), \
                 mock.patch.object(guard, "EXPECTED_GROUP_RECORDS", 1), \
                 mock.patch.object(guard, "_base_run", return_value=base_result):
                result = guard.run({"dry_run": False})
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["failure_reason"], "proxy16_metric_count_mismatch")


if __name__ == "__main__":
    unittest.main()
