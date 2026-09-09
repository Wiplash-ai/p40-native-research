#!/usr/bin/env python3
"""Local contracts for T31's dedicated remote profile and client."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GUARD_SPEC = importlib.util.spec_from_file_location(
    "p40_t31_stage_attribution_guard",
    ROOT / "remote/p40-t31-qwen-w4a8-stage-attribution-shadow-guard.py",
)
guard = importlib.util.module_from_spec(GUARD_SPEC)
assert GUARD_SPEC and GUARD_SPEC.loader
sys.modules[GUARD_SPEC.name] = guard
GUARD_SPEC.loader.exec_module(guard)

CLIENT_SPEC = importlib.util.spec_from_file_location(
    "p40_t31_stage_attribution_client",
    ROOT / "scripts/p40_t31_qwen_stage_attribution_shadow_client.py",
)
client = importlib.util.module_from_spec(CLIENT_SPEC)
assert CLIENT_SPEC and CLIENT_SPEC.loader
sys.modules[CLIENT_SPEC.name] = client
CLIENT_SPEC.loader.exec_module(client)


class StageAttributionGuardTests(unittest.TestCase):
    def test_pins_the_stage_shadow_binary_selector_and_oracle(self):
        self.assertEqual(guard.EXPECTED_ORIGINAL_COMMAND, "p40-t31-qwen-w4a8-stage-attribution-shadow")
        self.assertEqual(guard.PROFILE_ID, "t31-w4a8-real-expert-stage-attribution-shadow-16")
        self.assertEqual(guard.ENGINE_SHA256, "d6f66c7afc13f962ded4ad9e45e491342ef9e7e308101508b4c7fbf9036847b1")
        self.assertEqual(guard.EXPECTED_GROUP_RECORDS, 2397)
        self.assertIn("COLI_CUDA_W4A8_DP4A=groupwise-stage-shadow", guard.model_argv())
        self.assertFalse(guard.run({"dry_run": True})["cuda_initialized"])

    def test_client_can_only_select_t31_forced_commands(self):
        identity = Path("/tmp/p40-t31-test-key")
        self.assertEqual(client.ssh_argv("host", identity)[-1], guard.EXPECTED_ORIGINAL_COMMAND)
        self.assertEqual(client.ssh_argv("host", identity, read_results=True)[-1], guard.RESULTS_ORIGINAL_COMMAND)

    def test_requires_each_stage_to_report_the_fixed_number_of_finite_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "qwen36"; engine.write_bytes(b"t31"); engine.chmod(0o755)
            stderr = root / "stderr.txt"
            stderr.write_text(
                "[w4a8-stage-input] rows=1 width=2 rel_l2=0.1 max_abs=0.2 cosine=0.9 finite=1\n"
                "[w4a8-stage-hidden] rows=1 width=2 rel_l2=0.3 max_abs=0.4 cosine=0.8 finite=1\n"
            )
            exact = hashlib.sha256(b"exact").hexdigest()
            base_result = {"status": "pass", "stdout": {"sha256": exact}, "stderr": {"path": str(stderr)}}
            with mock.patch.object(guard, "ENGINE", engine), \
                 mock.patch.object(guard, "ENGINE_SHA256", hashlib.sha256(b"t31").hexdigest()), \
                 mock.patch.object(guard, "EXPECTED_STDOUT_SHA256", exact), \
                 mock.patch.object(guard, "EXPECTED_GROUP_RECORDS", 1), \
                 mock.patch.object(guard, "_base_run", return_value=base_result):
                result = guard.run({"dry_run": False})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(len(result["shadow_metrics"]["input"]), 1)
        self.assertEqual(len(result["shadow_metrics"]["hidden"]), 1)

    def test_fails_closed_when_one_stage_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "qwen36"; engine.write_bytes(b"t31"); engine.chmod(0o755)
            stderr = root / "stderr.txt"
            stderr.write_text("[w4a8-stage-input] rows=1 width=2 rel_l2=0.1 max_abs=0.2 cosine=0.9 finite=1\n")
            exact = hashlib.sha256(b"exact").hexdigest()
            base_result = {"status": "pass", "stdout": {"sha256": exact}, "stderr": {"path": str(stderr)}}
            with mock.patch.object(guard, "ENGINE", engine), \
                 mock.patch.object(guard, "ENGINE_SHA256", hashlib.sha256(b"t31").hexdigest()), \
                 mock.patch.object(guard, "EXPECTED_STDOUT_SHA256", exact), \
                 mock.patch.object(guard, "EXPECTED_GROUP_RECORDS", 1), \
                 mock.patch.object(guard, "_base_run", return_value=base_result):
                result = guard.run({"dry_run": False})
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["failure_reason"], "stage_shadow_metric_count_mismatch")


if __name__ == "__main__":
    unittest.main()
