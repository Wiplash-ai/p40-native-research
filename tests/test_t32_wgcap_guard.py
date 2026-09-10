#!/usr/bin/env python3
"""Local contracts for T32's fixed capture profile and result validator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
GUARD_SPEC = importlib.util.spec_from_file_location("p40_t32_guard", ROOT / "remote/p40-t32-qwen-wgcap-gateup-guard.py")
guard = importlib.util.module_from_spec(GUARD_SPEC)
assert GUARD_SPEC and GUARD_SPEC.loader
sys.modules[GUARD_SPEC.name] = guard
GUARD_SPEC.loader.exec_module(guard)
CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t32_client", ROOT / "scripts/p40_t32_qwen_wgcap_client.py")
client = importlib.util.module_from_spec(CLIENT_SPEC)
assert CLIENT_SPEC and CLIENT_SPEC.loader
sys.modules[CLIENT_SPEC.name] = client
CLIENT_SPEC.loader.exec_module(client)
WGCAP_SPEC = importlib.util.spec_from_file_location("test_wgcap_helpers", ROOT / "tests/test_wgcap.py")
helpers = importlib.util.module_from_spec(WGCAP_SPEC)
assert WGCAP_SPEC and WGCAP_SPEC.loader
sys.modules[WGCAP_SPEC.name] = helpers
WGCAP_SPEC.loader.exec_module(helpers)


def fixed_records() -> list[tuple[int, int, int, int, int, int, float]]:
    return [
        (layer, step, rank % 2, 100 + rank, rank, 8, 1.0 / 8)
        for step in range(4) for layer in (0, 20, 39) for rank in range(8)
    ]


class T32GuardTests(unittest.TestCase):
    def test_pins_capture_selector_geometry_oracle_and_commands(self):
        self.assertEqual(guard.ENGINE_SHA256, "bcdf3e8859011deefeb79967b8b4b1e27910bc03a344d44fc0fc07c4856176aa")
        self.assertEqual(guard.CAPTURE_BYTES, 154_540_864)
        self.assertEqual(guard.RECORD_COUNT, 96)
        self.assertIn("COLI_W4_CAPTURE=gateup-r1", guard.model_argv())
        self.assertIn(f"COLI_W4_CAPTURE_FILE={guard.CAPTURE_PATH}", guard.model_argv())
        identity = Path("/tmp/t32-test-key")
        self.assertEqual(client.ssh_argv("host", identity)[-1], guard.EXPECTED_ORIGINAL_COMMAND)
        self.assertEqual(client.ssh_argv("host", identity, read_results=True)[-1], guard.RESULTS_ORIGINAL_COMMAND)

    def test_accepts_only_the_complete_fixed_capture_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.wgcap"
            helpers.write_capture(path, hidden=2048, intermediate=512, group=256, records=fixed_records(),
                                  target_layer_count=3, target_step_count=4)
            summary = guard.capture_summary(path)
        self.assertEqual(summary["records"], 96)
        self.assertEqual(summary["calibration_records"], 48)
        self.assertEqual(summary["holdout_records"], 48)
        bad = fixed_records(); bad[-1] = bad[-2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.wgcap"
            helpers.write_capture(path, hidden=2048, intermediate=512, group=256, records=bad,
                                  target_layer_count=3, target_step_count=4)
            with self.assertRaisesRegex(ValueError, "route"):
                guard.capture_summary(path)

    def test_fails_closed_when_the_source_summary_does_not_report_full_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "qwen36"; engine.write_bytes(b"t32"); engine.chmod(0o755)
            stderr = root / "stderr.txt"
            stderr.write_text("[wgcap] records=95 expected=96 errors=0 bytes=154540864 path=/tmp/capture.wgcap\n")
            base_result = {"status": "pass", "stdout": {"sha256": guard.EXPECTED_STDOUT_SHA256}, "stderr": {"path": str(stderr)}}
            with mock.patch.object(guard, "ENGINE", engine), \
                 mock.patch.object(guard, "ENGINE_SHA256", hashlib.sha256(b"t32").hexdigest()), \
                 mock.patch.object(guard, "RESULT_DIRECTORY", root), \
                 mock.patch.object(guard, "CAPTURE_PATH", root / "capture.wgcap"), \
                 mock.patch.object(guard, "_base_run", return_value=base_result):
                result = guard.run({"dry_run": False})
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["failure_reason"], "capture_source_summary_mismatch")


if __name__ == "__main__":
    unittest.main()
