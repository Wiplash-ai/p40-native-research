import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import guard
import snapshot
import summarize

PYTHON = [sys.executable, "-c"]

class T01HarnessTests(unittest.TestCase):
    def setUp(self): self.temp = tempfile.TemporaryDirectory(); self.lock = str(Path(self.temp.name) / "host.lock")
    def tearDown(self): self.temp.cleanup()
    def sleeping_command(self): return PYTHON + ["import time; time.sleep(30)"]
    def fresh(self): return guard.Telemetry(time.monotonic(), {"0": 35.0, "1": 35.0})
    def test_snapshot_is_allowlisted_and_hashes_fixture(self):
        old = os.environ.get("SECRET_FOR_TEST"); os.environ["SECRET_FOR_TEST"] = "nope"
        data = snapshot.collect(model=str(ROOT / "fixtures/prompt.txt"))
        self.assertNotIn("SECRET_FOR_TEST", data["environment"]); self.assertEqual(len(data["model"]["sha256"]), 64)
        if old is None: del os.environ["SECRET_FOR_TEST"]
    def test_stale_polling_terminates_fake_job(self):
        result = guard.run(self.sleeping_command(), lambda: guard.Telemetry(0, {"0": 30}), lock_path=self.lock, timeout=2)
        self.assertEqual(result["failure_reason"], "stale_telemetry"); self.assertEqual(result["cleanup"]["status"], "complete")
    def test_rising_temperature_terminates_fake_job(self):
        clock = iter([10.0, 20.1]); samples = iter([guard.Telemetry(10, {"0":55}), guard.Telemetry(20.1, {"0":61})])
        watchdog = guard.Watchdog(now=lambda: next(clock))
        self.assertIsNone(watchdog.check(next(samples))); self.assertEqual(watchdog.check(next(samples)), "temperature_slope")
    def test_telemetry_jsonl_parser_rejects_incomplete_collector_record(self):
        sample = guard.parse_telemetry('{"timestamp": 1, "temperatures": {"0": 31}}')
        self.assertEqual(sample.temperatures["0"], 31.0)
        with self.assertRaises(ValueError): guard.parse_telemetry('{"timestamp": 1}')

    def test_telemetry_jsonl_parser_fails_closed_on_any_bad_record(self):
        telemetry_file = Path(self.temp.name) / "telemetry.jsonl"
        telemetry_file.write_text('{"timestamp": 1, "temperatures": {"0": 31}}\n{"timestamp": 2}\n', encoding="utf-8")
        with self.assertRaises(ValueError): guard.parse_telemetry_jsonl(str(telemetry_file))
    def test_fan_critical_prevents_continued_job(self):
        result = guard.run(self.sleeping_command(), lambda: guard.Telemetry(time.monotonic(), {"0":30}, fan_critical=True), lock_path=self.lock)
        self.assertEqual(result["failure_reason"], "fan_critical")
    def test_device_error_prevents_continued_job(self):
        result = guard.run(self.sleeping_command(), lambda: guard.Telemetry(time.monotonic(), {"0":30}, device_error=True), lock_path=self.lock)
        self.assertEqual(result["failure_reason"], "device_error")
    def test_subprocess_failure_is_recorded(self):
        result = guard.run(PYTHON + ["raise SystemExit(7)"], self.fresh, lock_path=self.lock)
        self.assertEqual(result["failure_reason"], "subprocess_failure")
    def test_watchdog_timeout_is_recorded(self):
        result = guard.run(self.sleeping_command(), self.fresh, lock_path=self.lock, timeout=.01, poll_seconds=.02)
        self.assertEqual(result["failure_reason"], "watchdog_timeout")
    def test_failed_power_restore_is_durable_failure(self):
        def broken_restore(): raise RuntimeError("mock restore failed")
        result = guard.run(PYTHON + ["pass"], self.fresh, lock_path=self.lock, restore_power=broken_restore)
        self.assertEqual(result["status"], "fail"); self.assertEqual(result["cleanup"]["status"], "restore_failed")
    def test_lock_protects_host_wide_runner(self):
        with guard.HostLock(self.lock):
            result = guard.run(PYTHON + ["pass"], self.fresh, lock_path=self.lock)
        self.assertEqual(result["failure_reason"], "benchmark_lock_held")
        self.assertEqual(result["status"], "fail")
    def test_cleanup_does_not_touch_unrelated_process(self):
        unrelated = subprocess.Popen(self.sleeping_command())
        try:
            result = guard.run(self.sleeping_command(), lambda: guard.Telemetry(0, {}), lock_path=self.lock)
            self.assertEqual(result["cleanup"]["status"], "complete"); self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate(); unrelated.wait()
    def test_disconnect_cleanup_primitive_only_targets_job_group(self):
        process = subprocess.Popen(self.sleeping_command(), start_new_session=True)
        record = guard.cleanup_process_group(process, grace_seconds=.01)
        self.assertEqual(record["status"], "complete"); self.assertIsNotNone(process.returncode)
    def test_dry_run_never_initializes_cuda(self):
        output = subprocess.check_output([sys.executable, str(ROOT / "scripts/run_matrix.py"), "--dry-run"], text=True)
        data = json.loads(output); self.assertFalse(data["cuda_initialized"]); self.assertIn("argv", data["cases"][0])

    def test_guard_cli_fails_closed_without_dry_run(self):
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/guard.py"), "--", "/bin/echo", "inert"], text=True, capture_output=True)
        self.assertNotEqual(completed.returncode, 0); self.assertIn("--dry-run only", completed.stderr)
    def test_result_validator_accepts_guard_result(self):
        result = guard.run(PYTHON + ["pass"], self.fresh, lock_path=self.lock)
        self.assertEqual(summarize.validate(result), [])

if __name__ == "__main__": unittest.main()
