import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "remote/p40-canary-guard.py"
SPEC = importlib.util.spec_from_file_location("p40_canary_guard", REMOTE)
guard = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def request(**changes):
    value = {
        "primitive": "P01-copy", "gpu": 0, "duration_seconds": 1,
        "bytes": 64 * 1024 * 1024, "memory_cap_mib": 256, "seed": 1, "dry_run": True,
    }
    value.update(changes)
    return value


class CanaryGuardTests(unittest.TestCase):
    def test_dry_run_profile_has_no_cuda_initialization(self):
        result = guard.run(guard.parse_request(json.dumps(request())))
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse(result["cuda_initialized"])

    def test_result_path_is_uuid_not_request_controlled(self):
        result = guard.run(guard.parse_request(json.dumps(request())))
        self.assertNotIn("result_path", result)
        self.assertRegex(result["run_id"], r"^[0-9a-f-]{36}$")

    def test_command_is_fixed_to_allowlisted_benchmark(self):
        argv = guard.benchmark_argv(request(primitive="P03-dp4a-gemv", gpu=1))
        self.assertEqual(argv[0], str(guard.BENCHMARK))
        self.assertNotIn("sh", argv)
        self.assertIn("P03-dp4a-gemv", argv)

    def test_unrecognized_or_oversized_profile_is_rejected(self):
        with self.assertRaises(guard.UnsafeRequest): guard.parse_request(json.dumps(request(command="whoami")))
        with self.assertRaises(guard.UnsafeRequest): guard.parse_request(json.dumps(request(bytes=257 * 1024 * 1024)))
        with self.assertRaises(guard.UnsafeRequest): guard.parse_request(json.dumps(request(duration_seconds=11)))

    def test_new_critical_sel_event_is_detected_as_delta_not_history(self):
        self.assertEqual(guard.new_critical_events({"200b"}, {"200b", "200c"}), {"200c"})

    def test_safety_predicate_rejects_hot_or_failed_fans(self):
        gpus = [{"temperature_c": 65, "memory_used_mib": 0}]
        fans = [{"rpm": 2000} for _ in range(8)]
        self.assertEqual(guard.unsafe_reason(gpus, fans, require_idle=True), "temperature_limit")
        gpus[0]["temperature_c"] = 35; fans[0]["rpm"] = 400
        self.assertEqual(guard.unsafe_reason(gpus, fans, require_idle=True), "fan_critical")

    def test_safety_predicate_requires_cool_empty_start(self):
        gpus = [{"temperature_c": 41, "memory_used_mib": 0}]
        fans = [{"rpm": 2000} for _ in range(8)]
        self.assertEqual(guard.unsafe_reason(gpus, fans, require_idle=True), "not_cool_or_idle")

    def test_cooldown_requires_minimum_time_and_temperature_recovery(self):
        hot = [{"index": 0, "temperature_c": 41.0, "memory_used_mib": 0.0}]
        cool = [{"index": 0, "temperature_c": 40.0, "memory_used_mib": 0.0}]
        fans = [{"rpm": 2000} for _ in range(8)]
        samples = []
        with mock.patch.object(guard, "gpu_state", side_effect=[hot, cool]), \
             mock.patch.object(guard, "fan_state", return_value=fans), \
             mock.patch.object(guard, "critical_fan_events", return_value=set()), \
             mock.patch.object(guard.time, "monotonic", side_effect=[0.0, 300.0, 301.0]), \
             mock.patch.object(guard.time, "sleep"):
            self.assertIsNone(guard.cool_down(samples, set()))
        self.assertEqual(len(samples), 2)

    def test_cooldown_times_out_only_at_extended_deadline(self):
        hot = [{"index": 0, "temperature_c": 41.0, "memory_used_mib": 0.0}]
        fans = [{"rpm": 2000} for _ in range(8)]
        with mock.patch.object(guard, "gpu_state", return_value=hot), \
             mock.patch.object(guard, "fan_state", return_value=fans), \
             mock.patch.object(guard, "critical_fan_events", return_value=set()), \
             mock.patch.object(guard.time, "monotonic", side_effect=[0.0, 900.0]), \
             mock.patch.object(guard.time, "sleep"):
            self.assertEqual(guard.cool_down([], set()), "cooldown_timeout")
