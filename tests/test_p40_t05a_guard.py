import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p40_t05a_guard", ROOT / "remote/p40-t05a-q8-guard.py")
guard = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)

CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t05a_client", ROOT / "scripts/p40_t05a_control_client.py")
client = importlib.util.module_from_spec(CLIENT_SPEC)
assert CLIENT_SPEC and CLIENT_SPEC.loader
sys.modules[CLIENT_SPEC.name] = client
CLIENT_SPEC.loader.exec_module(client)

TRIPLET_SPEC = importlib.util.spec_from_file_location("p40_t05b_triplet_guard", ROOT / "remote/p40-t05b-triplet-guard.py")
triplet = importlib.util.module_from_spec(TRIPLET_SPEC)
assert TRIPLET_SPEC and TRIPLET_SPEC.loader
sys.modules[TRIPLET_SPEC.name] = triplet
TRIPLET_SPEC.loader.exec_module(triplet)

TRIPLET_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t05b_client", ROOT / "scripts/p40_t05b_triplet_client.py")
triplet_client = importlib.util.module_from_spec(TRIPLET_CLIENT_SPEC)
assert TRIPLET_CLIENT_SPEC and TRIPLET_CLIENT_SPEC.loader
sys.modules[TRIPLET_CLIENT_SPEC.name] = triplet_client
TRIPLET_CLIENT_SPEC.loader.exec_module(triplet_client)


class T05AGuardTests(unittest.TestCase):
    def test_dry_run_never_initializes_cuda(self):
        result = guard.run(guard.parse_request('{"dry_run": true}'))
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse(result["cuda_initialized"])

    def test_request_and_command_are_fixed(self):
        for request in ({}, {"dry_run": 1}, {"dry_run": True, "command": "sh"}):
            with self.assertRaises(guard.UnsafeRequest):
                guard.parse_request(json.dumps(request))
        self.assertEqual(guard.benchmark_argv()[0], str(guard.BENCHMARK))
        self.assertEqual(guard.benchmark_argv()[1:], list(guard.FIXED_ARGUMENTS))
        self.assertEqual(guard.GPU, 0)
        self.assertEqual(guard.SAFE_POWER_W, 125)
        self.assertEqual(len(guard.BENCHMARK_SHA256), 64)

    def test_client_only_selects_t05a_commands(self):
        identity = Path("/tmp/p40-t05a-test-key")
        self.assertEqual(client.ssh_argv("host", identity)[-1], "p40-t05a-q8")
        self.assertEqual(client.ssh_argv("host", identity, read_results=True)[-1], "p40-t05a-q8-results")

    def test_triplet_identity_cannot_reuse_the_t05a_binary_or_command(self):
        self.assertEqual(triplet.t05a.EXPECTED_ORIGINAL_COMMAND, "p40-t05b-triplet")
        self.assertEqual(triplet.t05a.PROFILE_ID, "t05b-dn-triplet-2048-8192-4096")
        self.assertIn("qwen_dn_triplet_control", str(triplet.t05a.BENCHMARK))
        self.assertIn("dn-triplet-2048-8192-4096", triplet.t05a.FIXED_ARGUMENTS)
        identity = Path("/tmp/p40-t05b-test-key")
        self.assertEqual(triplet_client.ssh_argv("host", identity)[-1], "p40-t05b-triplet")

    def test_mocked_run_caps_restores_and_records_output(self):
        class FinishedProcess:
            returncode = 0
            def poll(self): return 0
            def wait(self, timeout): return 0

        gpus = [
            {"index": 0, "temperature_c": 35.0, "power_limit_w": 250.0, "memory_used_mib": 0.0, "utilization_percent": 0.0},
            {"index": 1, "temperature_c": 36.0, "power_limit_w": 250.0, "memory_used_mib": 0.0, "utilization_percent": 0.0},
        ]
        fans = [{"name": f"FAN{i}", "rpm": 2000.0} for i in range(1, 9)]
        calls = []
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "qwen_dn_q8_control"; binary.write_text("fixture"); binary.chmod(0o755)
            with mock.patch.object(guard, "BENCHMARK", binary), \
                 mock.patch.object(guard, "BENCHMARK_SHA256", guard.digest(binary)), \
                 mock.patch.object(guard, "RESULT_DIRECTORY", root / "results"), \
                 mock.patch.object(guard.base, "gpu_state", return_value=gpus), \
                 mock.patch.object(guard.base, "fan_state", return_value=fans), \
                 mock.patch.object(guard.base, "critical_fan_events", return_value=set()), \
                 mock.patch.object(guard.base, "cool_down", return_value=None), \
                 mock.patch.object(guard.base, "set_power_limit", side_effect=lambda gpu, watts: calls.append((gpu, watts))), \
                 mock.patch.object(guard.subprocess, "Popen", return_value=FinishedProcess()):
                result = guard.run({"dry_run": False})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(calls, [(0, 125), (0, 250.0)])
        self.assertEqual(result["stdout"]["bytes"], 0)
        self.assertEqual(result["stderr"]["bytes"], 0)

    def test_hot_preflight_does_not_launch_fixture(self):
        gpus = [
            {"index": 0, "temperature_c": 65.0, "power_limit_w": 250.0, "memory_used_mib": 0.0, "utilization_percent": 0.0},
            {"index": 1, "temperature_c": 36.0, "power_limit_w": 250.0, "memory_used_mib": 0.0, "utilization_percent": 0.0},
        ]
        fans = [{"name": f"FAN{i}", "rpm": 2000.0} for i in range(1, 9)]
        with TemporaryDirectory() as temporary:
            binary = Path(temporary) / "qwen_dn_q8_control"; binary.write_text("fixture"); binary.chmod(0o755)
            with mock.patch.object(guard, "BENCHMARK", binary), \
                 mock.patch.object(guard, "BENCHMARK_SHA256", guard.digest(binary)), \
                 mock.patch.object(guard.base, "gpu_state", return_value=gpus), \
                 mock.patch.object(guard.base, "fan_state", return_value=fans), \
                 mock.patch.object(guard.base, "critical_fan_events", return_value=set()), \
                 mock.patch.object(guard.subprocess, "Popen") as popen:
                result = guard.run({"dry_run": False})
        self.assertEqual(result["failure_reason"], "temperature_limit")
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
