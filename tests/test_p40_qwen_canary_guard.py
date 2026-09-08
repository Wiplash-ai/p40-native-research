import importlib.util
import json
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p40_qwen_canary_guard", ROOT / "remote/p40-qwen-canary-guard.py")
guard = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)

CONTROL_SPEC = importlib.util.spec_from_file_location("p40_qwen_control_64", ROOT / "remote/p40-qwen-control-64.py")
control = importlib.util.module_from_spec(CONTROL_SPEC)
assert CONTROL_SPEC and CONTROL_SPEC.loader
sys.modules[CONTROL_SPEC.name] = control
CONTROL_SPEC.loader.exec_module(control)

CONTROL_256_SPEC = importlib.util.spec_from_file_location("p40_qwen_control_256", ROOT / "remote/p40-qwen-control-256.py")
control_256 = importlib.util.module_from_spec(CONTROL_256_SPEC)
assert CONTROL_256_SPEC and CONTROL_256_SPEC.loader
sys.modules[CONTROL_256_SPEC.name] = control_256
CONTROL_256_SPEC.loader.exec_module(control_256)

CLIENT_SPEC = importlib.util.spec_from_file_location("p40_qwen_control_client", ROOT / "scripts/p40_qwen_control_client.py")
client = importlib.util.module_from_spec(CLIENT_SPEC)
assert CLIENT_SPEC and CLIENT_SPEC.loader
sys.modules[CLIENT_SPEC.name] = client
CLIENT_SPEC.loader.exec_module(client)


class QwenCanaryGuardTests(unittest.TestCase):
    def test_base_guard_prefers_the_root_owned_install_name(self):
        self.assertEqual(guard.SOURCE.name, "p40-canary-guard.py")
        installed_source = (ROOT / "remote/p40-qwen-canary-guard.py").with_name("p40-canary-guard")
        self.assertEqual(installed_source.name, "p40-canary-guard")
        extensionless_loader = guard.SourceFileLoader("p40_primitive_guard", str(installed_source))
        self.assertIsNotNone(importlib.util.spec_from_loader("p40_primitive_guard", extensionless_loader))

    def test_dry_run_never_initializes_cuda(self):
        result = guard.run(guard.parse_request(json.dumps({"dry_run": True})))
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse(result["cuda_initialized"])

    def test_request_has_no_command_surface(self):
        for request in ({}, {"dry_run": 1}, {"dry_run": True, "command": "sh"}):
            with self.assertRaises(guard.UnsafeRequest): guard.parse_request(json.dumps(request))

    def test_command_is_fixed_to_pinned_model_and_two_gpus(self):
        argv = guard.model_argv()
        self.assertEqual(argv[0:2], ["/usr/bin/env", "-i"])
        self.assertIn("COLI_GPUS=0,1", argv)
        self.assertIn("COLI_CUDA_PROFILE=1", argv)
        self.assertIn(str(guard.ENGINE), argv)
        self.assertIn(f"SNAP={guard.MODEL}", argv)
        self.assertNotIn("sh", argv)

    def test_control_profile_has_a_distinct_fixed_64_output_identity(self):
        self.assertEqual(control.EXPECTED_ORIGINAL_COMMAND, "p40-qwen-control-64")
        self.assertEqual(control.qwen.OUTPUT_TOKENS, 64)
        self.assertEqual(control.qwen.PROFILE_ID, "control-64")
        self.assertEqual(control.qwen.SCHEMA_VERSION, "p40-qwen-control-64-v1")
        self.assertIn("N_NEW=64", control.qwen.model_argv())
        self.assertEqual(control.qwen.parse_request('{"dry_run": true}'), {"dry_run": True})

    def test_plateau_profile_has_a_distinct_fixed_256_output_identity(self):
        self.assertEqual(control_256.EXPECTED_ORIGINAL_COMMAND, "p40-qwen-control-256")
        self.assertEqual(control_256.qwen.OUTPUT_TOKENS, 256)
        self.assertEqual(control_256.qwen.PROFILE_ID, "control-256")
        self.assertEqual(control_256.qwen.SCHEMA_VERSION, "p40-qwen-control-256-v1")
        self.assertIn("N_NEW=256", control_256.qwen.model_argv())

    def test_client_only_selects_forced_profile_commands(self):
        identity = Path("/tmp/p40-test-key")
        self.assertEqual(client.ssh_argv("host", identity, "control-64")[-1], "p40-qwen-control-64")
        self.assertEqual(client.ssh_argv("host", identity, "control-256")[-1], "p40-qwen-control-256")
        self.assertEqual(client.ssh_argv("host", identity, "canary-16", read_results=True)[-1], "p40-qwen-canary-results")

    def test_mocked_run_caps_and_restores_both_gpus_and_records_output(self):
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
        with self.subTest("safety cleanup"):
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                engine = root / "qwen36"; engine.write_text("placeholder"); engine.chmod(0o755)
                model = root / "model"; model.mkdir()
                prompt = root / "prompt.txt"; prompt.write_text("fixed prompt")
                with mock.patch.object(guard, "ENGINE", engine), \
                     mock.patch.object(guard, "MODEL", model), \
                     mock.patch.object(guard, "PROMPT", prompt), \
                     mock.patch.object(guard, "RESULT_DIRECTORY", root / "results"), \
                     mock.patch.object(guard.base, "gpu_state", return_value=gpus), \
                     mock.patch.object(guard.base, "fan_state", return_value=fans), \
                     mock.patch.object(guard.base, "critical_fan_events", return_value=set()), \
                     mock.patch.object(guard.base, "new_critical_events", return_value=set()), \
                     mock.patch.object(guard.base, "cool_down", return_value=None), \
                     mock.patch.object(guard.base, "set_power_limit", side_effect=lambda gpu, watts: calls.append((gpu, watts))), \
                     mock.patch.object(guard.subprocess, "Popen", return_value=FinishedProcess()):
                    result = guard.run({"dry_run": False})
                self.assertEqual(result["status"], "pass")
                self.assertEqual(calls, [(0, 125), (1, 125), (0, 250.0), (1, 250.0)])
                self.assertEqual(result["stdout"]["bytes"], 0)
                self.assertEqual(result["stderr"]["bytes"], 0)

    def test_unsafe_start_telemetry_fails_closed_before_model_process(self):
        from tempfile import TemporaryDirectory
        hot_gpus = [
            {"index": 0, "temperature_c": 65.0, "power_limit_w": 250.0, "memory_used_mib": 0.0, "utilization_percent": 0.0},
            {"index": 1, "temperature_c": 36.0, "power_limit_w": 250.0, "memory_used_mib": 0.0, "utilization_percent": 0.0},
        ]
        fans = [{"name": f"FAN{i}", "rpm": 2000.0} for i in range(1, 9)]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "qwen36"; engine.write_text("placeholder"); engine.chmod(0o755)
            model = root / "model"; model.mkdir()
            prompt = root / "prompt.txt"; prompt.write_text("fixed prompt")
            with mock.patch.object(guard, "ENGINE", engine), \
                 mock.patch.object(guard, "MODEL", model), \
                 mock.patch.object(guard, "PROMPT", prompt), \
                 mock.patch.object(guard.base, "gpu_state", return_value=hot_gpus), \
                 mock.patch.object(guard.base, "fan_state", return_value=fans), \
                 mock.patch.object(guard.base, "critical_fan_events", return_value=set()), \
                 mock.patch.object(guard.subprocess, "Popen") as popen:
                result = guard.run({"dry_run": False})
        self.assertEqual(result["failure_reason"], "temperature_limit")
        self.assertEqual(result["status"], "fail")
        popen.assert_not_called()

    def test_new_fan_critical_event_terminates_and_restores_both_gpus(self):
        class RunningProcess:
            returncode = -15

            def poll(self): return None

        from tempfile import TemporaryDirectory
        gpus = [
            {"index": 0, "temperature_c": 35.0, "power_limit_w": 250.0, "memory_used_mib": 0.0, "utilization_percent": 0.0},
            {"index": 1, "temperature_c": 36.0, "power_limit_w": 250.0, "memory_used_mib": 0.0, "utilization_percent": 0.0},
        ]
        fans = [{"name": f"FAN{i}", "rpm": 2000.0} for i in range(1, 9)]
        calls = []
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "qwen36"; engine.write_text("placeholder"); engine.chmod(0o755)
            model = root / "model"; model.mkdir()
            prompt = root / "prompt.txt"; prompt.write_text("fixed prompt")
            with mock.patch.object(guard, "ENGINE", engine), \
                 mock.patch.object(guard, "MODEL", model), \
                 mock.patch.object(guard, "PROMPT", prompt), \
                 mock.patch.object(guard, "RESULT_DIRECTORY", root / "results"), \
                 mock.patch.object(guard.base, "gpu_state", return_value=gpus), \
                 mock.patch.object(guard.base, "fan_state", return_value=fans), \
                 mock.patch.object(guard.base, "critical_fan_events", side_effect=[set(), {"42"}]), \
                 mock.patch.object(guard.base, "terminate", return_value=["term_process_group"]), \
                 mock.patch.object(guard.base, "cool_down", return_value=None), \
                 mock.patch.object(guard.base, "set_power_limit", side_effect=lambda gpu, watts: calls.append((gpu, watts))), \
                 mock.patch.object(guard.subprocess, "Popen", return_value=RunningProcess()), \
                 mock.patch.object(guard.time, "monotonic", side_effect=[0.0, 0.0, 0.0, 0.0]):
                result = guard.run({"dry_run": False})
        self.assertEqual(result["failure_reason"], "new_fan_critical_event")
        self.assertEqual(result["cleanup"]["actions"], ["term_process_group"])
        self.assertTrue(result["cleanup"]["power_restored"])
        self.assertEqual(calls, [(0, 125), (1, 125), (0, 250.0), (1, 250.0)])


if __name__ == "__main__": unittest.main()
