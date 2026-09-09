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

SWEEP_SPEC = importlib.util.spec_from_file_location("p40_t05c_sweep_guard", ROOT / "remote/p40-t05c-sweep-guard.py")
sweep = importlib.util.module_from_spec(SWEEP_SPEC)
assert SWEEP_SPEC and SWEEP_SPEC.loader
sys.modules[SWEEP_SPEC.name] = sweep
SWEEP_SPEC.loader.exec_module(sweep)

SWEEP_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t05c_client", ROOT / "scripts/p40_t05c_sweep_client.py")
sweep_client = importlib.util.module_from_spec(SWEEP_CLIENT_SPEC)
assert SWEEP_CLIENT_SPEC and SWEEP_CLIENT_SPEC.loader
sys.modules[SWEEP_CLIENT_SPEC.name] = sweep_client
SWEEP_CLIENT_SPEC.loader.exec_module(sweep_client)

DIAGNOSTIC_SPEC = importlib.util.spec_from_file_location("p40_t05d_diagnostic_guard", ROOT / "remote/p40-t05d-diagnostic-guard.py")
diagnostic = importlib.util.module_from_spec(DIAGNOSTIC_SPEC)
assert DIAGNOSTIC_SPEC and DIAGNOSTIC_SPEC.loader
sys.modules[DIAGNOSTIC_SPEC.name] = diagnostic
DIAGNOSTIC_SPEC.loader.exec_module(diagnostic)

DIAGNOSTIC_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t05d_client", ROOT / "scripts/p40_t05d_diagnostic_client.py")
diagnostic_client = importlib.util.module_from_spec(DIAGNOSTIC_CLIENT_SPEC)
assert DIAGNOSTIC_CLIENT_SPEC and DIAGNOSTIC_CLIENT_SPEC.loader
sys.modules[DIAGNOSTIC_CLIENT_SPEC.name] = diagnostic_client
DIAGNOSTIC_CLIENT_SPEC.loader.exec_module(diagnostic_client)

CPUORDER_SPEC = importlib.util.spec_from_file_location("p40_t05e_cpuorder_guard", ROOT / "remote/p40-t05e-cpuorder-guard.py")
cpuorder = importlib.util.module_from_spec(CPUORDER_SPEC)
assert CPUORDER_SPEC and CPUORDER_SPEC.loader
sys.modules[CPUORDER_SPEC.name] = cpuorder
CPUORDER_SPEC.loader.exec_module(cpuorder)

CPUORDER_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t05e_client", ROOT / "scripts/p40_t05e_cpuorder_client.py")
cpuorder_client = importlib.util.module_from_spec(CPUORDER_CLIENT_SPEC)
assert CPUORDER_CLIENT_SPEC and CPUORDER_CLIENT_SPEC.loader
sys.modules[CPUORDER_CLIENT_SPEC.name] = cpuorder_client
CPUORDER_CLIENT_SPEC.loader.exec_module(cpuorder_client)

SHUFFLE_SPEC = importlib.util.spec_from_file_location("p40_t05f_shuffle_guard", ROOT / "remote/p40-t05f-shuffle-guard.py")
shuffle = importlib.util.module_from_spec(SHUFFLE_SPEC)
assert SHUFFLE_SPEC and SHUFFLE_SPEC.loader
sys.modules[SHUFFLE_SPEC.name] = shuffle
SHUFFLE_SPEC.loader.exec_module(shuffle)

SHUFFLE_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t05f_client", ROOT / "scripts/p40_t05f_shuffle_client.py")
shuffle_client = importlib.util.module_from_spec(SHUFFLE_CLIENT_SPEC)
assert SHUFFLE_CLIENT_SPEC and SHUFFLE_CLIENT_SPEC.loader
sys.modules[SHUFFLE_CLIENT_SPEC.name] = shuffle_client
SHUFFLE_CLIENT_SPEC.loader.exec_module(shuffle_client)

FULL_SWEEP_SPEC = importlib.util.spec_from_file_location("p40_t05g_full_sweep_guard", ROOT / "remote/p40-t05g-full-sweep-guard.py")
full_sweep = importlib.util.module_from_spec(FULL_SWEEP_SPEC)
assert FULL_SWEEP_SPEC and FULL_SWEEP_SPEC.loader
sys.modules[FULL_SWEEP_SPEC.name] = full_sweep
FULL_SWEEP_SPEC.loader.exec_module(full_sweep)

FULL_SWEEP_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t05g_client", ROOT / "scripts/p40_t05g_full_sweep_client.py")
full_sweep_client = importlib.util.module_from_spec(FULL_SWEEP_CLIENT_SPEC)
assert FULL_SWEEP_CLIENT_SPEC and FULL_SWEEP_CLIENT_SPEC.loader
sys.modules[FULL_SWEEP_CLIENT_SPEC.name] = full_sweep_client
FULL_SWEEP_CLIENT_SPEC.loader.exec_module(full_sweep_client)

T08_SPEC = importlib.util.spec_from_file_location("p40_t08_lmhead_guard", ROOT / "remote/p40-t08-lmhead-cpuorder-guard.py")
t08 = importlib.util.module_from_spec(T08_SPEC)
assert T08_SPEC and T08_SPEC.loader
sys.modules[T08_SPEC.name] = t08
T08_SPEC.loader.exec_module(t08)

T08_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t08_lmhead_client", ROOT / "scripts/p40_t08_lmhead_cpuorder_client.py")
t08_client = importlib.util.module_from_spec(T08_CLIENT_SPEC)
assert T08_CLIENT_SPEC and T08_CLIENT_SPEC.loader
sys.modules[T08_CLIENT_SPEC.name] = t08_client
T08_CLIENT_SPEC.loader.exec_module(t08_client)

T11_SPEC = importlib.util.spec_from_file_location("p40_t11_attention_guard", ROOT / "remote/p40-t11-attention-cpuorder-guard.py")
t11 = importlib.util.module_from_spec(T11_SPEC)
assert T11_SPEC and T11_SPEC.loader
sys.modules[T11_SPEC.name] = t11
T11_SPEC.loader.exec_module(t11)

T11_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t11_attention_client", ROOT / "scripts/p40_t11_attention_cpuorder_client.py")
t11_client = importlib.util.module_from_spec(T11_CLIENT_SPEC)
assert T11_CLIENT_SPEC and T11_CLIENT_SPEC.loader
sys.modules[T11_CLIENT_SPEC.name] = t11_client
T11_CLIENT_SPEC.loader.exec_module(t11_client)

T14_SPEC = importlib.util.spec_from_file_location("p40_t14_shared_expert_guard", ROOT / "remote/p40-t14-shared-expert-cpuorder-guard.py")
t14 = importlib.util.module_from_spec(T14_SPEC)
assert T14_SPEC and T14_SPEC.loader
sys.modules[T14_SPEC.name] = t14
T14_SPEC.loader.exec_module(t14)

T14_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t14_shared_expert_client", ROOT / "scripts/p40_t14_shared_expert_cpuorder_client.py")
t14_client = importlib.util.module_from_spec(T14_CLIENT_SPEC)
assert T14_CLIENT_SPEC and T14_CLIENT_SPEC.loader
sys.modules[T14_CLIENT_SPEC.name] = t14_client
T14_CLIENT_SPEC.loader.exec_module(t14_client)

T17_SPEC = importlib.util.spec_from_file_location("p40_t17_w4a8_dp4a_guard", ROOT / "remote/p40-t17-w4a8-dp4a-guard.py")
t17 = importlib.util.module_from_spec(T17_SPEC)
assert T17_SPEC and T17_SPEC.loader
sys.modules[T17_SPEC.name] = t17
T17_SPEC.loader.exec_module(t17)

T17_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t17_w4a8_dp4a_client", ROOT / "scripts/p40_t17_w4a8_dp4a_client.py")
t17_client = importlib.util.module_from_spec(T17_CLIENT_SPEC)
assert T17_CLIENT_SPEC and T17_CLIENT_SPEC.loader
sys.modules[T17_CLIENT_SPEC.name] = t17_client
T17_CLIENT_SPEC.loader.exec_module(t17_client)


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

    def test_sweep_identity_pins_the_30_layer_binary_and_command(self):
        self.assertEqual(sweep.t05a.EXPECTED_ORIGINAL_COMMAND, "p40-t05c-sweep")
        self.assertEqual(sweep.t05a.PROFILE_ID, "t05c-dn-sweep-30x-triplet-2048-8192-4096")
        self.assertIn("qwen_dn_sweep_control", str(sweep.t05a.BENCHMARK))
        self.assertIn("dn-sweep-30x-triplet-2048-8192-4096", sweep.t05a.FIXED_ARGUMENTS)
        self.assertEqual(sweep.t05a.WATCHDOG_SECONDS, 240)
        identity = Path("/tmp/p40-t05c-test-key")
        self.assertEqual(sweep_client.ssh_argv("host", identity)[-1], "p40-t05c-sweep")

    def test_diagnostic_identity_is_distinct_from_the_rejected_t05c_run(self):
        self.assertEqual(diagnostic.t05a.EXPECTED_ORIGINAL_COMMAND, "p40-t05d-diagnostic")
        self.assertEqual(diagnostic.t05a.PROFILE_ID, "t05d-dn-error-attribution-30x-triplet")
        self.assertEqual(diagnostic.t05a.BENCHMARK_SHA256, "3d8043ee2846e2d5942a823bdbeaa43fffea331e9175da92d5d8c790233115a7")
        self.assertIn("qwen_dn_sweep_control", str(diagnostic.t05a.BENCHMARK))
        identity = Path("/tmp/p40-t05d-test-key")
        self.assertEqual(diagnostic_client.ssh_argv("host", identity)[-1], "p40-t05d-diagnostic")

    def test_cpuorder_identity_pins_the_standalone_out_projection_control(self):
        self.assertEqual(cpuorder.t05a.EXPECTED_ORIGINAL_COMMAND, "p40-t05e-cpuorder")
        self.assertEqual(cpuorder.t05a.PROFILE_ID, "t05e-dn-out-cpuorder-4096x2048")
        self.assertIn("qwen_dn_out_cpuorder_control", str(cpuorder.t05a.BENCHMARK))
        self.assertIn("dn-out-cpuorder-4096x2048", cpuorder.t05a.FIXED_ARGUMENTS)
        identity = Path("/tmp/p40-t05e-test-key")
        self.assertEqual(cpuorder_client.ssh_argv("host", identity)[-1], "p40-t05e-cpuorder")

    def test_shuffle_identity_pins_the_optimized_exact_order_binary(self):
        self.assertEqual(shuffle.t05a.EXPECTED_ORIGINAL_COMMAND, "p40-t05f-shuffle")
        self.assertEqual(shuffle.t05a.PROFILE_ID, "t05f-dn-out-cpuorder-shuffle-4096x2048")
        self.assertEqual(shuffle.t05a.BENCHMARK_SHA256, "c2b2873420ac96a57d21f5a04ae1cdc1221c127642e2cf25cee175f2f8dc5e64")
        self.assertIn("qwen_dn_out_cpuorder_control", str(shuffle.t05a.BENCHMARK))
        identity = Path("/tmp/p40-t05f-test-key")
        self.assertEqual(shuffle_client.ssh_argv("host", identity)[-1], "p40-t05f-shuffle")

    def test_full_sweep_identity_pins_the_exact_30_layer_binary(self):
        self.assertEqual(full_sweep.t05a.EXPECTED_ORIGINAL_COMMAND, "p40-t05g-full-sweep")
        self.assertEqual(full_sweep.t05a.PROFILE_ID, "t05g-dn-cpuorder-30x-triplet-2048-8192-4096")
        self.assertEqual(full_sweep.t05a.BENCHMARK_SHA256, "b1d9dcbad1d2efb008a851782bace32707f620bc752bf3cbd8ad6d5b2f248a9e")
        self.assertIn("qwen_dn_sweep_cpuorder_control", str(full_sweep.t05a.BENCHMARK))
        identity = Path("/tmp/p40-t05g-test-key")
        self.assertEqual(full_sweep_client.ssh_argv("host", identity)[-1], "p40-t05g-full-sweep")

    def test_t08_identity_pins_the_real_lmhead_shape_and_binary(self):
        self.assertEqual(t08.t05.EXPECTED_ORIGINAL_COMMAND, "p40-t08-lmhead-cpuorder")
        self.assertEqual(t08.t05.PROFILE_ID, "t08-lmhead-cpuorder-2048x248044")
        self.assertEqual(t08.t05.BENCHMARK_SHA256, "25d36744dca6761d93082500f8c49b70779835c5ca1001fbd45391fc2d70d777")
        self.assertIn("qwen_lmhead_cpuorder_control", str(t08.t05.BENCHMARK))
        self.assertEqual(t08.t05.FIXED_ARGUMENTS[-6:], ("--repetitions", "3", "--calls-per-sample", "1", "--memory-cap-mib", "512", "--seed", "1")[-6:])
        identity = Path("/tmp/p40-t08-test-key")
        self.assertEqual(t08_client.ssh_argv("host", identity)[-1], "p40-t08-lmhead-cpuorder")
        self.assertEqual(t08_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t08-lmhead-cpuorder-results")

    def test_t11_identity_pins_the_ten_layer_attention_projection_sweep(self):
        self.assertEqual(t11.t05.EXPECTED_ORIGINAL_COMMAND, "p40-t11-attention-cpuorder")
        self.assertEqual(t11.t05.PROFILE_ID, "t11-attention-projections-cpuorder-10x")
        self.assertEqual(t11.t05.BENCHMARK_SHA256, "b7d74121b78bf7a7003c83eea88dcacf68754e5de27f122b470e4e83f9427ca9")
        self.assertIn("qwen_attention_cpuorder_control", str(t11.t05.BENCHMARK))
        self.assertIn("attention-projections-cpuorder-10x-2048-8192-512-4096-2048", t11.t05.FIXED_ARGUMENTS)
        self.assertEqual(t11.t05.FIXED_ARGUMENTS[-6:], ("--repetitions", "3", "--calls-per-sample", "1", "--memory-cap-mib", "288", "--seed", "1")[-6:])
        identity = Path("/tmp/p40-t11-test-key")
        self.assertEqual(t11_client.ssh_argv("host", identity)[-1], "p40-t11-attention-cpuorder")
        self.assertEqual(t11_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t11-attention-cpuorder-results")

    def test_t14_identity_pins_the_40_layer_shared_expert_sequence(self):
        self.assertEqual(t14.t05.EXPECTED_ORIGINAL_COMMAND, "p40-t14-shared-expert-cpuorder")
        self.assertEqual(t14.t05.PROFILE_ID, "t14-shared-expert-cpuorder-40x")
        self.assertEqual(t14.t05.BENCHMARK_SHA256, "3e6b1ef1c65ec5cb2e937e91089763047bf9755655a2870209683773ddb86228")
        self.assertIn("qwen_shared_expert_cpuorder_control", str(t14.t05.BENCHMARK))
        self.assertIn("shared-expert-cpuorder-40x-2048-512-2048", t14.t05.FIXED_ARGUMENTS)
        self.assertEqual(t14.t05.FIXED_ARGUMENTS[-6:], ("--repetitions", "3", "--calls-per-sample", "1", "--memory-cap-mib", "160", "--seed", "1")[-6:])
        identity = Path("/tmp/p40-t14-test-key")
        self.assertEqual(t14_client.ssh_argv("host", identity)[-1], "p40-t14-shared-expert-cpuorder")
        self.assertEqual(t14_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t14-shared-expert-cpuorder-results")

    def test_t17_identity_pins_the_w4a8_dp4a_projection_control(self):
        self.assertEqual(t17.t05.EXPECTED_ORIGINAL_COMMAND, "p40-t17-w4a8-dp4a")
        self.assertEqual(t17.t05.PROFILE_ID, "t17-w4a8-dp4a-expert-projection-4x")
        self.assertEqual(t17.t05.BENCHMARK_SHA256, "6787701b7acd642f88a5128a8111b262ce0ccea3765bf0351ca4ec7076b5fafe")
        self.assertIn("qwen_w4a8_dp4a_control", str(t17.t05.BENCHMARK))
        self.assertIn("w4a8-dp4a-expert-proj-4x-2048-512", t17.t05.FIXED_ARGUMENTS)
        self.assertEqual(t17.t05.FIXED_ARGUMENTS[-6:], ("--repetitions", "3", "--calls-per-sample", "64", "--memory-cap-mib", "32", "--seed", "1")[-6:])
        identity = Path("/tmp/p40-t17-test-key")
        self.assertEqual(t17_client.ssh_argv("host", identity)[-1], "p40-t17-w4a8-dp4a")
        self.assertEqual(t17_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t17-w4a8-dp4a-results")

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
        popen_kwargs = {}
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
                 mock.patch.object(guard.subprocess, "Popen", side_effect=lambda *args, **kwargs: (popen_kwargs.update(kwargs), FinishedProcess())[1]):
                result = guard.run({"dry_run": False})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(calls, [(0, 125), (0, 250.0)])
        self.assertEqual(popen_kwargs["env"], guard.RUN_ENV)
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
