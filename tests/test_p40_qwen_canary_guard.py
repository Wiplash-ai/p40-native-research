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

T06_SPEC = importlib.util.spec_from_file_location("p40_t06_qwen_dn_cpuorder", ROOT / "remote/p40-t06-qwen-dn-cpuorder-guard.py")
t06 = importlib.util.module_from_spec(T06_SPEC)
assert T06_SPEC and T06_SPEC.loader
sys.modules[T06_SPEC.name] = t06
T06_SPEC.loader.exec_module(t06)

T06B_SPEC = importlib.util.spec_from_file_location("p40_t06b_qwen_dn_cpuorder", ROOT / "remote/p40-t06b-qwen-dn-cpuorder-guard.py")
t06b = importlib.util.module_from_spec(T06B_SPEC)
assert T06B_SPEC and T06B_SPEC.loader
sys.modules[T06B_SPEC.name] = t06b
T06B_SPEC.loader.exec_module(t06b)

T06C_SPEC = importlib.util.spec_from_file_location("p40_t06c_qwen_dn_cpuorder", ROOT / "remote/p40-t06c-qwen-dn-cpuorder-guard.py")
t06c = importlib.util.module_from_spec(T06C_SPEC)
assert T06C_SPEC and T06C_SPEC.loader
sys.modules[T06C_SPEC.name] = t06c
T06C_SPEC.loader.exec_module(t06c)

T07_SPEC = importlib.util.spec_from_file_location("p40_t07_qwen_dn_cpuorder", ROOT / "remote/p40-t07-qwen-dn-cpuorder-guard.py")
t07 = importlib.util.module_from_spec(T07_SPEC)
assert T07_SPEC and T07_SPEC.loader
sys.modules[T07_SPEC.name] = t07
T07_SPEC.loader.exec_module(t07)

T09_SPEC = importlib.util.spec_from_file_location("p40_t09_qwen_lmhead", ROOT / "remote/p40-t09-qwen-lmhead-cpuorder-guard.py")
t09 = importlib.util.module_from_spec(T09_SPEC)
assert T09_SPEC and T09_SPEC.loader
sys.modules[T09_SPEC.name] = t09
T09_SPEC.loader.exec_module(t09)

T10_SPEC = importlib.util.spec_from_file_location("p40_t10_qwen_lmhead", ROOT / "remote/p40-t10-qwen-lmhead-cpuorder-guard.py")
t10 = importlib.util.module_from_spec(T10_SPEC)
assert T10_SPEC and T10_SPEC.loader
sys.modules[T10_SPEC.name] = t10
T10_SPEC.loader.exec_module(t10)

T12_SPEC = importlib.util.spec_from_file_location("p40_t12_qwen_attention", ROOT / "remote/p40-t12-qwen-attention-cpuorder-guard.py")
t12 = importlib.util.module_from_spec(T12_SPEC)
assert T12_SPEC and T12_SPEC.loader
sys.modules[T12_SPEC.name] = t12
T12_SPEC.loader.exec_module(t12)

T13_SPEC = importlib.util.spec_from_file_location("p40_t13_qwen_attention", ROOT / "remote/p40-t13-qwen-attention-cpuorder-guard.py")
t13 = importlib.util.module_from_spec(T13_SPEC)
assert T13_SPEC and T13_SPEC.loader
sys.modules[T13_SPEC.name] = t13
T13_SPEC.loader.exec_module(t13)

T15_SPEC = importlib.util.spec_from_file_location("p40_t15_qwen_shared", ROOT / "remote/p40-t15-qwen-shared-cpuorder-guard.py")
t15 = importlib.util.module_from_spec(T15_SPEC)
assert T15_SPEC and T15_SPEC.loader
sys.modules[T15_SPEC.name] = t15
T15_SPEC.loader.exec_module(t15)

T16_SPEC = importlib.util.spec_from_file_location("p40_t16_qwen_shared", ROOT / "remote/p40-t16-qwen-shared-cpuorder-guard.py")
t16 = importlib.util.module_from_spec(T16_SPEC)
assert T16_SPEC and T16_SPEC.loader
sys.modules[T16_SPEC.name] = t16
T16_SPEC.loader.exec_module(t16)

CLIENT_SPEC = importlib.util.spec_from_file_location("p40_qwen_control_client", ROOT / "scripts/p40_qwen_control_client.py")
client = importlib.util.module_from_spec(CLIENT_SPEC)
assert CLIENT_SPEC and CLIENT_SPEC.loader
sys.modules[CLIENT_SPEC.name] = client
CLIENT_SPEC.loader.exec_module(client)

T06_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t06_qwen_client", ROOT / "scripts/p40_t06_qwen_dn_cpuorder_client.py")
t06_client = importlib.util.module_from_spec(T06_CLIENT_SPEC)
assert T06_CLIENT_SPEC and T06_CLIENT_SPEC.loader
sys.modules[T06_CLIENT_SPEC.name] = t06_client
T06_CLIENT_SPEC.loader.exec_module(t06_client)

T06B_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t06b_qwen_client", ROOT / "scripts/p40_t06b_qwen_dn_cpuorder_client.py")
t06b_client = importlib.util.module_from_spec(T06B_CLIENT_SPEC)
assert T06B_CLIENT_SPEC and T06B_CLIENT_SPEC.loader
sys.modules[T06B_CLIENT_SPEC.name] = t06b_client
T06B_CLIENT_SPEC.loader.exec_module(t06b_client)

T06C_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t06c_qwen_client", ROOT / "scripts/p40_t06c_qwen_dn_cpuorder_client.py")
t06c_client = importlib.util.module_from_spec(T06C_CLIENT_SPEC)
assert T06C_CLIENT_SPEC and T06C_CLIENT_SPEC.loader
sys.modules[T06C_CLIENT_SPEC.name] = t06c_client
T06C_CLIENT_SPEC.loader.exec_module(t06c_client)

T07_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t07_qwen_client", ROOT / "scripts/p40_t07_qwen_dn_cpuorder_client.py")
t07_client = importlib.util.module_from_spec(T07_CLIENT_SPEC)
assert T07_CLIENT_SPEC and T07_CLIENT_SPEC.loader
sys.modules[T07_CLIENT_SPEC.name] = t07_client
T07_CLIENT_SPEC.loader.exec_module(t07_client)

T09_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t09_qwen_client", ROOT / "scripts/p40_t09_qwen_lmhead_cpuorder_client.py")
t09_client = importlib.util.module_from_spec(T09_CLIENT_SPEC)
assert T09_CLIENT_SPEC and T09_CLIENT_SPEC.loader
sys.modules[T09_CLIENT_SPEC.name] = t09_client
T09_CLIENT_SPEC.loader.exec_module(t09_client)

T10_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t10_qwen_client", ROOT / "scripts/p40_t10_qwen_lmhead_cpuorder_client.py")
t10_client = importlib.util.module_from_spec(T10_CLIENT_SPEC)
assert T10_CLIENT_SPEC and T10_CLIENT_SPEC.loader
sys.modules[T10_CLIENT_SPEC.name] = t10_client
T10_CLIENT_SPEC.loader.exec_module(t10_client)

T12_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t12_qwen_client", ROOT / "scripts/p40_t12_qwen_attention_cpuorder_client.py")
t12_client = importlib.util.module_from_spec(T12_CLIENT_SPEC)
assert T12_CLIENT_SPEC and T12_CLIENT_SPEC.loader
sys.modules[T12_CLIENT_SPEC.name] = t12_client
T12_CLIENT_SPEC.loader.exec_module(t12_client)

T13_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t13_qwen_client", ROOT / "scripts/p40_t13_qwen_attention_cpuorder_client.py")
t13_client = importlib.util.module_from_spec(T13_CLIENT_SPEC)
assert T13_CLIENT_SPEC and T13_CLIENT_SPEC.loader
sys.modules[T13_CLIENT_SPEC.name] = t13_client
T13_CLIENT_SPEC.loader.exec_module(t13_client)

T15_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t15_qwen_client", ROOT / "scripts/p40_t15_qwen_shared_cpuorder_client.py")
t15_client = importlib.util.module_from_spec(T15_CLIENT_SPEC)
assert T15_CLIENT_SPEC and T15_CLIENT_SPEC.loader
sys.modules[T15_CLIENT_SPEC.name] = t15_client
T15_CLIENT_SPEC.loader.exec_module(t15_client)

T16_CLIENT_SPEC = importlib.util.spec_from_file_location("p40_t16_qwen_client", ROOT / "scripts/p40_t16_qwen_shared_cpuorder_client.py")
t16_client = importlib.util.module_from_spec(T16_CLIENT_SPEC)
assert T16_CLIENT_SPEC and T16_CLIENT_SPEC.loader
sys.modules[T16_CLIENT_SPEC.name] = t16_client
T16_CLIENT_SPEC.loader.exec_module(t16_client)


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

    def test_t06_identity_pins_exact_delta_q8_path_and_output_oracle(self):
        self.assertEqual(t06.EXPECTED_ORIGINAL_COMMAND, "p40-t06-qwen-dn-cpuorder")
        self.assertEqual(t06.PROFILE_ID, "t06-dn-cpuorder-16")
        self.assertEqual(len(t06.ENGINE_SHA256), 64)
        self.assertEqual(len(t06.EXPECTED_STDOUT_SHA256), 64)
        argv = t06.model_argv()
        self.assertIn("COLI_CUDA_DN_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_DN_DEVICE=0", argv)
        identity = Path("/tmp/p40-t06-test-key")
        self.assertEqual(t06_client.ssh_argv("host", identity)[-1], "p40-t06-qwen-dn-cpuorder")
        self.assertEqual(t06_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t06-qwen-dn-cpuorder-results")

    def test_t06_cuda_diagnostic_allows_device_inventory_only(self):
        self.assertFalse(t06.cuda_runtime_diagnostic(
            b"[CUDA] device 0: Tesla P40, 25.6 GB VRAM, sm_61\n[CUDA] device 1: Tesla P40, 25.6 GB VRAM, sm_61\n"))
        self.assertTrue(t06.cuda_runtime_diagnostic(
            b"[CUDA] expert group issue launch: invalid resource handle\n"))
        self.assertTrue(t06.cuda_runtime_diagnostic(b"[dn-cpuorder] CUDA failure; permanently falling back to CPU\n"))

    def test_t06b_identity_is_distinct_and_pins_the_dedicated_stream_binary(self):
        self.assertEqual(t06b.EXPECTED_ORIGINAL_COMMAND, "p40-t06b-qwen-dn-cpuorder")
        self.assertEqual(t06b.PROFILE_ID, "t06b-dn-cpuorder-16-dedicated-stream")
        self.assertEqual(t06b.ENGINE_SHA256, "bbee1f3c74cd57e1165d2de97c955b16a12c42f193644e4bf20d70873de19ffb")
        identity = Path("/tmp/p40-t06b-test-key")
        self.assertEqual(t06b_client.ssh_argv("host", identity)[-1], "p40-t06b-qwen-dn-cpuorder")
        self.assertEqual(t06b_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t06b-qwen-dn-cpuorder-results")

    def test_t06c_identity_uses_a_new_backend_selector_binary(self):
        self.assertEqual(t06c.EXPECTED_ORIGINAL_COMMAND, "p40-t06c-qwen-dn-cpuorder")
        self.assertEqual(t06c.PROFILE_ID, "t06c-dn-cpuorder-16-backend-device-selector")
        self.assertEqual(t06c.ENGINE_SHA256, "8bef44d2ff680cc3fb35756c95bfdef343fafa989e67009b1ffff43011d49d4c")
        identity = Path("/tmp/p40-t06c-test-key")
        self.assertEqual(t06c_client.ssh_argv("host", identity)[-1], "p40-t06c-qwen-dn-cpuorder")
        self.assertEqual(t06c_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t06c-qwen-dn-cpuorder-results")

    def test_t07_identity_changes_only_output_length_and_oracle(self):
        self.assertEqual(t07.EXPECTED_ORIGINAL_COMMAND, "p40-t07-qwen-dn-cpuorder")
        self.assertEqual(t07.PROFILE_ID, "t07-dn-cpuorder-64-backend-device-selector")
        self.assertEqual(t07.ENGINE_SHA256, t06c.ENGINE_SHA256)
        self.assertEqual(t07.EXPECTED_STDOUT_SHA256, "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f")
        argv = t07.model_argv()
        self.assertIn("N_NEW=64", argv)
        self.assertNotIn("N_NEW=16", argv)
        self.assertIn("COLI_CUDA_DN_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_DN_DEVICE=0", argv)
        identity = Path("/tmp/p40-t07-test-key")
        self.assertEqual(t07_client.ssh_argv("host", identity)[-1], "p40-t07-qwen-dn-cpuorder")
        self.assertEqual(t07_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t07-qwen-dn-cpuorder-results")

    def test_t09_identity_pins_lmhead_binary_and_required_cache_marker(self):
        self.assertEqual(t09.EXPECTED_ORIGINAL_COMMAND, "p40-t09-qwen-lmhead-cpuorder")
        self.assertEqual(t09.PROFILE_ID, "t09-dn-lmhead-cpuorder-16")
        self.assertEqual(t09.ENGINE_SHA256, "857940ee9bf68844e81bc80b8dc0921909842bcf83bd0ea76cd12c265d239af6")
        self.assertIn(b"plus LM head", t09.CACHE_MARKER)
        argv = t09.model_argv()
        self.assertIn("COLI_CUDA_DN_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_DN_DEVICE=0", argv)
        self.assertIn("COLI_CUDA_LMHEAD_CPUORDER=1", argv)
        self.assertIn(str(t09.ENGINE), argv)
        identity = Path("/tmp/p40-t09-test-key")
        self.assertEqual(t09_client.ssh_argv("host", identity)[-1], "p40-t09-qwen-lmhead-cpuorder")
        self.assertEqual(t09_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t09-qwen-lmhead-cpuorder-results")

    def test_t10_identity_changes_only_length_and_uses_the_64_output_oracle(self):
        self.assertEqual(t10.EXPECTED_ORIGINAL_COMMAND, "p40-t10-qwen-lmhead-cpuorder")
        self.assertEqual(t10.PROFILE_ID, "t10-dn-lmhead-cpuorder-64")
        self.assertEqual(t10.t09.ENGINE_SHA256, t09.ENGINE_SHA256)
        self.assertEqual(t10.EXPECTED_STDOUT_SHA256, "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f")
        argv = t10.model_argv()
        self.assertIn("N_NEW=64", argv)
        self.assertNotIn("N_NEW=16", argv)
        self.assertIn("COLI_CUDA_DN_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_LMHEAD_CPUORDER=1", argv)
        identity = Path("/tmp/p40-t10-test-key")
        self.assertEqual(t10_client.ssh_argv("host", identity)[-1], "p40-t10-qwen-lmhead-cpuorder")
        self.assertEqual(t10_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t10-qwen-lmhead-cpuorder-results")

    def test_t12_identity_pins_attention_binary_marker_and_16_output_oracle(self):
        self.assertEqual(t12.EXPECTED_ORIGINAL_COMMAND, "p40-t12-qwen-attention-cpuorder")
        self.assertEqual(t12.PROFILE_ID, "t12-dn-lmhead-attention-cpuorder-16")
        self.assertEqual(t12.ENGINE_SHA256, "cded60fd2b68980c858577354b1ada9fc89ea8bb813a95deb50b999fff6b7dad")
        self.assertIn(b"40 attention Q/K/V/O", t12.CACHE_MARKER)
        argv = t12.model_argv()
        self.assertIn("COLI_CUDA_DN_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_LMHEAD_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_ATTN_CPUORDER=1", argv)
        self.assertIn(str(t12.ENGINE), argv)
        identity = Path("/tmp/p40-t12-test-key")
        self.assertEqual(t12_client.ssh_argv("host", identity)[-1], "p40-t12-qwen-attention-cpuorder")
        self.assertEqual(t12_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t12-qwen-attention-cpuorder-results")

    def test_t13_changes_only_length_and_uses_the_established_64_output_oracle(self):
        self.assertEqual(t13.EXPECTED_ORIGINAL_COMMAND, "p40-t13-qwen-attention-cpuorder")
        self.assertEqual(t13.PROFILE_ID, "t13-dn-lmhead-attention-cpuorder-64")
        self.assertEqual(t13.t12.ENGINE_SHA256, t12.ENGINE_SHA256)
        self.assertEqual(t13.EXPECTED_STDOUT_SHA256, "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f")
        argv = t13.model_argv()
        self.assertIn("N_NEW=64", argv)
        self.assertNotIn("N_NEW=16", argv)
        self.assertIn("COLI_CUDA_DN_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_LMHEAD_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_ATTN_CPUORDER=1", argv)
        identity = Path("/tmp/p40-t13-test-key")
        self.assertEqual(t13_client.ssh_argv("host", identity)[-1], "p40-t13-qwen-attention-cpuorder")
        self.assertEqual(t13_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t13-qwen-attention-cpuorder-results")

    def test_t15_pins_shared_mlp_binary_marker_and_16_output_oracle(self):
        self.assertEqual(t15.EXPECTED_ORIGINAL_COMMAND, "p40-t15-qwen-shared-cpuorder")
        self.assertEqual(t15.PROFILE_ID, "t15-dn-lmhead-attention-shared-cpuorder-16")
        self.assertEqual(t15.ENGINE_SHA256, "30c9f07bf029209cf7c2931c0b351e89bf15b7411a7fab6b7c2d21cd1c8ca9d8")
        self.assertIn(b"120 shared MLP matrices", t15.CACHE_MARKER)
        argv = t15.model_argv()
        self.assertIn("COLI_CUDA_DN_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_LMHEAD_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_ATTN_CPUORDER=1", argv)
        self.assertIn("COLI_CUDA_SHARED_CPUORDER=1", argv)
        self.assertIn(str(t15.ENGINE), argv)
        identity = Path("/tmp/p40-t15-test-key")
        self.assertEqual(t15_client.ssh_argv("host", identity)[-1], "p40-t15-qwen-shared-cpuorder")
        self.assertEqual(t15_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t15-qwen-shared-cpuorder-results")

    def test_t16_changes_only_length_and_uses_the_established_64_output_oracle(self):
        self.assertEqual(t16.EXPECTED_ORIGINAL_COMMAND, "p40-t16-qwen-shared-cpuorder")
        self.assertEqual(t16.PROFILE_ID, "t16-dn-lmhead-attention-shared-cpuorder-64")
        self.assertEqual(t16.t15.ENGINE_SHA256, t15.ENGINE_SHA256)
        self.assertEqual(t16.EXPECTED_STDOUT_SHA256, "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f")
        argv = t16.model_argv()
        self.assertIn("N_NEW=64", argv)
        self.assertNotIn("N_NEW=16", argv)
        self.assertIn("COLI_CUDA_SHARED_CPUORDER=1", argv)
        identity = Path("/tmp/p40-t16-test-key")
        self.assertEqual(t16_client.ssh_argv("host", identity)[-1], "p40-t16-qwen-shared-cpuorder")
        self.assertEqual(t16_client.ssh_argv("host", identity, read_results=True)[-1], "p40-t16-qwen-shared-cpuorder-results")

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
