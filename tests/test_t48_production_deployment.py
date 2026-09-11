#!/usr/bin/env python3
"""Static contracts for the local-only accepted-Q8 deployment files."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
UNIT = (ROOT / "deployment/colibri-qwen36.service").read_text()
ENV = (ROOT / "deployment/colibri-qwen36-t24.env").read_text()
KIMI_UNIT = (ROOT / "deployment/colibri-kimi-k3.service.template").read_text()
OLLAMA_8B_40K = (ROOT / "deployment/ollama-qwen3-8b-40k.Modelfile").read_text()
SUPERVISOR = ROOT / "deployment/colibri-thermal-supervisor.py"
FAN_GUARD = (ROOT / "deployment/wiplash-gpu-fan-guard").read_text()


class ProductionDeploymentTests(unittest.TestCase):
    def test_listener_is_loopback_and_has_one_kv_slot(self):
        self.assertIn("--host 127.0.0.1", UNIT)
        self.assertIn("--kv-slots 1", UNIT)
        self.assertIn("--ctx 32768", UNIT)
        self.assertIn("--ngen 16384", UNIT)
        self.assertNotIn("--gpu", UNIT)
        self.assertIn("colibri-thermal-supervisor.py", UNIT)

    def test_accepted_exact_path_is_explicit(self):
        for setting in (
            "COLI_CUDA=1", "COLI_GPUS=0,1", "COLI_DENSE_I8=1",
            "COLI_CUDA_DN_CPUORDER=1", "COLI_CUDA_DN_PAIR=1",
            "COLI_CUDA_LMHEAD_CPUORDER=1", "COLI_CUDA_ATTN_CPUORDER=1",
            "COLI_CUDA_SHARED_CPUORDER=1",
        ):
            self.assertIn(setting, ENV)

    def test_long_output_templates_match_their_supported_context_windows(self):
        self.assertIn("--ctx ${K3_MAXT} --ngen ${K3_MAXT}", KIMI_UNIT)
        self.assertNotIn("--ngen 512", KIMI_UNIT)
        self.assertIn("FROM qwen3:8b", OLLAMA_8B_40K)
        self.assertIn("PARAMETER num_ctx 40960", OLLAMA_8B_40K)

    def test_hot_telemetry_refuses_to_start_child_without_failure_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "nvidia-smi"
            marker = Path(temp) / "child-was-started"
            fake.write_text("#!/bin/sh\nprintf '71\\n'\n")
            fake.chmod(0o755)
            env = os.environ | {
                "COLIBRI_THERMAL_C": "70",
                "COLIBRI_THERMAL_POLL_SECONDS": "0.01",
                "COLIBRI_THERMAL_STOP_SECONDS": "0.1",
                "COLIBRI_NVIDIA_SMI": str(fake),
            }
            completed = subprocess.run(
                [sys.executable, str(SUPERVISOR), sys.executable, "-c",
                 f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                env=env, text=True, capture_output=True, timeout=3,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("71 C >= 70 C; refusing service start", completed.stderr)
            self.assertFalse(marker.exists())

    def test_fan_pwm_uses_the_board_verified_percent_domain(self):
        self.assertIn("printf '%02x' \"$pwm\"", FAN_GUARD)
        self.assertNotIn("pwm * 255", FAN_GUARD)


if __name__ == "__main__":
    unittest.main()
