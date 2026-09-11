import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.ollama_bench import BenchmarkError, benchmark, require_safe_start


SAFE = [
    {"index": 0, "temperature_c": 40, "power_w": 30.0, "memory_mib": 6144, "utilization_pct": 70},
    {"index": 1, "temperature_c": 42, "power_w": 25.0, "memory_mib": 0, "utilization_pct": 0},
]


class OllamaBenchmarkTests(unittest.TestCase):
    @patch("swarm.ollama_bench.post_json")
    @patch("swarm.ollama_bench.nvidia_snapshot", return_value=SAFE)
    def test_records_decode_rate_and_requires_gpu_residency(self, _snapshot, post):
        post.return_value = {
            "eval_count": 64, "eval_duration": 2_000_000_000,
            "prompt_eval_count": 10, "prompt_eval_duration": 100_000_000,
            "response": "deterministic test response",
        }
        result = benchmark(
            url="http://test/api/generate", model="qwen3:8b", context=4096,
            completion_limit=64, timeout=10, limit_c=70,
        )
        self.assertEqual(result.eval_tokens_per_s, 32.0)
        self.assertTrue(result.gpu_observed)
        self.assertEqual(result.response_text, "deterministic test response")
        self.assertEqual(post.call_args.args[1]["keep_alive"], "0s")

    @patch("swarm.ollama_bench.nvidia_snapshot")
    def test_hot_preflight_fails_before_request(self, snapshot):
        snapshot.return_value = [
            {"index": 0, "temperature_c": 70, "power_w": 10.0, "memory_mib": 0, "utilization_pct": 0},
            {"index": 1, "temperature_c": 40, "power_w": 10.0, "memory_mib": 0, "utilization_pct": 0},
        ]
        with self.assertRaises(BenchmarkError):
            require_safe_start(70)


if __name__ == "__main__":
    unittest.main()
