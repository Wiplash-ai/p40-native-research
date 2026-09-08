import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import colibri_t03


class T03CapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "research/capabilities/qwen36-colibri-1.10.1.json").read_text())
        cls.settings = {row["name"]: row for row in cls.manifest["settings"]}

    def test_all_audited_settings_have_one_valid_classification(self):
        allowed = set(self.manifest["classification_definitions"])
        self.assertGreaterEqual(len(self.settings), 43)
        self.assertTrue(all(row["classification"] in allowed and row["detail"] for row in self.settings.values()))

    def test_generic_noop_regression_guard(self):
        for name in ("CUDA_DENSE", "COLI_CUDA_ATTN", "COLI_CUDA_ATTN_SHARD", "COLI_CUDA_PIPE", "DRAFT", "COLI_CUDA_MTP"):
            self.assertEqual(self.settings[name]["classification"], "unsupported", name)

    def test_reached_backend_controls_do_not_expand_to_wrapper_controls(self):
        self.assertEqual(self.settings["COLI_CUDA_W4_PACKED"]["classification"], "backend-reached")
        self.assertEqual(self.settings["COLI_CUDA_DUAL_PROJ"]["classification"], "backend-reached")
        self.assertEqual(self.settings["COLI_CUDA_PROFILE"]["classification"], "unsupported")
        self.assertEqual(self.settings["COLI_CUDA_ASYNC"]["classification"], "unsupported")

    def test_timer_counter_fixture_with_overlap_is_not_double_counted(self):
        text = "\n".join([
            'COLI_T03_TIMER {"name":"head","duration":8,"unit":"ns","relation":"none"}',
            'COLI_T03_TIMER {"name":"issue","duration":3,"unit":"ns","relation":"overlap","parent":"moe"}',
            'COLI_T03_COUNTER {"name":"cuda_launches","value":2,"unit":"count"}',
            'COLI_T03_COUNTER {"name":"h2d_bytes","value":16384,"unit":"bytes"}',
        ])
        summary = colibri_t03.summarize(colibri_t03.parse_direct_output(text))
        self.assertEqual(summary["additive_duration_ns"], 8)
        self.assertEqual(summary["counters"]["cuda_launches"], 2)
        self.assertEqual(summary["counter_units"]["h2d_bytes"], "bytes")

    def test_parser_rejects_negative_duration_and_bad_counter_unit(self):
        with self.assertRaises(ValueError):
            colibri_t03.parse_timer('COLI_T03_TIMER {"name":"issue","duration":-1,"unit":"ns","relation":"none"}')
        with self.assertRaises(ValueError):
            colibri_t03.parse_counter('COLI_T03_COUNTER {"name":"d2h_bytes","value":4,"unit":"ms"}')

    def test_direct_records_are_required(self):
        with self.assertRaises(ValueError): colibri_t03.parse_direct_output("gateway request complete")


if __name__ == "__main__":
    unittest.main()
