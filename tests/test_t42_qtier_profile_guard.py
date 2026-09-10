#!/usr/bin/env python3
"""Static contracts for the guarded T42 exact-Qwen profiler run."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUARD = (ROOT / "remote/p40-t42-qwen-qtier-profile-guard.py").read_text()


class QTierProfileGuardTest(unittest.TestCase):
    def test_uses_the_isolated_engine_and_canonical_output_oracle(self):
        self.assertIn("colibri-t42-qtier-profile/c/qwen36", GUARD)
        self.assertIn("3a30488a24b0ee8c816e8f82b6ad6f4d406ad5cf810ba81bd6c8faddd585021f", GUARD)
        self.assertIn("5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f", GUARD)

    def test_enables_only_the_opt_in_profile_and_requires_both_devices(self):
        self.assertIn('result.append("COLI_QTIER_PROFILE=1")', GUARD)
        self.assertIn("PROFILE_LINE = re.compile", GUARD)
        self.assertIn("len(profiles) == 2", GUARD)
        self.assertIn('"qtier_profile_missing_or_zero"', GUARD)


if __name__ == "__main__":
    unittest.main()
