#!/usr/bin/env python3
"""Static contracts for the guarded T43 exact-Qwen host-take profile."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUARD = (ROOT / "remote/p40-t43-qwen-qtier-take-profile-guard.py").read_text()


class QTierTakeProfileGuardTest(unittest.TestCase):
    def test_uses_the_isolated_engine_and_exact_output_oracle(self):
        self.assertIn("colibri-t43-qtier-take-profile/c/qwen36", GUARD)
        self.assertIn("2aae5e44ea5bbec269c649e35ea5c578cf2abbdec8e459b6b363f19bf787e193", GUARD)
        self.assertIn("5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f", GUARD)

    def test_requests_only_host_take_attribution_and_two_devices(self):
        self.assertIn('result.append("COLI_QTIER_TAKE_PROFILE=1")', GUARD)
        self.assertNotIn('COLI_QTIER_PROFILE=1', GUARD)
        self.assertIn("HOST_TAKE_LINE = re.compile", GUARD)
        self.assertIn("len(profiles) != 2", GUARD)


if __name__ == "__main__":
    unittest.main()
