import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p40_telemetry_gateway as gateway


def sample():
    return {
        "schema_version": "p40-telemetry-v1",
        "gpus": [{"index": 0}, {"index": 1}],
        "fans": [{"name": f"FAN{i}"} for i in range(1, 9)],
        "device_error": False,
        "fan_critical": False,
    }


class TelemetryGatewayTests(unittest.TestCase):
    def test_ssh_command_is_forced_and_identity_is_explicit(self):
        self.assertEqual(
            gateway.ssh_argv("host", "/key", 8),
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "IdentitiesOnly=yes", "-i", "/key", "host", "p40-telemetry"],
        )

    def test_valid_snapshot_is_accepted(self):
        self.assertEqual(gateway.parse_snapshot(__import__("json").dumps(sample()))["gpus"][1]["index"], 1)

    def test_missing_safety_fields_fail_closed(self):
        broken = sample(); del broken["fan_critical"]
        with self.assertRaises(gateway.TelemetryProtocolError):
            gateway.parse_snapshot(__import__("json").dumps(broken))

    def test_malformed_json_fails_closed(self):
        with self.assertRaises(gateway.TelemetryProtocolError):
            gateway.parse_snapshot("not-json")
