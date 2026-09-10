#!/usr/bin/env python3
"""Contracts for the bounded T32 real-Qwen capture container."""
from __future__ import annotations

import importlib.util
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wgcap", ROOT / "scripts/wgcap.py")
wgcap = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = wgcap
SPEC.loader.exec_module(wgcap)


def write_capture(path: Path, *, hidden: int = 8, intermediate: int = 4, group: int = 4,
                  records: list[tuple[int, int, int, int, int, int, float]] | None = None,
                  target_layer_count: int = 1, target_step_count: int = 1,
                  trailing: bytes = b"") -> None:
    records = records or [(0, 0, 0, 7, 0, 8, 0.5)]
    rb = wgcap.record_bytes(hidden, intermediate, group)
    header = struct.pack(
        wgcap.HEADER_FORMAT, wgcap.MAGIC, wgcap.VERSION, wgcap.HEADER_BYTES,
        wgcap.BYTE_ORDER_MARKER, hidden, intermediate, group, rb, len(records),
        40, target_layer_count, target_step_count, 1, 0, 0,
    )
    body = bytearray(header)
    gw = wgcap.packed_w4_bytes(intermediate, hidden)
    dw = wgcap.packed_w4_bytes(hidden, intermediate)
    for layer, step, device, eid, rank, bundle, route_weight in records:
        body.extend(struct.pack(wgcap.RECORD_HEADER_FORMAT, layer, step, device, eid, rank, bundle,
                                gw, dw, route_weight, 0))
        body.extend(bytes(rb - wgcap.RECORD_HEADER_BYTES))
    path.write_bytes(bytes(body) + trailing)


class WgcapTests(unittest.TestCase):
    def test_round_trip_and_payload_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.wgcap"
            write_capture(path)
            capture = wgcap.parse_capture(path)
        self.assertEqual(capture.header.hidden, 8)
        self.assertEqual(len(capture.records), 1)
        fields = dict(wgcap.iter_payload_slices(capture, capture.records[0]))
        self.assertEqual(len(fields["x"]), 32)
        self.assertEqual(len(fields["q"]), 8)
        self.assertEqual(len(fields["gate_w4"]), 16)
        self.assertEqual(len(fields["down_w4"]), 16)

    def test_rejects_trailing_data_and_nonfinite_route_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.wgcap"
            write_capture(path, trailing=b"x")
            with self.assertRaisesRegex(wgcap.CaptureError, "truncated or trailing"):
                wgcap.parse_capture(path)
            write_capture(path, records=[(0, 0, 0, 7, 0, 8, math.nan)])
            with self.assertRaisesRegex(wgcap.CaptureError, "non-finite"):
                wgcap.parse_capture(path)

    def test_t32_layout_requires_all_fixed_routes_once(self):
        records = [
            (layer, step, rank % 2, 100 + rank, rank, 8, 1.0 / 8)
            for step in wgcap.P40_TARGET_STEPS
            for layer in wgcap.P40_TARGET_LAYERS
            for rank in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t32.wgcap"
            write_capture(path, hidden=wgcap.P40_HIDDEN, intermediate=wgcap.P40_INTERMEDIATE,
                          group=wgcap.P40_GROUP_WIDTH, records=records)
            capture = wgcap.parse_capture(path)
        wgcap.validate_t32_layout(capture)
        self.assertEqual(wgcap.manifest(capture)["calibration_records"], 48)
        self.assertEqual(wgcap.manifest(capture)["holdout_records"], 48)
        duplicate = list(records); duplicate[-1] = duplicate[-2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.wgcap"
            write_capture(path, hidden=wgcap.P40_HIDDEN, intermediate=wgcap.P40_INTERMEDIATE,
                          group=wgcap.P40_GROUP_WIDTH, records=duplicate)
            capture = wgcap.parse_capture(path)
        with self.assertRaisesRegex(wgcap.CaptureError, "exactly one"):
            wgcap.validate_t32_layout(capture)


if __name__ == "__main__":
    unittest.main()
