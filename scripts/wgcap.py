#!/usr/bin/env python3
"""Strict reader and manifest generator for the bounded WGCAP v1 sidecar.

The reader deliberately uses only the standard library: it validates capture
provenance without importing a numerical runtime or accepting a loose pickle /
NumPy container.  Numerical replay belongs to a separate, CPU-only T33 tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MAGIC = b"WGCAP01\0"
VERSION = 1
BYTE_ORDER_MARKER = 0x01020304
HEADER_FORMAT = "<8s14I"
RECORD_HEADER_FORMAT = "<8IfI"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
RECORD_HEADER_BYTES = struct.calcsize(RECORD_HEADER_FORMAT)
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
MAX_RECORDS = 96
P40_HIDDEN = 2048
P40_INTERMEDIATE = 512
P40_GROUP_WIDTH = 256
P40_TARGET_LAYERS = (0, 20, 39)
P40_TARGET_STEPS = (0, 1, 2, 3)


class CaptureError(ValueError):
    """A malformed, unsafe, or non-T32 capture artifact."""


@dataclass(frozen=True)
class CaptureHeader:
    version: int
    header_bytes: int
    byte_order_marker: int
    hidden: int
    intermediate: int
    quant_group_width: int
    record_bytes: int
    expected_records: int
    model_layers: int
    target_layer_count: int
    target_step_count: int
    source_schema: int


@dataclass(frozen=True)
class CaptureRecord:
    layer: int
    decode_step: int
    device: int
    expert_id: int
    route_rank: int
    route_bundle_size: int
    gate_up_weight_bytes: int
    down_weight_bytes: int
    router_weight: float
    offset: int


@dataclass(frozen=True)
class Capture:
    path: Path
    sha256: str
    data: bytes
    header: CaptureHeader
    records: tuple[CaptureRecord, ...]


def packed_w4_bytes(rows: int, columns: int) -> int:
    if rows <= 0 or columns <= 0 or columns % 2:
        raise CaptureError("invalid packed W4 shape")
    return rows * (columns // 2)


def record_bytes(hidden: int, intermediate: int, quant_group_width: int) -> int:
    if hidden <= 0 or intermediate <= 0 or quant_group_width <= 0:
        raise CaptureError("non-positive capture geometry")
    if hidden % 2 or intermediate % 2:
        raise CaptureError("WGCAP requires even packed dimensions")
    groups = math.ceil(hidden / quant_group_width)
    matrices = 2 * packed_w4_bytes(intermediate, hidden) + packed_w4_bytes(hidden, intermediate)
    vectors = (hidden + 3 * intermediate + hidden) * 4
    # Exact input, raw gate/up/hidden, exact output plus Q8 input/scales.
    return RECORD_HEADER_BYTES + vectors + hidden + groups * 4 + matrices + (2 * intermediate + hidden) * 4


def _read_header(data: bytes) -> CaptureHeader:
    if len(data) < HEADER_BYTES:
        raise CaptureError("truncated WGCAP header")
    fields = struct.unpack_from(HEADER_FORMAT, data)
    magic = fields[0]
    if magic != MAGIC:
        raise CaptureError("unrecognized WGCAP magic")
    header = CaptureHeader(*fields[1:13])
    if header.version != VERSION:
        raise CaptureError("unsupported WGCAP version")
    if header.header_bytes != HEADER_BYTES:
        raise CaptureError("unexpected WGCAP header size")
    if header.byte_order_marker != BYTE_ORDER_MARKER:
        raise CaptureError("non-little-endian WGCAP artifact")
    if header.expected_records < 1 or header.expected_records > MAX_RECORDS:
        raise CaptureError("unsafe WGCAP record count")
    if header.record_bytes != record_bytes(
        header.hidden, header.intermediate, header.quant_group_width
    ):
        raise CaptureError("inconsistent WGCAP record geometry")
    return header


def parse_capture(path: Path, *, max_bytes: int = MAX_CAPTURE_BYTES) -> Capture:
    path = path.resolve()
    stat = path.stat()
    if stat.st_size > max_bytes:
        raise CaptureError("WGCAP artifact exceeds byte cap")
    data = path.read_bytes()
    header = _read_header(data)
    expected_size = header.header_bytes + header.expected_records * header.record_bytes
    if len(data) != expected_size:
        raise CaptureError("truncated or trailing WGCAP payload")
    records: list[CaptureRecord] = []
    for index in range(header.expected_records):
        offset = header.header_bytes + index * header.record_bytes
        raw = struct.unpack_from(RECORD_HEADER_FORMAT, data, offset)
        layer, step, device, eid, rank, bundle, guw, dw, router_weight, reserved = raw
        if reserved != 0:
            raise CaptureError("nonzero WGCAP reserved record field")
        expected_gw = packed_w4_bytes(header.intermediate, header.hidden)
        expected_dw = packed_w4_bytes(header.hidden, header.intermediate)
        if (guw, dw) != (expected_gw, expected_dw):
            raise CaptureError("inconsistent WGCAP packed-weight byte count")
        if not math.isfinite(router_weight):
            raise CaptureError("non-finite WGCAP router weight")
        records.append(CaptureRecord(layer, step, device, eid, rank, bundle, guw, dw, router_weight, offset))
    return Capture(path, hashlib.sha256(data).hexdigest(), data, header, tuple(records))


def iter_payload_slices(capture: Capture, record: CaptureRecord) -> Iterator[tuple[str, memoryview]]:
    """Yield named raw field views in the fixed v1 record order."""
    h, i, group = capture.header.hidden, capture.header.intermediate, capture.header.quant_group_width
    groups = math.ceil(h / group)
    fields = (
        ("x", h * 4), ("q", h), ("q_scale", groups * 4),
        ("gate", i * 4), ("up", i * 4), ("hidden", i * 4), ("output", h * 4),
        ("gate_w4", record.gate_up_weight_bytes), ("up_w4", record.gate_up_weight_bytes),
        ("down_w4", record.down_weight_bytes), ("gate_scale", i * 4),
        ("up_scale", i * 4), ("down_scale", h * 4),
    )
    start = record.offset + RECORD_HEADER_BYTES
    view = memoryview(capture.data)
    for name, count in fields:
        end = start + count
        yield name, view[start:end]
        start = end
    if start != record.offset + capture.header.record_bytes:
        raise CaptureError("WGCAP payload cursor mismatch")


def validate_t32_layout(capture: Capture) -> None:
    h = capture.header
    if (h.hidden, h.intermediate, h.quant_group_width, h.expected_records) != (
        P40_HIDDEN, P40_INTERMEDIATE, P40_GROUP_WIDTH, 96
    ):
        raise CaptureError("artifact does not have the fixed T32 geometry")
    expected = {(step, layer, rank) for step in P40_TARGET_STEPS for layer in P40_TARGET_LAYERS for rank in range(8)}
    actual = {(r.decode_step, r.layer, r.route_rank) for r in capture.records}
    if actual != expected or len(actual) != len(capture.records):
        raise CaptureError("T32 capture does not contain exactly one record per fixed route")
    for record in capture.records:
        if record.route_bundle_size != 8 or record.device not in (0, 1):
            raise CaptureError("unexpected T32 bundle metadata")


def manifest(capture: Capture) -> dict:
    by_device: dict[str, int] = {}
    for record in capture.records:
        by_device[str(record.device)] = by_device.get(str(record.device), 0) + 1
    return {
        "format": "WGCAP-v1",
        "path": str(capture.path),
        "bytes": len(capture.data),
        "sha256": capture.sha256,
        "header": capture.header.__dict__,
        "records": len(capture.records),
        "by_device": by_device,
        "calibration_records": sum(record.decode_step < 2 for record in capture.records),
        "holdout_records": sum(record.decode_step >= 2 for record in capture.records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--t32", action="store_true", help="require the fixed real-Qwen T32 layout")
    args = parser.parse_args()
    try:
        capture = parse_capture(args.capture)
        if args.t32:
            validate_t32_layout(capture)
    except (OSError, CaptureError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "ok", **manifest(capture)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
