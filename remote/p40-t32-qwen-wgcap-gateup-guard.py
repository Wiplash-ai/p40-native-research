#!/usr/bin/env python3
"""Forced-command identity for T32's exact-output WGCAP collection run."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import math
import os
import re
import struct
import sys
import uuid
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-qwen-canary-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-qwen-canary-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t32_base", SourceFileLoader("p40_t32_base", str(SOURCE)))
assert SPEC and SPEC.loader
qwen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen
SPEC.loader.exec_module(qwen)


EXPECTED_ORIGINAL_COMMAND = "p40-t32-qwen-wgcap-gateup"
RESULTS_ORIGINAL_COMMAND = "p40-t32-qwen-wgcap-gateup-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t32-gateup-capture/c/qwen36")
ENGINE_SHA256 = "bcdf3e8859011deefeb79967b8b4b1e27910bc03a344d44fc0fc07c4856176aa"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t32-qwen-wgcap-gateup")
CAPTURE_PATH = RESULT_DIRECTORY / "qwen-gateup.wgcap"
PROFILE_ID = "t32-real-qwen-gateup-wgcap-16"
SCHEMA_VERSION = "p40-t32-qwen-wgcap-gateup-v1"
EXPECTED_STDOUT_SHA256 = "43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a"
MAGIC = b"WGCAP01\0"
HEADER = struct.Struct("<8s14I")
RECORD = struct.Struct("<8IfI")
HIDDEN, INTERMEDIATE, GROUP_WIDTH, RECORD_COUNT = 2048, 512, 256, 96
RECORD_BYTES = 1_609_800
CAPTURE_BYTES = 154_540_864
SUMMARY = re.compile(
    rb"^\[wgcap\] records=(\d+) expected=(\d+) errors=(\d+) bytes=(\d+) path=(.+)$",
    re.MULTILINE,
)


def engine_digest() -> str:
    return hashlib.sha256(ENGINE.read_bytes()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_summary(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size != CAPTURE_BYTES:
        raise ValueError("capture_byte_count_mismatch")
    expected_routes = {(step, layer, rank) for step in range(4) for layer in (0, 20, 39) for rank in range(8)}
    actual_routes: set[tuple[int, int, int]] = set()
    by_device: dict[str, int] = {}
    with path.open("rb") as source:
        raw_header = source.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise ValueError("capture_header_truncated")
        magic, version, header_bytes, marker, hidden, intermediate, group_width, record_bytes, records, model_layers, layer_count, step_count, source_schema, reserve0, reserve1 = HEADER.unpack(raw_header)
        if (magic, version, header_bytes, marker, hidden, intermediate, group_width, record_bytes, records,
            model_layers, layer_count, step_count, source_schema, reserve0, reserve1) != (
                MAGIC, 1, HEADER.size, 0x01020304, HIDDEN, INTERMEDIATE, GROUP_WIDTH, RECORD_BYTES,
                RECORD_COUNT, 40, 3, 4, 1, 0, 0):
            raise ValueError("capture_header_mismatch")
        for _ in range(RECORD_COUNT):
            raw_record = source.read(RECORD.size)
            if len(raw_record) != RECORD.size:
                raise ValueError("capture_record_truncated")
            layer, step, device, eid, rank, bundle, gate_up_bytes, down_bytes, router_weight, reserved = RECORD.unpack(raw_record)
            if (bundle, gate_up_bytes, down_bytes, reserved) != (8, 524_288, 524_288, 0):
                raise ValueError("capture_record_metadata_mismatch")
            if device not in (0, 1) or eid >= 256 or not math.isfinite(router_weight):
                raise ValueError("capture_record_value_mismatch")
            route = (step, layer, rank)
            if route not in expected_routes or route in actual_routes:
                raise ValueError("capture_route_layout_mismatch")
            actual_routes.add(route)
            by_device[str(device)] = by_device.get(str(device), 0) + 1
            source.seek(RECORD_BYTES - RECORD.size, os.SEEK_CUR)
        if source.read(1):
            raise ValueError("capture_trailing_data")
    if actual_routes != expected_routes:
        raise ValueError("capture_route_count_mismatch")
    return {
        "path": str(path), "bytes": CAPTURE_BYTES, "sha256": _file_digest(path),
        "records": RECORD_COUNT, "calibration_records": 48, "holdout_records": 48,
        "by_device": by_device,
    }


_base_model_argv = qwen.model_argv
_base_run = qwen.run


def model_argv() -> list[str]:
    output: list[str] = []
    for item in _base_model_argv():
        output.append(item)
        if item == "COLI_CUDA_PROFILE=1":
            output.extend(("COLI_W4_CAPTURE=gateup-r1", f"COLI_W4_CAPTURE_FILE={CAPTURE_PATH}"))
    return output


def run(request: dict) -> dict:
    if not request["dry_run"]:
        if not ENGINE.is_file() or engine_digest() != ENGINE_SHA256:
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            return {
                "schema_version": SCHEMA_VERSION, "profile_id": PROFILE_ID, "run_id": str(uuid.uuid4()),
                "request": request, "command": model_argv(), "started_at": now, "finished_at": now,
                "status": "fail", "failure_reason": "pinned_engine_missing_or_changed", "telemetry": [],
                "cleanup": {"actions": [], "power_restored": False}, "cuda_initialized": False,
            }
        RESULT_DIRECTORY.mkdir(mode=0o700, parents=True, exist_ok=True)
        CAPTURE_PATH.unlink(missing_ok=True)
    result = _base_run(request)
    if request["dry_run"]:
        return result
    if result.get("stdout", {}).get("sha256") != EXPECTED_STDOUT_SHA256:
        result["failure_reason"] = "exact_output_hash_mismatch"
        result["status"] = "fail"
        return result
    if result.get("status") != "pass":
        return result
    matches = list(SUMMARY.finditer(Path(result["stderr"]["path"]).read_bytes()))
    if len(matches) != 1:
        result["failure_reason"] = "capture_summary_missing_or_ambiguous"
        result["status"] = "fail"
        return result
    match = matches[0]
    if tuple(int(match.group(index)) for index in range(1, 5)) != (RECORD_COUNT, RECORD_COUNT, 0, CAPTURE_BYTES):
        result["failure_reason"] = "capture_source_summary_mismatch"
        result["status"] = "fail"
        return result
    if Path(match.group(5).decode("utf-8", "strict")) != CAPTURE_PATH:
        result["failure_reason"] = "capture_source_path_mismatch"
        result["status"] = "fail"
        return result
    try:
        result["capture"] = capture_summary(CAPTURE_PATH)
    except (OSError, ValueError) as error:
        result["failure_reason"] = f"capture_invalid:{error}"
        result["status"] = "fail"
    return result


qwen.ENGINE = ENGINE
qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
qwen.RESULT_DIRECTORY = RESULT_DIRECTORY
qwen.PROFILE_ID = PROFILE_ID
qwen.SCHEMA_VERSION = SCHEMA_VERSION
qwen.model_argv = model_argv
qwen.run = run


def main() -> int:
    return qwen.main()


if __name__ == "__main__":
    raise SystemExit(main())
