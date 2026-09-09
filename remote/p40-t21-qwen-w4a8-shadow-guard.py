#!/usr/bin/env python3
"""Forced-command identity for exact-output real-expert W4A8 shadow canary."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import re
import sys
import uuid
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-qwen-canary-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-qwen-canary-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t21_base", SourceFileLoader("p40_t21_base", str(SOURCE)))
assert SPEC and SPEC.loader
qwen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qwen
SPEC.loader.exec_module(qwen)

EXPECTED_ORIGINAL_COMMAND = "p40-t21-qwen-w4a8-shadow"
RESULTS_ORIGINAL_COMMAND = "p40-t21-qwen-w4a8-shadow-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t21-w4a8-shadow/c/qwen36")
ENGINE_SHA256 = "b58d4181e836af3b26c8970e24e32e17b69df617217f40f9a65b2508888d2127"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t21-qwen-w4a8-shadow")
PROFILE_ID = "t21-w4a8-real-expert-shadow-16"
SCHEMA_VERSION = "t21-qwen-w4a8-shadow-v1"
EXPECTED_STDOUT_SHA256 = "43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a"
SHADOW = re.compile(rb"^\[w4a8-shadow\] rows=(\d+) width=(\d+) rel_l2=([^ ]+) max_abs=([^ ]+) cosine=([^ ]+) finite=(\d+)$", re.MULTILINE)


def engine_digest() -> str:
    return hashlib.sha256(ENGINE.read_bytes()).hexdigest()


_base_model_argv = qwen.model_argv


def model_argv() -> list[str]:
    argv = _base_model_argv()
    output: list[str] = []
    for item in argv:
        output.append(item)
        if item == "COLI_CUDA_PROFILE=1":
            output.append("COLI_CUDA_W4A8_DP4A=shadow")
    return output


_base_run = qwen.run


def run(request: dict) -> dict:
    if not request["dry_run"] and (not ENGINE.is_file() or engine_digest() != ENGINE_SHA256):
        return {"schema_version": SCHEMA_VERSION, "profile_id": PROFILE_ID, "run_id": str(uuid.uuid4()),
                "request": request, "command": model_argv(), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "fail",
                "failure_reason": "pinned_engine_missing_or_changed", "telemetry": [],
                "cleanup": {"actions": [], "power_restored": False}, "cuda_initialized": False}
    result = _base_run(request)
    if request["dry_run"]:
        return result
    stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
    matches = list(SHADOW.finditer(stderr))
    if result.get("stdout", {}).get("sha256") != EXPECTED_STDOUT_SHA256:
        result["failure_reason"] = "exact_output_hash_mismatch"
        result["status"] = "fail"
    elif not matches:
        result["failure_reason"] = "w4a8_shadow_not_active"
        result["status"] = "fail"
    elif any(match.group(6) != b"1" for match in matches):
        result["failure_reason"] = "w4a8_shadow_nonfinite"
        result["status"] = "fail"
    else:
        result["shadow_metrics"] = [{"rows": int(m.group(1)), "width": int(m.group(2)),
            "relative_l2": float(m.group(3)), "max_absolute_error": float(m.group(4)),
            "cosine_similarity": float(m.group(5))} for m in matches]
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
