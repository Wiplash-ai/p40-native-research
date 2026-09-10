#!/usr/bin/env python3
"""Forced-command identity for exact Qwen shared/expert timeline profiling v2."""
from __future__ import annotations

import importlib.util
import re
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t46_base", SourceFileLoader("p40_t46_base", str(SOURCE)))
assert SPEC and SPEC.loader
t24 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t24
SPEC.loader.exec_module(t24)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t46-qwen-shared-expert-timeline"
RESULTS_ORIGINAL_COMMAND = "p40-t46-qwen-shared-expert-timeline-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t46-shared-expert-timeline/c/qwen36")
ENGINE_SHA256 = "b930f61378bddab23a957276f1f015e8a4d53b92cf1a44bad0012579819698a2"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t46-qwen-shared-expert-timeline")
PROFILE_ID = "t46-exact-qwen-shared-expert-timeline-64"
SCHEMA_VERSION = "p40-t46-qwen-shared-expert-timeline-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"
ACTIVE_MARKER = b"[shared-expert-timeline]"
TIMELINE_LINE = re.compile(
    rb"\[shared-expert-timeline\]\s+(\d+) windows \| dense-stream "
    rb"[0-9.]+ ms \| dense-end-to-expert-done [+-]?[0-9.]+ ms \| no-gpu0 (\d+) \| failures (\d+)"
)
EXPECTED_WINDOWS = 63 * 40


_base_model_argv = t24.model_argv
_base_run = t24.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_PROFILE=1":
            result.append("COLI_SHARED_EXPERT_TIMELINE=1")
    return result


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        if ACTIVE_MARKER not in stderr:
            result["failure_reason"] = "shared_expert_timeline_marker_missing"
            result["status"] = "fail"
        elif (b"unexpected window collection failure" in stderr or
              b"event elapsed query failed" in stderr or
              b"expert event creation failed" in stderr or
              b"expert event record failed" in stderr):
            result["failure_reason"] = "shared_expert_timeline_event_failure"
            result["status"] = "fail"
        else:
            match = TIMELINE_LINE.search(stderr)
            if (not match or int(match.group(1)) + int(match.group(2)) != EXPECTED_WINDOWS or
                    int(match.group(3)) != 0):
                result["failure_reason"] = "shared_expert_timeline_window_count"
                result["status"] = "fail"
    return result


def configure(module) -> None:
    module.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
    module.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
    module.ENGINE = ENGINE
    module.ENGINE_SHA256 = ENGINE_SHA256
    module.RESULT_DIRECTORY = RESULT_DIRECTORY
    module.PROFILE_ID = PROFILE_ID
    module.SCHEMA_VERSION = SCHEMA_VERSION
    module.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
    module.model_argv = model_argv


configure(t24)
configure(t24.t23)
configure(t24.t23.t15)
configure(t24.t23.t15.t12)
configure(t24.t23.t15.t12.t09)
configure(t24.t23.t15.t12.t09.t06)
configure(t24.t23.t15.t12.t09.t06.qwen)
t24.t23.t15.t12.t09.t06.qwen.OUTPUT_TOKENS = OUTPUT_TOKENS
t24.t23.t15.t12.t09.t06.qwen.run = run


def main() -> int:
    return t24.t23.t15.main()


if __name__ == "__main__":
    raise SystemExit(main())
