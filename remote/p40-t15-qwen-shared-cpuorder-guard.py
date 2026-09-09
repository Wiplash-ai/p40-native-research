#!/usr/bin/env python3
"""Forced-command identity for the exact-Q8 shared-MLP Qwen canary."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t12-qwen-attention-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t12-qwen-attention-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t15_base", SourceFileLoader("p40_t15_base", str(SOURCE)))
assert SPEC and SPEC.loader
t12 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t12
SPEC.loader.exec_module(t12)

EXPECTED_ORIGINAL_COMMAND = "p40-t15-qwen-shared-cpuorder"
RESULTS_ORIGINAL_COMMAND = "p40-t15-qwen-shared-cpuorder-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t15-shared/c/qwen36")
ENGINE_SHA256 = "30c9f07bf029209cf7c2931c0b351e89bf15b7411a7fab6b7c2d21cd1c8ca9d8"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t15-qwen-shared-cpuorder")
PROFILE_ID = "t15-dn-lmhead-attention-shared-cpuorder-16"
SCHEMA_VERSION = "p40-t15-qwen-shared-cpuorder-v1"
EXPECTED_STDOUT_SHA256 = "43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a"
CACHE_MARKER = b"[dn-cpuorder] exact Q8 cache active: 90 DeltaNet matrices, LM head, 40 attention Q/K/V/O matrices, and 120 shared MLP matrices on GPU 0"

_base_model_argv = t12.model_argv
# Skip the T09/T12 exact marker checks: this variant has a deliberately more
# specific marker, while retaining T06's engine/output/CUDA diagnostic gates.
_base_run = t12.t09.t06.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_ATTN_CPUORDER=1":
            result.append("COLI_CUDA_SHARED_CPUORDER=1")
    return result


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        if CACHE_MARKER not in stderr:
            result["failure_reason"] = "shared_cpuorder_not_active"
            result["status"] = "fail"
    return result


t12.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t12.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t12.ENGINE = ENGINE
t12.ENGINE_SHA256 = ENGINE_SHA256
t12.RESULT_DIRECTORY = RESULT_DIRECTORY
t12.PROFILE_ID = PROFILE_ID
t12.SCHEMA_VERSION = SCHEMA_VERSION
t12.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t12.model_argv = model_argv
t12.t09.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t12.t09.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t12.t09.ENGINE = ENGINE
t12.t09.ENGINE_SHA256 = ENGINE_SHA256
t12.t09.RESULT_DIRECTORY = RESULT_DIRECTORY
t12.t09.PROFILE_ID = PROFILE_ID
t12.t09.SCHEMA_VERSION = SCHEMA_VERSION
t12.t09.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t12.t09.model_argv = model_argv
t12.t09.t06.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t12.t09.t06.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t12.t09.t06.ENGINE = ENGINE
t12.t09.t06.ENGINE_SHA256 = ENGINE_SHA256
t12.t09.t06.RESULT_DIRECTORY = RESULT_DIRECTORY
t12.t09.t06.PROFILE_ID = PROFILE_ID
t12.t09.t06.SCHEMA_VERSION = SCHEMA_VERSION
t12.t09.t06.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t12.t09.t06.model_argv = model_argv
t12.t09.t06.qwen.ENGINE = ENGINE
t12.t09.t06.qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t12.t09.t06.qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t12.t09.t06.qwen.RESULT_DIRECTORY = RESULT_DIRECTORY
t12.t09.t06.qwen.PROFILE_ID = PROFILE_ID
t12.t09.t06.qwen.SCHEMA_VERSION = SCHEMA_VERSION
t12.t09.t06.qwen.OUTPUT_TOKENS = 16
t12.t09.t06.qwen.model_argv = model_argv
t12.t09.t06.qwen.run = run


def main() -> int:
    return t12.main()


if __name__ == "__main__":
    raise SystemExit(main())
