#!/usr/bin/env python3
"""Forced-command identity for the T09 exact-Q8 LM-head Qwen canary."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t06-qwen-dn-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t06-qwen-dn-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t09_base", SourceFileLoader("p40_t09_base", str(SOURCE)))
assert SPEC and SPEC.loader
t06 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t06
SPEC.loader.exec_module(t06)

EXPECTED_ORIGINAL_COMMAND = "p40-t09-qwen-lmhead-cpuorder"
RESULTS_ORIGINAL_COMMAND = "p40-t09-qwen-lmhead-cpuorder-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t09-lmhead/c/qwen36")
ENGINE_SHA256 = "857940ee9bf68844e81bc80b8dc0921909842bcf83bd0ea76cd12c265d239af6"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t09-qwen-lmhead-cpuorder")
PROFILE_ID = "t09-dn-lmhead-cpuorder-16"
SCHEMA_VERSION = "p40-t09-qwen-lmhead-cpuorder-v1"
EXPECTED_STDOUT_SHA256 = "43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a"
CACHE_MARKER = b"[dn-cpuorder] exact Q8 cache active: 90 DeltaNet matrices plus LM head on GPU 0"

_base_model_argv = t06.model_argv
_base_run = t06.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_DN_DEVICE=0":
            result.append("COLI_CUDA_LMHEAD_CPUORDER=1")
    return result


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        if CACHE_MARKER not in stderr:
            result["failure_reason"] = "lmhead_cpuorder_not_active"
            result["status"] = "fail"
    return result


t06.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t06.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t06.ENGINE = ENGINE
t06.ENGINE_SHA256 = ENGINE_SHA256
t06.RESULT_DIRECTORY = RESULT_DIRECTORY
t06.PROFILE_ID = PROFILE_ID
t06.SCHEMA_VERSION = SCHEMA_VERSION
t06.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t06.model_argv = model_argv
t06.qwen.ENGINE = ENGINE
t06.qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t06.qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t06.qwen.RESULT_DIRECTORY = RESULT_DIRECTORY
t06.qwen.PROFILE_ID = PROFILE_ID
t06.qwen.SCHEMA_VERSION = SCHEMA_VERSION
t06.qwen.model_argv = model_argv
t06.qwen.run = run


def main() -> int:
    return t06.main()


if __name__ == "__main__":
    raise SystemExit(main())
