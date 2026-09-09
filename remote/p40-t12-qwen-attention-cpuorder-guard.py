#!/usr/bin/env python3
"""Forced-command identity for the T12 exact-Q8 attention Qwen canary."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t09-qwen-lmhead-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t09-qwen-lmhead-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t12_base", SourceFileLoader("p40_t12_base", str(SOURCE)))
assert SPEC and SPEC.loader
t09 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t09
SPEC.loader.exec_module(t09)

EXPECTED_ORIGINAL_COMMAND = "p40-t12-qwen-attention-cpuorder"
RESULTS_ORIGINAL_COMMAND = "p40-t12-qwen-attention-cpuorder-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t12-attention/c/qwen36")
ENGINE_SHA256 = "cded60fd2b68980c858577354b1ada9fc89ea8bb813a95deb50b999fff6b7dad"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t12-qwen-attention-cpuorder")
PROFILE_ID = "t12-dn-lmhead-attention-cpuorder-16"
SCHEMA_VERSION = "p40-t12-qwen-attention-cpuorder-v1"
EXPECTED_STDOUT_SHA256 = "43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a"
CACHE_MARKER = b"[dn-cpuorder] exact Q8 cache active: 90 DeltaNet matrices, LM head, and 40 attention Q/K/V/O matrices on GPU 0"

_base_model_argv = t09.model_argv


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_LMHEAD_CPUORDER=1":
            result.append("COLI_CUDA_ATTN_CPUORDER=1")
    return result


_base_run = t09.run


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        if CACHE_MARKER not in stderr:
            result["failure_reason"] = "attention_cpuorder_not_active"
            result["status"] = "fail"
    return result


t09.ENGINE = ENGINE
t09.ENGINE_SHA256 = ENGINE_SHA256
t09.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t09.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t09.RESULT_DIRECTORY = RESULT_DIRECTORY
t09.PROFILE_ID = PROFILE_ID
t09.SCHEMA_VERSION = SCHEMA_VERSION
t09.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t09.CACHE_MARKER = CACHE_MARKER
t09.model_argv = model_argv
t09.t06.ENGINE = ENGINE
t09.t06.ENGINE_SHA256 = ENGINE_SHA256
t09.t06.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t09.t06.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t09.t06.RESULT_DIRECTORY = RESULT_DIRECTORY
t09.t06.PROFILE_ID = PROFILE_ID
t09.t06.SCHEMA_VERSION = SCHEMA_VERSION
t09.t06.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t09.t06.model_argv = model_argv
t09.t06.qwen.ENGINE = ENGINE
t09.t06.qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t09.t06.qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t09.t06.qwen.RESULT_DIRECTORY = RESULT_DIRECTORY
t09.t06.qwen.PROFILE_ID = PROFILE_ID
t09.t06.qwen.SCHEMA_VERSION = SCHEMA_VERSION
t09.t06.qwen.OUTPUT_TOKENS = 16
t09.t06.qwen.model_argv = model_argv
t09.t06.qwen.run = run


def main() -> int:
    return t09.main()


if __name__ == "__main__":
    raise SystemExit(main())
