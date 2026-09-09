#!/usr/bin/env python3
"""Forced-command identity for the exact T23 DeltaNet pair canary."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t15-qwen-shared-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t15-qwen-shared-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t23_base", SourceFileLoader("p40_t23_base", str(SOURCE)))
assert SPEC and SPEC.loader
t15 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t15
SPEC.loader.exec_module(t15)

EXPECTED_ORIGINAL_COMMAND = "p40-t23-qwen-deltanet-pair"
RESULTS_ORIGINAL_COMMAND = "p40-t23-qwen-deltanet-pair-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t23-dn-pair/c/qwen36")
ENGINE_SHA256 = "c014b491249b6692995a95f8d3e3ed5d0ae3dae657df4e5ae98e1c501dea21d7"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t23-qwen-deltanet-pair")
PROFILE_ID = "t23-exact-qwen-deltanet-pair-16"
SCHEMA_VERSION = "p40-t23-qwen-deltanet-pair-v1"
EXPECTED_STDOUT_SHA256 = "43095be2395844a0c7f321ea2496689325d5ae890f91ab53bb31c55e37a0877a"
PAIR_MARKER = b"[dn-cpuorder] exact Q8 QKV/Z pair issue/join active; CPU B/A overlap enabled"


_base_model_argv = t15.model_argv
_base_run = t15.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_DN_CPUORDER=1":
            result.append("COLI_CUDA_DN_PAIR=1")
    return result


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        if PAIR_MARKER not in stderr:
            result["failure_reason"] = "deltanet_pair_not_active"
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


configure(t15)
configure(t15.t12)
configure(t15.t12.t09)
configure(t15.t12.t09.t06)
configure(t15.t12.t09.t06.qwen)
t15.t12.t09.t06.qwen.OUTPUT_TOKENS = 16
t15.t12.t09.t06.qwen.run = run


def main() -> int:
    return t15.main()


if __name__ == "__main__":
    raise SystemExit(main())
