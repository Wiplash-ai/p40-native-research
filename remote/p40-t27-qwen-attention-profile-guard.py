#!/usr/bin/env python3
"""Forced-command identity for the exact-Qwen T27 attention timing control."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t24-qwen-deltanet-pair-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t27_base", SourceFileLoader("p40_t27_base", str(SOURCE)))
assert SPEC and SPEC.loader
t24 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t24
SPEC.loader.exec_module(t24)

EXPECTED_ORIGINAL_COMMAND = "p40-t27-qwen-attention-profile"
RESULTS_ORIGINAL_COMMAND = "p40-t27-qwen-attention-profile-results"
ENGINE = Path("/home/jordanculver/p40-native-research/colibri-t27-attn-profile/c/qwen36")
ENGINE_SHA256 = "1c98079d523efcdb536fa66e4772abc0b024a25927ce525f1b3058946405b34f"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t27-qwen-attention-profile")
PROFILE_ID = "t27-exact-qwen-attention-subprofile-64"
SCHEMA_VERSION = "p40-t27-qwen-attention-profile-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"
ATTN_PROFILE_MARKER = b"[timers]   attn-sub:"


_base_model_argv = t24.model_argv
_base_run = t24.run


def model_argv() -> list[str]:
    result: list[str] = []
    for argument in _base_model_argv():
        result.append(argument)
        if argument == "COLI_CUDA_ATTN_CPUORDER=1":
            result.append("COLI_ATTN_PROFILE=1")
    return result


def run(request: dict) -> dict:
    result = _base_run(request)
    if not request["dry_run"] and result.get("status") == "pass":
        stderr = Path(result.get("stderr", {}).get("path", "")).read_bytes()
        if ATTN_PROFILE_MARKER not in stderr:
            result["failure_reason"] = "attention_subprofile_missing"
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
t24.t23.t15.t12.t09.t06.qwen.OUTPUT_TOKENS = 64
t24.t23.t15.t12.t09.t06.qwen.run = run


def main() -> int:
    return t24.t23.t15.main()


if __name__ == "__main__":
    raise SystemExit(main())
