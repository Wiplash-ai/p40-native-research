#!/usr/bin/env python3
"""Forced-command identity for the T12 exact-Q8 attention 64-output comparison."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t12-qwen-attention-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t12-qwen-attention-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t13_base", SourceFileLoader("p40_t13_base", str(SOURCE)))
assert SPEC and SPEC.loader
t12 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t12
SPEC.loader.exec_module(t12)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t13-qwen-attention-cpuorder"
RESULTS_ORIGINAL_COMMAND = "p40-t13-qwen-attention-cpuorder-results"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t13-qwen-attention-cpuorder")
PROFILE_ID = "t13-dn-lmhead-attention-cpuorder-64"
SCHEMA_VERSION = "p40-t13-qwen-attention-cpuorder-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"

_base_model_argv = t12.model_argv


def model_argv() -> list[str]:
    return [f"N_NEW={OUTPUT_TOKENS}" if argument.startswith("N_NEW=") else argument
            for argument in _base_model_argv()]


t12.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t12.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t12.RESULT_DIRECTORY = RESULT_DIRECTORY
t12.PROFILE_ID = PROFILE_ID
t12.SCHEMA_VERSION = SCHEMA_VERSION
t12.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t12.model_argv = model_argv
t12.t09.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t12.t09.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t12.t09.RESULT_DIRECTORY = RESULT_DIRECTORY
t12.t09.PROFILE_ID = PROFILE_ID
t12.t09.SCHEMA_VERSION = SCHEMA_VERSION
t12.t09.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t12.t09.model_argv = model_argv
t12.t09.t06.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t12.t09.t06.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t12.t09.t06.RESULT_DIRECTORY = RESULT_DIRECTORY
t12.t09.t06.PROFILE_ID = PROFILE_ID
t12.t09.t06.SCHEMA_VERSION = SCHEMA_VERSION
t12.t09.t06.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t12.t09.t06.model_argv = model_argv
t12.t09.t06.qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t12.t09.t06.qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t12.t09.t06.qwen.RESULT_DIRECTORY = RESULT_DIRECTORY
t12.t09.t06.qwen.PROFILE_ID = PROFILE_ID
t12.t09.t06.qwen.SCHEMA_VERSION = SCHEMA_VERSION
t12.t09.t06.qwen.OUTPUT_TOKENS = OUTPUT_TOKENS
t12.t09.t06.qwen.model_argv = model_argv


def main() -> int:
    return t12.main()


if __name__ == "__main__":
    raise SystemExit(main())
