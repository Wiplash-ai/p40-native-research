#!/usr/bin/env python3
"""Forced-command identity for the fixed 64-output T06C timing comparison."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t06-qwen-dn-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t06-qwen-dn-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t07_base", SourceFileLoader("p40_t07_base", str(SOURCE)))
assert SPEC and SPEC.loader
t06 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t06
SPEC.loader.exec_module(t06)

OUTPUT_TOKENS = 64
EXPECTED_ORIGINAL_COMMAND = "p40-t07-qwen-dn-cpuorder"
RESULTS_ORIGINAL_COMMAND = "p40-t07-qwen-dn-cpuorder-results"
ENGINE_SHA256 = "8bef44d2ff680cc3fb35756c95bfdef343fafa989e67009b1ffff43011d49d4c"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t07-qwen-dn-cpuorder")
PROFILE_ID = "t07-dn-cpuorder-64-backend-device-selector"
SCHEMA_VERSION = "p40-t07-qwen-dn-cpuorder-v1"
EXPECTED_STDOUT_SHA256 = "5909fad89de2faac1b77af72bf8b25f56b8f33853df59d35cb04120e9ce1f35f"

_base_model_argv = t06.model_argv


def model_argv() -> list[str]:
    return [f"N_NEW={OUTPUT_TOKENS}" if argument.startswith("N_NEW=") else argument
            for argument in _base_model_argv()]


t06.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t06.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t06.ENGINE_SHA256 = ENGINE_SHA256
t06.RESULT_DIRECTORY = RESULT_DIRECTORY
t06.PROFILE_ID = PROFILE_ID
t06.SCHEMA_VERSION = SCHEMA_VERSION
t06.EXPECTED_STDOUT_SHA256 = EXPECTED_STDOUT_SHA256
t06.model_argv = model_argv
t06.qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t06.qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t06.qwen.RESULT_DIRECTORY = RESULT_DIRECTORY
t06.qwen.PROFILE_ID = PROFILE_ID
t06.qwen.SCHEMA_VERSION = SCHEMA_VERSION
t06.qwen.OUTPUT_TOKENS = OUTPUT_TOKENS
t06.qwen.model_argv = model_argv


def main() -> int:
    return t06.main()


if __name__ == "__main__":
    raise SystemExit(main())
