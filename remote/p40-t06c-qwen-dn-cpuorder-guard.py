#!/usr/bin/env python3
"""Forced-command identity for T06C backend-owned CUDA device selection."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t06-qwen-dn-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t06-qwen-dn-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t06c_base", SourceFileLoader("p40_t06c_base", str(SOURCE)))
assert SPEC and SPEC.loader
t06 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t06
SPEC.loader.exec_module(t06)

EXPECTED_ORIGINAL_COMMAND = "p40-t06c-qwen-dn-cpuorder"
RESULTS_ORIGINAL_COMMAND = "p40-t06c-qwen-dn-cpuorder-results"
ENGINE_SHA256 = "8bef44d2ff680cc3fb35756c95bfdef343fafa989e67009b1ffff43011d49d4c"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t06c-qwen-dn-cpuorder")
PROFILE_ID = "t06c-dn-cpuorder-16-backend-device-selector"
SCHEMA_VERSION = "p40-t06c-qwen-dn-cpuorder-v1"

t06.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t06.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t06.ENGINE_SHA256 = ENGINE_SHA256
t06.RESULT_DIRECTORY = RESULT_DIRECTORY
t06.PROFILE_ID = PROFILE_ID
t06.SCHEMA_VERSION = SCHEMA_VERSION
t06.qwen.EXPECTED_ORIGINAL_COMMAND = EXPECTED_ORIGINAL_COMMAND
t06.qwen.RESULTS_ORIGINAL_COMMAND = RESULTS_ORIGINAL_COMMAND
t06.qwen.RESULT_DIRECTORY = RESULT_DIRECTORY
t06.qwen.PROFILE_ID = PROFILE_ID
t06.qwen.SCHEMA_VERSION = SCHEMA_VERSION


def main() -> int:
    return t06.main()


if __name__ == "__main__":
    raise SystemExit(main())
