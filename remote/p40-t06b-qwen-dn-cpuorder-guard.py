#!/usr/bin/env python3
"""Forced-command identity for T06B's dedicated-stream Qwen integration."""
from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t06-qwen-dn-cpuorder-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t06-qwen-dn-cpuorder-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t06b_base", SourceFileLoader("p40_t06b_base", str(SOURCE)))
assert SPEC and SPEC.loader
t06 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t06
SPEC.loader.exec_module(t06)

EXPECTED_ORIGINAL_COMMAND = "p40-t06b-qwen-dn-cpuorder"
RESULTS_ORIGINAL_COMMAND = "p40-t06b-qwen-dn-cpuorder-results"
ENGINE_SHA256 = "bbee1f3c74cd57e1165d2de97c955b16a12c42f193644e4bf20d70873de19ffb"
RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t06b-qwen-dn-cpuorder")
PROFILE_ID = "t06b-dn-cpuorder-16-dedicated-stream"
SCHEMA_VERSION = "p40-t06b-qwen-dn-cpuorder-v1"

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
