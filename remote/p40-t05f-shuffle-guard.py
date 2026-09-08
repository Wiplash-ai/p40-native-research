#!/usr/bin/env python3
"""Distinct forced-command identity for the T05F shuffle-reduction control."""
from __future__ import annotations

from importlib.machinery import SourceFileLoader
import importlib.util
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t05a-q8-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t05a-q8-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t05f_base", SourceFileLoader("p40_t05f_base", str(SOURCE)))
assert SPEC and SPEC.loader
t05a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t05a
SPEC.loader.exec_module(t05a)

t05a.EXPECTED_ORIGINAL_COMMAND = "p40-t05f-shuffle"
t05a.RESULTS_ORIGINAL_COMMAND = "p40-t05f-shuffle-results"
t05a.BENCHMARK = Path("/home/jordanculver/p40-native-research/colibri-t05a/p40_benchmarks/qwen_dn_out_cpuorder_control")
t05a.BENCHMARK_SHA256 = "c2b2873420ac96a57d21f5a04ae1cdc1221c127642e2cf25cee175f2f8dc5e64"
t05a.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t05f-shuffle-controls")
t05a.PROFILE_ID = "t05f-dn-out-cpuorder-shuffle-4096x2048"
t05a.SCHEMA_VERSION = "p40-t05f-shuffle-v1"
t05a.FIXED_ARGUMENTS = (
    "--profile", "dn-out-cpuorder-4096x2048", "--gpu", "0", "--repetitions", "5",
    "--calls-per-sample", "8", "--memory-cap-mib", "48", "--seed", "1",
)


def main() -> int:
    return t05a.main()


if __name__ == "__main__":
    raise SystemExit(main())
