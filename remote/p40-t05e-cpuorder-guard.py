#!/usr/bin/env python3
"""Distinct forced-command identity for the standalone T05E Q8 kernel."""
from __future__ import annotations

from importlib.machinery import SourceFileLoader
import importlib.util
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("p40-t05a-q8-guard")
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("p40-t05a-q8-guard.py")
SPEC = importlib.util.spec_from_loader("p40_t05e_base", SourceFileLoader("p40_t05e_base", str(SOURCE)))
assert SPEC and SPEC.loader
t05a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t05a
SPEC.loader.exec_module(t05a)

t05a.EXPECTED_ORIGINAL_COMMAND = "p40-t05e-cpuorder"
t05a.RESULTS_ORIGINAL_COMMAND = "p40-t05e-cpuorder-results"
t05a.BENCHMARK = Path("/home/jordanculver/p40-native-research/colibri-t05a/p40_benchmarks/qwen_dn_out_cpuorder_control")
t05a.BENCHMARK_SHA256 = "4dc8f05ffb36cb963f84cf93a9f178509a6b2aba40e1665b36f608024b154c76"
t05a.RESULT_DIRECTORY = Path("/home/jordanculver/p40-native-research/results/t05e-cpuorder-controls")
t05a.PROFILE_ID = "t05e-dn-out-cpuorder-4096x2048"
t05a.SCHEMA_VERSION = "p40-t05e-cpuorder-v1"
t05a.FIXED_ARGUMENTS = (
    "--profile", "dn-out-cpuorder-4096x2048", "--gpu", "0", "--repetitions", "5",
    "--calls-per-sample", "8", "--memory-cap-mib", "48", "--seed", "1",
)


def main() -> int:
    return t05a.main()


if __name__ == "__main__":
    raise SystemExit(main())
