"""Tiny deterministic repository tasks for bounded executor research."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureTask:
    identifier: str
    objective: str
    hypothesis: str
    files: dict[str, str]


TASKS: tuple[FixtureTask, ...] = (
    FixtureTask(
        identifier="parity",
        objective="Correct the parity implementation so the supplied unit tests pass.",
        hypothesis="The modulo equality is inverted.",
        files={
            "calculator.py": "def is_even(value):\n    return value % 2 == 1\n",
            "tests/test_calculator.py": (
                "import unittest\n\nfrom calculator import is_even\n\n"
                "class ParityTest(unittest.TestCase):\n"
                "    def test_even_values(self):\n"
                "        self.assertTrue(is_even(2))\n        self.assertTrue(is_even(0))\n\n"
                "    def test_odd_values(self):\n"
                "        self.assertFalse(is_even(1))\n        self.assertFalse(is_even(-3))\n"
            ),
        },
    ),
    FixtureTask(
        identifier="display-name-whitespace",
        objective="Normalize a display name so the supplied unit tests pass.",
        hypothesis="Splitting on a literal space preserves repeated whitespace.",
        files={
            "names.py": (
                "def display_name(value):\n"
                "    return \" \".join(part.capitalize() for part in value.strip().split(\" \"))\n"
            ),
            "tests/test_names.py": (
                "import unittest\n\nfrom names import display_name\n\n"
                "class DisplayNameTest(unittest.TestCase):\n"
                "    def test_collapses_whitespace(self):\n"
                "        self.assertEqual(display_name(\"  ada   lovelace  \"), \"Ada Lovelace\")\n\n"
                "    def test_capitalizes_words(self):\n"
                "        self.assertEqual(display_name(\"grace hopper\"), \"Grace Hopper\")\n"
            ),
        },
    ),
    FixtureTask(
        identifier="currency-grouping",
        objective="Format currency with thousands grouping so the supplied unit tests pass.",
        hypothesis="The numeric format specifier omits the grouping comma.",
        files={
            "currency.py": "def format_cents(cents):\n    return f\"${cents / 100:.2f}\"\n",
            "tests/test_currency.py": (
                "import unittest\n\nfrom currency import format_cents\n\n"
                "class CurrencyTest(unittest.TestCase):\n"
                "    def test_groups_thousands(self):\n"
                "        self.assertEqual(format_cents(123456), \"$1,234.56\")\n\n"
                "    def test_small_amount(self):\n"
                "        self.assertEqual(format_cents(99), \"$0.99\")\n"
            ),
        },
    ),
    FixtureTask(
        identifier="multifile-timeout",
        objective="Return the configured timeout in seconds so the supplied unit tests pass.",
        hypothesis="Integer floor division loses the fractional portion of a millisecond timeout.",
        files={
            "settings.py": "DEFAULT_TIMEOUT_MS = 250\n",
            "service.py": (
                "from settings import DEFAULT_TIMEOUT_MS\n\n"
                "def timeout_seconds():\n"
                "    return DEFAULT_TIMEOUT_MS // 1000\n"
            ),
            "tests/test_service.py": (
                "import unittest\n\nfrom service import timeout_seconds\n\n"
                "class TimeoutTest(unittest.TestCase):\n"
                "    def test_preserves_fractional_seconds(self):\n"
                "        self.assertEqual(timeout_seconds(), 0.25)\n"
            ),
        },
    ),
    FixtureTask(
        identifier="two-file-required",
        objective="Normalize both user-facing labels so the supplied unit tests pass.",
        hypothesis="Both implementation files need the same lowercase normalization.",
        files={
            "primary.py": "def primary_label(value):\n    return value.strip()\n",
            "secondary.py": "def secondary_label(value):\n    return value.strip()\n",
            "tests/test_labels.py": (
                "import unittest\n\nfrom primary import primary_label\nfrom secondary import secondary_label\n\n"
                "class LabelTest(unittest.TestCase):\n"
                "    def test_primary(self):\n        self.assertEqual(primary_label(\" Ada \"), \"ada\")\n\n"
                "    def test_secondary(self):\n        self.assertEqual(secondary_label(\" Grace \"), \"grace\")\n"
            ),
        },
    ),
    FixtureTask(
        identifier="already-green",
        objective="Make an unnecessary improvement.",
        hypothesis="The baseline gate must reject a task whose tests already pass.",
        files={
            "healthy.py": "def identity(value):\n    return value\n",
            "tests/test_healthy.py": (
                "import unittest\n\nfrom healthy import identity\n\n"
                "class HealthyTest(unittest.TestCase):\n"
                "    def test_identity(self):\n        self.assertEqual(identity(3), 3)\n"
            ),
        },
    ),
)


def get_task(identifier: str) -> FixtureTask:
    for task in TASKS:
        if task.identifier == identifier:
            return task
    raise ValueError(f"unknown fixture task: {identifier}")
