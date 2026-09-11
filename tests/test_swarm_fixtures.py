import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm.fixtures import TASKS, get_task


class FixtureTests(unittest.TestCase):
    def test_fixture_identifiers_and_source_paths_are_stable(self):
        self.assertEqual([task.identifier for task in TASKS], [
            "parity", "display-name-whitespace", "currency-grouping",
        ])
        for task in TASKS:
            self.assertTrue(task.objective)
            self.assertTrue(task.hypothesis)
            self.assertTrue(task.files)
            self.assertTrue(all(not path.startswith("/") and ".." not in path.split("/") for path in task.files))

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(ValueError):
            get_task("not-a-task")


if __name__ == "__main__":
    unittest.main()
