"""Unit tests for the optional Q&A history store."""

import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from src.history import append_interaction


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.base = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_append_creates_file_and_parent_dir(self):
        history_file = self.base / "nested" / "qa_history.jsonl"
        self.assertFalse(history_file.parent.exists())

        append_interaction(history_file, "Q1?", "A1", ["a.txt"])

        self.assertTrue(history_file.parent.exists())
        self.assertTrue(history_file.exists())

    def test_append_writes_one_jsonl_record_per_call(self):
        history_file = self.base / "qa_history.jsonl"

        append_interaction(history_file, "Q1?", "A1", ["a.txt"])
        append_interaction(history_file, "Q2?", "A2", [])

        lines = history_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)

        first = json.loads(lines[0])
        second = json.loads(lines[1])

        self.assertEqual(set(first), {"timestamp", "question", "answer", "sources"})
        self.assertEqual(first["question"], "Q1?")
        self.assertEqual(first["answer"], "A1")
        self.assertEqual(first["sources"], ["a.txt"])
        self.assertEqual(second["question"], "Q2?")
        self.assertEqual(second["sources"], [])

    def test_timestamp_is_iso_utc(self):
        history_file = self.base / "qa_history.jsonl"

        append_interaction(history_file, "Q?", "A", [])

        record = json.loads(
            history_file.read_text(encoding="utf-8").splitlines()[0]
        )
        # Parses as ISO 8601 and carries timezone info (UTC).
        ts = datetime.fromisoformat(record["timestamp"])
        self.assertIsNotNone(ts.tzinfo)


if __name__ == "__main__":
    unittest.main()
