import json
import tempfile
import unittest
from pathlib import Path

from app import atomic_write_json


class AppPersistenceTests(unittest.TestCase):
    def test_atomic_write_json_replaces_file_without_tmp_leftover(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest.json"
            atomic_write_json(path, {"version": 1})
            atomic_write_json(path, {"version": 2, "text": "안정성"})

            data = json.loads(path.read_text(encoding="utf-8"))
            leftovers = list(Path(temp_dir).glob("*.tmp"))

            self.assertEqual(data["version"], 2)
            self.assertEqual(data["text"], "안정성")
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
