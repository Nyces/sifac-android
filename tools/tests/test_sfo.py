from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from sifac.sfo import parse_sfo, read_sfo

from tests.helpers import build_sfo


class SfoTests(unittest.TestCase):
    def test_parse_values(self) -> None:
        raw = build_sfo({"TITLE_ID": "CUSA24620", "APP_VER": "01.00", "COUNT": 3})
        values = parse_sfo(raw)
        self.assertEqual(values["TITLE_ID"], "CUSA24620")
        self.assertEqual(values["APP_VER"], "01.00")
        self.assertEqual(values["COUNT"], 3)

    def test_bad_magic(self) -> None:
        with self.assertRaises(ValueError):
            parse_sfo(b"not a sfo")

    def test_read_file(self) -> None:
        raw = build_sfo({"TITLE_ID": "CUSA24620"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "param.sfo"
            path.write_bytes(raw)
            self.assertEqual(read_sfo(path)["TITLE_ID"], "CUSA24620")


if __name__ == "__main__":
    unittest.main()
