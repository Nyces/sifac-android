from __future__ import annotations

import unittest

from sifac.bmar import BMAR_MAGIC, BmarArchive, BmarError

from tests.helpers import build_arc


class BmarTests(unittest.TestCase):
    def test_parse_bmar(self) -> None:
        arc = build_arc([("BML\0", "scene", b"BML\0payload")], name="BladeModelPac")
        raw = BMAR_MAGIC + arc
        archive = BmarArchive(raw)
        self.assertEqual(len(archive.entries), 1)
        entry = archive.entries[0]
        self.assertEqual(entry.type, "BML")
        self.assertEqual(entry.name, "scene")
        self.assertEqual(entry.payload_magic, "BML")
        self.assertTrue(archive.entry_data(0).startswith(b"BML\0"))

    def test_bad_magic(self) -> None:
        with self.assertRaises(BmarError):
            BmarArchive(b"nope")


if __name__ == "__main__":
    unittest.main()
