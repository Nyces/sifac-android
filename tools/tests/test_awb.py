from __future__ import annotations

import unittest

from sifac.awb import Afs2Archive, AwbError

from tests.helpers import build_awb


class AwbTests(unittest.TestCase):
    def test_parse_entries(self) -> None:
        raw = build_awb([(0, b"first"), (7, b"second")])
        archive = Afs2Archive(raw)
        self.assertEqual(archive.count, 2)
        self.assertEqual(archive.ids, (0, 7))
        self.assertEqual(len(archive.entries), 2)
        self.assertEqual(archive.entries[0].magic, "HCA")
        self.assertTrue(archive.entry_data(0).startswith(b"HCA\0first"))

    def test_bad_magic(self) -> None:
        with self.assertRaises(AwbError):
            Afs2Archive(b"nope")


if __name__ == "__main__":
    unittest.main()
