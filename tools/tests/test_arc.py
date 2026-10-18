from __future__ import annotations

import struct
import unittest

from sifac.arc import ArcArchive, ArcError
from sifac.binary import unwrap_section

from tests.helpers import build_arc


class ArcTests(unittest.TestCase):
    def test_parse_entries(self) -> None:
        data = build_arc(
            [
                ("mdl", "model.pac", b"BMAR104\0body"),
                ("bin", "data.pac", b"POF0payload"),
            ]
        )
        arc = ArcArchive(data)
        self.assertEqual(arc.name, "test.pac")
        self.assertEqual(len(arc), 2)
        self.assertEqual([entry.name for entry in arc], ["model.pac", "data.pac"])
        self.assertEqual([entry.type for entry in arc], ["mdl", "bin"])
        self.assertTrue(arc.entry_data(0).startswith(b"BMAR104\0body"))
        self.assertIsNone(arc.find("missing"))

    def test_unwrap_section(self) -> None:
        arc_bytes = build_arc([("bin", "x.bin", b"POF0x")])
        wrapped = b"\0" * 0x80 + arc_bytes
        payload, skipped = unwrap_section(wrapped)
        self.assertEqual(skipped, 0x80)
        self.assertTrue(payload.startswith(b"ARC\0"))
        nested = ArcArchive(payload)
        self.assertEqual(nested.entries[0].name, "x.bin")

    def test_unwrap_section_without_high_bit(self) -> None:
        arc_bytes = build_arc([("bin", "x.bin", b"POF0x")])
        wrapped = struct.pack("<I", len(arc_bytes)) + b"\0" * 0x7C + arc_bytes
        payload, skipped = unwrap_section(wrapped)
        self.assertEqual(skipped, 0x80)
        self.assertTrue(payload.startswith(b"ARC\0"))

    def test_bad_magic(self) -> None:
        with self.assertRaises(ArcError):
            ArcArchive(b"not an arc")


if __name__ == "__main__":
    unittest.main()
