from __future__ import annotations

import unittest

from sifac.blbn import BlbnError, BlbnFile, salted_hash

from .helpers import build_blbn


class BlbnTests(unittest.TestCase):
    def test_reads_symbols_after_the_section_header(self) -> None:
        blob, offsets = build_blbn(
            {"s_live_datas": b"\x01" * 16, "s_music_datas": b"\x02" * 8}
        )
        container = BlbnFile(blob)
        self.assertEqual(
            sorted(container.symbols), ["s_live_datas", "s_music_datas"]
        )
        self.assertEqual(container.read("s_live_datas"), b"\x01" * 16)
        self.assertEqual(container.read("s_music_datas"), b"\x02" * 8)
        self.assertEqual(container.symbols["s_live_datas"].offset, offsets["s_live_datas"])
        self.assertEqual(container.symbols["s_live_datas"].hash, salted_hash("s_live_datas"))

    def test_text_reads_base_relative_strings(self) -> None:
        blob, offsets = build_blbn({"s_stage_datas": b"%s"}, strings=["School Gym"])
        container = BlbnFile(blob)
        self.assertEqual(container.text(offsets["School Gym"]), "School Gym")

    def test_missing_symbol_raises(self) -> None:
        blob, _ = build_blbn({"s_live_datas": b"\0" * 4})
        with self.assertRaises(KeyError):
            BlbnFile(blob).read("s_music_datas")

    def test_rejects_other_payloads(self) -> None:
        with self.assertRaises(BlbnError):
            BlbnFile(b"ARC\0" + bytes(64))

    def test_salted_hash_is_order_sensitive(self) -> None:
        self.assertNotEqual(salted_hash("ab"), salted_hash("ba"))
        self.assertEqual(salted_hash(""), 0)


if __name__ == "__main__":
    unittest.main()
