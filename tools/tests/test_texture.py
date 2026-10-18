from __future__ import annotations

import struct
import unittest

from sifac.texture import (
    FORMAT_BC7,
    FORMAT_RGBA8,
    BtxError,
    BtxTexture,
    count_levels,
    parse_header,
)

from tests.helpers import build_btx


class BtxTests(unittest.TestCase):
    def test_rgba8_header_and_pixels(self) -> None:
        pixels = bytes(
            [
                255, 0, 0, 255,
                0, 255, 0, 255,
                0, 0, 255, 255,
                255, 255, 255, 255,
            ]
        )
        raw = build_btx("chip", 2, 2, FORMAT_RGBA8, pixels)
        texture = BtxTexture(raw)
        self.assertEqual(texture.header.name, "chip")
        self.assertEqual(texture.header.width, 2)
        self.assertEqual(texture.header.height, 2)
        self.assertEqual(texture.header.format_name, "rgba8")
        self.assertEqual(texture.levels, 1)
        width, height, rgba = texture.to_rgba()
        self.assertEqual((width, height), (2, 2))
        self.assertEqual(rgba, pixels)
        png = texture.to_png()
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_bc7_sizing(self) -> None:
        raw = build_btx("bc", 8, 8, FORMAT_BC7, b"\0" * 16 * 4)
        texture = BtxTexture(raw)
        self.assertEqual(texture.header.format_name, "bc7")
        self.assertEqual(texture.levels, 1)
        self.assertEqual(len(texture.level_bytes(0)), 64)

    def test_unknown_format_is_reported(self) -> None:
        raw = build_btx("face", 4, 4, 0x1B, b"\0" * 4 * 4 * 8)
        texture = BtxTexture(raw)
        self.assertEqual(texture.header.format_name, "unknown-0x1b")
        with self.assertRaises(BtxError):
            texture.to_rgba()

    def test_mip_levels(self) -> None:
        level0 = 16 * 16 * 4
        level1 = 8 * 8 * 4
        raw = build_btx("mip", 16, 16, FORMAT_RGBA8, b"\1" * (level0 + level1))
        texture = BtxTexture(raw)
        self.assertEqual(texture.levels, 2)
        self.assertEqual(len(texture.level_bytes(1)), level1)

    def test_bad_magic(self) -> None:
        with self.assertRaises(BtxError):
            parse_header(b"nope" + b"\0" * 200)

    def test_count_levels_falls_back_to_one(self) -> None:
        self.assertEqual(count_levels(0, 4, 4, FORMAT_RGBA8), 1)


if __name__ == "__main__":
    unittest.main()
