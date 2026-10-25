from __future__ import annotations

import struct
import unittest

from sifac.anim import BmaAnimation, BmaError

from tests.helpers import _chunk


def build_bma(version: bytes = b"103\0", constant: float = 60.0) -> bytes:
    payload = version + struct.pack("<f", constant) + b"\0" * 4
    return _chunk("ANHD", payload) + _chunk("ANSK", bytes(16))


class AnimTests(unittest.TestCase):
    def test_header(self) -> None:
        animation = BmaAnimation(build_bma())
        self.assertEqual(animation.version, "103")
        self.assertAlmostEqual(animation.duration, 60.0, places=4)
        self.assertEqual(animation.tracks, [])

    def test_header_after_prefix(self) -> None:
        raw = b"\0" * 0x80 + build_bma()
        animation = BmaAnimation(raw)
        self.assertEqual(animation.version, "103")

    def test_bad_magic(self) -> None:
        with self.assertRaises(BmaError):
            BmaAnimation(b"nope" * 8)


if __name__ == "__main__":
    unittest.main()
