from __future__ import annotations

import struct
import unittest

from sifac.anim import BmaAnimation, BmaError, RotationKeys

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


class RotationKeyTests(unittest.TestCase):
    def keys(self) -> RotationKeys:
        return RotationKeys(
            times=[0.0, 10.0, 20.0],
            quaternions=[(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)],
        )

    def test_interpolates_between_keys(self) -> None:
        self.assertEqual(self.keys().sample(5.0), (0.5, 0.0, 0.0, 0.5))

    def test_clamps_outside_the_range(self) -> None:
        keys = self.keys()
        self.assertEqual(keys.sample(-1.0), keys.quaternions[0])
        self.assertEqual(keys.sample(99.0), keys.quaternions[-1])

    def test_hits_a_key_exactly(self) -> None:
        self.assertEqual(self.keys().sample(10.0), (1.0, 0.0, 0.0, 0.0))

    def test_empty_keys_is_identity(self) -> None:
        self.assertEqual(RotationKeys([], []).sample(3.0), (0.0, 0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
