from __future__ import annotations

import math
import struct
import unittest

from sifac.bscam import BscamAnimation, lens_to_fov


def build_bscam(frames: int = 4, lens: float = 30.17) -> bytes:
    payload = bytearray(0x48)
    payload[0:4] = b"101\0"
    struct.pack_into("<2f", payload, 0x04, 60.0, float(frames))
    struct.pack_into("<3f", payload, 0x20, 1.5, 11.0, 1.4)
    struct.pack_into("<f", payload, 0x40, lens)
    data = bytearray(b"BSCM" + struct.pack("<2I", len(payload), 0) + bytes(payload))
    bodies = (
        b"".join(
            struct.pack("<5f", 0.0, 0.0, index * 0.5, 2.0 + index, 10.0)
            for index in range(frames)
        ),
        b"".join(
            struct.pack("<6f", 0.0, 0.0, 0.0, 0.0, 1.0, 0.0) for _ in range(frames)
        ),
        b"".join(struct.pack("<3f", 0.0, 0.0, 20.0 + index) for index in range(frames)),
    )
    for body in bodies:
        data += b"DATA" + struct.pack("<2I", len(body), 0) + body
    return bytes(data)


class BscamTests(unittest.TestCase):
    def test_lens_to_fov(self) -> None:
        self.assertAlmostEqual(lens_to_fov(10.125), 90.0, places=3)
        self.assertAlmostEqual(lens_to_fov(30.17), 37.1, delta=0.2)
        self.assertAlmostEqual(lens_to_fov(0.0), 90.0)

    def test_state_reads_the_three_tables(self) -> None:
        animation = BscamAnimation(build_bscam())
        state = animation.state(2)
        self.assertAlmostEqual(state.height, 4.0)
        self.assertAlmostEqual(state.distance, 10.0)
        self.assertAlmostEqual(state.position[0], math.sin(1.0) * 10.0, places=5)
        self.assertAlmostEqual(state.position[1], -math.cos(1.0) * 10.0, places=5)
        self.assertEqual(state.target[0], 0.0)
        self.assertEqual(state.target[1], 0.0)
        self.assertAlmostEqual(state.target[2], 1.4, places=5)
        self.assertAlmostEqual(state.fov_degrees, lens_to_fov(22.0), places=5)

    def test_frame_is_clamped(self) -> None:
        animation = BscamAnimation(build_bscam())
        self.assertEqual(animation.state(-5).height, animation.state(0).height)
        self.assertEqual(animation.state(999).height, animation.state(3).height)

    def test_rejects_other_payloads(self) -> None:
        with self.assertRaises(Exception):
            BscamAnimation(bytes(64))


if __name__ == "__main__":
    unittest.main()
