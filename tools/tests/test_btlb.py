from __future__ import annotations

import struct
import unittest

from sifac.btlb import Btlb, BtlbError


def build_record(
    position,
    end_tick: int = 7095,
    marker: float = 15.0,
    cues: list[tuple[int, int, int]] | None = None,
) -> bytes:
    record = bytearray(0x60)
    struct.pack_into("<4I", record, 0x00, 0, end_tick, 1, 0x00440031)
    struct.pack_into("<4f", record, 0x10, 0.0, 0.002, 0.089, 0.082)
    struct.pack_into("<3f", record, 0x20, *position)
    struct.pack_into("<f", record, 0x2C, marker)
    for ident, start_tick, length in cues or []:
        record += struct.pack(
            "<7I",
            0x000C000A,
            0,
            ident,
            0x00070064,
            0x3F800000,
            start_tick,
            length,
        )
    return bytes(record)


def build_btlb(records: list[bytes], channels: int = 2) -> bytes:
    data = bytearray(0xA0)
    data[0x80:0x84] = b"BTL\0"
    struct.pack_into("<7I", data, 0x84, channels, 948, 474, 1, 19912, 1, 1)
    for index in range(channels):
        data += struct.pack("<4I", 0x64, index * 0x10, 1, index)
    data += bytes(16)
    for record in records:
        data += record
    return bytes(data)


class BtlbTests(unittest.TestCase):
    def test_header_and_channels(self) -> None:
        timeline = Btlb(build_btlb([], channels=3))
        self.assertEqual(timeline.channel_count, 3)
        self.assertEqual(timeline.payload_size, 19912)
        self.assertEqual(len(timeline.channels), 3)
        self.assertEqual(timeline.channels[1].offset, 0x10)

    def test_chara_records(self) -> None:
        timeline = Btlb(
            build_btlb(
                [
                    build_record((-0.418, 1.202, 2.385)),
                    build_record((3.062, 1.442, 2.075)),
                ]
            )
        )
        records = timeline.chara_records()
        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(records[0].position[0], -0.418, places=3)
        self.assertAlmostEqual(records[1].position[2], 2.075, places=3)
        self.assertEqual(records[0].end_tick, 7095)
        self.assertEqual(records[0].ids, (49, 68))
        self.assertAlmostEqual(records[0].marker, 15.0)

    def test_marker_is_required(self) -> None:
        timeline = Btlb(build_btlb([build_record((1.0, 2.0, 3.0), marker=16.0)]))
        self.assertEqual(timeline.chara_records(), [])

    def test_cues_are_parsed(self) -> None:
        timeline = Btlb(
            build_btlb(
                [
                    build_record((-0.4, 1.2, 2.4), cues=[(72, 518, 48), (73, 621, 30)]),
                    build_record((1.2, 1.5, 2.4), cues=[(81, 1485, 108)]),
                ]
            )
        )
        records = timeline.chara_records()
        self.assertEqual(len(records), 2)
        self.assertEqual([cue.ident for cue in records[0].cues], [72, 73])
        self.assertEqual(records[0].cues[0].start_tick, 518)
        self.assertEqual(records[0].cues[0].length, 48)
        self.assertAlmostEqual(records[0].cues[0].weight, 1.0)
        self.assertEqual(len(records[1].cues), 1)
        self.assertEqual(records[1].cues[0].ident, 81)

    def test_rejects_other_payloads(self) -> None:
        with self.assertRaises(BtlbError):
            Btlb(bytes(0x200))


if __name__ == "__main__":
    unittest.main()
