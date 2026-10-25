"""Blade timeline (``.btlb``) reader.

A payload starts with an 0x80-byte header, then a ``BTL\\0`` block::

    0x80  "BTL\\0"
    0x84  u32 channel count (10 in the retail dump)
    0x88  u32 sum of the per-channel sizes (948)
    0x8c  u32 half of that sum (474)
    0x90  u32 1
    0x94  u32 payload size (19912)
    0x98  u32 1
    0x9c  u32 1
    0xa0  channel table, ``count`` entries of 16 bytes:
              u32 size, u32 offset, u32 flags, u32 index

Every live song ships four timelines::

    scene_<song>_000_default.btlb     scene defaults
    scene_stage_<song>_<n>.btlb       stage events
    score_<song>_lyrics_<title>.btlb  lyrics cues
    scene_chara_<song>_<n>.btlb       one record per performer

The character timeline holds one 0x40-byte record per performer, followed by
that performer's keyframe tables::

    +0x00 u32 0
    +0x04 u32 last cue tick
    +0x08 u32 1
    +0x0c u16 u16   (49, 68 in every record of song 0010)
    +0x10 f32 0, f32 a, f32 b, f32 c
    +0x20 f32 x, f32 y, f32 z    performer placement
    +0x2c f32 15.0
    +0x30 16 zero bytes

The placement axes are **not** verified yet: song 0010 keeps one axis near 1.4
for all nine performers (chest/head height, or a depth offset) while the other
two spread over 7.0 m and 1.8 m. Song 0020 spreads all three axes, so the
interpretation cannot be settled from the data alone; a RenderDoc capture of a
live frame would show the performer world matrices directly. In the meantime
the records are exposed verbatim rather than guessed: every retail song checked
carries exactly nine of them, matching the nine performers.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .binary import u32

MAGIC = b"BTL\0"
HEADER_SIZE = 0x80
CHANNEL_TABLE = 0xA0
CHANNEL_ENTRY_SIZE = 16
RECORD_SIZE = 0x40
RECORD_MARKER = 15.0


class BtlbError(ValueError):
    """Raised when a .btlb payload is malformed."""


@dataclass(frozen=True)
class BtlbChannel:
    size: int
    offset: int
    flags: int
    index: int


@dataclass(frozen=True)
class CharaRecord:
    """One performer placement record of a character timeline."""

    offset: int
    end_tick: int
    ids: tuple[int, int]
    blend: tuple[float, float, float, float]
    position: tuple[float, float, float]
    marker: float


class Btlb:
    def __init__(self, data: bytes) -> None:
        if len(data) < CHANNEL_TABLE or data[HEADER_SIZE : HEADER_SIZE + 4] != MAGIC:
            raise BtlbError("not a .btlb timeline")
        self.data = data
        (
            self.channel_count,
            self.channel_size,
            self.channel_half,
            self.flags,
            self.payload_size,
            self.tail_a,
            self.tail_b,
        ) = struct.unpack_from("<7I", data, HEADER_SIZE + 4)
        self.channels = tuple(self._read_channels())

    def _read_channels(self) -> list[BtlbChannel]:
        channels: list[BtlbChannel] = []
        for index in range(min(self.channel_count, 0x40)):
            base = CHANNEL_TABLE + index * CHANNEL_ENTRY_SIZE
            if base + CHANNEL_ENTRY_SIZE > len(self.data):
                break
            size, offset, flags, track = struct.unpack_from("<4I", self.data, base)
            if size > len(self.data) or offset > len(self.data):
                break
            channels.append(BtlbChannel(size, offset, flags, track))
        return channels

    def chara_records(self) -> list[CharaRecord]:
        """Return every performer placement record found in the payload."""
        records: list[CharaRecord] = []
        position = CHANNEL_TABLE
        limit = len(self.data) - RECORD_SIZE
        while position <= limit:
            if self._is_chara_record(position):
                head = struct.unpack_from("<4I", self.data, position)
                ids = struct.unpack_from("<2H", self.data, position + 0x0C)
                blend = struct.unpack_from("<4f", self.data, position + 0x10)
                place = struct.unpack_from("<3f", self.data, position + 0x20)
                marker = struct.unpack_from("<f", self.data, position + 0x2C)[0]
                records.append(
                    CharaRecord(position, head[1], ids, blend, place, marker)
                )
                position += RECORD_SIZE
                continue
            position += 4
        return records

    def _is_chara_record(self, offset: int) -> bool:
        if u32(self.data, offset) != 0 or u32(self.data, offset + 8) != 1:
            return False
        marker = struct.unpack_from("<f", self.data, offset + 0x2C)[0]
        if abs(marker - RECORD_MARKER) > 1e-6:
            return False
        return self.data[offset + 0x30 : offset + 0x3C] == bytes(12)


def load_timelines(pac_path: str | Path) -> dict[str, Btlb]:
    """Load every ``.btlb`` stored in a song pack's ``timeline.pac``."""
    from .arc import ArcArchive, ArcError
    from .binary import unwrap_section
    from .pac import read_pac

    image = read_pac(pac_path)
    top = ArcArchive(image.data)
    entry = top.find("timeline.pac")
    if entry is None:
        return {}
    payload, _ = unwrap_section(top.entry_data(entry))
    try:
        timeline = ArcArchive(payload)
    except ArcError:
        return {}
    timelines: dict[str, Btlb] = {}
    for item in timeline.entries:
        if not item.name.lower().endswith(".btlb"):
            continue
        try:
            timelines[item.name] = Btlb(timeline.entry_data(item))
        except BtlbError:
            continue
    return timelines
