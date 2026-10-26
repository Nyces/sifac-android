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
0x20 reserved bytes and that performer's cue list::

    +0x00 u32 0
    +0x04 u32 last cue tick
    +0x08 u32 1
    +0x0c u16 u16   (49, 68 in every record of song 0010)
    +0x10 f32 0, f32 a, f32 b, f32 c
    +0x20 f32 x, f32 y, f32 z    performer placement
    +0x2c f32 marker            15.0 for songs 0010/0020, 13.0 for 0030
    +0x30 16 zero bytes
    +0x60 cue list, 28 bytes per cue::

              +0x00 u32 0x000c000a   (channel 12, kind 10)
              +0x04 u32 0
              +0x08 u32 ident        (48..98 in song 0010)
              +0x0c u16 7, u16 100   (99 in a few cues)
              +0x10 f32 weight       (always 1.0 so far)
              +0x14 u32 start tick
              +0x18 u32 length (ticks, clamped at 12000 for the last cue)

The cue intervals are disjoint and shared across the nine performers, so they
are timeline *segments* rather than per-performer timing; the ``ident`` differs
per performer inside a segment.

The ``ident`` is a **facial preset id**: ``common/character_common.pac`` holds
``facial_preset_datas.bin`` whose ``s_facial_preset_datas`` is 62 records of 72
bytes keyed by exactly this id namespace (see :mod:`sifac.facial`). All 34
idents used across the dump exist in that table, and two songs quoting the same
ident therefore play the same face. A cue reads "hold facial preset N from tick
t for len ticks". It still cannot select a dance take: each live character pack
holds exactly one 6.5 MB motion clip.

Placement is only verified for song 0010, where one axis stays near 1.4 m for
all nine performers (chest/head height, or a depth offset) while the other two
spread over 7.0 m and 1.8 m; taking the two spread axes as the floor plan keeps
every pair at least 1.32 m apart, whereas either alternative puts two performers
0.14 m or 0.05 m apart.

The record signature is `u32 0, u32 end tick, u32 1` plus a 1..64 float at +0x2c
and a non-zero placement, which finds nine records in songs 0010, 0020 and 0050
and none or one in others (0030/0040: one, 0060: none, 0070: five) even though
every song ships nine character packs. Those timelines use a slightly different
record — 0030's only match carries extra floats at +0x34 and a four-byte gap
before its cue list — so the layout is not fully general yet and consumers must
fall back to a placeholder layout whenever fewer records than performers appear.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .binary import u32

MAGIC = b"BTL\0"
HEADER_SIZE = 0x80
CHANNEL_TABLE = 0xA0
CHANNEL_ENTRY_SIZE = 16
RECORD_SIZE = 0x40
CUE_START = 0x60
CUE_SIZE = 0x1C
CUE_CHANNEL = 0x000C000A
CUE_LIMIT = 4096


class BtlbError(ValueError):
    """Raised when a .btlb payload is malformed."""


@dataclass(frozen=True)
class BtlbChannel:
    size: int
    offset: int
    flags: int
    index: int


@dataclass(frozen=True)
class Cue:
    """One channel cue of a performer's timeline segment."""

    channel: int
    ident: int
    tag: int
    weight: float
    start_tick: int
    length: int


@dataclass(frozen=True)
class CharaRecord:
    """One performer record of a character timeline."""

    offset: int
    end_tick: int
    ids: tuple[int, int]
    blend: tuple[float, float, float, float]
    position: tuple[float, float, float]
    marker: float
    cues: tuple[Cue, ...] = field(default_factory=tuple)


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
        """Return every performer record found in the payload."""
        records: list[CharaRecord] = []
        offsets: list[int] = []
        position = CHANNEL_TABLE
        limit = len(self.data) - RECORD_SIZE
        while position <= limit:
            if self._is_chara_record(position):
                head = struct.unpack_from("<4I", self.data, position)
                ids = struct.unpack_from("<2H", self.data, position + 0x0C)
                blend = struct.unpack_from("<4f", self.data, position + 0x10)
                place = struct.unpack_from("<3f", self.data, position + 0x20)
                marker = struct.unpack_from("<f", self.data, position + 0x2C)[0]
                offsets.append(position)
                records.append(
                    CharaRecord(position, head[1], ids, blend, place, marker)
                )
                position += RECORD_SIZE
                continue
            position += 4
        if not records:
            return records
        bounds = offsets[1:] + [len(self.data)]
        return [
            CharaRecord(
                record.offset,
                record.end_tick,
                record.ids,
                record.blend,
                record.position,
                record.marker,
                tuple(self._read_cues(record.offset + CUE_START, end)),
            )
            for record, end in zip(records, bounds)
        ]

    def _read_cues(self, start: int, end: int) -> list[Cue]:
        cues: list[Cue] = []
        position = start
        while position + CUE_SIZE <= end and len(cues) < CUE_LIMIT:
            channel, zero, ident, tag = struct.unpack_from("<4I", self.data, position)
            if channel != CUE_CHANNEL or zero != 0:
                break
            weight = struct.unpack_from("<f", self.data, position + 0x10)[0]
            start_tick, length = struct.unpack_from(
                "<2I", self.data, position + 0x14
            )
            cues.append(Cue(channel, ident, tag, weight, start_tick, length))
            position += CUE_SIZE
        return cues

    def _is_chara_record(self, offset: int) -> bool:
        if u32(self.data, offset) != 0 or u32(self.data, offset + 8) != 1:
            return False
        tick = u32(self.data, offset + 4)
        if not 0 < tick <= 0x10000:
            return False
        marker = struct.unpack_from("<f", self.data, offset + 0x2C)[0]
        if not (1.0 <= marker <= 64.0):
            return False
        place = struct.unpack_from("<3f", self.data, offset + 0x20)
        return max(abs(value) for value in place) >= 0.3


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
