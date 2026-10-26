"""Blade animation (BMA) reader.

A BMA payload is a chunk chain like BML::

    ANHD  version string ("103"), constant (f32, 60.0 in every retail clip and
          probably the frame rate rather than a duration)
    ANSK  node table size and key counts
    DATA  ... one block per animated node

The first DATA block (right after ANSK) is the track table. Each entry is
24 bytes::

    u32 name pointer
    u32 reserved
    u32 data pointer
    u32 reserved
    u32 name hash
    u32 key count

The data pointer of a track points at another small DATA block holding the
per-key blocks (typically 12-byte float triples for translation and rotation).
"""

from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass
from pathlib import Path

from .binary import cstring, u32

CHUNK_HEADER = 12
TRACK_ENTRY_SIZE = 24


class BmaError(ValueError):
    """Raised when a BMA buffer is malformed."""


@dataclass(frozen=True)
class AnimChunk:
    tag: str
    offset: int
    size: int

    @property
    def payload_offset(self) -> int:
        return self.offset + CHUNK_HEADER


@dataclass(frozen=True)
class Track:
    name: str
    hash: int
    key_count: int
    data_offset: int


@dataclass(frozen=True)
class RotationKeys:
    """Key times (frames) and matching quaternions (x, y, z, w)."""

    times: list[float]
    quaternions: list[tuple[float, float, float, float]]

    def sample(self, frame: float) -> tuple[float, float, float, float]:
        if not self.times:
            return (0.0, 0.0, 0.0, 1.0)
        if frame <= self.times[0]:
            return self.quaternions[0]
        if frame >= self.times[-1]:
            return self.quaternions[-1]
        index = bisect.bisect_left(self.times, frame)
        if index <= 0:
            return self.quaternions[0]
        t0, t1 = self.times[index - 1], self.times[index]
        q0, q1 = self.quaternions[index - 1], self.quaternions[index]
        if abs(t1 - t0) < 1e-6:
            return q1
        alpha = (frame - t0) / (t1 - t0)
        return tuple(a + (b - a) * alpha for a, b in zip(q0, q1))  # type: ignore[return-value]


class BmaAnimation:
    def __init__(self, data: bytes) -> None:
        offset = data.find(b"ANHD")
        if offset < 0:
            raise BmaError("not a BMA animation")
        if offset:
            data = data[offset:]
        self.data = data
        self.chunks = tuple(self._walk())
        self.version = ""
        self.duration = 0.0
        self.tracks: list[Track] = []
        self._rotations: dict[int, RotationKeys | None] = {}
        self._read_header()
        self._read_tracks()

    def _walk(self) -> list[AnimChunk]:
        chunks: list[AnimChunk] = []
        position = 0
        while position + CHUNK_HEADER <= len(self.data):
            raw = self.data[position : position + 4]
            ok = all(65 <= b <= 90 for b in raw[:3]) and (65 <= raw[3] <= 90 or raw[3] == 0)
            if not ok:
                break
            size = u32(self.data, position + 4)
            if size > 0x2000000 or position + CHUNK_HEADER + size > len(self.data):
                break
            chunks.append(
                AnimChunk(raw.rstrip(b"\0").decode("ascii", "replace"), position, size)
            )
            position += CHUNK_HEADER + ((size + 3) & ~3)
        return chunks

    def _read_header(self) -> None:
        for chunk in self.chunks:
            if chunk.tag == "ANHD":
                self.version = cstring(self.data, chunk.payload_offset, 8)
                if chunk.size >= 8:
                    self.duration = struct.unpack_from(
                        "<f", self.data, chunk.payload_offset + 4
                    )[0]
                break

    def _read_tracks(self) -> None:
        data_chunks = [c for c in self.chunks if c.tag == "DATA"]
        if not data_chunks:
            return
        table = data_chunks[0]
        count = table.size // TRACK_ENTRY_SIZE
        for index in range(count):
            base = table.payload_offset + index * TRACK_ENTRY_SIZE
            name_ptr = u32(self.data, base)
            data_ptr = u32(self.data, base + 8)
            name_hash = u32(self.data, base + 16)
            key_count = u32(self.data, base + 20)
            if not 0 < name_ptr < len(self.data):
                continue
            name = cstring(self.data, name_ptr, 64)
            if not name:
                continue
            self.tracks.append(Track(name, name_hash, key_count, data_ptr))

    def track(self, name: str) -> Track | None:
        for track in self.tracks:
            if track.name == name:
                return track
        return None

    def payload(self, pointer: int) -> tuple[int, bytes] | None:
        """Return (size, bytes) when *pointer* targets a DATA payload."""
        if pointer < 16 or pointer + 12 > len(self.data):
            return None
        if self.data[pointer - 12 : pointer - 8] != b"DATA":
            return None
        size = u32(self.data, pointer - 8)
        if size <= 0 or pointer + size > len(self.data):
            return None
        return size, self.data[pointer : pointer + size]

    def rotation_keys(self, track: Track) -> RotationKeys | None:
        """Decode a track's rotation channel (time table + quaternions).

        Decoding walks every key of the track, so the result is memoised: a
        nine-performer frame used to re-decode the same 9000-key clip nine
        times per frame.
        """
        cached = self._rotations.get(track.data_offset, ...)
        if cached is not ...:
            return cached  # type: ignore[return-value]
        keys = self._decode_rotation(track)
        self._rotations[track.data_offset] = keys
        return keys

    def _decode_rotation(self, track: Track) -> RotationKeys | None:
        times: list[float] | None = None
        quaternions: list[tuple[float, float, float, float]] | None = None
        for index in range(track.key_count):
            base = track.data_offset + index * 16
            if base + 16 > len(self.data):
                break
            first, _reserved, second, _kind = struct.unpack_from("<4I", self.data, base)
            for pointer in (first, second):
                got = self.payload(pointer)
                if got is None:
                    continue
                size, body = got
                if size % 16 == 0 and size >= 32:
                    count = size // 16
                    values = [struct.unpack_from("<4f", body, i * 16) for i in range(count)]
                    if all(abs(sum(v * v for v in q) - 1.0) < 0.02 for q in values[:8]):
                        quaternions = values
                elif size % 12 == 0 and size >= 24:
                    count = size // 12
                    entries = [struct.unpack_from("<fII", body, i * 12) for i in range(count)]
                    candidate = [entry[0] for entry in entries]
                    ordered = all(
                        candidate[i] <= candidate[i + 1] for i in range(len(candidate) - 1)
                    )
                    if ordered and candidate[-1] > candidate[0]:
                        times = candidate
        if quaternions is None:
            return None
        if times is None or len(times) != len(quaternions):
            times = [float(i) for i in range(len(quaternions))]
        return RotationKeys(times, quaternions)

    def key_blocks(self, track: Track) -> list[bytes]:
        """Split a track's data block into its per-key payloads.

        Each key record is four u32: two payload pointers and two counts.
        """
        blocks: list[bytes] = []
        position = track.data_offset
        for _ in range(track.key_count):
            if position + 16 > len(self.data):
                break
            first, _reserved, second, _kind = struct.unpack_from(
                "<4I", self.data, position
            )
            for pointer in (first, second):
                if 8 <= pointer <= len(self.data) - 12:
                    blocks.append(self.data[pointer : pointer + 12])
            position += 16
        return blocks

    def channels(self, track: Track) -> list[tuple[int, int, int, list[float]]]:
        """Return (first_ptr, second_ptr, kind, first 3 floats of each payload)."""
        result = []
        position = track.data_offset
        for _ in range(track.key_count):
            if position + 16 > len(self.data):
                break
            first, _reserved, second, kind = struct.unpack_from(
                "<4I", self.data, position
            )
            values: list[float] = []
            for pointer in (first, second):
                if 8 <= pointer <= len(self.data) - 12:
                    values.extend(struct.unpack_from("<3f", self.data, pointer))
            result.append((first, second, kind, values))
            position += 16
        return result


def find_bma_in_pac(path: str | Path, character: str | None = None) -> BmaAnimation:
    from .bmar import find_bmar_in_pac

    bmar = find_bmar_in_pac(path, character)
    for index, entry in enumerate(bmar.entries):
        if entry.type == "BMA":
            return BmaAnimation(bmar.entry_data(index))
    raise BmaError(f"no BMA animation in {path}")


def sample_pose(
    animation: "BmaAnimation",
    node_names: set[str],
    frame: float,
) -> dict[str, tuple[float, float, float, float]]:
    """Sample every rotation track that names a node in the model."""
    pose: dict[str, tuple[float, float, float, float]] = {}
    for track in animation.tracks:
        if track.name not in node_names:
            continue
        keys = animation.rotation_keys(track)
        if keys is None:
            continue
        pose[track.name] = keys.sample(frame)
    return pose


def load_live_motion(path: str | Path) -> BmaAnimation:
    """Load the motion stored in a ``live/<song>/cXX_<song>.pac`` pack.

    Those packs hold a single ``mot`` entry whose payload is a BMAR wrapping one
    BMA (unlike the character packs, which nest a ``motion.pac`` ARC).
    """
    from .arc import ArcArchive
    from .bmar import BmarArchive
    from .binary import unwrap_section
    from .pac import read_pac

    image = read_pac(path)
    top = ArcArchive(image.data)
    if not top.entries:
        raise BmaError(f"no entries in {path}")
    payload, _ = unwrap_section(top.entry_data(top.entries[0]))
    if payload.startswith(b"BMAR104\0"):
        bmar = BmarArchive(payload)
        for index, entry in enumerate(bmar.entries):
            if entry.type == "BMA":
                return BmaAnimation(bmar.entry_data(index))
        raise BmaError(f"no BMA inside the BMAR of {path}")
    return BmaAnimation(payload)


def load_motion(path: str | Path, name: str) -> BmaAnimation:
    """Load one animation from the character's ``motion.pac`` by entry name."""
    from .arc import ArcArchive
    from .binary import unwrap_section
    from .pac import read_pac

    image = read_pac(path)
    top = ArcArchive(image.data)
    motion_entry = top.find("motion.pac")
    if motion_entry is None:
        raise BmaError(f"motion.pac not found in {path}")
    motion_bytes, _ = unwrap_section(top.entry_data(motion_entry))
    motion_arc = ArcArchive(motion_bytes)
    entry = motion_arc.find(name)
    if entry is None:
        raise BmaError(f"motion not found: {name}")
    return BmaAnimation(motion_arc.entry_data(entry))


def list_motions(path: str | Path, character: str | None = None) -> list[dict]:
    """List every animation in the character's ``motion.pac``."""
    from .arc import ArcArchive
    from .binary import unwrap_section
    from .pac import read_pac

    image = read_pac(path)
    top = ArcArchive(image.data)
    motion_entry = top.find("motion.pac")
    if motion_entry is None:
        return []
    motion_bytes, _ = unwrap_section(top.entry_data(motion_entry))
    motion_arc = ArcArchive(motion_bytes)
    result = []
    for entry in motion_arc.entries:
        raw = motion_arc.entry_data(entry)
        if b"ANHD" not in raw[:0x400]:
            continue
        try:
            animation = BmaAnimation(raw)
        except BmaError:
            continue
        result.append(
            {
                "name": entry.name,
                "size": entry.size,
                "duration": animation.duration,
                "tracks": len(animation.tracks),
            }
        )
    return result
