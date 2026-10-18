"""ARC archive reader used by nearly every SIFAC .pac file."""

from __future__ import annotations

from dataclasses import dataclass

from .binary import align_up, cstring, u16, u32, unwrap_section


class ArcError(ValueError):
    """Raised when an ARC header or entry table is malformed."""


@dataclass(frozen=True)
class ArcEntry:
    type: str
    index: int
    hash: int
    name: str
    offset: int  # absolute offset in the buffer handed to ArcArchive
    size: int

    @property
    def is_empty(self) -> bool:
        return self.size == 0


class ArcArchive:
    """A read-only view of an ARC archive.

    Offsets in the file are relative to the ARC header, not to the start of the
    Python buffer. ``base`` is the header offset inside the buffer.
    """

    HEADER_SIZE = 0x30
    NAME_SIZE = 0x20

    def __init__(self, data: bytes, base: int = 0) -> None:
        self.data = data
        self.base = base
        if len(data) < base + self.HEADER_SIZE:
            raise ArcError("buffer is smaller than an ARC header")
        if data[base : base + 4] != b"ARC\0":
            raise ArcError("not an ARC archive")
        self.version = u16(data, base + 4)
        self.count = u16(data, base + 6)
        self.count2 = u16(data, base + 8)
        self.name = cstring(data, base + 0x10, self.NAME_SIZE)
        self.offsets = self._read_offsets()
        self.entries = tuple(self._read_entries())

    def _read_offsets(self) -> tuple[int, ...]:
        if self.count == 0:
            return ()
        if self.count > 0x10000:
            raise ArcError(f"unreasonable entry count: {self.count}")
        table = self.base + self.HEADER_SIZE
        if table + self.count * 4 > len(self.data):
            raise ArcError("offset table is truncated")
        offsets = tuple(u32(self.data, table + i * 4) for i in range(self.count))
        payload_size = len(self.data) - self.base
        previous = 0
        for offset in offsets:
            if offset < 0 or offset >= payload_size:
                raise ArcError(f"entry offset outside archive: {offset:#x}")
            if offset < previous:
                raise ArcError("entry offsets are not sorted")
            previous = offset
        return offsets

    def _read_entries(self) -> list[ArcEntry]:
        if self.count == 0:
            return []
        position = self.base + align_up(self.HEADER_SIZE + self.count * 4, 0x20)
        entries: list[ArcEntry] = []
        for index in range(self.count):
            if position + 20 > len(self.data):
                raise ArcError(f"entry table truncated at entry {index}")
            raw_type = self.data[position : position + 4]
            entry_type = raw_type.split(b"\0", 1)[0].decode("ascii", "replace")
            entry_index = u16(self.data, position + 6)
            entry_hash = u32(self.data, position + 8)
            name_length = u16(self.data, position + 14)
            if name_length == 0 or name_length > 0x200:
                raise ArcError(
                    f"invalid name length {name_length} for entry {index} at {position:#x}"
                )
            name_start = position + 20
            name_end = name_start + name_length - 1
            if name_end > len(self.data):
                raise ArcError(f"entry name truncated at entry {index}")
            name = self.data[name_start:name_end].decode("utf-8", "replace")
            entry_offset = self.base + self.offsets[index]
            if index + 1 < self.count:
                entry_size = self.offsets[index + 1] - self.offsets[index]
            else:
                entry_size = len(self.data) - entry_offset
            if entry_size < 0:
                raise ArcError(f"negative entry size for {name!r}")
            entries.append(
                ArcEntry(
                    type=entry_type,
                    index=entry_index,
                    hash=entry_hash,
                    name=name,
                    offset=entry_offset,
                    size=entry_size,
                )
            )
            position += align_up(20 + name_length, 0x20)
        return entries

    def entry_data(self, entry: ArcEntry | int, unwrap: bool = False) -> bytes:
        if isinstance(entry, int):
            entry = self.entries[entry]
        data = self.data[entry.offset : entry.offset + entry.size]
        if unwrap:
            data, _ = unwrap_section(data)
        return data

    def find(self, name: str) -> ArcEntry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


def open_arc(data: bytes, base: int = 0, unwrap: bool = False) -> ArcArchive:
    """Open an ARC, optionally stripping a section header first."""
    if unwrap:
        data, _ = unwrap_section(data)
        base = 0
    return ArcArchive(data, base=base)
