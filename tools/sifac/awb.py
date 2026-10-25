"""CRI AFS2/AWB container reader (audio payloads stay encoded)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .binary import align_up, u16, u32

# Retail SIFAC AWBs store the raw stream offset, but every HCA payload starts at
# the next 0x20-byte boundary after it (verified on all 10,839 entries).
DATA_ALIGN = 0x20


class AwbError(ValueError):
    """Raised for a malformed AFS2/AWB file."""


@dataclass(frozen=True)
class Afs2Entry:
    index: int
    id: int
    offset: int
    size: int
    magic: str


class Afs2Archive:
    def __init__(self, data: bytes) -> None:
        self.data = data
        if data[:4] != b"AFS2":
            raise AwbError("not an AFS2 container")
        self.version = data[4]
        self.count = u32(data, 8)
        self.header_size = u32(data, 0x0C)
        if self.count > 0x100000:
            raise AwbError(f"unreasonable AWB entry count: {self.count}")
        id_start = 0x10
        id_end = id_start + self.count * 2
        if id_end > len(data):
            raise AwbError("AWB id table is truncated")
        self.ids = tuple(u16(data, id_start + i * 2) for i in range(self.count))
        offset_start = id_end
        offset_end = offset_start + self.count * 4
        if offset_end > len(data):
            raise AwbError("AWB offset table is truncated")
        self.offsets = tuple(
            u32(data, offset_start + i * 4) for i in range(self.count)
        )
        previous = 0
        for offset in self.offsets:
            if offset < previous or offset >= len(data):
                raise AwbError(f"invalid AWB entry offset: {offset:#x}")
            previous = offset
        self.starts = tuple(align_up(offset, DATA_ALIGN) for offset in self.offsets)
        for start in self.starts:
            if start >= len(data):
                raise AwbError(f"AWB entry start is past the end: {start:#x}")
        self.entries = tuple(self._read_entries())

    def _read_entries(self) -> list[Afs2Entry]:
        entries: list[Afs2Entry] = []
        for index, (entry_id, start) in enumerate(zip(self.ids, self.starts)):
            end = (
                self.starts[index + 1]
                if index + 1 < self.count
                else len(self.data)
            )
            head = self.data[start : start + 0x40]
            magic = "?"
            for candidate in (b"HCA\0", b"ADX\0", b"AFS2", b"RIFF"):
                if candidate in head:
                    magic = candidate.decode("ascii", "replace").rstrip("\0")
                    break
            entries.append(
                Afs2Entry(
                    index=index,
                    id=entry_id,
                    offset=start,
                    size=end - start,
                    magic=magic,
                )
            )
        return entries

    def entry_data(self, index: int) -> bytes:
        entry = self.entries[index]
        return self.data[entry.offset : entry.offset + entry.size]


def read_awb(path: str | Path) -> Afs2Archive:
    return Afs2Archive(Path(path).read_bytes())
