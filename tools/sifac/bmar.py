"""BMAR/Blade model archive reader.

A BMAR file is a short ``BMAR104`` signature followed by a normal ARC archive
whose entries are BML/BMD/BMA/BMT chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .arc import ArcArchive, ArcEntry
from .binary import find_magic, unwrap_section
from .pac import read_pac

BMAR_MAGIC = b"BMAR104\0"


class BmarError(ValueError):
    """Raised for a malformed BMAR file."""


@dataclass(frozen=True)
class BmarEntry:
    type: str
    name: str
    offset: int
    size: int
    payload_magic: str
    payload_skipped: int


class BmarArchive:
    def __init__(self, data: bytes) -> None:
        if not data.startswith(BMAR_MAGIC):
            raise BmarError("not a BMAR104 archive")
        self.data = data
        self.arc = ArcArchive(data, base=len(BMAR_MAGIC))
        self.entries = tuple(self._read_entries())

    def _read_entries(self) -> list[BmarEntry]:
        result: list[BmarEntry] = []
        for entry in self.arc.entries:
            payload, skipped = unwrap_section(self.arc.entry_data(entry))
            magic = "?"
            for candidate in (b"BML\0", b"BMD\0", b"BMA\0", b"BMT\0", b"MTXP", b"MDL\0"):
                if payload.startswith(candidate):
                    magic = candidate.decode("ascii", "replace").rstrip("\0")
                    break
            result.append(
                BmarEntry(
                    type=entry.type,
                    name=entry.name,
                    offset=entry.offset,
                    size=entry.size,
                    payload_magic=magic,
                    payload_skipped=skipped,
                )
            )
        return result

    def entry_data(self, index: int, unwrap: bool = True) -> bytes:
        entry = self.arc.entries[index]
        data = self.arc.entry_data(entry)
        if unwrap:
            data, _ = unwrap_section(data)
        return data

    def find(self, name: str) -> BmarEntry | None:
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None


def find_bmar_in_pac(path: str | Path) -> BmarArchive:
    """Find the first BMAR model inside a .pac that contains ``model.pac``."""
    image = read_pac(path)
    top = ArcArchive(image.data)
    model_entry = top.find("model.pac")
    if model_entry is None:
        raise BmarError(f"model.pac not found in {path}")
    model_arc_bytes, _ = unwrap_section(top.entry_data(model_entry))
    model_arc = ArcArchive(model_arc_bytes)
    for entry in model_arc.entries:
        if entry.type in {"mdl", "mdls"} or entry.name.startswith("mod_"):
            bmar_bytes, _ = unwrap_section(model_arc.entry_data(entry))
            if bmar_bytes.startswith(BMAR_MAGIC):
                return BmarArchive(bmar_bytes)
    raise BmarError(f"no BMAR model entry found in {path}")


def scan_magics(data: bytes, limit: int | None = None) -> list[tuple[int, str]]:
    """List known payload magics; useful when reverse engineering a new chunk."""
    stop = len(data) if limit is None else min(len(data), limit)
    hits: list[tuple[int, str]] = []
    for magic in (b"BMAR104\0", b"ARC\0", b"BML\0", b"BMD\0", b"BMA\0", b"BMT\0",
                  b"MTXP", b"MDL\0", b"ANHD", b"ANSK", b"POF0"):
        position = 0
        while True:
            position = data.find(magic, position, stop)
            if position < 0:
                break
            hits.append((position, magic.decode("ascii", "replace").rstrip("\0")))
            position += 1
    return sorted(hits)
