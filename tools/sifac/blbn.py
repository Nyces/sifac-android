"""Reader for the BLBN symbol container used by SIFAC data tables.

``live_datas.bin`` and ``stage_datas.bin`` are not magic-tagged binary blobs;
they start with a 0x80-byte section header followed by a ``BLBN`` container.
The container is a chain of 12-byte blocks:

* ``BLBN``   root block, its payload is the block chain itself;
* ``SYM\\0``  one symbol per block, ``size`` is the 0x18-byte entry length;
* anything else (zero magic) is a raw data block, skipped by the walker.

Every symbol entry is 24 bytes (``<QQII``): the data offset relative to the
``BLBN`` header, the name offset relative to the same base, a salted name hash
and the data size. ``eboot``'s ``sub_4BB4C0`` walks exactly this layout and
returns ``base + offset`` after matching the hash.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .binary import align_up, cstring

MAGIC = b"BLBN"
SYMBOL_MAGIC = b"SYM\0"
BLOCK_HEADER = 12
SYMBOL_STRUCT = struct.Struct("<QQII")


class BlbnError(ValueError):
    """Raised when a BLBN container cannot be walked."""


@dataclass(frozen=True)
class Symbol:
    name: str
    offset: int  # relative to the BLBN header
    size: int
    hash: int


def salted_hash(name: str) -> int:
    """Name hash used by the game's symbol lookup (``sub_4BB4C0``)."""
    value = 0
    for byte in name.encode("ascii"):
        value = (byte + 33 * value) & 0xFFFFFFFF
    return (value + (value >> 5)) & 0xFFFFFFFF


class BlbnFile:
    """Read-only view of a BLBN container.

    ``base`` default to the offset of the ``BLBN`` magic, which strips the
    section header in front of the container. Symbol offsets and name offsets
    are relative to that base.
    """

    def __init__(self, data: bytes, base: int | None = None) -> None:
        self.data = data
        self.base = data.find(MAGIC) if base is None else base
        if self.base < 0:
            raise BlbnError("not a BLBN container")
        self.symbols: dict[str, Symbol] = {}
        self._walk()

    def _walk(self) -> None:
        position = self.base
        while position + BLOCK_HEADER <= len(self.data):
            magic = self.data[position : position + 4]
            size = struct.unpack_from("<I", self.data, position + 4)[0]
            flags = self.data[position + 9]
            alignment = struct.unpack_from("<H", self.data, position + 10)[0]
            if flags & 1:
                size = int.from_bytes(self.data[position + 4 : position + 8], "big")
                alignment = int.from_bytes(
                    self.data[position + 10 : position + 12], "big"
                )
            if magic == b"\0\0\0\0" and size == 0:
                return
            payload = align_up(position + BLOCK_HEADER, alignment or 4)
            if payload + size > len(self.data):
                raise BlbnError(f"block at {position:#x} runs past the container")
            if magic == SYMBOL_MAGIC:
                if size < SYMBOL_STRUCT.size:
                    raise BlbnError(f"symbol at {position:#x} is truncated")
                offset, name_offset, name_hash, length = SYMBOL_STRUCT.unpack_from(
                    self.data, payload
                )
                symbol = Symbol(self.text(name_offset), offset, length, name_hash)
                self.symbols[symbol.name] = symbol
            position = align_up(payload + size, 4)

    def text(self, offset: int) -> str:
        """Read a NUL-terminated string at a base-relative offset."""
        start = self.base + offset
        if not 0 <= start < len(self.data):
            raise BlbnError(f"string offset {offset:#x} is outside the container")
        return cstring(self.data, start)

    def read(self, name: str) -> bytes:
        symbol = self.symbols.get(name)
        if symbol is None:
            raise KeyError(f"{name!r} is not a symbol of this container")
        start = self.base + symbol.offset
        end = start + symbol.size
        if end > len(self.data):
            raise BlbnError(f"{name!r} runs past the container")
        return self.data[start:end]
