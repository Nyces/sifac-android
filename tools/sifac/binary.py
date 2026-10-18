"""Small binary helpers shared by the SIFAC format readers."""

from __future__ import annotations

import struct
from typing import Iterable, Sequence

# Magics that appear at the start of a useful payload. Section wrappers in the
# game insert 0x80/0x200 bytes of header before one of these.
KNOWN_MAGICS: tuple[bytes, ...] = (
    b"ARC\0",
    b"BMAR104\0",
    b"BML\0",
    b"BMD\0",
    b"BMA\0",
    b"BMT\0",
    b"MTXP",
    b"MDL\0",
    b"ANHD",
    b"ANSK",
    b"POF0",
    b"AFS2",
    b"HCA\0",
    b"ADX\0",
    b"USM\0",
    b"\x89PNG",
    b"DDS ",
)


def align_up(value: int, alignment: int = 0x20) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (value + alignment - 1) & ~(alignment - 1)


def u16(data: bytes | bytearray | memoryview, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def u32(data: bytes | bytearray | memoryview, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def u64(data: bytes | bytearray | memoryview, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def cstring(data: bytes, offset: int, limit: int | None = None) -> str:
    """Read a NUL-terminated string, bounded by *limit* when given."""
    end = data.find(b"\0", offset)
    if end < 0:
        end = len(data)
    if limit is not None:
        end = min(end, offset + limit)
    return data[offset:end].decode("utf-8", "replace")


def starts_with_any(data: bytes, magics: Sequence[bytes]) -> bool:
    return any(data.startswith(magic) for magic in magics)


def find_magic(
    data: bytes,
    magics: Sequence[bytes] = KNOWN_MAGICS,
    limit: int = 0x400,
    alignment: int = 4,
) -> tuple[int, bytes] | tuple[None, None]:
    """Find the first known payload magic at a 4-byte aligned offset."""
    stop = min(len(data), limit)
    for offset in range(0, stop + 1, alignment):
        for magic in magics:
            if data.startswith(magic, offset):
                return offset, magic
    return None, None


def unwrap_section(
    data: bytes,
    magics: Sequence[bytes] = KNOWN_MAGICS,
    limit: int = 0x400,
) -> tuple[bytes, int]:
    """Strip an optional section header before a known payload magic.

    Several SIFAC archives wrap each payload in a 0x80-byte section record whose
    first u32 is a size. Some records set bit 31, others do not, so the reliable
    check is "does a known magic begin at an aligned offset near the start?".
    Returns (payload, skipped_bytes).
    """
    if starts_with_any(data, magics):
        return data, 0
    offset, _ = find_magic(data, magics=magics, limit=limit)
    if offset is None:
        return data, 0
    return data[offset:], offset


def hexdump(data: bytes, offset: int = 0, length: int = 0x100) -> Iterable[str]:
    """Yield 16-byte hexdump lines for CLI debugging."""
    end = min(len(data), offset + length)
    for pos in range(offset, end, 16):
        chunk = data[pos : pos + 16]
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        yield f"{pos:08x}: {chunk.hex(' '):<47}  {text}"
