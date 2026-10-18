"""PS4 param.sfo reader (only the fields needed to verify a dump)."""

from __future__ import annotations

import struct
from pathlib import Path


class SfoError(ValueError):
    """Raised for a malformed param.sfo."""


def parse_sfo(data: bytes) -> dict[str, object]:
    if len(data) < 20:
        raise SfoError("truncated SFO header")
    magic, version, key_table, data_table, count = struct.unpack_from(
        "<4sIIII", data, 0
    )
    if magic != b"\0PSF":
        raise SfoError("bad SFO signature")
    result: dict[str, object] = {}
    for index in range(count):
        offset = 20 + index * 16
        if offset + 16 > len(data):
            raise SfoError("truncated SFO index table")
        key_offset, fmt, length, _capacity, value_offset = struct.unpack_from(
            "<HHIII", data, offset
        )
        key_start = key_table + key_offset
        key_end = data.find(b"\0", key_start)
        if key_end < 0:
            raise SfoError("unterminated SFO key")
        key = data[key_start:key_end].decode("utf-8", "replace")
        value_start = data_table + value_offset
        value = data[value_start : value_start + length]
        if fmt == 0x0404:
            if len(value) != 4:
                raise SfoError(f"bad integer value for {key}")
            result[key] = struct.unpack("<I", value)[0]
        elif fmt == 0x0204:
            result[key] = value.rstrip(b"\0").decode("utf-8", "replace")
        else:
            result[key] = value
    return result


def read_sfo(path: str | Path) -> dict[str, object]:
    return parse_sfo(Path(path).read_bytes())
