"""Synthetic builders used by the unit tests (no game assets involved)."""

from __future__ import annotations

import struct

from sifac.binary import align_up


def build_arc(entries: list[tuple[str, str, bytes]], name: str = "test.pac") -> bytes:
    """Build a minimal ARC archive.

    entries: [(type, name, payload)]
    Offsets are relative to the ARC header, matching the game's format.
    """
    count = len(entries)
    header = bytearray(0x30)
    header[0:4] = b"ARC\0"
    struct.pack_into("<HHH", header, 4, 1, count, count)
    header[0x10 : 0x10 + len(name)] = name.encode("ascii")

    offset_table = bytearray(count * 4)
    entry_start = align_up(0x30 + count * 4, 0x20)

    entry_table = bytearray()
    for index, (entry_type, entry_name, _payload) in enumerate(entries):
        record = bytearray(20)
        record[0:4] = entry_type.encode("ascii").ljust(4, b"\0")
        struct.pack_into("<HH", record, 4, 0, index)
        struct.pack_into("<I", record, 8, 0x12345678 + index)
        struct.pack_into("<HH", record, 12, 0, len(entry_name) + 1)
        record += entry_name.encode("ascii") + b"\0"
        record += b"\0" * (align_up(len(record), 0x20) - len(record))
        entry_table += record

    data_start = align_up(entry_start + len(entry_table), 0x20)
    payloads = bytearray()
    offsets: list[int] = []
    for _entry_type, _entry_name, payload in entries:
        offsets.append(data_start + len(payloads))
        payloads += payload
        payloads += b"\0" * (align_up(len(payloads), 0x10) - len(payloads))

    for index, offset in enumerate(offsets):
        struct.pack_into("<I", offset_table, index * 4, offset)

    result = bytearray(data_start + len(payloads))
    result[0:0x30] = header
    result[0x30 : 0x30 + len(offset_table)] = offset_table
    result[entry_start : entry_start + len(entry_table)] = entry_table
    result[data_start:] = payloads
    return bytes(result)


def build_sfo(values: dict[str, object]) -> bytes:
    count = len(values)
    keys = bytearray()
    data = bytearray()
    index = bytearray()
    key_offsets: dict[str, int] = {}
    value_offsets: dict[str, int] = {}
    for key in values:
        key_offsets[key] = len(keys)
        keys += key.encode("utf-8") + b"\0"
    for key, value in values.items():
        value_offsets[key] = len(data)
        if isinstance(value, str):
            raw = value.encode("utf-8") + b"\0"
            fmt = 0x0204
        else:
            raw = struct.pack("<I", int(value))
            fmt = 0x0404
        data += raw
        index += struct.pack(
            "<HHIII", key_offsets[key], fmt, len(raw), len(raw), value_offsets[key]
        )
    keys_offset = 20 + len(index)
    data_offset = align_up(keys_offset + len(keys), 4)
    result = bytearray(data_offset + len(data))
    struct.pack_into("<4sIIII", result, 0, b"\0PSF", 0x0101, keys_offset, data_offset, count)
    result[20 : 20 + len(index)] = index
    result[keys_offset : keys_offset + len(keys)] = keys
    result[data_offset:] = data
    return bytes(result)


def build_awb(entries: list[tuple[int, bytes]], magic: bytes = b"HCA\0") -> bytes:
    count = len(entries)
    header_size = 0x20
    data_start = align_up(0x10 + count * 2 + count * 4, 0x20)
    offsets: list[int] = []
    payloads = bytearray()
    for entry_id, payload in entries:
        offsets.append(data_start + len(payloads))
        payloads += magic + payload
    result = bytearray(data_start + len(payloads))
    result[0:4] = b"AFS2"
    struct.pack_into("<BBH", result, 4, 2, 4, 2)
    struct.pack_into("<II", result, 8, count, header_size)
    for index, (entry_id, _payload) in enumerate(entries):
        struct.pack_into("<H", result, 0x10 + index * 2, entry_id)
        struct.pack_into("<I", result, 0x10 + count * 2 + index * 4, offsets[index])
    result[data_start:] = payloads
    return bytes(result)


def build_btx(name: str, width: int, height: int, fmt: int, payload: bytes) -> bytes:
    """Build a synthetic btx texture with a 0x80-byte header."""
    header = bytearray(0x80)
    header[0:4] = b"btx\0"
    struct.pack_into("<H", header, 4, 0x0102)
    struct.pack_into("<HH", header, 8, width, height)
    struct.pack_into("<I", header, 0x0C, width * 4)
    struct.pack_into("<I", header, 0x10, 0x01000000 | fmt)
    struct.pack_into("<I", header, 0x24, 0x30)
    encoded = name.encode("ascii") + b"\0"
    header[0x30 : 0x30 + len(encoded)] = encoded
    return bytes(header) + payload
