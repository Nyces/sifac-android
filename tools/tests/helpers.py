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


def build_scd(records: list[tuple[int, int, int, int, int, int]], channel: int = 1) -> bytes:
    """Build a synthetic score file.

    records: [(id, tick, lane_raw, constant, value, channel)]
    """
    size = 0xE0 + len(records) * 0x20 + 0x20
    data = bytearray(size)
    struct.pack_into("<I", data, 0x00, size)
    struct.pack_into("<I", data, 0x04, 0x80)
    data[0x80:0x84] = b"Scor"
    struct.pack_into("<I", data, 0x84, size)
    struct.pack_into("<I", data, 0x88, len(records) + 1)
    struct.pack_into("<I", data, 0x8C, 0x45C12)
    struct.pack_into("<I", data, 0x90, 480)
    struct.pack_into("<I", data, 0x9C, channel)
    for index, record in enumerate(records):
        struct.pack_into("<6I", data, 0xE0 + index * 0x20, *record)
    return bytes(data)


def _chunk(tag: str, payload: bytes) -> bytes:
    body = bytearray(payload)
    body += b"\0" * ((4 - len(body) % 4) % 4)
    return tag.encode("ascii").ljust(4, b"\0") + struct.pack("<II", len(payload), 0x40000) + bytes(body)


def build_bml_mesh(
    name: str = "test_mesh",
    positions: list[tuple[float, float, float]] | None = None,
    uvs: list[tuple[float, float]] | None = None,
) -> bytes:
    """Build a minimal BML with one MESH chunk and its four DATA blocks."""
    positions = positions or [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
    uvs = uvs or [(0.0, 0.0), (1.0, 1.0)]
    stride = 84
    vertex_count = len(positions)
    indices = [0, 1, 0] if vertex_count >= 2 else [0, 0, 0]

    name_payload = name.encode("ascii") + b"\0"
    vertex_payload = bytearray(vertex_count * stride)
    for i, (position, uv) in enumerate(zip(positions, uvs)):
        offset = i * stride
        struct.pack_into("<6f", vertex_payload, offset, *position, 0.0, 0.0, 1.0)
        struct.pack_into("<2H", vertex_payload, offset + 0x40,
                         int(uv[0] * 65535), int(uv[1] * 65535))
    index_payload = struct.pack(f"<{len(indices)}H", *indices)
    counts_payload = struct.pack("<4I", vertex_count, 0, len(indices), 0)

    name_chunk = _chunk("DATA", name_payload)
    vertex_chunk = _chunk("DATA", bytes(vertex_payload))
    index_chunk = _chunk("DATA", index_payload)
    counts_chunk = _chunk("DATA", counts_payload)

    header = _chunk("BML", b"\0" * 16)
    cursor = len(header) + 12 + 0x68
    name_ptr = cursor + 12
    cursor += len(name_chunk)
    vertex_ptr = cursor + 12
    cursor += len(vertex_chunk)
    index_ptr = cursor + 12
    cursor += len(index_chunk)
    counts_ptr = cursor + 12

    payload = bytearray(0x68)
    struct.pack_into("<4Q", payload, 0x00, name_ptr, vertex_ptr, index_ptr, counts_ptr)
    struct.pack_into("<I", payload, 0x50, len(indices))
    struct.pack_into("<I", payload, 0x54, vertex_count)
    struct.pack_into("<HH", payload, 0x58, 2, stride)
    mesh_chunk = _chunk("MESH", bytes(payload))

    return header + mesh_chunk + name_chunk + vertex_chunk + index_chunk + counts_chunk
