"""Blade ``btx`` texture reader.

A btx payload starts with an 0x80-byte header::

    0x00  "btx\\0"
    0x04  u16 version (0x0102)
    0x08  u16 width
    0x0a  u16 height
    0x0c  u32 pitch (width * 4)
    0x10  u32 format   (low byte = pixel format, high byte = 0x01)
    0x14  u32
    0x18  u32          (usage flags)
    0x1c  u32
    0x20  u32          (extra payload size)
    0x24  u32 name offset (0x30)
    0x30  name, NUL terminated
    0x80  pixel data (level 0, then optional mip levels)

Formats confirmed against the retail dump:

    0x00  RGBA8, 4 bytes per pixel
    0x0d  BC7,   16 bytes per 4x4 block (1 byte per pixel)
    0x1b  RGBA16F, 8 bytes per pixel

The format ids match the engine's enum, whose names are in the eboot rodata at
0x63FF80: ``R8G8B8A8, R8G8B8, R5G6B5, R5G5B5A1, R4G4B4A4, P8, P4, BC1 ... BC7
(0x0d), D24S8, D15S1, L8, A8, L4A4, L8A8, YUV, PVRTII_4, PVRTII_2, U16U16,
F16F16, F32, RGBA_U32U32, A16B16G16R16 (0x1b), ...``. A RenderDoc capture of
the retail game confirms 0x1b is uploaded as ``VK_FORMAT_R16G16B16A16_SFLOAT``
(128x64 face-expression texture, 64 KiB, no mips).

Format ``0x1b`` is not a picture: it is a per-vertex morph target for the face
mesh. ``sifac.expression`` decodes it (positions in texels ``[0, 4096)``,
normals after that, channel 3 as a validity mask), so ``to_rgba`` here only
exists for inspection and clamps the half floats to 0..255.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from .binary import cstring, u16, u32

BTX_MAGIC = b"btx\0"
HEADER_SIZE = 0x80

FORMAT_RGBA8 = 0x00
FORMAT_BC7 = 0x0D
FORMAT_RGBA16F = 0x1B

FORMAT_NAMES = {
    FORMAT_RGBA8: "rgba8",
    FORMAT_BC7: "bc7",
    FORMAT_RGBA16F: "rgba16f",
}

# bytes per pixel for the formats we can size
BYTES_PER_PIXEL = {
    FORMAT_RGBA8: 4,
    FORMAT_BC7: 1,
    FORMAT_RGBA16F: 8,
}


class BtxError(ValueError):
    """Raised when a btx payload is malformed."""


@dataclass(frozen=True)
class BtxHeader:
    version: int
    width: int
    height: int
    pitch: int
    format: int
    name: str
    header_size: int = HEADER_SIZE

    @property
    def format_name(self) -> str:
        return FORMAT_NAMES.get(self.format, f"0x{self.format:02x}")

    @property
    def bytes_per_pixel(self) -> int | None:
        return BYTES_PER_PIXEL.get(self.format)


def parse_header(data: bytes, base: int = 0) -> BtxHeader:
    if len(data) < base + HEADER_SIZE:
        raise BtxError("payload is smaller than a btx header")
    if data[base : base + 4] != BTX_MAGIC:
        raise BtxError("not a btx texture")
    version = u16(data, base + 4)
    width = u16(data, base + 8)
    height = u16(data, base + 10)
    pitch = u32(data, base + 0x0C)
    fmt = u32(data, base + 0x10) & 0xFF
    name_offset = u32(data, base + 0x24)
    if name_offset == 0 or name_offset >= HEADER_SIZE:
        name_offset = 0x30
    name = cstring(data, base + name_offset, 0x40)
    if width == 0 or height == 0:
        raise BtxError(f"invalid dimensions {width}x{height}")
    return BtxHeader(version, width, height, pitch, fmt, name)


def level_size(width: int, height: int, fmt: int) -> int:
    if fmt == FORMAT_BC7:
        return ((width + 3) // 4) * ((height + 3) // 4) * 16
    bpp = BYTES_PER_PIXEL.get(fmt)
    if bpp is None:
        raise BtxError(f"unsupported btx format 0x{fmt:02x}")
    return width * height * bpp


def count_levels(payload_size: int, width: int, height: int, fmt: int) -> int:
    """Count how many mip levels fit exactly in the payload."""
    levels = 0
    remaining = payload_size
    w, h = width, height
    while w >= 1 and h >= 1:
        size = level_size(w, h, fmt)
        if size > remaining:
            break
        remaining -= size
        levels += 1
        if w == 1 and h == 1:
            break
        w = max(1, w // 2)
        h = max(1, h // 2)
    return max(levels, 1)


def _half_to_byte(value: float) -> int:
    """Clamp one half-float channel to 0..255 (NaN and negatives become 0)."""
    if not value > 0.0:
        return 0
    if value >= 1.0:
        return 255
    return int(value * 255)


class BtxTexture:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.header = parse_header(data)
        self.payload = data[HEADER_SIZE:]
        self.levels = count_levels(
            len(self.payload), self.header.width, self.header.height, self.header.format
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "BtxTexture":
        return cls(Path(path).read_bytes())

    def _require_supported(self) -> None:
        if self.header.format not in (FORMAT_RGBA8, FORMAT_BC7, FORMAT_RGBA16F):
            raise BtxError(
                f"btx format {self.header.format_name} is not decoded yet"
            )

    def level_bytes(self, level: int = 0) -> bytes:
        self._require_supported()
        offset = 0
        w, h = self.header.width, self.header.height
        for _ in range(level):
            offset += level_size(w, h, self.header.format)
            w = max(1, w // 2)
            h = max(1, h // 2)
        size = level_size(w, h, self.header.format)
        return self.payload[offset : offset + size]

    def to_rgba(self, level: int = 0) -> tuple[int, int, int, bytes]:
        """Return (width, height, RGBA bytes) for one level."""
        self._require_supported()
        width, height = self.header.width, self.header.height
        for _ in range(level):
            width = max(1, width // 2)
            height = max(1, height // 2)
        raw = self.level_bytes(level)
        if self.header.format == FORMAT_RGBA8:
            return width, height, raw
        if self.header.format == FORMAT_RGBA16F:
            halves = struct.unpack(f"<{width * height * 4}e", raw)
            return width, height, bytes(_half_to_byte(value) for value in halves)
        from texture2ddecoder import decode_bc7  # optional dependency

        bgra = decode_bc7(raw, width, height)
        rgba = bytearray(len(bgra))
        rgba[0::4] = bgra[2::4]
        rgba[1::4] = bgra[1::4]
        rgba[2::4] = bgra[0::4]
        rgba[3::4] = bgra[3::4]
        return width, height, bytes(rgba)

    def to_png(self, level: int = 0) -> bytes:
        width, height, rgba = self.to_rgba(level)
        return encode_png(width, height, rgba)


def encode_png(width: int, height: int, rgba: bytes) -> bytes:
    stride = width * 4
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: none
        raw += rgba[y * stride : (y + 1) * stride]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def find_textures(data: bytes, limit: int = 0x400) -> list[int]:
    """Return offsets of btx headers within a buffer."""
    offsets = []
    position = 0
    while True:
        position = data.find(BTX_MAGIC, position, limit)
        if position < 0:
            break
        offsets.append(position)
        position += 1
    return offsets


def scan_loose_textures(data: bytes) -> dict[str, "BtxTexture"]:
    """Every btx blob stored directly in a buffer, outside any ARC.

    Character packs keep the 45-47 face-expression maps as loose 64 KiB blobs
    (0x80 header + 128x64 RGBA16F) between the ARC payloads, so the texture
    ARCHive inside ``model.pac`` only holds the eight BC7 maps. Walking the raw
    buffer is the only way to reach them.
    """
    textures: dict[str, BtxTexture] = {}
    position = 0
    while True:
        position = data.find(BTX_MAGIC, position)
        if position < 0:
            break
        try:
            header = parse_header(data, position)
            size = HEADER_SIZE + level_size(header.width, header.height, header.format)
        except (BtxError, struct.error):
            position += 1
            continue
        if position + size > len(data):
            position += 1
            continue
        texture = BtxTexture(data[position : position + size])
        name = texture.header.name
        if name and name not in textures:
            textures[name] = texture
        # Jump past this blob: its payload can contain the magic by accident.
        position += size
    return textures


def load_texture_archive(pac_path: str | Path) -> dict[str, "BtxTexture"]:
    """Load every btx texture a model uses, keyed by both btx and ARC name.

    Three layouts exist:

    * characters — ``model.pac`` holds the BMAR and a ``texture`` ARC;
    * stages — ``model.pac`` holds only the BMAR and the textures live in a
      sibling top-level ``texture.pac``;
    * face expressions — loose blobs inside the character ``.pac`` itself,
      outside every ARC.

    The loose scan always runs, so callers see one flat name map.
    """
    from .arc import ArcArchive, ArcError
    from .binary import unwrap_section
    from .pac import read_pac

    image = read_pac(pac_path)
    textures: dict[str, BtxTexture] = {}
    for name, texture in scan_loose_textures(image.data).items():
        textures[name] = texture

    def collect(texture_bytes: bytes) -> dict[str, BtxTexture]:
        try:
            texture_arc = ArcArchive(texture_bytes)
        except ArcError:
            return {}
        found: dict[str, BtxTexture] = {}
        for entry in texture_arc.entries:
            payload, _ = unwrap_section(texture_arc.entry_data(entry))
            try:
                texture = BtxTexture(payload)
            except BtxError:
                continue
            name = texture.header.name or entry.name
            found[name] = texture
            found.setdefault(entry.name, texture)
        return found

    top = ArcArchive(image.data)
    sibling = top.find("texture.pac") or next(
        (entry for entry in top.entries if entry.type == "texs"), None
    )
    if sibling is not None:
        textures.update(collect(unwrap_section(top.entry_data(sibling))[0]))
        return textures

    model_entry = top.find("model.pac")
    if model_entry is None:
        return textures
    model_bytes, _ = unwrap_section(top.entry_data(model_entry))
    model_arc = ArcArchive(model_bytes)
    nested = model_arc.find("texture") or next(
        (entry for entry in model_arc.entries if entry.type == "texs"), None
    )
    if nested is not None:
        textures.update(collect(unwrap_section(model_arc.entry_data(nested))[0]))
    return textures

