"""Blade BML chunk parser and mesh extraction.

A BML chunk is ``magic(4) size(u32) flags(u32) payload(size)``. The chunks that
matter here:

``NODE``   scene-graph node (name + transform)
``MESH``   one drawable mesh; payload:

             +0x00 u64 name string offset
             +0x08 u64 vertex buffer offset
             +0x10 u64 index buffer offset
             +0x18 u64 counts chunk offset
             +0x50 u32 index count
             +0x54 u32 vertex count
             +0x58 u16 topology (2)
             +0x5a u16 vertex stride (84 in the retail dump)

``DATA``   raw payload (names, vertices, indices, counts)
``TXTR``   texture reference used by the model
``MATE``   material (name + shader parameters)
``DRAW``   draw call binding

Vertex layout for stride 84 (confirmed on ``mod_01_honoka``)::

    0x00  3 x f32 position
    0x0c  3 x f32 normal
    0x18  f32      (tangent w / unused)
    0x1c  4 bytes  constant
    0x20  12 bytes zeros
    0x38  2 x u16  UV (unorm16)
    0x34  4 bytes  constant (0x40,0x30,0x1f,0xff)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from .binary import cstring, u32
from .texture import encode_png

CHUNK_HEADER = 12
MESH_HEADER_FIELDS = {
    "name": 0x00,
    "vertices": 0x08,
    "indices": 0x10,
    "counts": 0x18,
}
TOPOLOGY_TRIANGLES = 2


class BmlError(ValueError):
    """Raised when a BML buffer is malformed."""


@dataclass(frozen=True)
class Chunk:
    tag: str
    offset: int
    size: int
    flags: int

    @property
    def payload_offset(self) -> int:
        return self.offset + CHUNK_HEADER


@dataclass
class Mesh:
    name: str
    vertex_count: int
    index_count: int
    stride: int
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not self.positions:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        zs = [p[2] for p in self.positions]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


class BmlArchive:
    def __init__(self, data: bytes) -> None:
        if data[:4] != b"BML\0":
            raise BmlError("not a BML buffer")
        self.data = data
        self.chunks = tuple(self._walk())

    def _walk(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        position = 0
        while position + CHUNK_HEADER <= len(self.data):
            raw = self.data[position : position + 4]
            tag_ok = (
                all(65 <= byte <= 90 for byte in raw[:3])
                and (65 <= raw[3] <= 90 or raw[3] == 0)
            )
            if not tag_ok:
                break
            size = u32(self.data, position + 4)
            flags = u32(self.data, position + 8)
            if size > 0x2000000 or position + CHUNK_HEADER + size > len(self.data):
                break
            chunks.append(
                Chunk(raw.rstrip(b"\0").decode("ascii", "replace"), position, size, flags)
            )
            position += CHUNK_HEADER + ((size + 3) & ~3)
        return chunks

    def find(self, tag: str) -> list[Chunk]:
        return [chunk for chunk in self.chunks if chunk.tag == tag]

    def string_at(self, offset: int) -> str:
        return cstring(self.data, offset, 0x40)

    def texture_names(self) -> list[str]:
        names = []
        for chunk in self.find("TXTR"):
            pointer = struct.unpack_from("<Q", self.data, chunk.payload_offset)[0]
            if 0 < pointer < len(self.data):
                names.append(self.string_at(pointer))
        return names

    def meshes(self) -> list[Mesh]:
        meshes: list[Mesh] = []
        for chunk in self.find("MESH"):
            mesh = self._read_mesh(chunk)
            if mesh is not None:
                meshes.append(mesh)
        return meshes

    def _read_mesh(self, chunk: Chunk) -> Mesh | None:
        payload = chunk.payload_offset
        if payload + 0x60 > len(self.data):
            return None
        name_ptr = struct.unpack_from("<Q", self.data, payload + 0x00)[0]
        vertex_ptr = struct.unpack_from("<Q", self.data, payload + 0x08)[0]
        index_ptr = struct.unpack_from("<Q", self.data, payload + 0x10)[0]
        index_count = u32(self.data, payload + 0x50)
        vertex_count = u32(self.data, payload + 0x54)
        topology = struct.unpack_from("<H", self.data, payload + 0x58)[0]
        stride = struct.unpack_from("<H", self.data, payload + 0x5A)[0]
        if topology != TOPOLOGY_TRIANGLES or stride < 0x20:
            return None
        if vertex_count == 0 or index_count == 0:
            return None
        if vertex_ptr + vertex_count * stride > len(self.data):
            return None
        if index_ptr + index_count * 2 > len(self.data):
            return None

        mesh = Mesh(
            name=self.string_at(name_ptr) if name_ptr else "",
            vertex_count=vertex_count,
            index_count=index_count,
            stride=stride,
        )
        for i in range(vertex_count):
            offset = vertex_ptr + i * stride
            x, y, z, nx, ny, nz, _w = struct.unpack_from("<7f", self.data, offset)
            mesh.positions.append((x, y, z))
            mesh.normals.append((nx, ny, nz))
            u_raw, v_raw = struct.unpack_from("<2H", self.data, offset + 0x38)
            mesh.uvs.append((u_raw / 65535.0, v_raw / 65535.0))
        mesh.indices = list(
            struct.unpack_from(f"<{index_count}H", self.data, index_ptr)
        )
        return mesh

    def to_obj(self, with_uv: bool = True) -> str:
        lines = ["# SIFAC Blade model (read-only extraction)"]
        base = 1
        for mesh in self.meshes():
            lines.append(f"o {mesh.name or 'mesh'}")
            # Blade uses Z as the up axis; OBJ expects Y up.
            for x, y, z in mesh.positions:
                lines.append(f"v {x:.6f} {z:.6f} {-y:.6f}")
            for nx, ny, nz in mesh.normals:
                lines.append(f"vn {nx:.6f} {nz:.6f} {-ny:.6f}")
            if with_uv:
                for u, v in mesh.uvs:
                    lines.append(f"vt {u:.6f} {1.0 - v:.6f}")
            for i in range(mesh.triangle_count):
                a, b, c = mesh.indices[i * 3 : i * 3 + 3]
                ia, ib, ic = base + a, base + b, base + c
                if with_uv:
                    lines.append(
                        f"f {ia}/{ia}/{ia} {ib}/{ib}/{ib} {ic}/{ic}/{ic}"
                    )
                else:
                    lines.append(f"f {ia}//{ia} {ib}//{ib} {ic}//{ic}")
            base += mesh.vertex_count
        return "\n".join(lines) + "\n"


def _set(pixels: bytearray, width: int, x: int, y: int, color: tuple[int, int, int, int]) -> None:
    height = len(pixels) // (width * 4)
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 4
        pixels[offset : offset + 4] = bytes(color)


def render_silhouette(meshes: list[Mesh], width: int = 512, height: int = 768) -> bytes:
    """Orthographic front (x/z) point plot, useful to validate geometry."""
    points = [p for mesh in meshes for p in mesh.positions]
    if not points:
        raise ValueError("no vertices to render")
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_z = min(p[2] for p in points)
    max_z = max(p[2] for p in points)
    span_x = max(max_x - min_x, 1e-6)
    span_z = max(max_z - min_z, 1e-6)
    scale = min((width - 32) / span_x, (height - 32) / span_z)
    pixels = bytearray(width * height * 4)
    for i in range(width * height):
        pixels[i * 4 : i * 4 + 4] = bytes((12, 12, 16, 255))
    for x, _y, z in points:
        px = int((x - min_x) * scale) + 16
        py = height - 1 - (int((z - min_z) * scale) + 16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                _set(pixels, width, px + dx, py + dy, (120, 200, 255, 255))
    return encode_png(width, height, bytes(pixels))


def dump_chunks(data: bytes) -> list[dict]:
    archive = BmlArchive(data)
    return [
        {"tag": c.tag, "offset": c.offset, "size": c.size, "flags": c.flags}
        for c in archive.chunks
    ]


def find_bml_in_pac(path: str | Path, character: str | None = None) -> BmlArchive:
    """Open the BML chunk list of a character model inside a .pac."""
    from .bmar import find_bmar_in_pac

    bmar = find_bmar_in_pac(path, character)
    for index, entry in enumerate(bmar.entries):
        if entry.type == "BML":
            return BmlArchive(bmar.entry_data(index))
    raise BmlError(f"no BML chunk found in {path}")
