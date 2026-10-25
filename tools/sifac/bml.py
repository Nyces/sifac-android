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
``NODE``   scene-graph node; payload:

             +0x00 u64 name string offset
             +0x08 u64 parent name string offset
             +0x10 3 x f32 translation
             +0x1c 3 x f32 scale
             +0x28 3 x f32 rotation (Euler radians, applied Z * Y * X)
             +0x34 3 x f32 bounding box min
             +0x40 3 x f32 bounding box max

``MATE``   material; first two pointers are the material name and its texture
``DRAW``   draw call: u32 node index, mesh index, material index, flags

Vertex layout (empirically validated per stride; attribute offsets differ because
meshes without skinning omit the tail block)::

    stride 84   0x00 3 x f32 position
                0x0c 3 x f32 normal
                0x18 f32      (attribute 1 .w, saved with the normal)
                0x1c 4 x u8   bone palette indices (attribute 3)
                0x40 2 x u16 UV (unorm16), followed by extra UV sets

    stride 60   0x00 3 x f32 position
                0x0c 3 x f32 normal
                0x18 2 x u16 UV (unorm16)

Vertex attributes as actually bound on the PS4 GPU (confirmed with a RenderDoc
capture of the retail game, draw of ``Honoka_hair``)::

    attr0  float4  position (w = 1)
    attr1  float4  normal, .w = 0.1528 on the first hair vertex
    attr2  float4  all zero on hair
    attr3  uint4   four bone indices (u8 x 4), e.g. (77, 32, 31, 30)
    attr4  uint4   zeros on hair
    attr5  float4  tangent
    attr6  float4  UV

The file's 84-byte vertex only carries position, normal, tangent and UV: the
bone index stream is **not** present in the model container, which matches the
earlier finding that no weight quad exists in the vertex tail. Real skinning
therefore needs either the GPU-side index stream or the engine's own binding
step; until then :mod:`sifac.skin` provides automatic weights.
"""

from __future__ import annotations

import math
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
# Offset of the first UV set, keyed by vertex stride. Meshes with skinning use
# a longer vertex and keep UVs further in.
UV_OFFSET_BY_STRIDE = {60: 0x18, 84: 0x40}


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


@dataclass(frozen=True)
class Node:
    index: int
    name: str
    parent: str
    translation: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotation: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]


@dataclass(frozen=True)
class Material:
    index: int
    name: str
    texture: str


@dataclass(frozen=True)
class DrawCall:
    node_index: int
    mesh_index: int
    material_index: int
    flags: int


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
    # Four bone palette indices (u8 x 4) stored at 0x1c of stride-84 vertices.
    bone_indices: list[tuple[int, int, int, int]] = field(default_factory=list)

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

    def nodes(self) -> list[Node]:
        nodes: list[Node] = []
        for index, chunk in enumerate(self.find("NODE")):
            payload = chunk.payload_offset
            name_ptr, parent_ptr = struct.unpack_from("<2Q", self.data, payload)
            translation = struct.unpack_from("<3f", self.data, payload + 0x10)
            scale = struct.unpack_from("<3f", self.data, payload + 0x1C)
            rotation = struct.unpack_from("<3f", self.data, payload + 0x28)
            bbox_min = struct.unpack_from("<3f", self.data, payload + 0x34)
            bbox_max = struct.unpack_from("<3f", self.data, payload + 0x40)
            nodes.append(
                Node(
                    index=index,
                    name=self.string_at(name_ptr) if name_ptr else "",
                    parent=self.string_at(parent_ptr) if parent_ptr else "",
                    translation=translation,
                    scale=scale,
                    rotation=rotation,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                )
            )
        return nodes

    def materials(self) -> list[Material]:
        materials: list[Material] = []
        for index, chunk in enumerate(self.find("MATE")):
            payload = chunk.payload_offset
            name_ptr, texture_ptr = struct.unpack_from("<2Q", self.data, payload)
            materials.append(
                Material(
                    index=index,
                    name=self.string_at(name_ptr) if name_ptr else "",
                    texture=self.string_at(texture_ptr) if texture_ptr else "",
                )
            )
        return materials

    def draw_calls(self) -> list[DrawCall]:
        calls: list[DrawCall] = []
        for chunk in self.find("DRAW"):
            payload = chunk.payload_offset
            node_index, mesh_index, material_index, flags = struct.unpack_from(
                "<4I", self.data, payload
            )
            calls.append(DrawCall(node_index, mesh_index, material_index, flags))
        return calls

    def mesh_materials(self) -> dict[int, Material]:
        """Map mesh index to its material via the DRAW table."""
        materials = self.materials()
        result: dict[int, Material] = {}
        for call in self.draw_calls():
            if 0 <= call.material_index < len(materials):
                result[call.mesh_index] = materials[call.material_index]
        return result

    def node_world_position(self, name: str) -> tuple[float, float, float] | None:
        """World position of a node in mesh space (Z up).

        Local transforms are ``T * Rz * Ry * Rx * S`` (validated by the mirrored
        eye positions and the head height on ``mod_01_honoka``). Node space is
        Y up, while mesh vertices are Z up, so the result is converted with
        ``(x, y, z) -> (x, -z, y)``.
        """
        nodes = {node.name: node for node in self.nodes()}
        chain: list[Node] = []
        current = nodes.get(name)
        while current is not None:
            chain.append(current)
            current = nodes.get(current.parent)
        if not chain:
            return None
        matrix = _identity()
        for node in reversed(chain):
            matrix = _mat_mul(matrix, _local_matrix(node))
        x, y, z = matrix[0][3], matrix[1][3], matrix[2][3]
        return (x, -z, y)

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
            x, y, z, nx, ny, nz = struct.unpack_from("<6f", self.data, offset)
            mesh.positions.append((x, y, z))
            mesh.normals.append((nx, ny, nz))
            uv_offset = UV_OFFSET_BY_STRIDE.get(stride, 0x40 if stride >= 0x44 else 0x18)
            u_raw, v_raw = struct.unpack_from("<2H", self.data, offset + uv_offset)
            mesh.uvs.append((u_raw / 65535.0, v_raw / 65535.0))
            if stride == 84:
                mesh.bone_indices.append(
                    struct.unpack_from("<4B", self.data, offset + 0x1C)
                )
        mesh.indices = list(
            struct.unpack_from(f"<{index_count}H", self.data, index_ptr)
        )
        return mesh

    def to_obj(self, with_uv: bool = True) -> str:
        return self.to_obj_for(self.meshes(), with_uv=with_uv)

    def to_obj_for(self, meshes: list["Mesh"], with_uv: bool = True) -> str:
        lines = ["# SIFAC Blade model (read-only extraction)"]
        materials = self.mesh_materials()
        base = 1
        for mesh_index, mesh in enumerate(meshes):
            lines.append(f"o {mesh.name or 'mesh'}")
            material = materials.get(mesh_index)
            if material:
                lines.append(f"usemtl {material.name}")
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

    def to_mtl(self) -> str:
        lines = ["# SIFAC Blade materials (texture names refer to btx exports)"]
        for material in self.materials():
            lines.append(f"newmtl {material.name}")
            lines.append("Kd 1.000000 1.000000 1.000000")
            if material.texture:
                lines.append(f"# btx texture: {material.texture}")
                lines.append(f"map_Kd {material.texture}.png")
            lines.append("")
        return "\n".join(lines)


def _set(pixels: bytearray, width: int, x: int, y: int, color: tuple[int, int, int, int]) -> None:
    height = len(pixels) // (width * 4)
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 4
        pixels[offset : offset + 4] = bytes(color)


def _identity() -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def _rot_x(angle: float) -> list[list[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]


def _rot_y(angle: float) -> list[list[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]


def _rot_z(angle: float) -> list[list[float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _local_matrix(node: Node) -> list[list[float]]:
    matrix = _identity()
    matrix[0][3], matrix[1][3], matrix[2][3] = node.translation
    rx, ry, rz = node.rotation
    rotation = _mat_mul(_mat_mul(_rot_z(rz), _rot_y(ry)), _rot_x(rx))
    matrix = _mat_mul(matrix, rotation)
    scaling = _identity()
    scaling[0][0], scaling[1][1], scaling[2][2] = node.scale
    return _mat_mul(matrix, scaling)


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
