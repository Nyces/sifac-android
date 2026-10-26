"""Face-expression morph targets (btx format ``0x1b``).

The 45-47 ``<Name>_face_<expression>`` textures a character ships are 128x64
RGBA16F arrays, but they are not pictures: they are per-vertex morph targets
indexed by the face mesh's vertex id.

Layout, verified on Honoka (2401-vertex face), Rin and Nico:

* texels ``[0, 4096)`` hold the **position delta** in metres, one texel per
  face vertex;
* texels ``[4096, 8192)`` hold the **normal delta**, same indexing;
* channel 3 of each texel is a validity mask (1 for 2395 of 2401 vertices).

Evidence: for every expression the non-zero texels fall inside those two
ranges (only a handful of strays appear at ``4095``); the first array peaks at
1.0-5.3 cm on a 20 cm face, the right scale for a facial deformation; the
second array is roughly perpendicular to the rest normal (mean dot -0.01) and
its magnitude sits well below 1, so it is an offset rather than a replacement
normal. Eye expressions start at vertex 0 and mouth expressions at vertex 233,
so the mesh's vertex order groups the face parts.

RenderDoc confirms the usage: in the retail capture the expression texture is
bound to the *vertex* shader of the draw that renders the 3370-triangle face
mesh (event 90, ``ResourceId::28522``), not to the pixel shader, which is what a
vertex-texture-fetch morph target looks like.

The expression is applied as ``p + A[i]`` and ``normalize(n + B[i])``.

Which expression plays when is still open: the character timeline's cue
``ident`` (34 distinct values across the dump) is the obvious candidate, but it
has not been matched to these names yet, so callers pick one by name.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .bml import Mesh
from .texture import FORMAT_RGBA16F, BtxTexture, BtxError

# Texels between the position array and the normal array. 128 x 64 = 8192
# texels hold both arrays, 32 rows each.
ARRAY_STRIDE = 4096
MASK_THRESHOLD = 0.5


class ExpressionError(ValueError):
    """Raised when a face-expression texture does not fit a face mesh."""


@dataclass
class FaceExpression:
    """Morph target of one face mesh, indexed by vertex id."""

    name: str
    positions: list[tuple[float, float, float]]
    normals: list[tuple[float, float, float]]
    _cache: dict[int, Mesh] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.positions)

    def apply(self, mesh: Mesh, name: str | None = None) -> Mesh:
        """``mesh`` with the deltas added; the result is cached per mesh.

        The texture always holds ``ARRAY_STRIDE`` entries; a face mesh uses the
        leading ones (2401 for Honoka, 2711 for Rin), so a shorter mesh simply
        reads a prefix of the array.
        """
        key = id(mesh)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        count = len(mesh.positions)
        if count > len(self.positions):
            raise ExpressionError(
                f"{self.name} holds {len(self.positions)} vertices but "
                f"{mesh.name} has {count}"
            )
        positions = [
            (p[0] + d[0], p[1] + d[1], p[2] + d[2])
            for p, d in zip(mesh.positions, self.positions)
        ]
        if mesh.normals and len(mesh.normals) == count:
            normals = []
            for normal, delta in zip(mesh.normals, self.normals):
                x = normal[0] + delta[0]
                y = normal[1] + delta[1]
                z = normal[2] + delta[2]
                length = math.sqrt(x * x + y * y + z * z) or 1.0
                normals.append((x / length, y / length, z / length))
        else:
            normals = list(mesh.normals)
        result = Mesh(
            name=name or mesh.name,
            vertex_count=mesh.vertex_count,
            index_count=mesh.index_count,
            stride=mesh.stride,
            positions=positions,
            normals=normals,
            uvs=mesh.uvs,
            indices=mesh.indices,
            bone_indices=mesh.bone_indices,
        )
        self._cache[key] = result
        return result


def decode(name: str, texture: BtxTexture) -> FaceExpression:
    """Read one expression texture into per-vertex deltas."""
    if texture.header.format != FORMAT_RGBA16F:
        raise ExpressionError(
            f"{name}: format {texture.header.format_name} is not an expression map"
        )
    width, height = texture.header.width, texture.header.height
    total = width * height
    if total < ARRAY_STRIDE * 2:
        raise ExpressionError(f"{name}: {width}x{height} is too small")
    payload = texture.level_bytes(0)
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    for texel in range(ARRAY_STRIDE):
        px, py, pz, mask = struct.unpack_from("<4e", payload, texel * 8)
        if mask > MASK_THRESHOLD:
            positions.append((px, py, pz))
        else:
            positions.append((0.0, 0.0, 0.0))
        nx, ny, nz, nmask = struct.unpack_from(
            "<4e", payload, (ARRAY_STRIDE + texel) * 8
        )
        normals.append((nx, ny, nz) if nmask > MASK_THRESHOLD else (0.0, 0.0, 0.0))
    return FaceExpression(name, positions, normals)


def expressions_from_textures(
    textures: dict[str, BtxTexture],
) -> dict[str, FaceExpression]:
    """Expressions among an already-loaded texture archive."""
    expressions: dict[str, FaceExpression] = {}
    for name, texture in textures.items():
        if texture.header.format != FORMAT_RGBA16F:
            continue
        try:
            expressions[name] = decode(name, texture)
        except (ExpressionError, BtxError):
            continue
    return expressions


def load_face_expressions(pac_path: str | Path) -> dict[str, FaceExpression]:
    """Every expression a character pack ships, keyed by its full btx name."""
    from .texture import load_texture_archive

    return expressions_from_textures(load_texture_archive(pac_path))


def expression_suffix(name: str) -> str:
    """``Honoka_face_eye_open`` -> ``eye_open``."""
    marker = "_face_"
    index = name.lower().find(marker)
    return name[index + len(marker) :] if index >= 0 else name


def find_expression(
    expressions: dict[str, FaceExpression], suffix: str
) -> FaceExpression | None:
    """Case-insensitive lookup by the part after ``_face_``."""
    wanted = suffix.lower()
    for name, expression in expressions.items():
        if expression_suffix(name).lower() == wanted:
            return expression
    return None
