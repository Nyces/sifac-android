"""GPU rasteriser for the stage/live preview (OpenGL 3.3 through moderngl).

The software renderer in :mod:`sifac.render` is a per-pixel Python loop; a
1080p live frame takes about five minutes. This module keeps the same scene
description (``(mesh, material)`` pairs plus a :class:`sifac.render.Camera`) and
shade model (``0.72 + 0.28 * |n . l|``, additive glow layers, mipmapped
sampling) but lets the GPU rasterise it.

Characters are skinned on the GPU: their bind-pose geometry and per-vertex bone
weights go into static vertex buffers, and every frame only a palette of
``pose_world * invert(bind_world)`` matrices is uploaded, so no vertex is ever
transformed in Python. Rigid parts (eyes, faces) and a character's floor spot
ride in a per-draw model matrix.

moderngl is an optional dependency: ``pip install moderngl numpy``. When it or
an OpenGL 3.3 context is missing, :class:`GpuScene` raises :class:`GpuUnavailable`
and the CLI keeps using the software path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .render import ADDITIVE_GAIN, BACKGROUND, _additive, _repair_uvs
from .texture import BtxTexture

VERTEX_SHADER = """
#version 330
uniform mat4 u_view_projection;
uniform mat4 u_model;
in vec3 in_position;
in vec2 in_uv;
in vec3 in_normal;
out vec2 v_uv;
out vec3 v_normal;

void main() {
    v_uv = in_uv;
    v_normal = in_normal;
    gl_Position = u_view_projection * u_model * vec4(in_position, 1.0);
}
"""


def skin_vertex_shader(max_bones: int) -> str:
    """Linear-blend skinning on the GPU.

    ``u_bones`` holds ``pose_world * invert(bind_world)`` per palette slot, so
    the shader only has to blend matrices; the vertex positions stay in bind
    pose in the VBO and never get rewritten on the CPU.
    """
    return f"""
#version 330
uniform mat4 u_view_projection;
uniform mat4 u_model;
layout(std140) uniform Bones {{
    mat4 u_bones[{max_bones}];
}};
in vec3 in_position;
in vec2 in_uv;
in vec3 in_normal;
in vec4 in_bone;
in vec4 in_weight;
out vec2 v_uv;
out vec3 v_normal;

void main() {{
    mat4 skin = mat4(0.0);
    for (int i = 0; i < 4; ++i) {{
        float weight = in_weight[i];
        if (weight == 0.0) {{
            continue;
        }}
        skin += u_bones[int(in_bone[i] + 0.5)] * weight;
    }}
    vec4 local = skin * vec4(in_position, 1.0);
    v_normal = mat3(skin) * in_normal;
    v_uv = in_uv;
    gl_Position = u_view_projection * u_model * local;
}}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_texture;
uniform vec3 u_light;
uniform float u_additive;
uniform float u_gain;
in vec2 v_uv;
in vec3 v_normal;
out vec4 out_color;

void main() {
    vec4 texel = texture(u_texture, vec2(v_uv.x, 1.0 - v_uv.y));
    float length_sq = dot(v_normal, v_normal);
    vec3 normal = length_sq > 0.0 ? v_normal * inversesqrt(length_sq) : vec3(0.0, 0.0, 1.0);
    float lambert = min(1.0, abs(dot(normal, u_light)));
    float shade = 0.72 + 0.28 * lambert;
    if (u_additive > 0.5) {
        out_color = vec4(texel.rgb * shade * u_gain, 0.0);
    } else {
        out_color = vec4(texel.rgb * shade, 1.0);
    }
}
"""


class GpuUnavailable(RuntimeError):
    """Raised when moderngl or an OpenGL 3.3 context is not available."""


# Uniform block budget: a character rig peaks at 168 bones (79 biped + 89 hair),
# so 192 mat4 slots (12 KiB) covers every pack with room to spare.
MAX_BONES = 192


@dataclass
class DrawItem:
    """One mesh to draw.

    ``model`` is an optional 4x4 row-major model matrix (rigid eyes and faces,
    or a character's floor spot). ``bones`` and ``skin`` are only set for
    skinned meshes: ``bones`` is an (n, 4, 4) row-major delta matrix array and
    ``skin`` is the pair of per-vertex (slot indices, weights) float arrays.
    """

    mesh: object
    material: object
    model: "np.ndarray | None" = None
    bones: "np.ndarray | None" = None
    skin: "tuple | None" = None

    @property
    def additive(self) -> bool:
        return _additive(self.material)


def material_texture_key(material) -> str:
    """Texture name a material references (mirrors ``render._decode``)."""
    if material is None:
        return ""
    return getattr(material, "texture", "") or getattr(material, "name", "")


class CharacterRig:
    """Bind-pose skin data for one character, reused across every frame.

    Character meshes stay in bind pose in their VBOs; each frame only the bone
    palette changes, so all the per-vertex work happens once here and then in
    the vertex shader.
    """

    def __init__(self, meshes, skeleton, weights_by_mesh) -> None:
        from .skin import invert

        self.skeleton = skeleton
        self.meshes = list(meshes)
        palette: list[str] = []
        index: dict[str, int] = {}
        self.skins: list[tuple | None] = []
        for mesh, weights in zip(self.meshes, weights_by_mesh):
            if weights is None:
                self.skins.append(None)
                continue
            count = len(mesh.positions)
            slots = np.zeros((count, 4), dtype=np.int32)
            weights_array = np.zeros((count, 4), dtype=np.float32)
            for vertex, (bones, values) in enumerate(weights):
                for rank in range(min(4, len(bones))):
                    name = bones[rank]
                    slot = index.get(name)
                    if slot is None:
                        slot = len(palette)
                        index[name] = slot
                        palette.append(name)
                    slots[vertex, rank] = slot
                    weights_array[vertex, rank] = values[rank]
            self.skins.append((slots, weights_array))
        self.palette = tuple(palette)
        self.overflow = len(self.palette) > MAX_BONES
        bind = skeleton.world_all()
        self._bind_inverse = np.asarray(
            [invert(bind[name]) if name in bind else np.eye(4) for name in self.palette],
            dtype=np.float32,
        ).reshape(len(self.palette), 4, 4)

    def palette_matrices(self, pose_quat) -> np.ndarray:
        """``pose_world * invert(bind_world)`` for every palette bone."""
        if not self.palette:
            return np.zeros((0, 4, 4), dtype=np.float32)
        world = self.skeleton.world_all(pose_quat=pose_quat)
        posed = np.asarray([world.get(name, np.eye(4)) for name in self.palette],
                           dtype=np.float32).reshape(len(self.palette), 4, 4)
        return np.matmul(posed, self._bind_inverse)



class GpuScene:
    """One offscreen framebuffer plus the shaders needed to fill it."""

    def __init__(
        self,
        width: int,
        height: int,
        background: tuple[int, int, int, int] = BACKGROUND,
    ) -> None:
        try:
            import moderngl
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise GpuUnavailable(
                "moderngl is not installed; run 'pip install moderngl' or drop --gpu"
            ) from exc
        self._moderngl = moderngl
        try:
            self.ctx = moderngl.create_standalone_context(require=330)
        except Exception as exc:  # pragma: no cover - depends on the machine
            raise GpuUnavailable(f"no OpenGL 3.3 context: {exc}") from exc
        self.width = width
        self.height = height
        self.background = background
        self.color = self.ctx.texture((width, height), 4)
        self.depth = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color], depth_attachment=self.depth
        )
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        self.program["u_texture"].value = 0
        self.program["u_light"].value = (-0.4, -0.85, 0.35)
        self.skin_program = self.ctx.program(
            vertex_shader=skin_vertex_shader(MAX_BONES),
            fragment_shader=FRAGMENT_SHADER,
        )
        self.skin_program["u_texture"].value = 0
        self.skin_program["u_light"].value = (-0.4, -0.85, 0.35)
        self.bones_buffer = self.ctx.buffer(
            reserve=MAX_BONES * 16 * 4, dynamic=True
        )
        self.skin_program["Bones"].binding = 0
        self.bones_buffer.bind_to_uniform_block(0)
        self._textures: dict[str, object] = {}
        self._geometry: dict[int, list] = {}
        self._skinned: dict[int, list] = {}
        self._generation = 0
        self._identity_blob = np.eye(4, dtype=np.float32).tobytes()
        self._fallback = self.ctx.texture((1, 1), 4, bytes((180, 180, 180, 255)))
        self._fallback.build_mipmaps()

    @property
    def info(self) -> str:
        ctx = self.ctx
        return f"{ctx.info['GL_RENDERER']} / OpenGL {ctx.info['GL_VERSION']}"

    def _upload(self, key: str, textures: dict[str, BtxTexture]):
        cached = self._textures.get(key)
        if cached is not None:
            return cached
        texture = textures.get(key)
        if texture is None and "." in key:
            texture = textures.get(key.rsplit(".", 1)[0])
        if texture is None:
            self._textures[key] = self._fallback
            return self._fallback
        try:
            width, height, rgba = texture.to_rgba(0)
        except Exception:
            self._textures[key] = self._fallback
            return self._fallback
        uploaded = self.ctx.texture((width, height), 4, rgba)
        uploaded.build_mipmaps()
        uploaded.filter = (
            self._moderngl.LINEAR_MIPMAP_LINEAR,
            self._moderngl.LINEAR,
        )
        uploaded.repeat_x = True
        uploaded.repeat_y = True
        self._textures[key] = uploaded
        return uploaded

    @staticmethod
    def _vertex_arrays(mesh, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(mesh.positions, dtype=np.float32)
        if mesh.uvs and len(mesh.uvs) == count:
            repaired = _repair_uvs(mesh)
            uvs = np.asarray(repaired if repaired is not None else mesh.uvs, dtype=np.float32)
        else:
            uvs = np.zeros((count, 2), dtype=np.float32)
        uvs = np.nan_to_num(uvs, nan=0.0, posinf=0.0, neginf=0.0)
        if mesh.normals and len(mesh.normals) == count:
            normals = np.asarray(mesh.normals, dtype=np.float32)
        else:
            normals = np.zeros((count, 3), dtype=np.float32)
            normals[:, 2] = 1.0
        normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
        return positions, uvs, normals

    @staticmethod
    def triangles_of(mesh) -> "np.ndarray | None":
        count = len(mesh.positions)
        if count == 0 or not mesh.indices:
            return None
        indices = np.asarray(mesh.indices, dtype=np.int32)
        triangles = indices[: (len(indices) // 3) * 3].reshape(-1, 3)
        triangles = triangles[(triangles >= 0).all(axis=1) & (triangles < count).all(axis=1)]
        return triangles if len(triangles) else None

    def _geometry_for(self, mesh, triangles):
        """VAO+VBO+EBO for a mesh, cached while the same object keeps showing up.

        Stage meshes are built once and reused by every frame, while posed
        character meshes are new objects each frame; ``render`` drops the
        entries the current frame did not touch, so nothing accumulates.
        """
        entry = self._geometry.get(id(mesh))
        if entry is not None and entry[0] is mesh:
            entry[4] = self._generation
            return entry[1]

        count = len(mesh.positions)
        positions, uvs, normals = self._vertex_arrays(mesh, count)
        vertex_data = np.hstack((positions, uvs, normals)).astype(np.float32)
        vbo = self.ctx.buffer(vertex_data.tobytes())
        ebo = self.ctx.buffer(triangles.reshape(-1).astype(np.int32).tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 2f 3f", "in_position", "in_uv", "in_normal")],
            ebo,
        )
        self._geometry[id(mesh)] = [mesh, vao, vbo, ebo, self._generation]
        return vao

    def _skinned_for(self, mesh, triangles, skin):
        """VAO for a bind-pose mesh plus its per-vertex bone slots and weights.

        The mesh object is stable for a character (bind pose), so this is built
        once and reused every frame; only the bone palette is re-uploaded.
        """
        entry = self._skinned.get(id(mesh))
        if entry is not None and entry[0] is mesh:
            entry[4] = self._generation
            return entry[1]

        count = len(mesh.positions)
        positions, uvs, normals = self._vertex_arrays(mesh, count)
        slots, weights = skin
        if len(slots) != count or len(weights) != count:
            return None
        vertex_data = np.hstack(
            (positions, uvs, normals, slots.astype(np.float32), weights.astype(np.float32))
        ).astype(np.float32)
        vbo = self.ctx.buffer(vertex_data.tobytes())
        ebo = self.ctx.buffer(triangles.reshape(-1).astype(np.int32).tobytes())
        vao = self.ctx.vertex_array(
            self.skin_program,
            [(vbo, "3f 2f 3f 4f 4f", "in_position", "in_uv", "in_normal",
              "in_bone", "in_weight")],
            ebo,
        )
        self._skinned[id(mesh)] = [mesh, vao, vbo, ebo, self._generation]
        return vao

    def _bind_model(self, program, model) -> None:
        if model is None:
            program["u_model"].write(self._identity_blob)
        else:
            program["u_model"].write(
                np.ascontiguousarray(np.asarray(model, dtype=np.float32).T).tobytes()
            )
        program["u_gain"].value = ADDITIVE_GAIN

    def _draw_item(self, item: DrawItem, texture) -> bool:
        mesh = item.mesh
        triangles = self.triangles_of(mesh)
        if triangles is None:
            return False
        if not np.isfinite(np.asarray(mesh.positions, dtype=np.float32)).all():
            return False
        additive = item.additive
        if item.skin is not None and item.bones is not None:
            vao = self._skinned_for(mesh, triangles, item.skin)
            if vao is None:
                return False
            palette = np.zeros((MAX_BONES, 4, 4), dtype=np.float32)
            bones = np.asarray(item.bones, dtype=np.float32)
            used = min(len(bones), MAX_BONES)
            palette[:used] = bones[:used]
            self.bones_buffer.write(
                np.ascontiguousarray(palette.transpose(0, 2, 1)).tobytes()
            )
            self._bind_model(self.skin_program, item.model)
            self.skin_program["u_additive"].value = 1.0 if additive else 0.0
            texture.use(0)
            vao.render(self._moderngl.TRIANGLES)
        else:
            vao = self._geometry_for(mesh, triangles)
            self._bind_model(self.program, item.model)
            self.program["u_additive"].value = 1.0 if additive else 0.0
            texture.use(0)
            vao.render(self._moderngl.TRIANGLES)
        return True

    def _drop_stale_geometry(self) -> None:
        for cache in (self._geometry, self._skinned):
            for key, entry in list(cache.items()):
                if entry[4] == self._generation:
                    continue
                entry[1].release()
                entry[2].release()
                entry[3].release()
                del cache[key]

    def render_items(self, items, textures, camera) -> bytes:
        """Draw a mixed list of :class:`DrawItem` and return top-down RGBA bytes."""
        ctx = self.ctx
        self._generation += 1
        self.fbo.use()
        red, green, blue, alpha = self.background
        ctx.clear(red / 255.0, green / 255.0, blue / 255.0, alpha / 255.0, depth=1.0)

        matrix = np.asarray(camera.matrices(self.width / max(self.height, 1)), dtype=np.float32)
        blob = np.ascontiguousarray(matrix.T).tobytes()
        self.program["u_view_projection"].write(blob)
        self.skin_program["u_view_projection"].write(blob)

        ctx.enable(self._moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.depth_mask = True
        ctx.disable(self._moderngl.BLEND)

        order = sorted(range(len(items)), key=lambda index: items[index].additive)
        for index in order:
            item = items[index]
            additive = item.additive
            key = material_texture_key(item.material)
            texture = self._upload(key, textures) if key else self._fallback
            if additive:
                ctx.enable(self._moderngl.BLEND)
                ctx.blend_func = (self._moderngl.ONE, self._moderngl.ONE)
                ctx.depth_mask = False
            else:
                ctx.disable(self._moderngl.BLEND)
                ctx.depth_mask = True
            self._draw_item(item, texture)
        ctx.depth_mask = True
        ctx.disable(self._moderngl.BLEND)
        self._drop_stale_geometry()

        data = self.fbo.read(components=4)
        rows = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
        return rows[::-1].tobytes()

    def render(self, wire, textures, camera) -> bytes:
        """Draw every (mesh, material) pair and return top-down RGBA bytes."""
        return self.render_items(
            [DrawItem(mesh, material) for mesh, material in wire], textures, camera
        )
