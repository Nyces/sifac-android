"""GPU rasteriser for the stage/live preview (OpenGL 3.3 through moderngl).

The software renderer in :mod:`sifac.render` is a per-pixel Python loop; a
1080p live frame takes about five minutes. This module keeps the same scene
description (``(mesh, material)`` pairs plus a :class:`sifac.render.Camera`) and
shade model (``0.72 + 0.28 * |n . l|``, additive glow layers, nearest-looking
samples) but lets the GPU rasterise it, so the same frame takes a fraction of a
second.

moderngl is an optional dependency: ``pip install moderngl``. When it or an
OpenGL 3.3 context is missing, :class:`GpuScene` raises :class:`GpuUnavailable`
and the CLI keeps using the software path.
"""

from __future__ import annotations

import numpy as np

from .render import ADDITIVE_GAIN, BACKGROUND, _additive, _repair_uvs
from .texture import BtxTexture

VERTEX_SHADER = """
#version 330
uniform mat4 u_view_projection;
in vec3 in_position;
in vec2 in_uv;
in vec3 in_normal;
out vec2 v_uv;
out vec3 v_normal;

void main() {
    v_uv = in_uv;
    v_normal = in_normal;
    gl_Position = u_view_projection * vec4(in_position, 1.0);
}
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


def material_texture_key(material) -> str:
    """Texture name a material references (mirrors ``render._decode``)."""
    if material is None:
        return ""
    return getattr(material, "texture", "") or getattr(material, "name", "")


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
        self._textures: dict[str, object] = {}
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

    def _draw(self, mesh, texture, additive: bool) -> None:
        count = len(mesh.positions)
        if count == 0 or not mesh.indices:
            return
        indices = np.asarray(mesh.indices, dtype=np.int32)
        triangles = indices[: (len(indices) // 3) * 3].reshape(-1, 3)
        triangles = triangles[(triangles >= 0).all(axis=1) & (triangles < count).all(axis=1)]
        if not len(triangles):
            return
        positions = np.asarray(mesh.positions, dtype=np.float32)
        if not np.isfinite(positions).all():
            return
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

        vertex_data = np.hstack((positions, uvs, normals)).astype(np.float32)
        index_data = triangles.reshape(-1).astype(np.int32)

        vbo = self.ctx.buffer(vertex_data.tobytes())
        ebo = self.ctx.buffer(index_data.tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 2f 3f", "in_position", "in_uv", "in_normal")],
            ebo,
        )
        self.program["u_additive"].value = 1.0 if additive else 0.0
        texture.use(0)
        vao.render(self._moderngl.TRIANGLES)
        vao.release()
        vbo.release()
        ebo.release()

    def render(self, wire, textures, camera) -> bytes:
        """Draw every (mesh, material) pair and return top-down RGBA bytes."""
        ctx = self.ctx
        self.fbo.use()
        red, green, blue, alpha = self.background
        ctx.clear(red / 255.0, green / 255.0, blue / 255.0, alpha / 255.0, depth=1.0)

        matrix = np.asarray(camera.matrices(self.width / max(self.height, 1)), dtype=np.float32)
        self.program["u_view_projection"].write(np.ascontiguousarray(matrix.T).tobytes())
        self.program["u_gain"].value = ADDITIVE_GAIN

        ctx.enable(self._moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.depth_mask = True
        ctx.disable(self._moderngl.BLEND)

        flags = [_additive(material) for _mesh, material in wire]
        order = sorted(range(len(wire)), key=lambda index: flags[index])
        for index in order:
            mesh, material = wire[index]
            key = material_texture_key(material)
            texture = self._upload(key, textures) if key else self._fallback
            if flags[index]:
                ctx.enable(self._moderngl.BLEND)
                ctx.blend_func = (self._moderngl.ONE, self._moderngl.ONE)
                ctx.depth_mask = False
            else:
                ctx.disable(self._moderngl.BLEND)
                ctx.depth_mask = True
            self._draw(mesh, texture, flags[index])
        ctx.depth_mask = True
        ctx.disable(self._moderngl.BLEND)

        data = self.fbo.read(components=4)
        rows = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
        return rows[::-1].tobytes()
