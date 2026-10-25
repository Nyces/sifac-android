"""Software textured preview for posed Blade meshes.

The body and hair meshes store world-space vertices and skin normally. Eye
meshes (stride 44) are stored in their node's local space, so they are placed
with that node's world matrix. The face mesh (stride 60) belongs to the game's
separate face pass (RenderDoc event 90 renders it into its own target), so it is
skipped instead of being drawn in the wrong space.
"""

from __future__ import annotations

from .texture import BtxTexture, encode_png


def _screen_bounds(meshes: list) -> tuple[float, float, float, float]:
    points = [p for mesh in meshes for p in mesh.positions]
    if not points:
        raise ValueError("no vertices to render")
    return (
        min(p[0] for p in points),
        max(p[0] for p in points),
        min(p[2] for p in points),
        max(p[2] for p in points),
    )


def render_textured(
    meshes: list,
    bindings: dict,
    textures: dict[str, BtxTexture],
    width: int = 512,
    height: int = 768,
    skip_strides: tuple[int, ...] = (60,),
) -> bytes:
    """Rasterise meshes with their btx textures and a simple directional light."""
    min_x, max_x, min_z, max_z = _screen_bounds(meshes)
    span_x = max(max_x - min_x, 1e-6)
    span_z = max(max_z - min_z, 1e-6)
    scale = min((width - 32) / span_x, (height - 32) / span_z)
    origin_x = min_x
    origin_z = min_z

    pixels = bytearray(width * height * 4)
    for i in range(width * height):
        pixels[i * 4 : i * 4 + 4] = bytes((16, 18, 24, 255))
    depth = [float("inf")] * (width * height)

    light = (-0.4, -0.85, 0.35)
    length = sum(component * component for component in light) ** 0.5
    light = tuple(component / length for component in light)

    decoded: dict[str, tuple[int, int, bytes]] = {}

    def texture_for(mesh_index: int) -> tuple[int, int, bytes] | None:
        material = bindings.get(mesh_index)
        if material is None:
            return None
        key = getattr(material, "texture", "") or getattr(material, "name", "")
        if not key:
            return None
        if key not in decoded:
            texture = textures.get(key)
            if texture is None:
                return None
            try:
                decoded[key] = texture.to_rgba()
            except Exception:
                return None
        return decoded[key]

    for mesh_index, mesh in enumerate(meshes):
        if mesh.stride in skip_strides or not mesh.indices:
            continue
        canvas = texture_for(mesh_index)
        for triangle in range(mesh.triangle_count):
            a, b, c = mesh.indices[triangle * 3 : triangle * 3 + 3]
            if a >= len(mesh.positions) or b >= len(mesh.positions) or c >= len(mesh.positions):
                continue
            pa, pb, pc = mesh.positions[a], mesh.positions[b], mesh.positions[c]
            ax = (pa[0] - origin_x) * scale + 16
            bx = (pb[0] - origin_x) * scale + 16
            cx = (pc[0] - origin_x) * scale + 16
            ay = height - 1 - ((pa[2] - origin_z) * scale + 16)
            by = height - 1 - ((pb[2] - origin_z) * scale + 16)
            cy = height - 1 - ((pc[2] - origin_z) * scale + 16)
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-9:
                continue
            min_px = max(0, int(min(ax, bx, cx)))
            max_px = min(width - 1, int(max(ax, bx, cx)) + 1)
            min_py = max(0, int(min(ay, by, cy)))
            max_py = min(height - 1, int(max(ay, by, cy)) + 1)
            if min_px > max_px or min_py > max_py:
                continue
            if mesh.uvs:
                ua, ub, uc = mesh.uvs[a], mesh.uvs[b], mesh.uvs[c]
            else:
                ua = ub = uc = (0.0, 0.0)
            if mesh.normals:
                na, nb, nc = mesh.normals[a], mesh.normals[b], mesh.normals[c]
            else:
                na = nb = nc = (0.0, 0.0, 1.0)
            inv_area = 1.0 / area
            for py in range(min_py, max_py + 1):
                fy = py + 0.5
                for px in range(min_px, max_px + 1):
                    fx = px + 0.5
                    w0 = ((bx - ax) * (fy - ay) - (by - ay) * (fx - ax)) * inv_area
                    w1 = ((cx - bx) * (fy - by) - (cy - by) * (fx - bx)) * inv_area
                    w2 = 1.0 - w0 - w1
                    if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                        continue
                    z = pa[1] * w0 + pb[1] * w1 + pc[1] * w2
                    index = py * width + px
                    if z >= depth[index]:
                        continue
                    if canvas is None:
                        color = (180, 180, 180, 255)
                    else:
                        tw, th, data = canvas
                        u = ua[0] * w0 + ub[0] * w1 + uc[0] * w2
                        v = 1.0 - (ua[1] * w0 + ub[1] * w1 + uc[1] * w2)
                        tx = min(tw - 1, max(0, int(u * tw)))
                        ty = min(th - 1, max(0, int(v * th)))
                        offset = (ty * tw + tx) * 4
                        color = tuple(data[offset : offset + 4])
                    if color[3] < 96 and canvas is not None:
                        continue
                    nx = na[0] * w0 + nb[0] * w1 + nc[0] * w2
                    ny = na[1] * w0 + nb[1] * w1 + nc[1] * w2
                    nz = na[2] * w0 + nb[2] * w1 + nc[2] * w2
                    normal_length = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
                    lambert = abs(
                        (nx * light[0] + ny * light[1] + nz * light[2]) / normal_length
                    )
                    shade = 0.72 + 0.28 * min(1.0, lambert)
                    depth[index] = z
                    pixels[index * 4 : index * 4 + 4] = bytes(
                        (
                            min(255, int(color[0] * shade)),
                            min(255, int(color[1] * shade)),
                            min(255, int(color[2] * shade)),
                            color[3],
                        )
                    )
    return encode_png(width, height, bytes(pixels))
