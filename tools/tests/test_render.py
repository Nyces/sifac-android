from __future__ import annotations

import unittest
from types import SimpleNamespace

from sifac.bml import Mesh
from sifac.render import (
    _additive,
    _repair_uvs,
    _uv_scale,
    encode_sheet,
    render_buffer,
    render_textured,
)
from sifac.texture import BtxTexture

from tests.helpers import build_btx


def make_quad() -> Mesh:
    return Mesh(
        name="quad",
        vertex_count=4,
        index_count=6,
        stride=84,
        positions=[(-1.0, 0.0, -1.0), (1.0, 0.0, -1.0), (1.0, 0.0, 1.0), (-1.0, 0.0, 1.0)],
        normals=[(0.0, -1.0, 0.0)] * 4,
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        indices=[0, 1, 2, 0, 2, 3],
    )


def make_grid() -> Mesh:
    """3x3 grid with a planar mapping: u = 0.1 * x, v = 0.1 * z."""
    positions = [(float(x), 0.0, float(z)) for z in range(3) for x in range(3)]
    uvs = [(0.1 * x, 0.1 * z) for z in range(3) for x in range(3)]
    indices = []
    for z in range(2):
        for x in range(2):
            a = z * 3 + x
            indices += [a, a + 1, a + 4, a, a + 4, a + 3]
    return Mesh(
        name="grid",
        vertex_count=9,
        index_count=len(indices),
        stride=48,
        positions=positions,
        normals=[(0.0, 0.0, 1.0)] * 9,
        uvs=uvs,
        indices=indices,
    )


class RenderTests(unittest.TestCase):
    def test_textured_quad(self) -> None:
        pixels = bytes((255, 0, 0, 255)) * 4
        texture = BtxTexture(build_btx("red", 2, 2, 0x00, pixels))
        mesh = make_quad()
        bindings = {0: SimpleNamespace(name="mat", texture="red")}
        png = render_textured([mesh], bindings, {"red": texture}, width=64, height=64)
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_face_stride_renders_once_anchored(self) -> None:
        mesh = make_quad()
        mesh.stride = 60
        png = render_textured([mesh], {}, {}, width=32, height=32)
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_missing_texture_falls_back_to_grey(self) -> None:
        mesh = make_quad()
        png = render_textured([mesh], {}, {}, width=32, height=32)
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_layer_reuse_keeps_base_pixels(self) -> None:
        mesh = make_quad()
        wire = [(mesh, None)]
        pixels, depth = render_buffer(wire, {}, width=16, height=16)
        self.assertEqual(len(pixels), 16 * 16 * 4)
        reused, _ = render_buffer([], {}, width=16, height=16, base=(pixels, depth))
        self.assertEqual(reused, pixels)

    def test_sheet_composes_tiles(self) -> None:
        tile = bytes((255, 0, 0, 255)) * (8 * 8)
        png = encode_sheet([tile, tile, tile], 8, 8, columns=2)
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_uv_repair_restores_planar_mapping(self) -> None:
        mesh = make_grid()
        mesh.uvs[4] = (5000.0, -32000.0)
        repaired = _repair_uvs(mesh)
        self.assertIsNotNone(repaired)
        self.assertAlmostEqual(repaired[4][0], 0.1, places=3)
        self.assertAlmostEqual(repaired[4][1], 0.1, places=3)

    def test_uv_repair_skips_coherent_mesh(self) -> None:
        self.assertIsNone(_repair_uvs(make_grid()))

    def test_uv_scale_is_median(self) -> None:
        mesh = make_grid()
        scale = _uv_scale(mesh, mesh.uvs, 512, 512)
        self.assertIsNotNone(scale)
        # Each grid cell spans 1 world unit and 0.1 UV -> 51.2 texels per unit.
        self.assertAlmostEqual(scale, 51.2, delta=1.0)

    def test_additive_layers_are_recognised(self) -> None:
        self.assertFalse(_additive(None))
        self.assertFalse(_additive(SimpleNamespace(name="wall_mt", texture="s01_walls")))
        self.assertTrue(_additive(SimpleNamespace(name="Glow_Floor_A_mt", texture="x")))
        self.assertTrue(_additive(SimpleNamespace(name="Floor_frame_mt", texture="x")))


if __name__ == "__main__":
    unittest.main()
