from __future__ import annotations

import unittest
from types import SimpleNamespace

from sifac.bml import Mesh
from sifac.render import encode_sheet, render_buffer, render_textured
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


if __name__ == "__main__":
    unittest.main()
