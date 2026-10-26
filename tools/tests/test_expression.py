from __future__ import annotations

import struct
import unittest

from sifac.bml import Mesh
from sifac.expression import (
    ARRAY_STRIDE,
    ExpressionError,
    decode,
    expression_suffix,
    expressions_from_textures,
    find_expression,
)
from sifac.texture import BtxTexture

from tests.helpers import build_btx


def build_expression(
    positions: list[tuple[float, float, float]],
    normals: list[tuple[float, float, float]] | None = None,
    masks: list[float] | None = None,
    name: str = "Test_face_eye_open",
) -> bytes:
    """128x64 RGBA16F expression map: positions in [0,4096), normals after."""
    normals = normals or [(0.0, 0.0, 0.0)] * len(positions)
    masks = masks or [1.0] * len(positions)
    payload = bytearray(ARRAY_STRIDE * 2 * 8)
    for index, (position, normal, mask) in enumerate(zip(positions, normals, masks)):
        struct.pack_into("<4e", payload, index * 8, *position, mask)
        struct.pack_into("<4e", payload, (ARRAY_STRIDE + index) * 8, *normal, 1.0)
    return build_btx(name, 128, 64, 0x1B, bytes(payload))


def mesh(name: str, positions: list[tuple[float, float, float]]) -> Mesh:
    return Mesh(
        name=name,
        vertex_count=len(positions),
        index_count=0,
        stride=60,
        positions=positions,
        normals=[(0.0, 0.0, 1.0) for _ in positions],
        uvs=[(0.0, 0.0) for _ in positions],
        indices=[],
    )


class DecodeTests(unittest.TestCase):
    def test_reads_the_two_arrays(self) -> None:
        raw = build_expression(
            [(0.01, 0.02, 0.03), (0.0, 0.0, 0.0)],
            [(0.0, 0.5, 0.0), (0.0, 0.0, 0.0)],
        )
        expression = decode("Test_face_eye_open", BtxTexture(raw))
        self.assertEqual(len(expression), ARRAY_STRIDE)
        self.assertAlmostEqual(expression.positions[0][0], 0.01, places=4)
        self.assertAlmostEqual(expression.positions[0][2], 0.03, places=4)
        self.assertEqual(expression.positions[1], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(expression.normals[0][1], 0.5, places=3)

    def test_mask_zero_means_no_delta(self) -> None:
        raw = build_expression([(0.4, 0.4, 0.4)], masks=[0.0])
        expression = decode("Test_face_eye_open", BtxTexture(raw))
        self.assertEqual(expression.positions[0], (0.0, 0.0, 0.0))

    def test_rejects_other_formats(self) -> None:
        raw = build_btx("plain", 128, 64, 0x00, bytes(128 * 64 * 4))
        with self.assertRaises(ExpressionError):
            decode("plain", BtxTexture(raw))


class ApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = build_expression(
            [(0.01, 0.0, 0.0), (0.0, 0.02, 0.0)],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        )
        self.expression = decode("Test_face_eye_open", BtxTexture(raw))

    def test_adds_deltas_to_a_short_mesh(self) -> None:
        source = mesh("face", [(0.0, 0.0, 1.5), (0.0, 0.0, 1.6)])
        morphed = self.expression.apply(source)
        self.assertAlmostEqual(morphed.positions[0][0], 0.01, places=4)
        self.assertAlmostEqual(morphed.positions[0][2], 1.5, places=4)
        self.assertAlmostEqual(morphed.positions[1][1], 0.02, places=4)
        # The normal delta is added and renormalised.
        self.assertAlmostEqual(morphed.normals[1][2], 1.0, places=4)
        self.assertIsNot(morphed, source)
        self.assertEqual(source.positions[0], (0.0, 0.0, 1.5))

    def test_result_is_cached_per_mesh(self) -> None:
        source = mesh("face", [(0.0, 0.0, 1.5)])
        self.assertIs(self.expression.apply(source), self.expression.apply(source))

    def test_too_many_vertices_is_an_error(self) -> None:
        big = mesh("face", [(0.0, 0.0, 1.5)] * (ARRAY_STRIDE + 1))
        with self.assertRaises(ExpressionError):
            self.expression.apply(big)


class LookupTests(unittest.TestCase):
    def test_suffix_and_lookup_are_case_insensitive(self) -> None:
        raw = build_expression([(0.0, 0.0, 0.0)], name="Rin_face_Default")
        table = expressions_from_textures(
            {"Rin_face_Default": BtxTexture(raw)}
        )
        self.assertEqual(expression_suffix("Rin_face_Default"), "Default")
        self.assertIsNotNone(find_expression(table, "default"))
        self.assertIsNone(find_expression(table, "eye_blink"))

    def test_non_expression_textures_are_skipped(self) -> None:
        plain = build_btx("plain", 128, 64, 0x00, bytes(128 * 64 * 4))
        table = expressions_from_textures({"plain": BtxTexture(plain)})
        self.assertEqual(table, {})


if __name__ == "__main__":
    unittest.main()
