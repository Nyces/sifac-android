from __future__ import annotations

import math
import unittest

from sifac.bml import Mesh, Node
from sifac.skin import Skeleton, auto_weights, deform, invert, mat_mul, transform_point


def node(index: int, name: str, parent: str, translation: tuple[float, float, float]) -> Node:
    return Node(
        index=index,
        name=name,
        parent=parent,
        translation=translation,
        scale=(1.0, 1.0, 1.0),
        rotation=(0.0, 0.0, 0.0),
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(0.0, 0.0, 0.0),
    )


def make_mesh() -> Mesh:
    return Mesh(
        name="test",
        vertex_count=2,
        index_count=3,
        stride=84,
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
        normals=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
        uvs=[(0.0, 0.0), (1.0, 1.0)],
        indices=[0, 1, 0],
    )


class SkinTests(unittest.TestCase):
    def setUp(self) -> None:
        # Root at the origin, two bones stacked along +Y in node space.
        self.skeleton = Skeleton(
            [
                node(0, "Bip001_Hips", "", (0.0, 0.0, 0.0)),
                node(1, "Bip001_Spine", "Bip001_Hips", (0.0, 1.0, 0.0)),
                node(2, "Bip001_Neck", "Bip001_Spine", (0.0, 1.0, 0.0)),
            ]
        )

    def test_segments(self) -> None:
        segments = {s.name: s for s in self.skeleton.segments()}
        self.assertIn("Bip001_Hips", segments)
        self.assertGreater(len(segments), 2)

    def test_weights_normalised(self) -> None:
        weights = auto_weights(make_mesh(), self.skeleton, 2)
        self.assertEqual(len(weights), 2)
        for bones, values in weights:
            self.assertTrue(bones)
            self.assertAlmostEqual(sum(values), 1.0, places=5)

    def test_identity_pose_keeps_positions(self) -> None:
        mesh = make_mesh()
        weights = auto_weights(mesh, self.skeleton, 2)
        posed = deform(mesh.positions, self.skeleton, weights, {})
        for before, after in zip(mesh.positions, posed):
            for a, b in zip(before, after):
                self.assertAlmostEqual(a, b, places=4)

    def test_rotation_moves_vertices(self) -> None:
        mesh = make_mesh()
        weights = auto_weights(mesh, self.skeleton, 2)
        posed = deform(mesh.positions, self.skeleton, weights, {"Bip001_Spine": (0.0, 0.0, math.pi / 2)})
        moved = sum(
            1
            for before, after in zip(mesh.positions, posed)
            if math.dist(before, after) > 1e-4
        )
        self.assertGreater(moved, 0)

    def test_invert_roundtrip(self) -> None:
        m = [[1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 4.0], [0.0, 0.0, 1.0, 5.0], [0.0, 0.0, 0.0, 1.0]]
        inv = invert(m)
        self.assertEqual(transform_point(inv, (3.0, 4.0, 5.0)), (0.0, 0.0, 0.0))
        identity = mat_mul(m, inv)
        for i in range(4):
            self.assertAlmostEqual(identity[i][i], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
