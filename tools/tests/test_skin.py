from __future__ import annotations

import math
import unittest

from sifac.bml import Mesh, Node
from sifac.skin import (
    Skeleton,
    anchor_rigid,
    auto_weights,
    deform,
    invert,
    mat_mul,
    palette_map,
    transform_point,
    weights_from_palette,
)


def node(
    index: int,
    name: str,
    parent: str,
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Node:
    return Node(
        index=index,
        name=name,
        parent=parent,
        translation=translation,
        scale=(1.0, 1.0, 1.0),
        rotation=rotation,
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


class BasisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skeleton = Skeleton(
            [node(0, "Bone", "", (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2))]
        )
        self.absolute = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]

    def test_absolute_ignores_bind_rotation(self) -> None:
        local = self.skeleton.local("Bone", absolute=self.absolute, basis="absolute")
        point = transform_point(local, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(point[0], 1.0, places=5)
        self.assertAlmostEqual(point[1], 0.0, places=5)

    def test_post_applies_bind_rotation(self) -> None:
        local = self.skeleton.local("Bone", absolute=self.absolute, basis="post")
        point = transform_point(local, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(point[0], 0.0, places=5)
        self.assertAlmostEqual(point[1], 1.0, places=5)

    def test_quaternion_pose_rotates_rigidly(self) -> None:
        skeleton = Skeleton(
            [node(0, "Bone", "", (0.0, 0.0, 0.0))]
        )
        position = (1.0, 0.0, 1.0)
        weights = [(["Bone"], [1.0])]
        quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        posed = deform([position], skeleton, weights, None, {"Bone": quat})
        pivot = (0.0, 0.0, 0.0)
        self.assertAlmostEqual(math.dist(posed[0], pivot), math.dist(position, pivot), places=5)
        self.assertGreater(math.dist(posed[0], position), 1e-3)


class PaletteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skeleton = Skeleton(
            [
                node(0, "Bip001_Hips", "", (0.0, 0.0, 0.0)),
                node(1, "Bip001_Spine", "Bip001_Hips", (0.0, 1.0, 0.0)),
                node(2, "Bip001_Neck", "Bip001_Spine", (0.0, 1.0, 0.0)),
            ]
        )
        self.mesh = Mesh(
            name="palette",
            vertex_count=2,
            index_count=0,
            stride=84,
            positions=[(0.0, 0.0, 0.5), (0.0, 0.0, 1.5)],
            normals=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
            uvs=[(0.0, 0.0), (1.0, 1.0)],
            indices=[],
            bone_indices=[(1, 0, 0, 0), (2, 0, 0, 0)],
        )

    def test_palette_map_recovers_bones(self) -> None:
        mapping = palette_map([self.mesh], self.skeleton)
        self.assertEqual(mapping[1], "Bip001_Hips")
        self.assertEqual(mapping[2], "Bip001_Spine")

    def test_weights_from_palette_uses_vertex_indices(self) -> None:
        mapping = {1: "Bip001_Hips", 2: "Bip001_Spine"}
        weights = weights_from_palette(self.mesh, self.skeleton, mapping)
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0][0], ["Bip001_Hips"])
        self.assertEqual(weights[1][0], ["Bip001_Spine"])
        for _bones, values in weights:
            self.assertAlmostEqual(sum(values), 1.0, places=5)


class AnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skeleton = Skeleton(
            [
                node(0, "Root", "", (0.0, 0.0, 0.0)),
                node(1, "Head", "Root", (0.0, 1.4, 0.0)),
            ]
        )
        self.nodes = list(self.skeleton.nodes.values())

    def mesh(self, stride: int, positions: list[tuple[float, float, float]]) -> Mesh:
        return Mesh(
            name=f"m{stride}",
            vertex_count=len(positions),
            index_count=0,
            stride=stride,
            positions=positions,
            normals=[(0.0, 0.0, 1.0) for _ in positions],
            uvs=[(0.0, 0.0) for _ in positions],
            indices=[],
        )

    def test_face_mesh_only_moves_by_node_origin(self) -> None:
        face = self.mesh(60, [(0.1, 0.0, 0.2), (-0.1, 0.05, 0.3)])
        placed = anchor_rigid([face], {0: 1}, self.nodes, self.skeleton)
        # Node translation (0, 1.4, 0) in node space becomes (0, 0, 1.4) in mesh space.
        self.assertAlmostEqual(placed[0].positions[0][2], 1.6, places=5)
        self.assertAlmostEqual(placed[0].positions[0][0], 0.1, places=5)
        self.assertAlmostEqual(placed[0].positions[1][2], 1.7, places=5)

    def test_eye_mesh_uses_full_matrix(self) -> None:
        eye = self.mesh(44, [(0.01, 0.02, 0.03)])
        placed = anchor_rigid([eye], {0: 1}, self.nodes, self.skeleton)
        self.assertAlmostEqual(placed[0].positions[0][2], 1.43, places=5)

    def test_skinned_mesh_is_untouched(self) -> None:
        body = self.mesh(84, [(0.0, 0.0, 1.0)])
        placed = anchor_rigid([body], {0: 1}, self.nodes, self.skeleton)
        self.assertIs(placed[0], body)

    def test_unmapped_mesh_is_untouched(self) -> None:
        face = self.mesh(60, [(0.0, 0.0, 0.0)])
        placed = anchor_rigid([face], {}, self.nodes, self.skeleton)
        self.assertIs(placed[0], face)


if __name__ == "__main__":
    unittest.main()
