from __future__ import annotations

import math
import unittest

from sifac.bml import Mesh, Node
from sifac.skin import (
    Skeleton,
    anchor_rigid,
    auto_weights,
    build_weights,
    deform,
    invert,
    mat_mul,
    mesh_candidates,
    palette_map,
    place_by_nodes,
    rigid_matrix,
    rig_groups,
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


class PlaceByNodesTests(unittest.TestCase):
    def test_local_matrix_moves_mesh_once(self) -> None:
        skeleton = Skeleton(
            [
                node(0, "root", "", (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2)),
                node(1, "prop", "root", (2.0, 0.0, 0.0)),
            ]
        )
        nodes = list(skeleton.nodes.values())
        mesh = Mesh(
            name="box",
            vertex_count=1,
            index_count=0,
            stride=48,
            positions=[(1.0, 0.0, 0.0)],
            normals=[(0.0, 0.0, 1.0)],
            uvs=[(0.0, 0.0)],
            indices=[],
        )
        placed = place_by_nodes([mesh], {0: 1}, nodes, skeleton)
        # Only the prop's own translation applies, not the root rotation.
        self.assertAlmostEqual(placed[0].positions[0][0], 3.0, places=5)
        self.assertAlmostEqual(placed[0].positions[0][1], 0.0, places=5)

    def test_unmapped_mesh_is_untouched(self) -> None:
        skeleton = Skeleton([node(0, "root", "", (0.0, 0.0, 0.0))])
        mesh = Mesh(
            name="box",
            vertex_count=1,
            index_count=0,
            stride=48,
            positions=[(1.0, 2.0, 3.0)],
            normals=[(0.0, 0.0, 1.0)],
            uvs=[(0.0, 0.0)],
            indices=[],
        )
        placed = place_by_nodes([mesh], {}, list(skeleton.nodes.values()), skeleton)
        self.assertIs(placed[0], mesh)


class WorldAllTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skeleton = Skeleton(
            [
                node(0, "Root", "", (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2)),
                node(1, "Spine", "Root", (1.0, 0.0, 0.0)),
                node(2, "Neck", "Spine", (0.0, 0.0, 1.0), (0.3, 0.0, 0.0)),
                node(3, "Hair_bon_1", "Neck", (0.1, 0.2, 0.3)),
            ]
        )

    def test_matches_world_for_every_node(self) -> None:
        quat = (0.0, 0.0, math.sin(0.4), math.cos(0.4))
        for pose_quat in (None, {"Neck": quat}):
            bulk = self.skeleton.world_all(pose_quat=pose_quat)
            for name in self.skeleton.nodes:
                single = self.skeleton.world(name, pose_quat=pose_quat)
                for row in range(4):
                    for column in range(4):
                        self.assertAlmostEqual(
                            bulk[name][row][column], single[row][column], places=9
                        )

    def test_order_puts_parents_first(self) -> None:
        order = self.skeleton.order()
        seen: set[str] = set()
        for name in order:
            parent = self.skeleton.nodes[name].parent
            if parent:
                self.assertIn(parent, seen)
            seen.add(name)
        self.assertEqual(len(order), len(self.skeleton.nodes))


class RigidMatrixTests(unittest.TestCase):
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

    def apply(self, matrix, positions):
        return [
            (
                matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2] * p[2] + matrix[0][3],
                matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2] * p[2] + matrix[1][3],
                matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2] * p[2] + matrix[2][3],
            )
            for p in positions
        ]

    def test_matches_anchor_rigid(self) -> None:
        quat = (0.0, 0.0, math.sin(0.3), math.cos(0.3))
        meshes = [
            self.mesh(60, [(0.1, 0.0, 0.2), (-0.1, 0.05, 0.3)]),
            self.mesh(44, [(0.01, 0.02, 0.03)]),
        ]
        anchored = anchor_rigid(meshes, {0: 1, 1: 1}, self.nodes, self.skeleton, {"Head": quat})
        world = self.skeleton.world_all(pose_quat={"Head": quat})
        for index, mesh in enumerate(meshes):
            matrix = rigid_matrix(
                index, mesh, {0: 1, 1: 1}, self.nodes, self.skeleton, posed_world=world
            )
            self.assertIsNotNone(matrix)
            moved = self.apply(matrix, mesh.positions)
            for expected, actual in zip(anchored[index].positions, moved):
                for a, b in zip(expected, actual):
                    self.assertAlmostEqual(a, b, places=6)

    def test_skinned_mesh_has_no_rigid_matrix(self) -> None:
        body = self.mesh(84, [(0.0, 0.0, 1.0)])
        self.assertIsNone(
            rigid_matrix(0, body, {0: 1}, self.nodes, self.skeleton)
        )


class BuildWeightsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.skeleton = Skeleton(
            [
                node(0, "Bip001_Hips", "", (0.0, 0.0, 0.0)),
                node(1, "Bip001_Spine", "Bip001_Hips", (0.0, 1.0, 0.0)),
                node(2, "Head_bon_1", "Bip001_Spine", (0.0, 0.2, 0.0)),
            ]
        )

    def test_groups_and_candidates(self) -> None:
        groups = rig_groups(self.skeleton)
        self.assertEqual(groups["biped"], ["Bip001_Hips", "Bip001_Spine"])
        self.assertEqual(groups["hair"], ["Head_bon_1"])
        body = Mesh(name="body", vertex_count=1, index_count=0, stride=84)
        hair = Mesh(name="hair", vertex_count=1, index_count=0, stride=84)
        self.assertEqual(mesh_candidates(groups, body), groups["biped"])
        self.assertEqual(
            mesh_candidates(groups, hair), groups["hair"] + groups["head"]
        )

    def test_rigid_meshes_get_no_weights(self) -> None:
        eye = Mesh(name="eye", vertex_count=1, index_count=0, stride=44)
        face = Mesh(name="face", vertex_count=1, index_count=0, stride=60)
        body = Mesh(
            name="body",
            vertex_count=1,
            index_count=0,
            stride=84,
            positions=[(0.0, 0.0, 1.0)],
        )
        _mapping, weights = build_weights([eye, face, body], self.skeleton, 2)
        self.assertIsNone(weights[0])
        self.assertIsNone(weights[1])
        self.assertIsNotNone(weights[2])
        self.assertAlmostEqual(sum(weights[2][0][1]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
