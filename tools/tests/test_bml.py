from __future__ import annotations

import struct
import unittest

from sifac.bml import BmlArchive, BmlError, render_silhouette

from tests.helpers import _chunk, build_bml_mesh


class BmlTests(unittest.TestCase):
    def test_chunks_and_mesh(self) -> None:
        raw = build_bml_mesh(
            "test_mesh",
            positions=[(0.0, 0.0, 0.0), (0.1, 0.2, 1.5)],
            uvs=[(0.0, 0.0), (0.5, 1.0)],
        )
        archive = BmlArchive(raw)
        self.assertEqual([c.tag for c in archive.chunks][:2], ["BML", "MESH"])
        meshes = archive.meshes()
        self.assertEqual(len(meshes), 1)
        mesh = meshes[0]
        self.assertEqual(mesh.name, "test_mesh")
        self.assertEqual(mesh.vertex_count, 2)
        self.assertEqual(mesh.triangle_count, 1)
        self.assertAlmostEqual(mesh.positions[1][2], 1.5)
        self.assertAlmostEqual(mesh.uvs[1][0], 0.5, places=3)
        self.assertAlmostEqual(mesh.uvs[1][1], 1.0, places=3)
        self.assertEqual(mesh.indices, [0, 1, 0])

    def test_bounds(self) -> None:
        raw = build_bml_mesh(positions=[(-1.0, 0.0, 0.0), (1.0, 1.0, 2.0)])
        mesh = BmlArchive(raw).meshes()[0]
        low, high = mesh.bounds()
        self.assertEqual(low, (-1.0, 0.0, 0.0))
        self.assertEqual(high, (1.0, 1.0, 2.0))

    def test_obj_export(self) -> None:
        raw = build_bml_mesh()
        obj = BmlArchive(raw).to_obj()
        self.assertIn("o test_mesh", obj)
        self.assertIn("\nv ", obj)
        self.assertIn("\nf ", obj)

    def test_silhouette(self) -> None:
        raw = build_bml_mesh()
        meshes = BmlArchive(raw).meshes()
        png = render_silhouette(meshes)
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_nodes_materials_draws(self) -> None:
        node = bytearray(0x68)
        struct.pack_into("<3f", node, 0x10, 1.0, 2.0, 3.0)
        struct.pack_into("<3f", node, 0x1C, 1.0, 1.0, 1.0)
        struct.pack_into("<3f", node, 0x28, 0.1, 0.2, 0.3)
        draw = struct.pack("<4I", 0, 0, 1, 0)
        raw = build_bml_mesh() + _chunk("NODE", bytes(node)) + _chunk("MATE", bytes(0x48)) + _chunk("DRAW", draw)
        archive = BmlArchive(raw)
        nodes = archive.nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].translation, (1.0, 2.0, 3.0))
        self.assertAlmostEqual(nodes[0].rotation[2], 0.3, places=5)
        self.assertEqual(len(archive.materials()), 1)
        calls = archive.draw_calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0].node_index, calls[0].mesh_index, calls[0].material_index), (0, 0, 1))
        self.assertIsNone(archive.node_world_position("missing"))

    def test_mtl_export(self) -> None:
        raw = build_bml_mesh() + _chunk("MATE", bytes(0x48))
        mtl = BmlArchive(raw).to_mtl()
        self.assertIn("newmtl", mtl)

    def test_bad_magic(self) -> None:
        with self.assertRaises(BmlError):
            BmlArchive(b"nope")


if __name__ == "__main__":
    unittest.main()
