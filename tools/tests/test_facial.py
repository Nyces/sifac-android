from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from sifac.facial import (
    PRESET_STRIDE,
    load_anims,
    load_presets,
    preset_name,
)

from .helpers import build_arc, build_blbn


def preset_record(preset_id: int, layers: list[tuple[int, float]]) -> bytes:
    record = bytearray(PRESET_STRIDE)
    struct.pack_into("<2I", record, 0, preset_id, len(layers))
    for index, (part, weight) in enumerate(layers):
        struct.pack_into("<If", record, 8 + index * 8, part, weight)
    return bytes(record)


def write_common(root: Path) -> Path:
    presets = b"".join(
        [
            preset_record(0, [(0, 1.0)]),
            preset_record(48, [(1, 1.0)]),
            preset_record(61, [(220, 1.0), (101, 0.5)]),
            preset_record(120, [(222, 1.0), (50, 1.0), (51, 1.0)]),
        ]
    )
    anims = b"".join(
        struct.pack("<4I", 1000 + index, index % 3, index % 2, 2 + index)
        for index in range(4)
    )
    preset_blob, _ = build_blbn({"s_facial_preset_datas": presets})
    anim_blob, _ = build_blbn({"s_facial_anim_datas": anims})
    inner = b"\0" * 0x80 + build_arc(
        [
            ("BIN\0", "facial_datas.bin", anim_blob),
            ("BIN\0", "facial_preset_datas.bin", preset_blob),
        ],
        "character_common.pac",
    )
    common = build_arc([("PAC\0", "character_common.pac", inner)], "common.pac")
    path = root / "common" / "common.pac"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(common)
    return path


class FacialTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.common = write_common(Path(self._temp.name))

    def test_reads_every_preset(self) -> None:
        presets = load_presets(self.common)
        self.assertEqual(sorted(presets), [0, 48, 61, 120])
        self.assertEqual(len(presets[61].layers), 2)
        self.assertEqual(presets[61].layers[0].part, 220)
        self.assertAlmostEqual(presets[61].layers[1].weight, 0.5)
        self.assertEqual(len(presets[120].layers), 3)

    def test_preset_name(self) -> None:
        presets = load_presets(self.common)
        self.assertEqual(preset_name(presets[61]), "220 + 101")
        self.assertEqual(preset_name(presets[48]), "1")

    def test_reads_the_anim_table(self) -> None:
        anims = load_anims(self.common)
        self.assertEqual(len(anims), 4)
        self.assertEqual(anims[0].anim_id, 1000)
        self.assertEqual(anims[3].anim_id, 1003)
        self.assertEqual(anims[3].length, 5)


if __name__ == "__main__":
    unittest.main()
