from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from sifac.livedata import (
    LIVE_STRIDE,
    STAGE_STRIDE,
    load_live_records,
    load_stage_records,
    stage_id_for,
)

from .helpers import build_arc, build_blbn


def _live_table(records: list[tuple[int, int, int, int, int, int]]) -> bytes:
    table = bytearray()
    for live_id, music_id, variant, slot, stage_id, flag in records:
        record = bytearray(LIVE_STRIDE)
        struct.pack_into("<3I", record, 0x00, live_id, music_id, variant)
        struct.pack_into("<2I", record, 0x14, slot, stage_id)
        struct.pack_into("<I", record, 0x28, flag)
        table += record
    blob, _ = build_blbn({"s_live_datas": bytes(table)})
    return blob


def _stage_table(records: list[tuple[int, int, list[str]]]) -> bytes:
    strings = [text for _id, _base, names in records for text in names]
    # String offsets do not depend on the table payload, so build once with a
    # placeholder to learn them and once with the real records.
    _blob, offsets = build_blbn({"s_stage_datas": bytes(4)}, strings=strings)
    table = bytearray()
    for stage_id, base_id, names in records:
        record = bytearray(STAGE_STRIDE)
        struct.pack_into("<2I", record, 0, stage_id, base_id)
        struct.pack_into("<4Q", record, 8, *(offsets[text] for text in names))
        table += record
    blob, _ = build_blbn({"s_stage_datas": bytes(table)}, strings=strings)
    return blob


def _write_dump(root: Path) -> Path:
    live_blob = _live_table(
        [
            (10000, 10, 0, 0, 2, 0),
            (10001, 10, 1, 1, 4, 0),
            (20000, 20, 0, 0, 1, 0),
        ]
    )
    stage_blob = _stage_table(
        [
            (1, 1, ["ステージ・オブ・ライブ", "Live Show Stage", "演唱会舞台", "演唱會舞台"]),
            (2, 2, ["学院体育館", "School Gym", "学校体育馆", "學院體育館"]),
            (4, 4, ["歩行者天国", "Pedestrian Mall", "步行者天国", "行人徒步區"]),
        ]
    )
    live_common = b"\0" * 0x80 + build_arc(
        [("BIN\0", "live_datas.bin", live_blob)], "live_common.pac"
    )
    stage_common = b"\0" * 0x80 + build_arc(
        [("BIN\0", "stage_datas.bin", stage_blob)], "stage_common.pac"
    )
    common = build_arc(
        [
            ("PAC\0", "live_common.pac", live_common),
            ("PAC\0", "stage_common.pac", stage_common),
        ],
        "common.pac",
    )
    path = root / "common" / "common.pac"
    path.parent.mkdir(parents=True)
    path.write_bytes(common)
    return path


class LiveDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.common = _write_dump(Path(self._temp.name))

    def test_reads_the_live_records(self) -> None:
        records = load_live_records(self.common)
        self.assertEqual([record.music_id for record in records], [10, 10, 20])
        self.assertEqual([record.slot for record in records], [0, 1, 0])
        self.assertEqual(records[0].live_id, 10000)
        self.assertEqual(records[1].variant, 1)

    def test_stage_id_for_slot(self) -> None:
        records = load_live_records(self.common)
        self.assertEqual(stage_id_for(records, 10, 0), 2)
        self.assertEqual(stage_id_for(records, 10, 1), 4)
        self.assertEqual(stage_id_for(records, 20, 0), 1)
        self.assertIsNone(stage_id_for(records, 10, 8))

    def test_reads_the_stage_names(self) -> None:
        stages = load_stage_records(self.common)
        self.assertEqual(sorted(stages), [1, 2, 4])
        self.assertEqual(stages[2].name("ja"), "学院体育館")
        self.assertEqual(stages[2].name("en"), "School Gym")
        self.assertEqual(stages[2].name("zh-Hant"), "學院體育館")
        self.assertEqual(stages[4].base_id, 4)

    def test_stage_name_falls_back_to_the_first_language(self) -> None:
        stages = load_stage_records(self.common)
        self.assertEqual(stages[1].name("xx"), "Live Show Stage")


if __name__ == "__main__":
    unittest.main()
