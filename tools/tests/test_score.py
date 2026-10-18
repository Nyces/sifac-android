from __future__ import annotations

import unittest

from sifac.chart import render_chart
from sifac.score import ScoreChart, ScoreError, parse_score_name

from tests.helpers import build_scd


class ScoreTests(unittest.TestCase):
    def test_parse_name(self) -> None:
        self.assertEqual(
            parse_score_name("score_0010_04_bokurara_ch1.scd"),
            ("0010", "04", "1"),
        )
        self.assertIsNone(parse_score_name("score_scene_0010_00_ch1.scd"))
        self.assertIsNone(parse_score_name("random.pac"))

    def test_notes_and_lanes(self) -> None:
        raw = build_scd(
            [
                (2, 0, 0x37, 100, 120, 1),
                (3, 480, 0x38, 100, 960, 1),
                (4, 960, 0x3F, 100, 120, 1),
            ]
        )
        chart = ScoreChart("score_0010_04_bokurara_ch1.scd", raw)
        self.assertEqual(chart.difficulty, "04")
        self.assertEqual(chart.difficulty_name, "master")
        self.assertEqual(len(chart.notes), 3)
        self.assertEqual([n.lane for n in chart.notes], [0, 1, 8])
        self.assertEqual(chart.notes[1].beat, 1.0)
        self.assertEqual(chart.notes[2].lane_raw, 0x3F)

    def test_zero_record_is_skipped(self) -> None:
        raw = build_scd([(2, 0, 0x37, 100, 120, 1)], channel=1)
        chart = ScoreChart("score_0010_00_x_ch1.scd", raw)
        self.assertEqual(len(chart.notes), 1)

    def test_csv(self) -> None:
        raw = build_scd([(2, 960, 0x37, 100, 120, 1)])
        chart = ScoreChart("score_0010_00_x_ch1.scd", raw)
        text = chart.to_csv()
        self.assertTrue(text.startswith("id,tick,beat,lane"))
        self.assertIn("2,960,2.0000,0,120,1", text)

    def test_bad_magic(self) -> None:
        with self.assertRaises(ScoreError):
            ScoreChart("score_0010_00_x_ch1.scd", b"\0" * 0x200)

    def test_render_png(self) -> None:
        raw = build_scd(
            [(2 + i, i * 480, 0x37 + (i % 9), 100, 120, 1) for i in range(40)]
        )
        chart = ScoreChart("score_0010_04_x_ch1.scd", raw)
        png = render_chart(chart)
        self.assertTrue(png.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
