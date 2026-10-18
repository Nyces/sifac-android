"""Live song pack and score (.scd) parsing.

Score files (``score_<pack>_<difficulty>_<song>_ch<N>.scd``) start with an
0x80-byte header, then a ``Scor`` block::

    0x84  u32 total size
    0x88  u32 record count (includes a trailing all-zero record)
    0x8c  u32 tempo-related constant (0x45c12 in the retail dump)
    0x90  u32 ticks per beat (480)
    0x94  u32
    0x98  u32
    0x9c  u32 channel index
    0xa0  u32
    0xe0  records, 32 bytes each:
            u32 id
            u32 tick
            u32 lane_raw   (0x37 + lane, lanes 0..8)
            u32 constant   (100)
            u32 value      (note length / parameter in ticks)
            u32 channel
            u32 reserved
            u32 reserved

Channel 1 holds the playable notes for every difficulty; channels 2..5 keep
9/9/9/11 constant records that do not change with difficulty, and channels 6..7
are empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .arc import ArcArchive, ArcEntry
from .binary import unwrap_section
from .pac import read_pac

SCORE_MAGIC = b"Scor"
HEADER_SIZE = 0x80
RECORD_START = 0xE0
RECORD_SIZE = 0x20
TICKS_PER_BEAT = 480
LANE_BASE = 0x37
LANE_COUNT = 9

DIFFICULTY_NAMES = {
    "00": "easy",
    "01": "normal",
    "02": "hard",
    "03": "expert",
    "04": "master",
    "10": "normal-alt",
    "11": "hard-alt",
    "12": "expert-alt",
    "13": "master-alt",
}


class ScoreError(ValueError):
    """Raised when a live pack has no score.pac or a broken score archive."""


@dataclass(frozen=True)
class ScoreFile:
    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class Note:
    id: int
    tick: int
    lane: int
    lane_raw: int
    value: int
    channel: int

    @property
    def beat(self) -> float:
        return self.tick / TICKS_PER_BEAT


@dataclass(frozen=True)
class ScoreHeader:
    size: int
    record_count: int
    tempo_constant: int
    ticks_per_beat: int
    channel: int


def parse_score_name(name: str) -> tuple[str, str, str] | None:
    """Split ``score_0010_04_bokurara_ch1.scd`` into (pack, difficulty, channel)."""
    match = re.match(r"score_(\d+)_(\d+)_(.+)_ch(\d+)\.scd$", name, re.IGNORECASE)
    if not match:
        return None
    pack_id, difficulty, title, channel = match.groups()
    return pack_id, difficulty, channel


class ScoreChart:
    def __init__(self, name: str, data: bytes) -> None:
        if len(data) < RECORD_START:
            raise ScoreError(f"score file is too small: {name}")
        if data[HEADER_SIZE : HEADER_SIZE + 4] != SCORE_MAGIC:
            raise ScoreError(f"missing Scor magic: {name}")
        self.name = name
        self.data = data
        self.header = ScoreHeader(
            size=int.from_bytes(data[0x84:0x88], "little"),
            record_count=int.from_bytes(data[0x88:0x8C], "little"),
            tempo_constant=int.from_bytes(data[0x8C:0x90], "little"),
            ticks_per_beat=int.from_bytes(data[0x90:0x94], "little") or TICKS_PER_BEAT,
            channel=int.from_bytes(data[0x9C:0xA0], "little"),
        )
        parsed = parse_score_name(name)
        self.pack_id = parsed[0] if parsed else ""
        self.difficulty = parsed[1] if parsed else ""
        self.channel_name = parsed[2] if parsed else ""
        self.notes = tuple(self._read_notes())

    def _read_notes(self) -> list[Note]:
        notes: list[Note] = []
        available = (len(self.data) - RECORD_START) // RECORD_SIZE
        count = min(self.header.record_count, available)
        for index in range(count):
            offset = RECORD_START + index * RECORD_SIZE
            note_id, tick, lane_raw, _constant, value, channel = (
                int.from_bytes(self.data[offset + i * 4 : offset + i * 4 + 4], "little")
                for i in range(6)
            )
            if not any((note_id, tick, lane_raw, value, channel)):
                continue  # trailing zero record
            lane = lane_raw - LANE_BASE
            if not 0 <= lane < LANE_COUNT:
                lane = -1
            notes.append(Note(note_id, tick, lane, lane_raw, value, channel))
        return notes

    @property
    def difficulty_name(self) -> str:
        return DIFFICULTY_NAMES.get(self.difficulty, self.difficulty)

    def to_csv(self) -> str:
        lines = ["id,tick,beat,lane,value,channel"]
        for note in self.notes:
            lines.append(
                f"{note.id},{note.tick},{note.beat:.4f},{note.lane},{note.value},{note.channel}"
            )
        return "\n".join(lines) + "\n"


class LivePack:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        image = read_pac(self.path)
        self.pack = ArcArchive(image.data)
        score_entry = self.pack.find("score.pac")
        if score_entry is None:
            raise ScoreError(f"score.pac not found in {self.path}")
        score_bytes, _ = unwrap_section(self.pack.entry_data(score_entry))
        self.score_arc = ArcArchive(score_bytes)
        self.scores = tuple(
            ScoreFile(entry.name, entry.offset, entry.size)
            for entry in self.score_arc.entries
            if entry.name.lower().endswith(".scd")
        )

    def entry(self, score: ScoreFile | str) -> ArcEntry:
        name = score.name if isinstance(score, ScoreFile) else score
        entry = self.score_arc.find(name)
        if entry is None:
            raise ScoreError(f"score not found: {name}")
        return entry

    def score_data(self, score: ScoreFile | str, unwrap: bool = False) -> bytes:
        data = self.score_arc.entry_data(self.entry(score))
        if unwrap:
            data, _ = unwrap_section(data)
        return data

    def extract(self, out_dir: str | Path, unwrap: bool = False) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for score in self.scores:
            target = out_dir / _safe_name(score.name)
            target.write_bytes(self.score_data(score, unwrap=unwrap))
            written.append(target)
        return written

    def chart(self, difficulty: str = "04", channel: str = "1") -> ScoreChart:
        """Load one channel of one difficulty, e.g. ``chart("04", "1")``."""
        suffix = f"_{difficulty}_"
        wanted = f"_ch{channel}.scd"
        for score in self.scores:
            if suffix in score.name and score.name.endswith(wanted) and "lyrics" not in score.name:
                return ScoreChart(score.name, self.score_data(score))
        raise ScoreError(f"chart not found: difficulty {difficulty}, channel {channel}")

    def charts(self, channel: str = "1") -> list[ScoreChart]:
        wanted = f"_ch{channel}.scd"
        charts = []
        for score in self.scores:
            if score.name.endswith(wanted) and "lyrics" not in score.name:
                charts.append(ScoreChart(score.name, self.score_data(score)))
        return charts



def _safe_name(name: str) -> str:
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)


def iter_live_packs(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    live = root / "live"
    if not live.is_dir():
        return
    for path in sorted(live.rglob("*.pac")):
        if path.is_file():
            yield path
