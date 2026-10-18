"""Live song pack and score (.scd) listing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .arc import ArcArchive, ArcEntry
from .binary import unwrap_section
from .pac import read_pac


class ScoreError(ValueError):
    """Raised when a live pack has no score.pac or a broken score archive."""


@dataclass(frozen=True)
class ScoreFile:
    name: str
    offset: int
    size: int


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
