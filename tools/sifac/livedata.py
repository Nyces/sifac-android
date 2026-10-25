"""Song/live/stage tables that ``common.pac`` feeds into ``eboot``.

``common.pac → live_common.pac → live_datas.bin`` is a BLBN container with two
arrays:

* ``s_music_datas`` 125 x 0x340 bytes, first dword is the music id;
* ``s_live_datas`` 1125 x 0x388 bytes, nine records per song.

``live_datas.bin``'s live record layout (verified against ``sub_94590`` and the
song-select builder at 0x113880):

* ``+0x00`` live id (``music_id * 1000 + variant``);
* ``+0x04`` music id, the foreign key into ``s_music_datas``;
* ``+0x08`` variant (0..4 and 10..13, the chart suffix in ``score.pac``);
* ``+0x14`` slot 0..8, the order the song-select list uses;
* ``+0x18`` stage id, see below;
* ``+0x28`` flag (1 on some slots).

The stage id indexes ``stage_datas.bin`` (``stage_common.pac``) and, through
``sub_28D9F0``'s ``"stage_%03d"``, the ``stage/st%03d.pac`` package (the path is
literal in the eboot rodata at 0x5FDF10). The stage record holds 4 language
names (Japanese, English, Simplified Chinese, Traditional Chinese) as offsets
into the same BLBN container.

The eboot rodata at 0x5EDB75 holds nine adjacent labels ("AC EASY", "AC
NORMAL", "AC HARD", "AC EXTREME", "AC CHALLENGE", "BIGINNER", "STANDARD",
"HYPER", "ADVENTURE") that match the nine chart suffixes, and the stage id
rises with the slot, so harder charts get the grander venues (song 0010 goes
from the school gym to the grand castle).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .arc import ArcArchive
from .binary import unwrap_section
from .blbn import BlbnFile
from .pac import read_pac

MUSIC_STRIDE = 0x340
LIVE_STRIDE = 0x388
STAGE_STRIDE = 0x148
STAGE_LANGUAGES = ("ja", "en", "zh-Hans", "zh-Hant")


@dataclass(frozen=True)
class LiveRecord:
    live_id: int
    music_id: int
    variant: int
    slot: int
    stage_id: int
    flag: int


@dataclass(frozen=True)
class StageRecord:
    stage_id: int
    base_id: int
    names: tuple[str, ...]

    def name(self, language: str = "en") -> str:
        try:
            index = STAGE_LANGUAGES.index(language)
        except ValueError:
            index = 1
        if 0 <= index < len(self.names) and self.names[index]:
            return self.names[index]
        return next((text for text in self.names if text), "")


def _nested_entry(path: Path, *names: str) -> bytes:
    data = unwrap_section(read_pac(path).data)[0]
    for name in names[:-1]:
        archive = ArcArchive(data)
        entry = archive.find(name)
        if entry is None:
            raise KeyError(f"{path.name} has no {name!r}")
        data = unwrap_section(archive.entry_data(entry))[0]
    archive = ArcArchive(data)
    entry = archive.find(names[-1])
    if entry is None:
        raise KeyError(f"{path.name} has no {names[-1]!r}")
    return archive.entry_data(entry)


def load_live_records(path: Path) -> list[LiveRecord]:
    """Read ``s_live_datas`` from a dump's ``common/common.pac``."""
    container = BlbnFile(_nested_entry(path, "live_common.pac", "live_datas.bin"))
    table = container.read("s_live_datas")
    records: list[LiveRecord] = []
    for offset in range(0, len(table) - LIVE_STRIDE + 1, LIVE_STRIDE):
        live_id, music_id, variant = struct.unpack_from("<3I", table, offset)
        slot = struct.unpack_from("<I", table, offset + 0x14)[0]
        stage_id = struct.unpack_from("<I", table, offset + 0x18)[0]
        flag = struct.unpack_from("<I", table, offset + 0x28)[0]
        if not music_id or not stage_id:
            continue
        records.append(LiveRecord(live_id, music_id, variant, slot, stage_id, flag))
    return records


def stage_id_for(records: list[LiveRecord], music_id: int, slot: int = 0) -> int | None:
    """Stage id the game uses for ``music_id`` at the given song-select slot."""
    for record in records:
        if record.music_id == music_id and record.slot == slot:
            return record.stage_id
    return None


def load_stage_records(path: Path) -> dict[int, StageRecord]:
    """Read ``s_stage_datas`` (id, base variant, 4 language names) by id."""
    container = BlbnFile(_nested_entry(path, "stage_common.pac", "stage_datas.bin"))
    table = container.read("s_stage_datas")
    records: dict[int, StageRecord] = {}
    for offset in range(0, len(table) - STAGE_STRIDE + 1, STAGE_STRIDE):
        stage_id, base_id = struct.unpack_from("<2I", table, offset)
        if not stage_id:
            continue
        names = tuple(
            container.text(struct.unpack_from("<Q", table, offset + 8 + index * 8)[0])
            for index in range(len(STAGE_LANGUAGES))
        )
        records[stage_id] = StageRecord(stage_id, base_id, names)
    return records
