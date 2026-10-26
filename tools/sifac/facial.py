"""Facial preset table that explains the character timeline's cue idents.

``common/character_common.pac`` holds two BLBN files:

``facial_datas.bin``
    ``s_facial_anim_datas``, 54 records of 16 bytes: ``(id, a, b, c)`` with
    ids 1000-1400 grouped by hundred.

``facial_preset_datas.bin``
    ``s_facial_preset_datas``, 62 records of **72 bytes**::

        +0x00 u32 preset id        0, 48..56, 60..65, 72..83, 96..98,
                                   100, 120..122
        +0x04 u32 layer count      1..3
        +0x08 layers, ``u32 part id, f32 weight`` per layer

    Verified by the stride check (1116 dwords / 18 = 62 exactly) and by every
    record starting with an id from that list followed by a small layer count
    and ``1.0`` weights.

That id namespace is exactly what the character timeline's cue carries in its
``ident`` field: the 34 idents used across the whole dump (0, 48..56, 60..65,
72..83, 96..98, 100, 120..122) are all preset ids here, and songs that share a
cue ident therefore play the same facial preset. So a cue means "hold facial
preset *N* from tick *t* for *len* ticks", which is why the intervals are shared
by all nine performers while the ident is what each performer plays.

Example presets: 61 = [220, 101], 62 = [200, 102], 120 = [222, 50, 51].

The layer part ids (1..11, 50/51, 100..102, 200..201, 210..211, 220..222,
231..232, 241, 250) are a namespace we have not located a table for yet, so
this module stops at decoding the presets; the morph textures themselves are
reached through :mod:`sifac.expression` by name.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .arc import ArcArchive
from .binary import unwrap_section
from .blbn import BlbnFile
from .pac import read_pac

PRESET_STRIDE = 72
ANIM_STRIDE = 16
MAX_LAYERS = 4


@dataclass(frozen=True)
class FacialLayer:
    part: int
    weight: float


@dataclass(frozen=True)
class FacialPreset:
    """One cue ident: a weighted stack of facial parts."""

    preset_id: int
    layers: tuple[FacialLayer, ...]


@dataclass(frozen=True)
class FacialAnim:
    anim_id: int
    count: int
    kind: int
    length: int


def _character_common(path: Path) -> ArcArchive:
    top = ArcArchive(read_pac(path).data)
    entry = top.find("character_common.pac")
    if entry is None:
        raise KeyError(f"{path.name} has no character_common.pac")
    return ArcArchive(unwrap_section(top.entry_data(entry))[0])


def load_presets(path: str | Path) -> dict[int, FacialPreset]:
    """``s_facial_preset_datas`` keyed by the id a timeline cue carries."""
    archive = _character_common(Path(path))
    entry = archive.find("facial_preset_datas.bin")
    if entry is None:
        raise KeyError("facial_preset_datas.bin is missing")
    body = BlbnFile(archive.entry_data(entry)).read("s_facial_preset_datas")
    presets: dict[int, FacialPreset] = {}
    for offset in range(0, len(body) - PRESET_STRIDE + 1, PRESET_STRIDE):
        preset_id, count = struct.unpack_from("<2I", body, offset)
        if count > MAX_LAYERS:
            continue
        layers = tuple(
            FacialLayer(*struct.unpack_from("<If", body, offset + 8 + index * 8))
            for index in range(count)
        )
        presets[preset_id] = FacialPreset(preset_id, layers)
    return presets


def load_anims(path: str | Path) -> list[FacialAnim]:
    """``s_facial_anim_datas``: ``(id, count, kind, length)``."""
    archive = _character_common(Path(path))
    entry = archive.find("facial_datas.bin")
    if entry is None:
        raise KeyError("facial_datas.bin is missing")
    body = BlbnFile(archive.entry_data(entry)).read("s_facial_anim_datas")
    return [
        FacialAnim(*struct.unpack_from("<4I", body, offset))
        for offset in range(0, len(body) - ANIM_STRIDE + 1, ANIM_STRIDE)
    ]


def preset_name(preset: FacialPreset) -> str:
    """Readable form for logs, e.g. ``61 = 220 + 101``."""
    return " + ".join(f"{layer.part:g}" for layer in preset.layers)
