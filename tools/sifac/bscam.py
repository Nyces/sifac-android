"""Blade stage camera animation (``.bscam``).

The file is a chunk chain::

    BSCM   header: version "101", f32 fps (60), f32 frame count (9060),
           defaults for height (1.2875), distance (11.048), target height
           (1.4985) and focal length (30.17 mm)
    DATA   20 bytes per frame: orbit yaw (+8), height (+12), distance (+16)
    DATA   24 bytes per frame: local rotation quaternion (+8, x/y/z/w)
    DATA   12 bytes per frame: focal length in mm (+8)
    POF0   index block

The camera rides a turntable: it orbits the stage centre at ``distance`` and
``height``, and the per-frame angle is ``yaw``::

    position = (sin(yaw) * distance, -cos(yaw) * distance, height)
    target   = (0, 0, default_target_height)

``yaw`` is an orbit angle, not an aim direction. Least-squares triangulation of
the per-frame aim over five separate shots of song 0010 (frames 1500, 2100,
3900, 4800 and 6600) returns (0.05, 0.07, 1.23), (-0.09, -0.11, 1.10),
(-0.07, 0.09, 1.30), (-0.04, 0.11, 1.20) and (0.19, -0.23, 1.10): the camera
always looks at the stage centre, at roughly the header's target height, so the
per-frame quaternion is *not* part of the aim (its rotation only spreads those
rays by up to 0.9 m). The quaternion is left parsed but unused until its role
is identified.

Everything is baked per frame, so a frame is a lookup. The third table holds a
**focal length in millimetres**, not an angle: read as degrees, song 0010's close
shots frame 0.4 m of empty floor and song 0050's 8.16 frame nothing at all. Read
as a 36 mm-wide sensor lens, 30.17 gives a 37 deg vertical field of view for the
opening wide shot and 8.16 gives a 103 deg wide-angle close-up, which is the
look the arcade game uses. Across song 0010's 229 close shots (<5 m) this keeps
a performer inside the frame 86% of the time against 59% for the degree reading.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

from .binary import u32

CHUNK_HEADER = 12
# The game renders at 1920x1080, so a 36 mm wide sensor is 20.25 mm tall.
SENSOR_HALF_HEIGHT = 10.125


class BscamError(ValueError):
    """Raised for a malformed .bscam file."""


@dataclass(frozen=True)
class CameraState:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_degrees: float
    distance: float
    height: float


class BscamAnimation:
    def __init__(self, data: bytes) -> None:
        offset = data.find(b"BSCM")
        if offset < 0:
            raise BscamError("not a .bscam camera animation")
        self.data = data
        self.header_offset = offset
        payload = offset + CHUNK_HEADER
        self.version = data[payload : payload + 4].rstrip(b"\0").decode("ascii", "replace")
        self.fps, self.frame_count = struct.unpack_from("<2f", data, payload + 4)
        (
            self.default_height,
            self.default_distance,
            self.default_target_height,
        ) = struct.unpack_from("<3f", data, payload + 0x20)
        self.default_lens = struct.unpack_from("<f", data, payload + 0x40)[0]
        self._frames: dict[int, int] = {}
        self._sizes: dict[int, int] = {}
        start = offset + CHUNK_HEADER + u32(data, offset + 4)
        self._walk(start + (4 - start % 4) % 4)

    def _walk(self, position: int) -> None:
        while position + CHUNK_HEADER <= len(self.data):
            tag = self.data[position : position + 4]
            if not all(32 <= byte < 127 for byte in tag):
                break
            size = u32(self.data, position + 4)
            body = position + CHUNK_HEADER
            if size == 0 or body + size > len(self.data):
                break
            if tag == b"DATA" and self.frame_count > 0:
                stride = round(size / self.frame_count)
                if stride * self.frame_count == size:
                    self._sizes[stride] = body
            position = body + size
            position += (4 - position % 4) % 4

    def _row(self, stride: int, frame: int) -> tuple[float, ...]:
        body = self._sizes.get(stride)
        if body is None:
            raise BscamError(f"no {stride}-byte camera table")
        index = max(0, min(int(frame), int(self.frame_count) - 1))
        return struct.unpack_from(f"<{stride // 4}f", self.data, body + index * stride)

    def state(self, frame: float) -> CameraState:
        yaw = 0.0
        height = self.default_height
        distance = self.default_distance
        if 20 in self._sizes:
            row = self._row(20, frame)
            yaw, height, distance = row[2], row[3], row[4]
        lens = self.default_lens
        if 12 in self._sizes:
            row = self._row(12, frame)
            if row[2] > 1.0:
                lens = row[2]
        position = (math.sin(yaw) * distance, -math.cos(yaw) * distance, height)
        target = (0.0, 0.0, self.default_target_height)
        return CameraState(position, target, lens_to_fov(lens), distance, height)


def lens_to_fov(lens_mm: float) -> float:
    """Vertical field of view for a focal length on the game's 16:9 sensor."""
    if lens_mm <= 0.5:
        return 90.0
    return math.degrees(2.0 * math.atan(SENSOR_HALF_HEIGHT / lens_mm))


def load_scene_camera(pac_path: str | Path) -> BscamAnimation | None:
    """Load the camera animation stored in a song pack's ``scene.pac``."""
    from .arc import ArcArchive, ArcError
    from .binary import unwrap_section
    from .pac import read_pac

    image = read_pac(pac_path)
    top = ArcArchive(image.data)
    scene_entry = top.find("scene.pac")
    if scene_entry is None:
        return None
    try:
        scene = ArcArchive(unwrap_section(top.entry_data(scene_entry))[0])
    except ArcError:
        return None
    for entry in scene.entries:
        if entry.name.lower().endswith(".bscam"):
            return BscamAnimation(scene.entry_data(entry))
    return None
