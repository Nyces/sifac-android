"""Blade stage camera animation (``.bscam``).

The file is a chunk chain::

    BSCM   header: version "101", f32 fps (60), f32 frame count (9060),
           defaults for height (1.2875), distance (11.048), target height
           (1.4985) and field of view (30.17 deg)
    DATA   20 bytes per frame: orbit yaw (+8), height (+12), distance (+16)
    DATA   24 bytes per frame: local rotation quaternion (+8, x/y/z/w)
    DATA   12 bytes per frame: fov in degrees (+8)
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

Everything is baked per frame, so a frame is a lookup. The third table is read
as a vertical field of view in degrees: song 0010 opens at 30.17 deg, which at
11 m frames the 7 m wide formation and the 13.6 m LED screen correctly.

That reading has one open counter-example: song 0050 reaches 8.16 deg at a 3 m
distance, which would frame a 0.4 m patch at the target and render an empty
stage, while the same table converges on the same (0, 0, 1.3) target as song
0010. Either the value is a focal length rather than an angle (8 mm on a 16:9
sensor is a very wide lens), or 0050's staging differs; a RenderDoc capture of
a live frame would settle both this and the performer placement.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

from .binary import u32

CHUNK_HEADER = 12


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
        self.default_fov = struct.unpack_from("<f", data, payload + 0x40)[0]
        self.fov = self.default_fov
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
        fov = self.default_fov
        if 12 in self._sizes:
            row = self._row(12, frame)
            if row[2] > 1.0:
                fov = row[2]
        position = (math.sin(yaw) * distance, -math.cos(yaw) * distance, height)
        target = (0.0, 0.0, self.default_target_height)
        return CameraState(position, target, fov, distance, height)


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
