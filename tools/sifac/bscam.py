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

``yaw`` is an orbit angle, not an aim direction: it always points the rig at
the stage centre. The quaternion is a *local* rotation inside that rig frame
(identity looks at the stage centre with Z up), so the aim direction is
``Rz(yaw) * q * (0, 1, 0)``. At frame 0 the quaternion pitches the camera
4.2 degrees down while the turntable itself is level.

Everything is baked per frame, so a frame is a lookup. The vertical field of
view is 16:9 (the game renders at 1920x1080).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

from .binary import u32

CHUNK_HEADER = 12
AIM_DISTANCE = 10.0


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
        local = (0.0, 1.0, 0.0)
        if 24 in self._sizes:
            row = self._row(24, frame)
            local = _rotate(row[2:6], (0.0, 1.0, 0.0))
        forward = (
            math.cos(yaw) * local[0] - math.sin(yaw) * local[1],
            math.sin(yaw) * local[0] + math.cos(yaw) * local[1],
            local[2],
        )
        fov = self.default_fov
        if 12 in self._sizes:
            row = self._row(12, frame)
            if row[2] > 1.0:
                fov = row[2]
        position = (math.sin(yaw) * distance, -math.cos(yaw) * distance, height)
        target = tuple(position[i] + forward[i] * AIM_DISTANCE for i in range(3))
        return CameraState(position, target, fov, distance, height)


def _rotate(quaternion: tuple[float, ...], vector: tuple[float, float, float]):
    """Rotate ``vector`` by an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = (quaternion + (0.0, 0.0, 0.0, 1.0))[:4]
    cx = y * vector[2] - z * vector[1]
    cy = z * vector[0] - x * vector[2]
    cz = x * vector[1] - y * vector[0]
    dx = 2.0 * w * cx
    dy = 2.0 * w * cy
    dz = 2.0 * w * cz
    return (
        vector[0] + dx + y * dz - z * dy,
        vector[1] + dy + z * dx - x * dz,
        vector[2] + dz + x * dy - y * dx,
    )


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
