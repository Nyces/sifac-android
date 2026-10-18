"""Render a parsed score chart to a PNG preview (no external dependencies)."""

from __future__ import annotations

from .score import LANE_COUNT, TICKS_PER_BEAT, ScoreChart
from .texture import encode_png

LANE_WIDTH = 40
MARGIN = 24
PIXELS_PER_BEAT = 6
MIN_HEIGHT = 480
MAX_HEIGHT = 6000


def render_chart(chart: ScoreChart, width_scale: int = 1) -> bytes:
    """Draw notes as a piano-roll: x = lane, y = time (top = start)."""
    if not chart.notes:
        raise ValueError("chart has no notes")
    last_tick = max(note.tick for note in chart.notes)
    beats = last_tick / TICKS_PER_BEAT
    height = int(beats * PIXELS_PER_BEAT) + MARGIN * 2
    height = max(MIN_HEIGHT, min(MAX_HEIGHT, height))
    lane_width = LANE_WIDTH * width_scale
    width = LANE_COUNT * lane_width + MARGIN * 2

    pixels = bytearray(width * height * 4)
    # background
    for i in range(width * height):
        pixels[i * 4 : i * 4 + 4] = bytes((18, 18, 24, 255))
    # lane separators
    for lane in range(LANE_COUNT + 1):
        x = MARGIN + lane * lane_width
        for y in range(MARGIN, height - MARGIN):
            _set(pixels, width, x, y, (44, 46, 60, 255))
    # beat lines every 4 beats
    tick_span = last_tick or 1
    usable = height - MARGIN * 2
    for beat in range(0, int(beats) + 1, 4):
        y = MARGIN + int(usable * (beat * TICKS_PER_BEAT) / tick_span)
        color = (52, 54, 70, 255) if beat % 16 == 0 else (36, 38, 50, 255)
        for x in range(MARGIN, width - MARGIN):
            _set(pixels, width, x, y, color)

    # notes
    for note in chart.notes:
        if note.lane < 0:
            continue
        x = MARGIN + note.lane * lane_width + lane_width // 2
        y = MARGIN + int(usable * note.tick / tick_span)
        length = int(usable * note.value / tick_span)
        color = (255, 214, 92, 255) if note.value <= TICKS_PER_BEAT else (110, 214, 255, 255)
        half = max(2, lane_width // 5)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                _set(pixels, width, x + dx, y + dy, color)
        if length > 2:
            for dy in range(0, min(length, height - y - MARGIN)):
                for dx in range(-1, 2):
                    _set(pixels, width, x + dx, y + dy, color)

    return encode_png(width, height, bytes(pixels))


def _set(pixels: bytearray, width: int, x: int, y: int, color: tuple[int, int, int, int]) -> None:
    if x < 0 or y < 0:
        return
    height = len(pixels) // (width * 4)
    if x >= width or y >= height:
        return
    offset = (y * width + x) * 4
    pixels[offset : offset + 4] = bytes(color)
