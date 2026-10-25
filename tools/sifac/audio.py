"""CRI HCA metadata and optional WAV export.

HCA headers are big-endian and chunked without an explicit length per chunk::

    "HCA\\0"  u16 version (0x0200)  u16 header_size
    "fmt\\0"  u8 channels  u24 sample_rate  u32 block_count  u32 reserved
    "comp"    u16 block_size  ... (or "dec\\0" for older files)
    "ciph"    u16 cipher_type  [u32 key when type != 0]
    "pad\\0"

Every retail SIFAC stream (``music.awb``, ``voice.awb`` and the DLC packs) has
``cipher_type = 0``, so the payloads need no key. Decoding itself is delegated to
the optional ``cricodecs`` package, which wraps the native CRI codec; install it
with ``pip install 'sifac-tools[audio]'``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .binary import be16, be24, be32

HCA_MAGIC = b"HCA\0"
SAMPLES_PER_BLOCK = 1024


class HcaError(ValueError):
    """Raised for a malformed HCA file or a missing decoder."""


@dataclass(frozen=True)
class HcaInfo:
    offset: int
    version: int
    header_size: int
    channels: int
    sample_rate: int
    block_count: int
    block_size: int
    cipher_type: int

    @property
    def duration(self) -> float:
        if not self.sample_rate:
            return 0.0
        return self.block_count * SAMPLES_PER_BLOCK / self.sample_rate

    @property
    def encrypted(self) -> bool:
        return self.cipher_type != 0


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_rate: int
    bits_per_sample: int
    data_size: int

    @property
    def duration(self) -> float:
        if not self.sample_rate or not self.channels:
            return 0.0
        bytes_per_sample = max(self.bits_per_sample // 8, 1)
        return self.data_size / (self.sample_rate * self.channels * bytes_per_sample)


def read_hca_info(data: bytes, offset: int = 0) -> HcaInfo:
    """Read the header of the HCA stream starting at *offset*."""
    if data[offset : offset + 4] != HCA_MAGIC:
        raise HcaError("not an HCA stream")
    version = be16(data, offset + 4)
    header_size = be16(data, offset + 6)
    if header_size < 16 or offset + header_size > len(data):
        raise HcaError(f"invalid HCA header size: {header_size:#x}")
    header_end = offset + header_size

    fmt = data.find(b"fmt\0", offset + 8, header_end)
    if fmt < 0 or fmt + 16 > header_end:
        raise HcaError("HCA fmt chunk missing")
    channels = data[fmt + 4]
    sample_rate = be24(data, fmt + 5)
    block_count = be32(data, fmt + 8)

    comp = data.find(b"comp", fmt + 16, header_end)
    if comp < 0:
        comp = data.find(b"dec\0", fmt + 16, header_end)
    if comp < 0 or comp + 6 > header_end:
        raise HcaError("HCA comp/dec chunk missing")
    block_size = be16(data, comp + 4)

    ciph = data.find(b"ciph", comp + 6, header_end)
    cipher_type = be16(data, ciph + 4) if ciph >= 0 and ciph + 6 <= header_end else 0
    return HcaInfo(
        offset=offset,
        version=version,
        header_size=header_size,
        channels=channels,
        sample_rate=sample_rate,
        block_count=block_count,
        block_size=block_size,
        cipher_type=cipher_type,
    )


def decode_hca(data: bytes, keycode: int | None = None) -> bytes:
    """Decode an HCA stream to RIFF/WAVE bytes."""
    try:
        from cricodecs import hca
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise HcaError(
            "HCA decoding needs the optional cricodecs package: "
            "pip install 'sifac-tools[audio]'"
        ) from exc
    try:
        if keycode is None:
            return hca.decode(data)
        return hca.decode(data, keycode=keycode)
    except Exception as exc:  # pragma: no cover - native decoder errors
        raise HcaError(f"HCA decode failed: {exc}") from exc


def read_wav_info(data: bytes) -> WavInfo:
    """Read the RIFF/WAVE header of the bytes returned by :func:`decode_hca`."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HcaError("not a RIFF/WAVE stream")
    channels = bits = sample_rate = data_size = 0
    position = 12
    while position + 8 <= len(data):
        tag = data[position : position + 4]
        size = struct.unpack_from("<I", data, position + 4)[0]
        body = position + 8
        if tag == b"fmt " and size >= 16:
            channels = struct.unpack_from("<H", data, body + 2)[0]
            sample_rate = struct.unpack_from("<I", data, body + 4)[0]
            bits = struct.unpack_from("<H", data, body + 14)[0]
        elif tag == b"data":
            data_size = min(size, len(data) - body)
            break
        position = body + size + (size & 1)
    if not channels or not sample_rate:
        raise HcaError("WAVE fmt chunk missing")
    return WavInfo(
        channels=channels,
        sample_rate=sample_rate,
        bits_per_sample=bits,
        data_size=data_size,
    )
