from __future__ import annotations

import importlib.util
import struct
import unittest

from sifac.audio import HcaError, decode_hca, read_hca_info, read_wav_info


def build_hca(
    channels: int = 2,
    sample_rate: int = 48000,
    block_count: int = 100,
    block_size: int = 0x2AA,
    cipher_type: int = 0,
) -> bytes:
    header = bytearray(0x60)
    header[0:4] = b"HCA\0"
    struct.pack_into(">HH", header, 4, 0x0200, 0x60)
    header[8:12] = b"fmt\0"
    header[12] = channels
    header[13:16] = sample_rate.to_bytes(3, "big")
    struct.pack_into(">I", header, 16, block_count)
    header[24:28] = b"comp"
    struct.pack_into(">H", header, 28, block_size)
    header[40:44] = b"ciph"
    struct.pack_into(">H", header, 44, cipher_type)
    header[46:50] = b"pad\0"
    return bytes(header) + b"\0" * (block_size * block_count)


def build_wav(channels: int = 2, sample_rate: int = 48000, frames: int = 4800) -> bytes:
    data = b"\0" * (frames * channels * 2)
    header = bytearray(44)
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, 36 + len(data))
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    struct.pack_into("<IHHIIHH", header, 16, 16, 1, channels, sample_rate,
                     sample_rate * channels * 2, channels * 2, 16)
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, len(data))
    return bytes(header) + data


class HcaTests(unittest.TestCase):
    def test_read_header(self) -> None:
        info = read_hca_info(build_hca())
        self.assertEqual(info.version, 0x0200)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.sample_rate, 48000)
        self.assertEqual(info.block_count, 100)
        self.assertEqual(info.block_size, 0x2AA)
        self.assertEqual(info.cipher_type, 0)
        self.assertFalse(info.encrypted)
        self.assertAlmostEqual(info.duration, 100 * 1024 / 48000, places=5)

    def test_encrypted_header(self) -> None:
        info = read_hca_info(build_hca(cipher_type=56))
        self.assertTrue(info.encrypted)

    def test_bad_magic(self) -> None:
        with self.assertRaises(HcaError):
            read_hca_info(b"not an hca")

    def test_read_wav_info(self) -> None:
        info = read_wav_info(build_wav())
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.sample_rate, 48000)
        self.assertEqual(info.bits_per_sample, 16)
        self.assertAlmostEqual(info.duration, 0.1, places=5)


@unittest.skipUnless(importlib.util.find_spec("cricodecs"), "cricodecs not installed")
class HcaDecodeTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self) -> None:
        from cricodecs import hca

        config = hca.HcaEncodeConfig()
        config.sample_rate = 48000
        config.channel_count = 2
        config.quality = hca.HcaQuality.HIGH
        encoded = hca.encode(build_wav(), config)
        info = read_hca_info(encoded)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.sample_rate, 48000)
        self.assertFalse(info.encrypted)

        decoded = decode_hca(encoded)
        wav = read_wav_info(decoded)
        self.assertEqual(wav.channels, 2)
        self.assertEqual(wav.sample_rate, 48000)
        self.assertEqual(wav.bits_per_sample, 16)


if __name__ == "__main__":
    unittest.main()
