from __future__ import annotations

import lzma
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from sifac.pac import PacError, read_header, read_pac

from tests.helpers import build_arc


class PacTests(unittest.TestCase):
    def _write(self, directory: str, name: str, data: bytes) -> Path:
        path = Path(directory) / name
        path.write_bytes(data)
        return path

    def test_plain_arc(self) -> None:
        data = build_arc([("bin", "a.bin", b"POF0a")])
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "plain.pac", data)
            header = read_header(path)
            self.assertEqual(header.kind, "arc")
            self.assertEqual(header.arc_count, 1)
            image = read_pac(path)
            self.assertEqual(image.data, data)

    def test_zlib(self) -> None:
        data = build_arc([("bin", "a.bin", b"POF0a")])
        payload = zlib.compress(data)
        raw = b"cmp\0zlib" + struct.pack("<II", len(payload), len(data)) + payload
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "zlib.pac", raw)
            header = read_header(path)
            self.assertEqual(header.kind, "zlib")
            image = read_pac(path)
            self.assertEqual(image.data, data)

    def test_lzma(self) -> None:
        data = build_arc([("bin", "a.bin", b"POF0a")])
        filters = [
            {
                "id": lzma.FILTER_LZMA1,
                "dict_size": 1 << 24,
                "lc": 3,
                "lp": 0,
                "pb": 2,
            }
        ]
        stream = lzma.compress(data, format=lzma.FORMAT_RAW, filters=filters)
        raw = (
            b"cmp\0lzma"
            + struct.pack("<II", 5 + len(stream), len(data))
            + bytes([0x5D])
            + struct.pack("<I", 1 << 24)
            + stream
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "lzma.pac", raw)
            header = read_header(path)
            self.assertEqual(header.kind, "lzma")
            image = read_pac(path)
            self.assertEqual(image.data, data)

    def test_bad_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "bad.pac", b"nope")
            with self.assertRaises(PacError):
                read_header(path)


if __name__ == "__main__":
    unittest.main()
