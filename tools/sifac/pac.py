"""Outer .pac container handling: plain ARC, cmp/zlib and cmp/lzma."""

from __future__ import annotations

import lzma
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .arc import ArcArchive, ArcError
from .binary import u32, unwrap_section


class PacError(ValueError):
    """Raised when a .pac container is malformed."""


@dataclass(frozen=True)
class PacHeader:
    path: Path
    kind: str
    file_size: int
    compressed_size: int | None
    uncompressed_size: int | None
    arc_name: str | None = None
    arc_count: int | None = None

    @property
    def ratio(self) -> float | None:
        if not self.compressed_size or not self.uncompressed_size:
            return None
        return self.uncompressed_size / self.compressed_size


@dataclass
class PacImage:
    path: Path
    kind: str
    data: bytes
    compressed_size: int
    uncompressed_size: int

    def open_arc(self) -> ArcArchive:
        return ArcArchive(self.data)


def _parse_lzma_props(raw: bytes) -> tuple[int, int, int, int]:
    props = raw[16]
    lc = props % 9
    rest = props // 9
    lp = rest % 5
    pb = rest // 5
    dict_size = u32(raw, 17)
    return lc, lp, pb, dict_size


def read_pac(path: str | Path) -> PacImage:
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] == b"ARC\0":
        return PacImage(path, "arc", raw, len(raw), len(raw))
    if raw[:8] == b"cmp\0zlib":
        if len(raw) < 16:
            raise PacError("truncated cmp/zlib header")
        compressed_size = u32(raw, 8)
        uncompressed_size = u32(raw, 12)
        payload = raw[16 : 16 + compressed_size]
        data = zlib.decompress(payload)
        if len(data) != uncompressed_size:
            raise PacError(
                f"zlib size mismatch: {len(data)} != {uncompressed_size}"
            )
        return PacImage(path, "zlib", data, compressed_size, uncompressed_size)
    if raw[:8] == b"cmp\0lzma":
        if len(raw) < 21:
            raise PacError("truncated cmp/lzma header")
        compressed_size = u32(raw, 8)
        uncompressed_size = u32(raw, 12)
        lc, lp, pb, dict_size = _parse_lzma_props(raw)
        payload = raw[21 : 16 + compressed_size]
        decompressor = lzma.LZMADecompressor(
            format=lzma.FORMAT_RAW,
            filters=[
                {
                    "id": lzma.FILTER_LZMA1,
                    "dict_size": dict_size,
                    "lc": lc,
                    "lp": lp,
                    "pb": pb,
                }
            ],
        )
        data = decompressor.decompress(payload, max_length=uncompressed_size)
        if len(data) != uncompressed_size:
            raise PacError(
                f"lzma size mismatch: {len(data)} != {uncompressed_size}"
            )
        return PacImage(path, "lzma", data, compressed_size, uncompressed_size)
    raise PacError(f"unsupported .pac header: {raw[:8]!r}")


def read_header(path: str | Path) -> PacHeader:
    """Read only the header; safe for scanning the whole dump."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(0x40)
    if head[:4] == b"ARC\0":
        count = struct.unpack_from("<H", head, 6)[0] if len(head) >= 8 else None
        name = head[0x10:0x30].split(b"\0", 1)[0].decode("utf-8", "replace")
        return PacHeader(path, "arc", size, size, size, name, count)
    if head[:8] == b"cmp\0zlib" and len(head) >= 16:
        return PacHeader(
            path, "zlib", size, u32(head, 8), u32(head, 12)
        )
    if head[:8] == b"cmp\0lzma" and len(head) >= 21:
        return PacHeader(
            path, "lzma", size, u32(head, 8), u32(head, 12)
        )
    raise PacError(f"unsupported .pac header: {head[:8]!r}")


def iter_pac_files(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pac":
            yield path


def scan_dump(root: str | Path) -> dict:
    """Scan every .pac header without decompressing the large archives."""
    root = Path(root)
    if not root.is_dir():
        raise PacError(f"not a directory: {root}")
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    arc_entries = 0
    arc_files = 0
    errors: list[dict[str, str]] = []
    total_pac_bytes = 0
    pac_count = 0
    for path in iter_pac_files(root):
        pac_count += 1
        try:
            header = read_header(path)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the scan
            errors.append({"path": str(path), "error": str(exc)})
            continue
        counts[header.kind] = counts.get(header.kind, 0) + 1
        sizes[header.kind] = sizes.get(header.kind, 0) + header.file_size
        total_pac_bytes += header.file_size
        if header.kind == "arc" and header.arc_count is not None:
            arc_files += 1
            arc_entries += header.arc_count
    return {
        "root": str(root),
        "pac_files": pac_count,
        "pac_bytes": total_pac_bytes,
        "kinds": counts,
        "kind_bytes": sizes,
        "arc_files_with_header": arc_files,
        "arc_top_level_entries": arc_entries,
        "errors": errors,
    }


def verify_dump(
    root: str | Path,
    max_depth: int = 2,
    limit: int | None = None,
) -> dict:
    """Deep-verify every .pac: outer container, ARC tables and nested ARC/BMAR.

    This reads and (for the 46 compressed files) decompresses the archives, so
    it is slower than ``scan_dump`` but catches malformed entry tables.
    """
    root = Path(root)
    report = {
        "root": str(root),
        "pacs": 0,
        "arcs": 0,
        "entries": 0,
        "bmar": 0,
        "errors": [],
    }
    for index, path in enumerate(iter_pac_files(root)):
        if limit is not None and index >= limit:
            break
        report["pacs"] += 1
        try:
            image = read_pac(path)
            stack: list[tuple[bytes, int]] = [(image.data, 0)]
            while stack:
                data, depth = stack.pop()
                arc = ArcArchive(data)
                report["arcs"] += 1
                report["entries"] += len(arc)
                for entry in arc.entries:
                    payload, _ = unwrap_section(arc.entry_data(entry))
                    if payload.startswith(b"BMAR104\0"):
                        report["bmar"] += 1
                    elif payload.startswith(b"ARC\0") and depth < max_depth:
                        stack.append((payload, depth + 1))
        except Exception as exc:  # noqa: BLE001 - report per file
            report["errors"].append({"path": str(path), "error": str(exc)})
    return report
