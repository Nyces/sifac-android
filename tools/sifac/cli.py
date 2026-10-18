"""Command line interface for the SIFAC verification tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arc import ArcArchive, ArcError
from .awb import Afs2Archive, AwbError
from .binary import hexdump, unwrap_section
from .bmar import BmarArchive, BmarError, find_bmar_in_pac, scan_magics
from .pac import PacError, read_header, read_pac, scan_dump, verify_dump
from .score import LivePack, ScoreError, iter_live_packs
from .sfo import SfoError, read_sfo


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _safe_name(name: str) -> str:
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(
        "_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in name
    )


def cmd_scan(args: argparse.Namespace) -> int:
    result = scan_dump(args.root)
    if args.json:
        _print_json(result)
    else:
        print(f"root: {result['root']}")
        print(f".pac files: {result['pac_files']}")
        print(f".pac bytes: {result['pac_bytes']:,}")
        for kind, count in sorted(result["kinds"].items()):
            size = result["kind_bytes"][kind]
            print(f"  {kind:8s} {count:6d}  {size:,} bytes")
        print(f"top-level ARC entries: {result['arc_top_level_entries']:,}")
        if result["errors"]:
            print(f"errors: {len(result['errors'])}")
            for error in result["errors"][:20]:
                print(f"  {error['path']}: {error['error']}")
    return 0 if not result["errors"] else 2


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_dump(args.root, max_depth=args.depth, limit=args.limit)
    if args.json:
        _print_json(result)
    else:
        print(f"root: {result['root']}")
        print(f"verified .pac files: {result['pacs']}")
        print(f"ARC archives: {result['arcs']}")
        print(f"ARC entries: {result['entries']}")
        print(f"BMAR models: {result['bmar']}")
        if result["errors"]:
            print(f"errors: {len(result['errors'])}")
            for error in result["errors"][:50]:
                print(f"  {error['path']}: {error['error']}")
    return 0 if not result["errors"] else 2


def cmd_info(args: argparse.Namespace) -> int:
    header = read_header(args.pac)
    payload = {
        "path": str(header.path),
        "kind": header.kind,
        "file_size": header.file_size,
        "compressed_size": header.compressed_size,
        "uncompressed_size": header.uncompressed_size,
        "ratio": header.ratio,
        "arc_name": header.arc_name,
        "arc_count": header.arc_count,
    }
    if args.json:
        _print_json(payload)
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    image = read_pac(args.pac)
    arc = ArcArchive(image.data)
    if args.json:
        _print_json(
            {
                "archive": arc.name,
                "count": len(arc),
                "entries": [
                    {
                        "type": entry.type,
                        "index": entry.index,
                        "hash": f"{entry.hash:08x}",
                        "name": entry.name,
                        "offset": entry.offset,
                        "size": entry.size,
                    }
                    for entry in arc.entries
                ],
            }
        )
        return 0
    print(f"{arc.name}: {len(arc)} entries")
    for entry in arc.entries:
        print(
            f"  {entry.index:3d} {entry.type:5s} {entry.name:40s} "
            f"off={entry.offset:#010x} size={entry.size:>10}"
        )
    return 0


def _tree(data: bytes, depth: int, max_depth: int, prefix: str = "") -> None:
    if depth > max_depth:
        return
    try:
        arc = ArcArchive(data)
    except ArcError:
        return
    for entry in arc.entries:
        payload, skipped = unwrap_section(arc.entry_data(entry))
        kind = "raw"
        for magic in (b"ARC\0", b"BMAR104\0", b"BML\0", b"BMD\0", b"BMA\0", b"BMT\0", b"MTXP", b"MDL\0", b"ANHD", b"ANSK", b"POF0"):
            if payload.startswith(magic):
                kind = magic.decode("ascii", "replace").rstrip("\0")
                break
        print(
            f"{prefix}{entry.type:5s} {entry.name:44s} "
            f"off={entry.offset:#010x} size={entry.size:>10} -> {kind}"
            + (f" (+{skipped:#x})" if skipped else "")
        )
        if payload.startswith(b"ARC\0"):
            _tree(payload, depth + 1, max_depth, prefix + "  ")
        elif payload.startswith(b"BMAR104\0") and depth + 1 <= max_depth:
            try:
                bmar = BmarArchive(payload)
            except BmarError:
                continue
            for sub in bmar.entries:
                print(
                    f"{prefix}  {sub.type:5s} {sub.name:44s} "
                    f"off={sub.offset:#010x} size={sub.size:>10} -> {sub.payload_magic}"
                )


def cmd_tree(args: argparse.Namespace) -> int:
    image = read_pac(args.pac)
    _tree(image.data, 0, args.depth)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    image = read_pac(args.pac)
    arc = ArcArchive(image.data)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in arc.entries:
        data = arc.entry_data(entry)
        if args.unwrap:
            data, _ = unwrap_section(data)
        target = out_dir / _safe_name(entry.name)
        if target.exists() and not args.force:
            print(f"skip existing: {target}")
            continue
        target.write_bytes(data)
        written += 1
        print(f"{target} ({len(data)} bytes)")
    print(f"wrote {written} file(s) to {out_dir}")
    return 0


def cmd_sfo(args: argparse.Namespace) -> int:
    values = read_sfo(args.file)
    if args.json:
        _print_json(values)
    else:
        for key, value in values.items():
            print(f"{key} = {value}")
    return 0


def cmd_awb(args: argparse.Namespace) -> int:
    archive = Afs2Archive(Path(args.file).read_bytes())
    if args.json:
        _print_json(
            {
                "version": archive.version,
                "count": archive.count,
                "entries": [
                    {
                        "index": entry.index,
                        "id": entry.id,
                        "offset": entry.offset,
                        "size": entry.size,
                        "magic": entry.magic,
                    }
                    for entry in archive.entries
                ],
            }
        )
        return 0
    print(
        f"{args.file}: AFS2 v{archive.version}, {archive.count} entries, "
        f"{len(archive.data):,} bytes"
    )
    for entry in archive.entries[: args.limit]:
        print(
            f"  {entry.index:3d} id={entry.id:5d} off={entry.offset:#010x} "
            f"size={entry.size:>10} magic={entry.magic}"
        )
    if args.extract:
        out_dir = Path(args.extract)
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry in archive.entries:
            target = out_dir / f"{entry.index:04d}_{entry.id:04d}.{entry.magic.lower() or 'bin'}"
            target.write_bytes(archive.entry_data(entry.index))
        print(f"wrote {len(archive.entries)} file(s) to {out_dir}")
    return 0


def cmd_bmar(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.suffix.lower() == ".pac":
        archive = find_bmar_in_pac(path)
    else:
        archive = BmarArchive(path.read_bytes())
    print(f"{path}: BMAR104 with {len(archive.entries)} entries")
    for entry in archive.entries:
        print(
            f"  {entry.type:5s} {entry.name:44s} off={entry.offset:#010x} "
            f"size={entry.size:>10} -> {entry.payload_magic}"
        )
    if args.magics:
        print("known magics:")
        for offset, magic in scan_magics(archive.data, limit=args.magics_limit):
            print(f"  {offset:#010x} {magic}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    pack = LivePack(args.pac)
    print(f"{args.pac}: {len(pack.scores)} .scd score file(s)")
    for score in pack.scores[: args.limit]:
        print(f"  {score.name:56s} off={score.offset:#010x} size={score.size:>9}")
    if args.extract:
        written = pack.extract(args.extract, unwrap=args.unwrap)
        print(f"wrote {len(written)} file(s) to {args.extract}")
    return 0


def cmd_hexdump(args: argparse.Namespace) -> int:
    data = Path(args.file).read_bytes()
    for line in hexdump(data, offset=args.offset, length=args.length):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sifac-tools",
        description="Read-only verification tools for a user-owned SIFAC dump.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan every .pac header in a dump")
    scan.add_argument("root")
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=cmd_scan)

    verify = sub.add_parser("verify", help="deep-verify every .pac header and nested ARC")
    verify.add_argument("root")
    verify.add_argument("--depth", type=int, default=2)
    verify.add_argument("--limit", type=int, default=None)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)

    info = sub.add_parser("info", help="show one .pac header")
    info.add_argument("pac")
    info.add_argument("--json", action="store_true")
    info.set_defaults(func=cmd_info)

    listing = sub.add_parser("list", help="list top-level ARC entries")
    listing.add_argument("pac")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    tree = sub.add_parser("tree", help="recursively show nested ARC/BMAR entries")
    tree.add_argument("pac")
    tree.add_argument("--depth", type=int, default=2)
    tree.set_defaults(func=cmd_tree)

    extract = sub.add_parser("extract", help="extract top-level ARC entries")
    extract.add_argument("pac")
    extract.add_argument("outdir")
    extract.add_argument("--unwrap", action="store_true")
    extract.add_argument("--force", action="store_true")
    extract.set_defaults(func=cmd_extract)

    sfo = sub.add_parser("sfo", help="parse param.sfo")
    sfo.add_argument("file")
    sfo.add_argument("--json", action="store_true")
    sfo.set_defaults(func=cmd_sfo)

    awb = sub.add_parser("awb", help="parse an AFS2/AWB audio container")
    awb.add_argument("file")
    awb.add_argument("--json", action="store_true")
    awb.add_argument("--limit", type=int, default=40)
    awb.add_argument("--extract")
    awb.set_defaults(func=cmd_awb)

    bmar = sub.add_parser("bmar", help="parse a BMAR model archive or a PAC containing one")
    bmar.add_argument("path")
    bmar.add_argument("--magics", action="store_true")
    bmar.add_argument("--magics-limit", type=int, default=0x2000)
    bmar.set_defaults(func=cmd_bmar)

    score = sub.add_parser("score", help="list .scd scores in a live pack")
    score.add_argument("pac")
    score.add_argument("--limit", type=int, default=40)
    score.add_argument("--extract")
    score.add_argument("--unwrap", action="store_true")
    score.set_defaults(func=cmd_score)

    dump = sub.add_parser("hexdump", help="hexdump a file")
    dump.add_argument("file")
    dump.add_argument("--offset", type=lambda value: int(value, 0), default=0)
    dump.add_argument("--length", type=lambda value: int(value, 0), default=0x100)
    dump.set_defaults(func=cmd_hexdump)

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (PacError, ArcError, BmarError, AwbError, ScoreError, SfoError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
