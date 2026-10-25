"""Command line interface for the SIFAC verification tools."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .arc import ArcArchive, ArcError
from .audio import HcaError
from .awb import Afs2Archive, AwbError
from .binary import hexdump, unwrap_section
from .bmar import BmarArchive, BmarError, find_bmar_in_pac, scan_magics
from .pac import PacError, read_header, read_pac, scan_dump, verify_dump
from .score import LivePack, ScoreError, iter_live_packs
from .sfo import SfoError, read_sfo
from .texture import BtxError, BtxTexture, find_textures


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


def cmd_tex(args: argparse.Namespace) -> int:
    image = read_pac(args.pac)
    out_dir = Path(args.outdir) if args.outdir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    exported = 0
    failures: list[str] = []

    def walk(data: bytes, depth: int, trail: str) -> None:
        nonlocal exported
        try:
            arc = ArcArchive(data)
        except ArcError:
            return
        for entry in arc.entries:
            payload, _ = unwrap_section(arc.entry_data(entry))
            label = f"{trail}/{entry.name}" if trail else entry.name
            hits = find_textures(payload, limit=0x100)
            if hits and payload.startswith(b"btx\0"):
                try:
                    texture = BtxTexture(payload)
                except BtxError as exc:
                    failures.append(f"{label}: {exc}")
                    continue
                name = texture.header.format_name
                stats[name] = stats.get(name, 0) + 1
                if out_dir and args.export:
                    if args.filter and args.filter.lower() not in texture.header.name.lower():
                        continue
                    try:
                        png = texture.to_png()
                    except BtxError as exc:
                        failures.append(f"{label}: {exc}")
                        continue
                    except ImportError:
                        failures.append(f"{label}: texture2ddecoder is required for BC7")
                        continue
                    target = out_dir / f"{_safe_name(texture.header.name or entry.name)}.png"
                    target.write_bytes(png)
                    exported += 1
            elif payload.startswith(b"ARC\0") and depth < args.depth:
                walk(payload, depth + 1, label)

    walk(image.data, 0, "")
    total = sum(stats.values())
    if args.json:
        _print_json({"textures": total, "formats": stats, "exported": exported,
                     "failures": failures[:50]})
    else:
        print(f"{args.pac}: {total} texture(s)")
        for name, count in sorted(stats.items()):
            print(f"  {name:12s} {count}")
        if out_dir and args.export:
            print(f"exported {exported} PNG file(s) to {out_dir}")
        for failure in failures[:20]:
            print(f"  skipped: {failure}")
    return 0


def cmd_chart(args: argparse.Namespace) -> int:
    from .chart import render_chart

    pack = LivePack(args.pac)
    if args.list:
        for chart in pack.charts(args.channel):
            print(
                f"{chart.difficulty_name:12s} {chart.name:46s} "
                f"notes={len(chart.notes):4d} last_tick={max((n.tick for n in chart.notes), default=0)}"
            )
        return 0
    chart = pack.chart(args.difficulty, args.channel)
    print(
        f"{chart.name}: {len(chart.notes)} notes, difficulty={chart.difficulty_name}, "
        f"ticks_per_beat={chart.header.ticks_per_beat}"
    )
    if args.csv:
        Path(args.csv).write_text(chart.to_csv(), encoding="utf-8")
        print(f"wrote {args.csv}")
    if args.png:
        Path(args.png).write_bytes(render_chart(chart))
        print(f"wrote {args.png}")
    if not args.csv and not args.png:
        for note in chart.notes[: args.limit]:
            print(
                f"  id={note.id:4d} tick={note.tick:7d} beat={note.beat:8.2f} "
                f"lane={note.lane} value={note.value}"
            )
    return 0


def cmd_mesh(args: argparse.Namespace) -> int:
    from .bml import BmlError, find_bml_in_pac, render_silhouette

    archive = find_bml_in_pac(args.pac, args.character)
    meshes = archive.meshes()
    materials = archive.materials()
    draw_calls = archive.draw_calls()
    print(f"{args.pac}: {len(meshes)} mesh(es), {sum(m.vertex_count for m in meshes)} vertices")
    bindings = archive.mesh_materials()
    for index, mesh in enumerate(meshes):
        low, high = mesh.bounds()
        material = bindings.get(index)
        print(
            f"  {mesh.name:26s} v={mesh.vertex_count:6d} tri={mesh.triangle_count:6d} "
            f"z=[{low[2]:.3f}, {high[2]:.3f}]"
            + (f" mat={material.name}({material.texture})" if material else "")
        )
    print(f"nodes={len(archive.nodes())} materials={len(materials)} draws={len(draw_calls)}")
    if args.obj:
        Path(args.obj).write_text(archive.to_obj(), encoding="utf-8")
        print(f"wrote {args.obj}")
    if args.mtl:
        Path(args.mtl).write_text(archive.to_mtl(), encoding="utf-8")
        print(f"wrote {args.mtl}")
    if args.png:
        Path(args.png).write_bytes(render_silhouette(meshes))
        print(f"wrote {args.png}")
    return 0


def cmd_anim(args: argparse.Namespace) -> int:
    from .anim import BmaError, find_bma_in_pac, list_motions

    if args.list:
        motions = list_motions(args.pac, args.character)
        print(f"{args.pac}: {len(motions)} animation(s)")
        for motion in motions[: args.limit]:
            print(
                f"  {motion['name']:40s} size={motion['size']:>8} "
                f"tracks={motion['tracks']:3d} anhd={motion['duration']:.2f}"
            )
        return 0
    animation = find_bma_in_pac(args.pac, args.character)
    print(
        f"{args.pac}: version={animation.version} duration={animation.duration:.2f} "
        f"tracks={len(animation.tracks)}"
    )
    for track in animation.tracks[: args.limit]:
        channels = animation.channels(track)
        kinds = ",".join(str(c[2]) for c in channels)
        print(f"  {track.name:34s} records={track.key_count} kinds=[{kinds}]")
    return 0


def _pose_meshes(
    archive: object,
    skeleton: object,
    meshes: list,
    pose: dict[str, tuple[float, float, float]] | None = None,
    pose_quat: dict[str, tuple[float, float, float, float]] | None = None,
    influences: int = 2,
) -> tuple[list, dict[int, str], int]:
    """Skin every mesh with either palette weights or automatic weights."""
    from .bml import Mesh
    from .skin import auto_weights, deform, palette_map, weights_from_palette

    biped = [name for name in skeleton.nodes if name.startswith("Bip001_")]
    hair = [name for name in skeleton.nodes if "_bon_" in name]
    head = [
        name for name in skeleton.nodes
        if name in ("Bip001_Head", "Bip001_Neck", "Honoka_head_set")
    ]
    mapping = palette_map(meshes, skeleton, biped + hair + head) if any(
        mesh.bone_indices for mesh in meshes
    ) else {}
    posed: list[Mesh] = []
    skipped = 0
    for mesh in meshes:
        lower = mesh.name.lower()
        if "hair" in lower:
            candidates = hair + head
        elif "face" in lower or "eye" in lower:
            candidates = head
        else:
            candidates = biped
        if mesh.stride in (44, 60):
            # Eye and face meshes are rigid: they are anchored to their node by
            # _attach_rigid_meshes instead of being skinned.
            posed.append(mesh)
            skipped += 1
            continue
        if not candidates:
            # Stage and prop models have no Biped rig; leave the geometry as is.
            posed.append(mesh)
            skipped += 1
            continue
        weights = (
            weights_from_palette(mesh, skeleton, mapping, candidates)
            if mesh.bone_indices and mapping
            else auto_weights(mesh, skeleton, influences, candidates)
        )
        positions = deform(mesh.positions, skeleton, weights, pose, pose_quat)
        posed.append(
            Mesh(
                name=mesh.name,
                vertex_count=mesh.vertex_count,
                index_count=mesh.index_count,
                stride=mesh.stride,
                positions=positions,
                normals=mesh.normals,
                uvs=mesh.uvs,
                indices=mesh.indices,
            )
        )
    return posed, mapping, skipped


def _attach_rigid_meshes(
    posed: list,
    archive: object,
    skeleton: object,
    pose_quat: dict[str, tuple[float, float, float, float]] | None = None,
) -> list:
    """Anchor eye and face meshes to their owning node."""
    from .skin import anchor_rigid

    nodes = archive.nodes()
    draw_nodes = {call.mesh_index: call.node_index for call in archive.draw_calls()}
    return anchor_rigid(posed, draw_nodes, nodes, skeleton, pose_quat)


def cmd_pose(args: argparse.Namespace) -> int:
    from .bml import find_bml_in_pac, render_silhouette
    from .skin import Skeleton

    archive = find_bml_in_pac(args.pac, args.character)
    skeleton = Skeleton(archive.nodes())
    meshes = archive.meshes()
    pose: dict[str, tuple[float, float, float]] = {}
    for spec in args.rotate:
        name, _, values = spec.partition("=")
        if not values:
            print(f"error: invalid --rotate {spec!r}, expected BONE=rx,ry,rz", file=sys.stderr)
            return 1
        parts = [float(v) for v in values.split(",")]
        if len(parts) != 3:
            print(f"error: invalid --rotate {spec!r}", file=sys.stderr)
            return 1
        pose[name.strip()] = tuple(math.radians(v) for v in parts)  # type: ignore[assignment]
    posed, mapping, skipped = _pose_meshes(archive, skeleton, meshes, pose, None, args.influences)
    posed = _attach_rigid_meshes(posed, archive, skeleton)
    print(
        f"{args.pac}: posed {len(posed)} mesh(es), {sum(m.vertex_count for m in posed)} vertices, "
        f"{len(pose)} bone(s) rotated, {skipped} rigid mesh(es) anchored, "
        f"palette={len(mapping)}"
    )
    if args.obj:
        Path(args.obj).write_text(archive.to_obj_for(posed), encoding="utf-8")
        print(f"wrote {args.obj}")
    if args.png:
        Path(args.png).write_bytes(render_silhouette(posed))
        print(f"wrote {args.png}")
    return 0


def cmd_animate(args: argparse.Namespace) -> int:
    from .anim import BmaError, load_motion
    from .bml import find_bml_in_pac, render_silhouette
    from .skin import Skeleton

    archive = find_bml_in_pac(args.pac, args.character)
    skeleton = Skeleton(archive.nodes())
    meshes = archive.meshes()
    animation = load_motion(args.pac, args.clip)
    pose_quat: dict[str, tuple[float, float, float, float]] = {}
    last_frame = 0.0
    missing = []
    for track in animation.tracks:
        if track.name not in skeleton.nodes:
            missing.append(track.name)
            continue
        keys = animation.rotation_keys(track)
        if keys is None:
            continue
        if keys.times:
            last_frame = max(last_frame, keys.times[-1])
        pose_quat[track.name] = keys.sample(args.frame)
    if not pose_quat:
        print(f"error: no rotation keys decoded from {args.clip}", file=sys.stderr)
        return 1
    posed, mapping, skipped = _pose_meshes(
        archive, skeleton, meshes, None, pose_quat, args.influences
    )
    posed = _attach_rigid_meshes(posed, archive, skeleton, pose_quat)
    print(
        f"{args.pac}: {args.clip} frame={args.frame:g}/{last_frame:g}, "
        f"{len(pose_quat)} track(s) applied, {len(missing)} unmatched, "
        f"palette={len(mapping)}, {skipped} rigid mesh(es) anchored"
    )
    if args.obj:
        Path(args.obj).write_text(archive.to_obj_for(posed), encoding="utf-8")
        print(f"wrote {args.obj}")
    if args.png:
        Path(args.png).write_bytes(render_silhouette(posed))
        print(f"wrote {args.png}")
    return 0


def cmd_hca(args: argparse.Namespace) -> int:
    from .audio import HcaError, decode_hca, read_hca_info, read_wav_info

    path = Path(args.file)
    raw = path.read_bytes()
    keycode = args.key
    is_awb = raw[:4] == b"AFS2"
    if is_awb:
        archive = Afs2Archive(raw)
        if args.list:
            print(f"{path}: AFS2 v{archive.version}, {archive.count} HCA stream(s)")
            for entry in archive.entries[: args.limit]:
                data = archive.entry_data(entry.index)
                try:
                    info = read_hca_info(data)
                except HcaError as exc:
                    print(f"  {entry.index:4d} id={entry.id:5d} {entry.size:>10} {exc}")
                    continue
                print(
                    f"  {entry.index:4d} id={entry.id:5d} off={entry.offset:#010x} "
                    f"size={entry.size:>10} {info.channels}ch {info.sample_rate}Hz "
                    f"{info.duration:7.2f}s cipher={info.cipher_type}"
                )
            return 0
        if args.all:
            out_dir = Path(args.outdir or (path.parent / f"{path.stem}_wav"))
            out_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for entry in archive.entries:
                data = archive.entry_data(entry.index)
                try:
                    info = read_hca_info(data)
                    wav = decode_hca(data, keycode)
                except HcaError as exc:
                    print(f"  skip {entry.index}: {exc}", file=sys.stderr)
                    continue
                target = out_dir / f"{entry.index:04d}_{entry.id:04d}.wav"
                target.write_bytes(wav)
                written += 1
                print(f"{target} ({info.duration:.2f}s, {len(wav):,} bytes)")
            print(f"wrote {written} WAV file(s) to {out_dir}")
            return 0
        if not 0 <= args.entry < len(archive.entries):
            print(f"error: entry {args.entry} out of range 0..{len(archive.entries) - 1}", file=sys.stderr)
            return 1
        entry = archive.entries[args.entry]
        data = archive.entry_data(args.entry)
        default_out = path.parent / f"{path.stem}_{entry.id:04d}.wav"
    else:
        data = raw
        default_out = path.with_suffix(".wav")

    info = read_hca_info(data)
    wav = decode_hca(data, keycode)
    wav_info = read_wav_info(wav)
    print(
        f"{path}: {info.channels}ch {info.sample_rate}Hz, {info.block_count} blocks "
        f"({info.duration:.2f}s), cipher={info.cipher_type}, block={info.block_size} bytes"
    )
    if not args.dry_run:
        target = Path(args.out) if args.out else default_out
        target.write_bytes(wav)
        print(
            f"wrote {target} ({wav_info.channels}ch {wav_info.sample_rate}Hz "
            f"{wav_info.bits_per_sample}-bit, {wav_info.duration:.2f}s)"
        )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .anim import load_motion
    from .bml import find_bml_in_pac
    from .render import render_textured
    from .skin import Skeleton
    from .texture import load_texture_archive

    archive = find_bml_in_pac(args.pac, args.character)
    skeleton = Skeleton(archive.nodes())
    meshes = archive.meshes()
    pose_quat: dict[str, tuple[float, float, float, float]] = {}
    if args.clip:
        animation = load_motion(args.pac, args.clip)
        for track in animation.tracks:
            if track.name not in skeleton.nodes:
                continue
            keys = animation.rotation_keys(track)
            if keys is not None:
                pose_quat[track.name] = keys.sample(args.frame)

    posed, mapping, skipped = _pose_meshes(
        archive, skeleton, meshes, None, pose_quat, args.influences
    )
    placed = _attach_rigid_meshes(posed, archive, skeleton, pose_quat)

    textures = load_texture_archive(args.pac)
    png = render_textured(
        placed,
        archive.mesh_materials(),
        textures,
        width=args.width,
        height=args.height,
    )
    target = Path(args.out)
    target.write_bytes(png)
    print(
        f"{args.pac}: rendered {len(placed)} mesh(es), {len(textures)} texture(s), "
        f"{skipped} rigid mesh(es) anchored, palette={len(mapping)} -> {target}"
    )
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    from .bml import find_bml_in_pac
    from .render import Camera, render_camera
    from .skin import Skeleton, place_by_nodes
    from .texture import load_texture_archive

    archive = find_bml_in_pac(args.pac, args.model)
    skeleton = Skeleton(archive.nodes())
    nodes = archive.nodes()
    draw_nodes = {call.mesh_index: call.node_index for call in archive.draw_calls()}
    all_meshes = place_by_nodes(archive.meshes(), draw_nodes, nodes, skeleton)
    bindings = archive.mesh_materials()
    keep = [
        index
        for index, mesh in enumerate(all_meshes)
        if not any(token.lower() in mesh.name.lower() for token in args.skip)
    ]
    meshes = [all_meshes[index] for index in keep]
    sub_bindings = {new: bindings[old] for new, old in enumerate(keep) if old in bindings}
    visible = [m for m in meshes if m.indices]
    lo = [min(m.bounds()[0][k] for m in visible) for k in range(3)]
    hi = [max(m.bounds()[1][k] for m in visible) for k in range(3)]
    center = [(lo[k] + hi[k]) * 0.5 for k in range(3)]
    radius = max(hi[k] - lo[k] for k in range(3)) * 0.5
    textures = load_texture_archive(args.pac)
    print(
        f"{args.pac}: {len(meshes)} mesh(es), {sum(m.triangle_count for m in meshes)} triangles, "
        f"{len(archive.materials())} materials, {len(textures)} textures"
    )
    print(
        f"  bounds x=[{lo[0]:.2f},{hi[0]:.2f}] y=[{lo[1]:.2f},{hi[1]:.2f}] z=[{lo[2]:.2f},{hi[2]:.2f}]"
    )
    if args.skip:
        print(f"  skipped names containing {args.skip}")
    if args.bounds_only:
        return 0
    eye = tuple(args.eye) if args.eye else (
        center[0],
        center[1] - radius * 1.6,
        center[2] + radius * 0.35,
    )
    target = tuple(args.target) if args.target else tuple(center)
    camera = Camera(position=eye, target=target, fov_degrees=args.fov)
    png = render_camera(
        meshes,
        sub_bindings,
        textures,
        camera,
        width=args.width,
        height=args.height,
    )
    Path(args.out).write_bytes(png)
    print(f"wrote {args.out} (eye={tuple(round(v, 2) for v in eye)})")
    return 0


def _formation(count: int, spacing: float = 2.2, depth: float = 1.7) -> list[tuple[float, float]]:
    """Lay the idols out in three rows, centred on the stage."""
    rows = 3
    columns = max(1, (count + rows - 1) // rows)
    spots: list[tuple[float, float]] = []
    for index in range(count):
        row = index % rows
        column = index // rows
        x = (column - (columns - 1) / 2.0) * spacing
        y = row * depth
        spots.append((x, y))
    return spots


def _formation_from_btlb(pac: Path, count: int) -> list[tuple[float, float]] | None:
    """Read the performer floor plan from a song's character timeline.

    Each record carries ``(x, focus height, depth)``. For song 0010 the ``(x,
    depth)`` plane spans 7.0 x 1.8 m with a minimum spacing of 1.32 m, while
    either alternative puts two performers 0.14 m apart, so those are the floor
    axes and the remaining one (about 1.4 m, i.e. chest height) is dropped.
    Songs whose plan is degenerate fall back to the placeholder formation.
    """
    from .btlb import load_timelines

    timelines = load_timelines(pac)
    records = next(
        (
            timeline.chara_records()
            for name, timeline in timelines.items()
            if "chara" in name and timeline.chara_records()
        ),
        [],
    )
    if len(records) < count:
        print(
            f"warning: {pac.name} holds {len(records)} performer record(s), need {count}",
            file=sys.stderr,
        )
        return None
    spots = [(record.position[0], record.position[2]) for record in records[:count]]
    spacing = min(
        (
            math.dist(a, b)
            for index, a in enumerate(spots)
            for b in spots[index + 1 :]
        ),
        default=0.0,
    )
    if spacing < 0.6:
        print(
            f"warning: timeline plan puts two performers {spacing:.2f} m apart, "
            "keeping the placeholder formation",
            file=sys.stderr,
        )
        return None
    print(
        f"formation: {count} performer(s) from the character timeline, "
        f"min spacing {spacing:.2f} m"
    )
    return spots


def cmd_live(args: argparse.Namespace) -> int:
    from .anim import load_live_motion, sample_pose
    from .bml import Mesh, find_bml_in_pac
    from .render import BACKGROUND, Camera, encode_sheet, render_buffer
    from .skin import Skeleton, place_by_nodes
    from .texture import encode_png, load_texture_archive

    live_dir = Path(args.live)
    song = args.song or live_dir.name
    root = Path(args.root) if args.root else live_dir.parent.parent

    stage_pac = root / "stage" / f"{args.stage}.pac"
    stage = find_bml_in_pac(stage_pac)
    stage_skeleton = Skeleton(stage.nodes())
    stage_nodes = stage.nodes()
    stage_draw = {call.mesh_index: call.node_index for call in stage.draw_calls()}
    stage_meshes = place_by_nodes(
        stage.meshes(), stage_draw, stage_nodes, stage_skeleton
    )
    stage_bindings = stage.mesh_materials()
    stage_wire: list[tuple[object, object]] = [
        (mesh, stage_bindings.get(index)) for index, mesh in enumerate(stage_meshes)
    ]
    textures = dict(load_texture_archive(stage_pac))

    floor_top = 0.0
    for mesh in stage_meshes:
        if "mainfloor" in mesh.name.lower():
            floor_top = max(floor_top, mesh.bounds()[1][2])
    print(f"stage {args.stage}: {len(stage_meshes)} mesh(es), dance floor z={floor_top:.2f}")

    packs = sorted(live_dir.glob(f"c??_{song}.pac"))
    if args.characters:
        packs = packs[: args.characters]
    spots = _formation(len(packs))
    if args.formation == "btlb":
        timeline_spots = _formation_from_btlb(live_dir / f"{song}.pac", len(packs))
        if timeline_spots:
            spots = timeline_spots
    print(f"song {song}: {len(packs)} character pack(s)")

    models = []
    for index, pack in enumerate(packs):
        character_id = pack.name[:3]
        char_pac = root / "chr" / character_id / f"{character_id}.pac"
        if not char_pac.exists():
            print(f"  skip {pack.name}: {char_pac} not found", file=sys.stderr)
            continue
        model = find_bml_in_pac(char_pac)
        skeleton = Skeleton(model.nodes())
        textures.update(load_texture_archive(char_pac))
        models.append(
            {
                "id": character_id,
                "model": model,
                "skeleton": skeleton,
                "motion": load_live_motion(pack),
                "bindings": model.mesh_materials(),
                "spot": spots[index],
            }
        )

    eye = tuple(args.eye) if args.eye else (0.0, -9.0, floor_top + 1.6)
    target = tuple(args.target) if args.target else (0.0, 1.6, floor_top + 0.6)
    camera = Camera(position=eye, target=target, fov_degrees=args.fov)

    camera_anim = None
    if args.camera_anim:
        from .bscam import BscamError, load_scene_camera

        song_pac = live_dir / f"{song}.pac"
        try:
            camera_anim = load_scene_camera(song_pac) if song_pac.exists() else None
        except (BscamError, ArcError, PacError) as exc:
            print(f"warning: {song_pac}: {exc}", file=sys.stderr)
        if camera_anim is None:
            print(
                f"warning: no .bscam camera in {song_pac}, using --eye/--target",
                file=sys.stderr,
            )
        else:
            state = camera_anim.state(0.0)
            print(
                f"camera animation: {camera_anim.version} {camera_anim.fps:.0f} fps, "
                f"{camera_anim.frame_count:.0f} frames, "
                f"frame 0 eye={tuple(round(v, 2) for v in state.position)} "
                f"fov={state.fov_degrees:.1f}"
            )

    stage_layer = None if camera_anim is not None else render_buffer(
        stage_wire, textures, args.width, args.height, camera=camera
    )

    frames = [args.frame]
    if args.frames:
        start, end, step = (int(v) for v in args.frames.split(":"))
        frames = list(range(start, end + 1, max(1, step)))
    print(f"frames: {len(frames)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tiles: list[bytes] = []
    for frame in frames:
        wire: list[tuple[object, object]] = []
        for entry in models:
            model = entry["model"]
            skeleton = entry["skeleton"]
            pose = sample_pose(entry["motion"], set(skeleton.nodes), frame)
            if not pose:
                continue
            posed, _, _ = _pose_meshes(
                model, skeleton, model.meshes(), None, pose, args.influences
            )
            posed = _attach_rigid_meshes(posed, model, skeleton, pose)
            offset_x, offset_y = entry["spot"]
            for mesh_index, mesh in enumerate(posed):
                wire.append(
                    (
                        Mesh(
                            name=f"{entry['id']}/{mesh.name}",
                            vertex_count=mesh.vertex_count,
                            index_count=mesh.index_count,
                            stride=mesh.stride,
                            positions=[
                                (p[0] + offset_x, p[1] + offset_y, p[2] + floor_top)
                                for p in mesh.positions
                            ],
                            normals=mesh.normals,
                            uvs=mesh.uvs,
                            indices=mesh.indices,
                        ),
                        entry["bindings"].get(mesh_index),
                    )
                )
        if camera_anim is not None:
            state = camera_anim.state(frame)
            frame_camera = Camera(
                position=state.position,
                target=state.target,
                fov_degrees=state.fov_degrees,
            )
            layer = None
            layer_wire = stage_wire + wire
        else:
            frame_camera = camera
            layer = stage_layer
            layer_wire = wire
        pixels, _depth = render_buffer(
            layer_wire,
            textures,
            args.width,
            args.height,
            camera=frame_camera,
            base=layer,
        )
        background = bytes(BACKGROUND[:3])
        filled = 0
        total = 0
        count = args.width * args.height
        for index in range(0, len(pixels), 4):
            red, green, blue = pixels[index], pixels[index + 1], pixels[index + 2]
            total += red + green + blue
            if pixels[index : index + 3] != background:
                filled += 1
        share = filled / count
        mean = total / (count * 3)
        # A live frame is filled by the stage or by a performer; song 0010's
        # opening shot covers 100% of the image, while a camera that misses the
        # stage (song 0050 on st001) covers 1.5%.
        if share < 0.2:
            eye = tuple(round(value, 2) for value in frame_camera.position)
            print(
                f"warning: frame {frame:g} covers {share:.1%} of the image at a mean "
                f"luminance of {mean:.0f} (eye={eye}, "
                f"fov={frame_camera.fov_degrees:.1f}, stage={args.stage}); the camera "
                "may sit outside the stage, or the stage may not be the one this "
                "song uses",
                file=sys.stderr,
            )
        tiles.append(pixels)
        label = f"frame {frame:g}"
        target_path = Path(args.out)
        if len(frames) > 1:
            target_path = target_path.with_name(f"{target_path.stem}_{frame:06d}{target_path.suffix}")
        target_path.write_bytes(encode_png(args.width, args.height, pixels))
        print(f"  {label} -> {target_path.name}")

    if args.sheet:
        columns = args.columns or min(4, len(tiles))
        Path(args.sheet).write_bytes(
            encode_sheet(tiles, args.width, args.height, columns)
        )
        print(f"wrote {args.sheet}")

    if args.audio:
        from .audio import decode_hca, read_hca_info, read_wav_info
        from .awb import Afs2Archive

        music = root / "sound" / "dlc" / f"MUSIC_{song}.awb"
        if not music.exists():
            print(f"warning: {music} not found, skipping audio", file=sys.stderr)
            return 0
        archive = Afs2Archive(music.read_bytes())
        data = archive.entry_data(0)
        info = read_hca_info(data)
        wav = decode_hca(data)
        wav_info = read_wav_info(wav)
        Path(args.audio).write_bytes(wav)
        print(
            f"wrote {args.audio} ({wav_info.channels}ch {wav_info.sample_rate}Hz, "
            f"{wav_info.duration:.2f}s)"
        )
        print(
            f"  frame {frames[0]:g} at 60 fps = {frames[0] / 60.0:.2f}s, "
            f"track length {info.duration:.2f}s"
        )
    return 0


def cmd_btlb(args: argparse.Namespace) -> int:
    from .btlb import load_timelines

    timelines = load_timelines(args.pac)
    if not timelines:
        print(f"{args.pac}: no .btlb timeline found", file=sys.stderr)
        return 1
    if args.json:
        _print_json(
            {
                name: {
                    "channels": timeline.channel_count,
                    "payload_size": timeline.payload_size,
                    "performers": [
                        {
                            "offset": record.offset,
                            "end_tick": record.end_tick,
                            "ids": list(record.ids),
                            "blend": list(record.blend),
                            "position": list(record.position),
                            "cues": [
                                {
                                    "ident": cue.ident,
                                    "tag": cue.tag,
                                    "start_tick": cue.start_tick,
                                    "length": cue.length,
                                }
                                for cue in record.cues
                            ],
                        }
                        for record in timeline.chara_records()
                    ],
                }
                for name, timeline in timelines.items()
            }
        )
        return 0
    for name, timeline in timelines.items():
        records = timeline.chara_records()
        print(
            f"{name}: {timeline.channel_count} channel(s), "
            f"payload={timeline.payload_size} bytes, {len(records)} performer record(s)"
        )
        for index, record in enumerate(records):
            x, y, z = record.position
            span = (
                f"{record.cues[0].start_tick}..{max(c.start_tick + c.length for c in record.cues)}"
                if record.cues
                else "-"
            )
            print(
                f"  [{index}] pos=({x:8.3f},{y:8.3f},{z:8.3f}) "
                f"end_tick={record.end_tick:6d} cues={len(record.cues):2d} span={span} "
                f"ids={record.ids} blend=({record.blend[1]:.3f},{record.blend[2]:.3f},{record.blend[3]:.3f})"
            )
            for cue in record.cues[: args.limit]:
                print(
                    f"        ident={cue.ident:3d} tick=[{cue.start_tick:5d},"
                    f"{cue.start_tick + cue.length:5d}] tag={cue.tag:#010x} "
                    f"weight={cue.weight:g}"
                )
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

    hca = sub.add_parser("hca", help="inspect or decode HCA audio in an AWB or .hca file")
    hca.add_argument("file")
    hca.add_argument("--entry", type=int, default=0)
    hca.add_argument("--list", action="store_true")
    hca.add_argument("--limit", type=int, default=40)
    hca.add_argument("--all", action="store_true")
    hca.add_argument("--outdir")
    hca.add_argument("--out")
    hca.add_argument("--key", type=lambda value: int(value, 0), default=None)
    hca.add_argument("--dry-run", action="store_true")
    hca.set_defaults(func=cmd_hca)

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

    tex = sub.add_parser("tex", help="list or export btx textures inside a .pac")
    tex.add_argument("pac")
    tex.add_argument("outdir", nargs="?")
    tex.add_argument("--export", action="store_true")
    tex.add_argument("--filter", default="")
    tex.add_argument("--depth", type=int, default=3)
    tex.add_argument("--json", action="store_true")
    tex.set_defaults(func=cmd_tex)

    chart = sub.add_parser("chart", help="parse a live chart and export CSV/PNG")
    chart.add_argument("pac")
    chart.add_argument("--list", action="store_true")
    chart.add_argument("--difficulty", default="04")
    chart.add_argument("--channel", default="1")
    chart.add_argument("--csv")
    chart.add_argument("--png")
    chart.add_argument("--limit", type=int, default=20)
    chart.set_defaults(func=cmd_chart)

    mesh = sub.add_parser("mesh", help="extract a character mesh to OBJ and a PNG preview")
    mesh.add_argument("pac")
    mesh.add_argument("--character", default=None)
    mesh.add_argument("--obj")
    mesh.add_argument("--mtl")
    mesh.add_argument("--png")
    mesh.set_defaults(func=cmd_mesh)

    anim = sub.add_parser("anim", help="inspect a character's animation tracks and motions")
    anim.add_argument("pac")
    anim.add_argument("--character", default=None)
    anim.add_argument("--list", action="store_true")
    anim.add_argument("--limit", type=int, default=15)
    anim.set_defaults(func=cmd_anim)

    pose = sub.add_parser("pose", help="apply an approximate pose and export the result")
    pose.add_argument("pac")
    pose.add_argument("--character", default=None)
    pose.add_argument("--rotate", action="append", default=[], metavar="BONE=rx,ry,rz")
    pose.add_argument("--influences", type=int, default=2)
    pose.add_argument("--obj")
    pose.add_argument("--png")
    pose.set_defaults(func=cmd_pose)

    animate = sub.add_parser("animate", help="apply a motion clip at one frame and export the result")
    animate.add_argument("pac")
    animate.add_argument("clip")
    animate.add_argument("--character", default=None)
    animate.add_argument("--frame", type=float, default=0.0)
    animate.add_argument("--influences", type=int, default=2)
    animate.add_argument("--obj")
    animate.add_argument("--png")
    animate.set_defaults(func=cmd_animate)

    render = sub.add_parser("render", help="textured software preview of a posed model")
    render.add_argument("pac")
    render.add_argument("--character", default=None)
    render.add_argument("--clip", default=None)
    render.add_argument("--frame", type=float, default=0.0)
    render.add_argument("--influences", type=int, default=2)
    render.add_argument("--width", type=int, default=512)
    render.add_argument("--height", type=int, default=768)
    render.add_argument("--out", required=True)
    render.set_defaults(func=cmd_render)

    stage = sub.add_parser("stage", help="render a stage model from a perspective camera")
    stage.add_argument("pac")
    stage.add_argument("--model", default=None)
    stage.add_argument("--eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    stage.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    stage.add_argument("--skip", action="append", default=[], metavar="NAME")
    stage.add_argument("--fov", type=float, default=45.0)
    stage.add_argument("--width", type=int, default=640)
    stage.add_argument("--height", type=int, default=360)
    stage.add_argument("--out", required=True)
    stage.add_argument("--bounds-only", action="store_true")
    stage.set_defaults(func=cmd_stage)

    live = sub.add_parser("live", help="compose a live stage frame (stage + idols + motion)")
    live.add_argument("live", help="song folder, e.g. <dump>/live/0010")
    live.add_argument("--root", default=None, help="dump root (defaults to two levels above the song folder)")
    live.add_argument("--song", default=None)
    live.add_argument("--stage", default="st001")
    live.add_argument("--frame", type=float, default=0.0)
    live.add_argument("--characters", type=int, default=0)
    live.add_argument("--influences", type=int, default=2)
    live.add_argument("--eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    live.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    live.add_argument("--fov", type=float, default=45.0)
    live.add_argument("--width", type=int, default=480)
    live.add_argument("--height", type=int, default=270)
    live.add_argument("--out", required=True)
    live.add_argument("--audio", default=None, metavar="WAV")
    live.add_argument("--frames", default=None, metavar="START:END:STEP")
    live.add_argument("--camera-anim", action="store_true", help="follow the song's .bscam camera")
    live.add_argument(
        "--formation",
        choices=("grid", "btlb"),
        default="grid",
        help="performer layout: placeholder grid or the song's character timeline",
    )
    live.add_argument("--sheet", default=None, metavar="PNG")
    live.add_argument("--columns", type=int, default=0)
    live.set_defaults(func=cmd_live)

    btl = sub.add_parser("btlb", help="inspect a song pack's .btlb timelines")
    btl.add_argument("pac")
    btl.add_argument("--limit", type=int, default=6, help="cues printed per performer")
    btl.add_argument("--json", action="store_true")
    btl.set_defaults(func=cmd_btlb)

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
    except (PacError, ArcError, BmarError, AwbError, HcaError, ScoreError, SfoError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
