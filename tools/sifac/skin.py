"""Approximate skinning for Blade models.

The retail model files do not expose per-vertex bone indices or weights (see
``bml.py`` for the negative results and the GPU-side attribute layout). A
RenderDoc capture confirms the GPU receives a ``uint4`` of bone indices as
attribute 3, but that stream is not stored in the model container. So this
module computes **automatic weights** from the bind-pose skeleton:

1. every bone becomes a segment from its node position to the average position
   of its children (a leaf falls back to a short segment along +Z);
2. each vertex is bound to its nearest ``max_influences`` segments with
   inverse-square distance weights, normalised to 1.

This is an approximation and is marked as such: it is good enough to pose and
animate a model, but joint deformation will not match the original game exactly.
Replacing :func:`auto_weights` with real data later requires no other change.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace

from .bml import Mesh, Node

# Node space is Y up, mesh vertices are Z up: (x, y, z) -> (x, -z, y).
_AXIS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
_AXIS_INV = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def identity() -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def rot_x(a: float) -> list[list[float]]:
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]


def rot_y(a: float) -> list[list[float]]:
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]


def rot_z(a: float) -> list[list[float]]:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def invert(m: list[list[float]]) -> list[list[float]]:
    """Invert an affine 4x4 matrix."""
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    det = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(det) < 1e-12:
        return identity()
    inv_det = 1.0 / det
    inv = [
        [
            (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * inv_det,
            (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * inv_det,
            (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * inv_det,
        ],
        [
            (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * inv_det,
            (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * inv_det,
            (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * inv_det,
        ],
        [
            (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * inv_det,
            (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * inv_det,
            (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * inv_det,
        ],
    ]
    t = [m[0][3], m[1][3], m[2][3]]
    result = identity()
    for i in range(3):
        for j in range(3):
            result[i][j] = inv[i][j]
    for i in range(3):
        result[i][3] = -sum(inv[i][k] * t[k] for k in range(3))
    return result


def transform_point(m: list[list[float]], p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )


def matrix_translation(m: list[list[float]]) -> tuple[float, float, float]:
    return (m[0][3], m[1][3], m[2][3])


@dataclass(frozen=True)
class BoneSegment:
    name: str
    head: tuple[float, float, float]
    tail: tuple[float, float, float]


class Skeleton:
    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = {node.name: node for node in nodes}
        self.children: dict[str, list[str]] = defaultdict(list)
        for node in nodes:
            if node.parent in self.nodes:
                self.children[node.parent].append(node.name)
        self._world: dict[str, list[list[float]]] = {}

    def local(
        self,
        name: str,
        extra: list[list[float]] | None = None,
        absolute: list[list[float]] | None = None,
        basis: str = "absolute",
    ) -> list[list[float]]:
        node = self.nodes[name]
        m = identity()
        m[0][3], m[1][3], m[2][3] = node.translation
        if absolute is not None:
            rotation = _apply_quat_basis(node, absolute, basis)
        else:
            rx, ry, rz = node.rotation
            rotation = mat_mul(mat_mul(rot_z(rz), rot_y(ry)), rot_x(rx))
            if extra is not None:
                rotation = mat_mul(extra, rotation)
        m = mat_mul(m, rotation)
        scale = identity()
        scale[0][0], scale[1][1], scale[2][2] = node.scale
        return mat_mul(m, scale)

    def world(
        self,
        name: str,
        pose: dict[str, tuple[float, float, float]] | None = None,
        pose_quat: dict[str, tuple[float, float, float, float]] | None = None,
        basis: str = "absolute",
    ) -> list[list[float]]:
        """World matrix in mesh space (Z up)."""
        chain: list[str] = []
        current: str | None = name
        while current and current in self.nodes:
            chain.append(current)
            current = self.nodes[current].parent
        matrix = identity()
        for bone in reversed(chain):
            extra = None
            absolute = None
            if pose_quat and bone in pose_quat:
                absolute = mat_from_quat(pose_quat[bone])
            elif pose and bone in pose:
                rx, ry, rz = pose[bone]
                extra = mat_mul(mat_mul(rot_z(rz), rot_y(ry)), rot_x(rx))
            matrix = mat_mul(matrix, self.local(bone, extra, absolute, basis))
        return mat_mul(mat_mul(_AXIS, matrix), _AXIS_INV)

    def segments(self) -> list[BoneSegment]:
        result: list[BoneSegment] = []
        for name in self.nodes:
            head = transform_point(self.world(name), (0.0, 0.0, 0.0))
            tails = [
                transform_point(self.world(child), (0.0, 0.0, 0.0))
                for child in self.children.get(name, [])
                if child in self.nodes
            ]
            if tails:
                tail = (
                    sum(t[0] for t in tails) / len(tails),
                    sum(t[1] for t in tails) / len(tails),
                    sum(t[2] for t in tails) / len(tails),
                )
            else:
                tail = (head[0], head[1], head[2] + 0.02)
            result.append(BoneSegment(name, head, tail))
        return result

    def leaf_names(self) -> list[str]:
        return [name for name in self.nodes if not self.children.get(name)]


def mat_from_quat(q: tuple[float, float, float, float]) -> list[list[float]]:
    x, y, z, w = q
    length = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / length, y / length, z / length, w / length
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _distance_to_segment(
    point: tuple[float, float, float], segment: BoneSegment
) -> float:
    ax, ay, az = segment.head
    bx, by, bz = segment.tail
    px, py, pz = point
    dx, dy, dz = bx - ax, by - ay, bz - az
    length_sq = dx * dx + dy * dy + dz * dz
    if length_sq < 1e-12:
        return math.dist(point, segment.head)
    t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / length_sq
    t = max(0.0, min(1.0, t))
    cx, cy, cz = ax + dx * t, ay + dy * t, az + dz * t
    return math.dist(point, (cx, cy, cz))


def auto_weights(
    mesh: Mesh,
    skeleton: Skeleton,
    max_influences: int = 2,
    candidates: list[str] | None = None,
) -> list[tuple[list[int], list[float]]]:
    """Per-vertex (bone indices, weights) using nearest bone segments."""
    segments = skeleton.segments()
    if candidates is not None:
        allowed = set(candidates)
        segments = [s for s in segments if s.name in allowed]
    if not segments:
        return [([], []) for _ in mesh.positions]
    names = [s.name for s in segments]
    result: list[tuple[list[int], list[float]]] = []
    for position in mesh.positions:
        scored = sorted(
            ((_distance_to_segment(position, s), index) for index, s in enumerate(segments)),
            key=lambda item: item[0],
        )[:max_influences]
        weights = [1.0 / (distance * distance + 1e-6) for distance, _ in scored]
        total = sum(weights) or 1.0
        result.append(
            (
                [names[index] for _, index in scored],
                [weight / total for weight in weights],
            )
        )
    return result


def palette_map(
    meshes: list[Mesh], skeleton: Skeleton, candidates: list[str] | None = None
) -> dict[int, str]:
    """Recover the file's bone palette indices to node names.

    The vertex stream stores four u8 palette indices per vertex, but the file
    does not say how that palette maps to scene-graph nodes. Grouping vertices
    by index and matching each group's centroid to the closest bone segment
    recovers the mapping well enough to drive the rig.
    """
    segments = skeleton.segments()
    if candidates is not None:
        allowed = set(candidates)
        segments = [s for s in segments if s.name in allowed]
    centroids: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for mesh in meshes:
        for position, indices in zip(mesh.positions, mesh.bone_indices):
            for index in indices:
                if index == 0:
                    continue
                acc = centroids.setdefault(index, [0.0, 0.0, 0.0])
                acc[0] += position[0]
                acc[1] += position[1]
                acc[2] += position[2]
                counts[index] = counts.get(index, 0) + 1
    mapping: dict[int, str] = {}
    for index, acc in centroids.items():
        n = counts[index]
        centroid = (acc[0] / n, acc[1] / n, acc[2] / n)
        best = min(segments, key=lambda s: _distance_to_segment(centroid, s))
        mapping[index] = best.name
    return mapping


def weights_from_palette(
    mesh: Mesh,
    skeleton: Skeleton,
    mapping: dict[int, str],
    fallback: list[str] | None = None,
) -> list[tuple[list[int], list[float]]]:
    """Per-vertex weights using the file's own palette indices as candidates."""
    segments = {s.name: s for s in skeleton.segments()}
    result: list[tuple[list[int], list[float]]] = []
    for position, indices in zip(mesh.positions, mesh.bone_indices):
        names = []
        for index in dict.fromkeys(indices):
            name = mapping.get(index)
            if name and name in segments and name not in names:
                names.append(name)
        if not names:
            if fallback:
                names = fallback[:1]
            else:
                result.append(([], []))
                continue
        scored = sorted(
            (1.0 / (_distance_to_segment(position, segments[n]) ** 2 + 1e-6), n)
            for n in names
        )
        total = sum(weight for weight, _ in scored) or 1.0
        result.append(([n for _, n in scored], [w / total for w, _ in scored]))
    return result


def place_by_nodes(
    meshes: list[Mesh],
    draw_nodes: dict[int, int],
    nodes: list[Node],
    skeleton: "Skeleton",
) -> list[Mesh]:
    """Apply each mesh's DRAW node transform (used by static stage models).

    Stage meshes are authored in stage space, so only the node's *local* matrix
    applies — walking the parent chain would rotate the whole stage (the root
    node carries the axis convention that the meshes already bake in). A
    ``LaserBox`` mesh is 25 cm wide and only lands on the truss once its node
    translation is applied.
    """
    placed: list[Mesh] = []
    for index, mesh in enumerate(meshes):
        node_index = draw_nodes.get(index)
        if node_index is None or node_index >= len(nodes):
            placed.append(mesh)
            continue
        matrix = skeleton.local(nodes[node_index].name)
        placed.append(
            replace(
                mesh,
                positions=[transform_point(matrix, p) for p in mesh.positions],
            )
        )
    return placed


def anchor_rigid(
    meshes: list[Mesh],
    draw_nodes: dict[int, int],
    nodes: list[Node],
    skeleton: "Skeleton",
    pose_quat: dict[str, tuple[float, float, float, float]] | None = None,
) -> list[Mesh]:
    """Place eye (stride 44) and face (stride 60) meshes with their owning node.

    Eye meshes are authored in node-local space and need the full world matrix.
    Face meshes are already Z-up and correctly scaled, but their origin sits at
    the character root instead of the head, so only the node translation applies.
    Verified on Honoka, Eli and Nico: every face ends up between the neck and the
    hair with its eyes at roughly half of the face height.
    """
    placed: list[Mesh] = []
    for index, mesh in enumerate(meshes):
        node_index = draw_nodes.get(index)
        if mesh.stride not in (44, 60) or node_index is None or node_index >= len(nodes):
            placed.append(mesh)
            continue
        node_name = nodes[node_index].name
        posed = skeleton.world(node_name, None, pose_quat)
        if mesh.stride == 44:
            positions = [transform_point(posed, p) for p in mesh.positions]
        else:
            # The face origin sits at the character root, so first shift it to
            # the bind anchor and then carry it along with the node's delta.
            bind = skeleton.world(node_name)
            delta = mat_mul(posed, invert(bind))
            tx, ty, tz = matrix_translation(bind)
            positions = [
                transform_point(delta, (p[0] + tx, p[1] + ty, p[2] + tz))
                for p in mesh.positions
            ]
        placed.append(replace(mesh, positions=positions))
    return placed


def _bind_rotation(node: Node) -> list[list[float]]:
    rx, ry, rz = node.rotation
    return mat_mul(mat_mul(rot_z(rz), rot_y(ry)), rot_x(rx))


def _apply_quat_basis(node: Node, q: list[list[float]], basis: str) -> list[list[float]]:
    """Combine a clip rotation with the node's bind rotation.

    The animation quaternions are authored in a different local frame than the
    bind pose, so several conventions are possible. ``basis`` selects one:
    ``absolute`` replaces the bind rotation, ``post`` applies the clip after it,
    ``pre`` before it, and ``sandwich`` treats the bind rotation as a change of
    basis (``R * q * R^-1``).
    """
    if basis == "absolute":
        return q
    r = _bind_rotation(node)
    if basis == "pre":
        return mat_mul(q, r)
    if basis == "sandwich":
        return mat_mul(mat_mul(r, q), invert(r))
    return mat_mul(r, q)


def deform(
    positions: list[tuple[float, float, float]],
    skeleton: Skeleton,
    weights: list[tuple[list[int], list[float]]],
    pose: dict[str, tuple[float, float, float]] | None = None,
    pose_quat: dict[str, tuple[float, float, float, float]] | None = None,
    basis: str = "absolute",
) -> list[tuple[float, float, float]]:
    """Apply a pose (bone name -> extra rotation)."""
    names = sorted({name for bones, _ in weights for name in bones})
    bind = {name: invert(skeleton.world(name)) for name in names}
    posed = {name: skeleton.world(name, pose, pose_quat, basis) for name in names}
    deltas = {name: mat_mul(posed[name], bind[name]) for name in names}
    out: list[tuple[float, float, float]] = []
    for position, (bones, bone_weights) in zip(positions, weights):
        x = y = z = 0.0
        for name, weight in zip(bones, bone_weights):
            px, py, pz = transform_point(deltas[name], position)
            x += px * weight
            y += py * weight
            z += pz * weight
        out.append((x, y, z))
    return out
