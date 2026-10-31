"""The building as a 3D model: the OSM footprint raised to its height, written as glTF binary (.glb).

Axes follow glTF: metres, Y up, X east, Z south. The origin is the footprint's centre at ground level, and
place.geojson says where that is on the map. Each wall has its own 0..1 texture coordinates, so the photo's
straightened front can go on one wall later.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from .look import Look

FLOOR = 3.0  # metres per floor, a common rule of thumb
GUESS_FLOORS = 2  # nothing says how tall it is


def parse_height(text: str | None) -> float | None:
    """OSM's height tag: "12", "12 m", "12.5", "40'" (feet). None if there's no number."""
    if not text:
        return None
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*(m|ft|')?", str(text))
    if not match:
        return None
    value = float(match.group(1))
    return value * 0.3048 if match.group(2) in ("ft", "'") else value


def building_height(tags: dict, floors: int | None = None) -> tuple[float, str]:
    """How tall, and where that came from: OSM height, OSM floors, Muse's floor count, or a guess."""
    height = parse_height(tags.get("height"))
    if height:
        return height, "osm height"
    levels = parse_height(tags.get("building:levels"))
    if levels:
        return levels * FLOOR, "osm floors"
    if isinstance(floors, int) and floors > 0:
        return floors * FLOOR, "muse floors"
    return GUESS_FLOORS * FLOOR, "guess"


@dataclass
class Part:
    """One piece of the model with one material."""

    name: str
    positions: np.ndarray  # (n, 3) float32, glTF axes
    normals: np.ndarray  # (n, 3) float32
    uvs: np.ndarray  # (n, 2) float32
    indices: np.ndarray  # (m,) uint32, counter-clockwise from outside
    color: tuple[float, float, float, float]
    texture: bytes | None = None  # JPEG or PNG, stretched over the 0..1 texture coordinates
    cutout: bool = False  # the PNG's transparent pixels are holes (a cut-out), and both sides show
    colors: np.ndarray | None = None  # (n, 3) float32 0..1, a colour per vertex instead of (or under) a texture
    roughness: float = 0.9  # 0 mirror-smooth .. 1 matt
    metallic: float = 0.0


WALL_COLOUR = (0.82, 0.80, 0.76, 1.0)
ROOF_COLOUR = (0.45, 0.42, 0.40, 1.0)


def _gltf(x: float, y: float, up: float) -> tuple[float, float, float]:
    """Map metres (east, north) and height to glTF axes."""
    return x, up, -y


def wall_edges(poly: Polygon) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The footprint's walls, counter-clockwise from above, so the outside is on each edge's right.
    Their order is the wall numbering used everywhere."""
    ring = list(orient(poly, 1.0).exterior.coords)
    return [(a, b) for a, b in zip(ring[:-1], ring[1:], strict=True) if math.dist(a, b) > 1e-6]


def _part(name, pos, nrm, uv, idx, color, texture=None) -> Part:
    return Part(name, np.array(pos, np.float32), np.array(nrm, np.float32), np.array(uv, np.float32),
                np.array(idx, np.uint32), color, texture)


# How far a window's glass, a door, a garage door sits back from the wall's face (metres).
RECESS = {"window": 0.12, "door": 0.10, "garage door": 0.15}
GLASS = {"roughness": 0.12, "metallic": 0.35}  # smooth and a little mirror-like, over the photo's own picture
GLASS_COLOUR = (0.05, 0.07, 0.09, 1.0)  # glass on a built (not pictured) building: dark, the sky shows in it
FRAME = 0.06  # metres: how wide and deep a window or door frame is
PANE = 0.9  # metres: a window wider than this is split by glazing bars into panes about this wide
TRANSOM = 1.5  # metres: a window taller than this gets a bar across near its top
SILL = (0.05, 0.06, 0.05)  # metres: a window sill's thickness, how far it stands out, how far past the window each side
RIB = 0.5  # metres between the panels of a garage door


@dataclass(frozen=True)
class Opening:
    """A window or door in a wall: from `start` to `end` along it (fractions of the wall's length, from its left end
    seen from outside), from `bottom` to `top` metres up. Cut into the wall and set back by RECESS[kind]."""

    kind: str
    start: float
    end: float
    bottom: float
    top: float


@dataclass
class WallPhoto:
    """A straightened picture for the stretch of a wall from `start` to `end` (fractions of its length, from its
    left end seen from outside), from `bottom` metres up to `top` (higher than the wall where a gable stands on it;
    None means the wall's own height). `bottom` is above 0 when the photo was taken uphill of the lowest ground: the
    wall below it stays plain, as a foundation."""

    jpeg: bytes
    start: float = 0.0
    end: float = 1.0
    top: float | None = None
    bottom: float = 0.0
    bare_gables: frozenset[int] = frozenset()  # gables on this wall (by RoofPlan.gables number) the picture misses
    openings: tuple[Opening, ...] = ()  # windows and doors in the picture, made real


MIRROR_TURN = 10.0  # degrees: a wall within this of facing exactly the other way counts as opposite
MIRROR_LENGTH = 0.15  # ... and within this share of its length


def mirror_walls(poly: Polygon, photos: dict[int, WallPhoto], gables=()) -> dict[int, tuple[int, WallPhoto]]:
    """A symmetric building (the person says so): each wall no photo saw takes the picture of the opposite wall (facing
    the other way, about as long), flipped left to right, as a mirror-image side looks from outside. Gables on the
    wall it goes to aren't painted from it. Returns {wall: (the wall it copies, its picture)}."""
    import io

    from PIL import Image, ImageOps

    edges = wall_edges(poly)
    lengths = [math.dist(a, b) for a, b in edges]
    bearings = [math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) for a, b in edges]
    out = {}
    for w in range(len(edges)):
        if w in photos:
            continue
        opposite = [s for s in photos
                    if abs(abs((bearings[s] - bearings[w] + 180) % 360 - 180) - 180) <= MIRROR_TURN
                    and abs(lengths[s] - lengths[w]) <= MIRROR_LENGTH * lengths[w]]
        if not opposite:
            continue
        s = min(opposite, key=lambda k: abs(lengths[k] - lengths[w]))
        p = photos[s]
        buf = io.BytesIO()
        ImageOps.mirror(Image.open(io.BytesIO(p.jpeg))).save(buf, "JPEG", quality=90)
        flipped = tuple(Opening(o.kind, 1 - o.end, 1 - o.start, o.bottom, o.top) for o in p.openings)
        bare = frozenset(k for k, g in enumerate(gables) if g.wall == w)
        out[w] = (s, WallPhoto(buf.getvalue(), 1 - p.end, 1 - p.start, p.top, p.bottom, bare, flipped))
    return out


class _Mesh:
    """Faces gathered into one part. Each face is flat and convex, its corners counter-clockwise from outside."""

    def __init__(self):
        self.pos, self.nrm, self.uv, self.idx = [], [], [], []

    def add(self, corners: list[tuple[float, float, float]], uvs: list[tuple[float, float]],
            normal: tuple[float, float, float] | None = None) -> None:
        if normal is None:
            a, b, c = (np.array(p, float) for p in corners[:3])
            n = np.cross(b - a, c - a)
            normal = tuple(n / (np.linalg.norm(n) or 1.0))
        base = len(self.pos)
        self.pos += corners
        self.nrm += [normal] * len(corners)
        self.uv += uvs
        for k in range(1, len(corners) - 1):
            self.idx += [base, base + k, base + k + 1]

    def add_facing(self, corners: list, uvs: list, facing: tuple[float, float, float]) -> None:
        """A face whose corners may be either way round: turned so it faces `facing`."""
        a, b, c = (np.array(p, float) for p in corners[:3])
        if np.dot(np.cross(b - a, c - a), facing) < 0:
            corners, uvs = corners[::-1], uvs[::-1]
        self.add(corners, uvs, facing)

    def part(self, name: str, color, texture: bytes | None = None) -> Part | None:
        return _part(name, self.pos, self.nrm, self.uv, self.idx, color, texture) if self.pos else None


def _wall_at(edge):
    """For a wall: where a point is (east, north, up) `t` of the way along it, `z` up and `back` metres in from its
    face; the direction along it (glTF axes); its length."""
    (x0, y0), (x1, y1) = edge
    length = math.dist((x0, y0), (x1, y1))
    nx, ny = (y1 - y0) / length, -(x1 - x0) / length

    def at(t: float, z: float, back: float = 0.0) -> tuple[float, float, float]:
        return x0 + t * (x1 - x0) - nx * back, y0 + t * (y1 - y0) - ny * back, z

    return at, _gltf((x1 - x0) / length, (y1 - y0) / length, 0.0), length


def _holed(wall: _Mesh, backs: dict[str, _Mesh], edge, stretch: tuple[float, float], rise: tuple[float, float],
           openings, uv, normal) -> None:
    """A stretch of a wall with its windows and doors cut out and set back: the wall in pieces round the holes; at
    the back of each hole a face into backs[kind] (or the wall's), framed by reveals in the wall's own material.
    `uv` maps a point (east, north, up) to the wall's picture."""
    s, e = stretch
    z0, z1 = rise
    holes = [o for o in openings if o.start >= s and o.end <= e and o.bottom >= z0 and o.top <= z1]
    at, along, _ = _wall_at(edge)

    ts = sorted({s, e, *(o.start for o in holes), *(o.end for o in holes)})
    zs = sorted({z0, z1, *(o.bottom for o in holes), *(o.top for o in holes)})
    for ta, tb in zip(ts[:-1], ts[1:], strict=True):
        for za, zb in zip(zs[:-1], zs[1:], strict=True):
            tm, zm = (ta + tb) / 2, (za + zb) / 2
            if any(o.start < tm < o.end and o.bottom < zm < o.top for o in holes):
                continue
            corners = [at(ta, za), at(tb, za), at(tb, zb), at(ta, zb)]
            wall.add([_gltf(*c) for c in corners], [uv(*c) for c in corners], normal)

    for o in holes:
        d = RECESS.get(o.kind, 0.1)
        back = [at(o.start, o.bottom, d), at(o.end, o.bottom, d), at(o.end, o.top, d), at(o.start, o.top, d)]
        backs.get(o.kind, wall).add([_gltf(*c) for c in back], [uv(*c) for c in back], normal)
        # The reveals: each side of the hole, facing into it.
        for (ta, za), (tb, zb), facing in (
            ((o.start, o.bottom), (o.start, o.top), along),  # left side, facing right
            ((o.end, o.bottom), (o.end, o.top), tuple(-v for v in along)),  # right side, facing left
            ((o.start, o.top), (o.end, o.top), (0.0, -1.0, 0.0)),  # the top, facing down
            ((o.start, o.bottom), (o.end, o.bottom), (0.0, 1.0, 0.0)),  # the sill, facing up
        ):
            corners = [at(ta, za), at(tb, zb), at(tb, zb, d), at(ta, za, d)]
            wall.add_facing([_gltf(*c) for c in corners], [uv(*c) for c in corners], facing)


def _box(mesh: _Mesh, at, normal, along, t: tuple[float, float], z: tuple[float, float],
         back: tuple[float, float]) -> None:
    """A box on a wall: `t` along it (fractions of its length), `z` up, `back` metres in from its face (negative
    stands out in front). `at`, `along` are _wall_at's; `normal` is the wall's outward one, glTF axes."""
    (t0, t1), (z0, z1), (b0, b1) = t, z, back
    away = tuple(-v for v in normal)
    left = tuple(-v for v in along)
    for corners, facing in (
        (((t0, z0, b0), (t1, z0, b0), (t1, z1, b0), (t0, z1, b0)), normal),
        (((t0, z0, b1), (t1, z0, b1), (t1, z1, b1), (t0, z1, b1)), away),
        (((t0, z1, b0), (t1, z1, b0), (t1, z1, b1), (t0, z1, b1)), (0.0, 1.0, 0.0)),
        (((t0, z0, b0), (t1, z0, b0), (t1, z0, b1), (t0, z0, b1)), (0.0, -1.0, 0.0)),
        (((t1, z0, b0), (t1, z1, b0), (t1, z1, b1), (t1, z0, b1)), along),
        (((t0, z0, b0), (t0, z1, b0), (t0, z1, b1), (t0, z0, b1)), left),
    ):
        mesh.add_facing([_gltf(*at(*c)) for c in corners], [(0.0, 0.0)] * 4, facing)


def _fittings(meshes: dict[str, _Mesh], edge, openings, normal) -> None:
    """What fills each opening, built rather than pictured: a frame round it (windows also glazing bars and a sill
    standing out below), a door its handle, a garage door ribs between its panels. The glass and door leaves
    themselves are _holed's back faces."""
    at, along, length = _wall_at(edge)
    f = FRAME / length
    for o in openings:
        d = RECESS.get(o.kind, 0.1)
        ring = [((o.start, o.start + f), (o.bottom, o.top)), ((o.end - f, o.end), (o.bottom, o.top)),
                ((o.start + f, o.end - f), (o.top - FRAME, o.top))]
        if o.kind == "window":
            ring.append(((o.start + f, o.end - f), (o.bottom, o.bottom + FRAME)))
            panes = max(1, round((o.end - o.start) * length / PANE))
            for k in range(1, panes):
                m = o.start + (o.end - o.start) * k / panes
                ring.append(((m - f / 3, m + f / 3), (o.bottom, o.top)))
            if o.top - o.bottom > TRANSOM:  # a tall window: a top light above a bar
                z = o.top - TRANSOM / 3
                ring.append(((o.start + f, o.end - f), (z - FRAME / 3, z + FRAME / 3)))
            high, out, past = SILL
            _box(meshes["frames"], at, normal, along, (o.start - past / length, o.end + past / length),
                 (o.bottom - high, o.bottom), (-out, 0.02))
        elif o.kind == "door":
            near = o.end - f - 0.06 / length
            _box(meshes["frames"], at, normal, along, (near - 0.12 / length, near), (o.bottom + 0.95, o.bottom + 1.0),
                 (d - 0.06, d))
        else:  # a garage door
            z = o.bottom + RIB
            while z < o.top - FRAME:
                _box(meshes["garage door"], at, normal, along, (o.start + f, o.end - f), (z - 0.02, z + 0.02),
                     (d - 0.025, d))
                z += RIB
        for t, z in ring:
            _box(meshes["frames"], at, normal, along, t, z, (d - FRAME, d))


def extrude(poly: Polygon, height: float, photos: dict[int, WallPhoto] | None = None, roof=None,
            look: Look | None = None) -> list[Part]:
    """Walls, and a roof: flat, or the pitched one planned by roof.plan_roof. Walls don't share corners, so each
    gets its own flat normal and texture.

    `photos` puts pictures on walls (by their wall_edges number); each of those is a part of its own, and the rest of
    that wall stays plain. A gable end standing on a wall with a picture is painted from that picture too.

    With a `look`, the building is built instead of pictured: every wall plain in the look's colours, and the
    photos' windows and doors (WallPhoto.openings) made of frames, glass and leaves.
    """
    photos = photos or {}
    if look is not None:
        return _built(poly, height, photos, roof, look)
    plain, painted, glass = _Mesh(), {i: _Mesh() for i in photos}, {i: _Mesh() for i in photos}
    edges = wall_edges(poly)

    def uv(i: int, x: float, y: float, z: float) -> tuple[float, float]:
        """Where a point on wall i falls in its picture."""
        (x0, y0), (x1, y1) = edges[i]
        t = ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2)
        photo = photos[i]
        span = (photo.top or height) - photo.bottom
        return (t - photo.start) / (photo.end - photo.start), 1 - (z - photo.bottom) / span

    for i, ((x0, y0), (x1, y1)) in enumerate(edges):
        length = math.dist((x0, y0), (x1, y1))
        normal = _gltf((y1 - y0) / length, -(x1 - x0) / length, 0.0)
        photo = photos.get(i)
        cuts = [0.0, photo.start, photo.end, 1.0] if photo else [0.0, 1.0]
        for s, e in zip(cuts[:-1], cuts[1:], strict=True):
            if e - s < 1e-6:
                continue
            a = (x0 + s * (x1 - x0), y0 + s * (y1 - y0))
            b = (x0 + e * (x1 - x0), y0 + e * (y1 - y0))
            # Seen from outside, the edge's start is on the left: bottom-left, bottom-right, top-right, top-left.
            painted_from = photo.bottom if photo and (s, e) == (photo.start, photo.end) else height
            for z0, z1 in ((0.0, painted_from), (painted_from, height)):
                if z1 - z0 < 1e-6:
                    continue
                corners = [(*a, z0), (*b, z0), (*b, z1), (*a, z1)]
                quad = [_gltf(*c) for c in corners]
                if z0 == painted_from:
                    _holed(painted[i], {"window": glass[i]}, edges[i], (s, e), (z0, z1), photo.openings,
                           lambda x, y, z, i=i: uv(i, x, y, z), normal)
                else:
                    plain.add(quad, [(0, 1), (1, 1), (1, 0), (0, 0)], normal)

    sloped = _Mesh()
    if roof is None or roof.kind == "flat":
        flats = [poly]
    else:
        flats = roof.flats
        for face in roof.slopes:
            sloped.add([_gltf(*c) for c in face], [(x, -y) for x, y, _ in face])  # a metre per unit, for tiling
        for k, g in enumerate(roof.gables):
            if g.wall in photos and k not in photos[g.wall].bare_gables:
                us = [uv(g.wall, *c) for c in g.points]
                if all(-0.02 <= u <= 1.02 for u, _ in us):  # wholly inside the wall's picture
                    painted[g.wall].add([_gltf(*c) for c in g.points], us)
                    continue
            plain.add([_gltf(*c) for c in g.points], [(0, 1), (1, 1), (0.5, 0)])
    _flats(sloped, flats, height)

    parts = [plain.part("walls", WALL_COLOUR)]
    parts += [painted[i].part(f"wall {i}", (1, 1, 1, 1), photos[i].jpeg) for i in sorted(painted)]
    for i in sorted(glass):
        part = glass[i].part(f"glass {i}", (1, 1, 1, 1), photos[i].jpeg)
        if part is not None:
            part.roughness, part.metallic = GLASS["roughness"], GLASS["metallic"]
            parts.append(part)
    parts.append(sloped.part("roof", ROOF_COLOUR))
    return [p for p in parts if p is not None]


def _flats(mesh: _Mesh, flats, height: float) -> None:
    for flat in flats:
        for tri in shapely.constrained_delaunay_triangles(flat).geoms:
            tri = orient(tri, 1.0)  # counter-clockwise from above faces up
            corners = [(x, y, height) for x, y in tri.exterior.coords[:3]]
            mesh.add([_gltf(*c) for c in corners], [(x, -y) for x, y, _ in corners], (0.0, 1.0, 0.0))


def _built(poly: Polygon, height: float, photos: dict[int, WallPhoto], roof, look: Look) -> list[Part]:
    """extrude with a look: plain walls with real windows and doors, every part one colour."""
    meshes = {k: _Mesh() for k in ("walls", "glass", "door", "garage door", "frames", "roof")}
    backs = {"window": meshes["glass"], "door": meshes["door"], "garage door": meshes["garage door"]}
    for i, edge in enumerate(wall_edges(poly)):
        (x0, y0), (x1, y1) = edge
        length = math.dist((x0, y0), (x1, y1))
        normal = _gltf((y1 - y0) / length, -(x1 - x0) / length, 0.0)
        openings = photos[i].openings if i in photos else ()
        _holed(meshes["walls"], backs, edge, (0.0, 1.0), (0.0, height), openings,
               lambda x, y, z: (0.0, 0.0), normal)
        _fittings(meshes, edge, openings, normal)
    if roof is None or roof.kind == "flat":
        flats = [poly]
    else:
        flats = roof.flats
        for face in roof.slopes:
            meshes["roof"].add([_gltf(*c) for c in face], [(x, -y) for x, y, _ in face])
        for g in roof.gables:
            meshes["walls"].add([_gltf(*c) for c in g.points], [(0, 1), (1, 1), (0.5, 0)])
    _flats(meshes["roof"], flats, height)

    colours = {"walls": look.walls, "glass": GLASS_COLOUR, "door": look.door, "garage door": look.garage_door,
               "frames": look.frames, "roof": look.roof}
    parts = []
    for name, mesh in meshes.items():
        part = mesh.part(name, colours[name])
        if part is None:
            continue
        if name == "glass":
            part.roughness, part.metallic = GLASS["roughness"], GLASS["metallic"]
        elif name in ("frames", "garage door"):
            part.roughness = 0.5
        parts.append(part)
    return parts


def write_glb(path: Path, parts: list[Part], name: str = "building") -> Path:
    """A self-contained glTF 2.0 binary: one mesh, one primitive and one material per part."""
    blob = bytearray()
    views, accessors, primitives, materials, images, textures = [], [], [], [], [], []

    def view(data: bytes, target: int | None = None) -> int:
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        if target:
            views[-1]["target"] = target
        blob.extend(data)
        blob.extend(b"\0" * (-len(blob) % 4))
        return len(views) - 1

    def add(array: np.ndarray, kind: str, target: int, bounds: bool = False) -> int:
        view(array.tobytes(), target)
        acc = {"bufferView": len(views) - 1, "count": len(array), "type": kind,
               "componentType": 5125 if array.dtype == np.uint32 else 5126}
        if bounds:
            acc["min"], acc["max"] = array.min(axis=0).tolist(), array.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for part in parts:
        attributes = {
            "POSITION": add(part.positions, "VEC3", 34962, bounds=True),
            "NORMAL": add(part.normals, "VEC3", 34962),
            "TEXCOORD_0": add(part.uvs, "VEC2", 34962),
        }
        if part.colors is not None:
            attributes["COLOR_0"] = add(part.colors, "VEC3", 34962)
        primitives.append({"attributes": attributes, "indices": add(part.indices, "SCALAR", 34963),
                           "material": len(materials)})
        pbr = {"baseColorFactor": list(part.color), "metallicFactor": part.metallic,
               "roughnessFactor": part.roughness}
        if part.texture:
            png = part.texture[1:4] == b"PNG"  # the PNG signature
            images.append({"bufferView": view(part.texture), "mimeType": "image/png" if png else "image/jpeg"})
            textures.append({"source": len(images) - 1, "sampler": 0})
            pbr["baseColorTexture"] = {"index": len(textures) - 1}
        material = {"name": part.name, "pbrMetallicRoughness": pbr}
        if part.cutout:
            material.update(alphaMode="MASK", alphaCutoff=0.5, doubleSided=True)
        materials.append(material)

    gltf = {
        "asset": {"version": "2.0", "generator": "placelens"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    if textures:  # smooth, and clamped at the wall's edges
        gltf.update(images=images, textures=textures,
                    samplers=[{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}])
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    chunks = struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(blob), 0x004E4942) + blob
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks)
    return path


def place_feature(outline: list[tuple[float, float]], anchor: tuple[float, float], height: float,
                  height_source: str, osm_id: str, tags: dict) -> dict:
    """place.geojson: the footprint where it really is, and the anchor the model's origin sits on."""
    ring = [[round(lon, 7), round(lat, 7)] for lat, lon in outline]
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "osm_id": osm_id,
                "name": tags.get("name"),
                "height_m": round(height, 2),
                "height_source": height_source,
                "anchor": [round(anchor[1], 7), round(anchor[0], 7)],  # lon, lat of the model's origin
                "model": "building.glb",
                "model_axes": "metres, Y up, X east, Z south",
            },
        }],
    }
