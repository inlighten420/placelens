"""Roofs. The footprint is cut into rectangular blocks at its inward corners, and each block gets a gable, hip or
flat roof. Muse says which kind, which way the ridges run and how steep; the footprint says where.

An L-shaped house becomes two blocks, each with its ridge along its length: two gables side by side, as most
L-shaped houses are built. Gable ends on the outside walls are part of those walls, so the photo paints them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import split

from .build import wall_edges

KINDS = ("gable", "hip", "flat")
PITCH = 30.0  # degrees, when nobody says
MIN_PITCH, MAX_PITCH = 10.0, 50.0
SQUARE = 0.95  # a block filling this much of its bounding rectangle is roofed as a rectangle
ON_WALL = 0.3  # metres: a gable end this close to a footprint wall stands on it

Point3 = tuple[float, float, float]  # metres east, north, up


@dataclass
class Gable:
    """A gable end: the triangle between the eaves and the ridge, on wall `wall` (None if inside the building,
    where one block meets another)."""

    wall: int | None
    points: list[Point3]
    block: int = 0  # which of the plan's blocks it ends


@dataclass
class RoofPlan:
    kind: str
    pitch: float
    blocks: list[Polygon]
    slopes: list[list[Point3]] = field(default_factory=list)  # each a flat, convex face, outside counter-clockwise
    gables: list[Gable] = field(default_factory=list)
    flats: list[Polygon] = field(default_factory=list)  # blocks too odd for a pitched roof: flat at the eaves

    def tops(self) -> dict[int, float]:
        """How high each wall reaches where a gable stands on it."""
        tops: dict[int, float] = {}
        for g in self.gables:
            if g.wall is not None:
                tops[g.wall] = max(tops.get(g.wall, 0.0), max(z for _, _, z in g.points))
        return tops


def _fill(p: Polygon) -> float:
    """How much of its bounding rectangle a shape fills, 0..1."""
    return p.area / p.minimum_rotated_rectangle.area


def _frame(p: Polygon) -> tuple[Point, list[tuple[tuple[float, float], float]]]:
    """Centre of the bounding rectangle and its two axes (unit vector, half length), long axis first."""
    rect = p.minimum_rotated_rectangle
    (x0, y0), (x1, y1), (x2, y2) = list(rect.exterior.coords)[:3]
    axes = []
    for dx, dy in ((x1 - x0, y1 - y0), (x2 - x1, y2 - y1)):
        length = math.hypot(dx, dy)
        axes.append(((dx / length, dy / length), length / 2))
    axes.sort(key=lambda a: -a[1])
    return rect.centroid, axes


def _aspect(p: Polygon) -> float:
    _, ((_, long), (_, short)) = _frame(p)
    return long / max(short, 1e-6)


def _reflex(poly: Polygon):
    """Inward corners, with the directions of the walls either side of them."""
    ring = list(orient(poly, 1.0).exterior.coords)[:-1]
    n = len(ring)
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        d_in, d_out = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
        if d_in[0] * d_out[1] - d_in[1] * d_out[0] < -1e-9:  # a right turn on a counter-clockwise ring
            yield b, d_in, d_out


def blocks(poly: Polygon, depth: int = 4) -> list[Polygon]:
    """The footprint cut into blocks that are as rectangular, then as long, as possible. Each cut runs on from a
    wall at an inward corner, the way an L or T is built from rectangles."""
    if depth == 0 or _fill(poly) >= SQUARE:
        return [poly]
    best = None
    reach = poly.length
    for (vx, vy), d_in, d_out in _reflex(poly):
        # Carry the wall coming in straight on, or the wall going out straight back: both lead inside. (A line
        # along the wall itself would run over the boundary, which doesn't cut.)
        for dx, dy in (d_in, (-d_out[0], -d_out[1])):
            n = math.hypot(dx, dy)
            ux, uy = dx / n, dy / n
            line = LineString([(vx, vy), (vx + ux * reach, vy + uy * reach)])
            pieces = [g for g in split(poly, line).geoms if g.area > 0.5]
            if len(pieces) < 2:
                continue
            fill = sum(_fill(g) * g.area for g in pieces) / poly.area
            aspect = sum(_aspect(g) * g.area for g in pieces) / poly.area
            key = (round(fill, 2), aspect)
            if best is None or key > best[0]:
                best = (key, pieces)
    if best is None:
        return [poly]
    return [b for p in best[1] for b in blocks(p, depth - 1)]


def _bearing(u: tuple[float, float]) -> float:
    return math.degrees(math.atan2(u[0], u[1])) % 180


def _outward(face: list[Point3], centre: Point3) -> list[Point3]:
    """The face's corners ordered counter-clockwise seen from outside: from the side away from `centre`."""
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = face[:3]
    nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay)
    ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az)
    nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    mx, my, mz = (sum(p[k] for p in face) / len(face) for k in range(3))
    away = (mx - centre[0]) * nx + (my - centre[1]) * ny + (mz - centre[2]) * nz
    return face if away > 0 else face[::-1]


def _on_wall(a: tuple[float, float], b: tuple[float, float], edges) -> int | None:
    """Which footprint wall the segment a-b lies along, if any."""
    for i, (p, q) in enumerate(edges):
        wall = LineString([p, q])
        if wall.distance(Point(a)) < ON_WALL and wall.distance(Point(b)) < ON_WALL:
            return i
    return None


def plan_roof(poly: Polygon, eave: float, kind: str = "flat", pitch: float = PITCH,
              ridge: float | None = None, pitches: dict[int, float] | None = None) -> RoofPlan:
    """The roof on a footprint whose walls are `eave` metres tall. `ridge` is the compass bearing the ridges run
    along (Muse's view, from the photos); a block whose axes are both a fair way off it, or without one, uses its
    long axis. `pitches` overrides `pitch` for some blocks (by number), where a photo measured them."""
    pitch = min(max(pitch, MIN_PITCH), MAX_PITCH)
    plan = RoofPlan(kind, pitch, blocks(poly) if kind != "flat" else [poly])
    if kind == "flat":
        plan.flats = [poly]
        return plan
    edges = wall_edges(poly)
    for n, block in enumerate(plan.blocks):
        block_pitch = min(max((pitches or {}).get(n, pitch), MIN_PITCH), MAX_PITCH)
        if _fill(block) < SQUARE:
            plan.flats.append(block)
            continue
        c, axes = _frame(block)
        if ridge is not None:  # the axis nearer Muse's ridge direction, unless both are 30 degrees or more off it
            off = [abs((_bearing(u) - ridge + 90) % 180 - 90) for u, _ in axes]
            if min(off) < 30:
                axes.sort(key=lambda a: abs((_bearing(a[0]) - ridge + 90) % 180 - 90))
        _roof_block(plan, n, c, axes, eave, math.tan(math.radians(block_pitch)), edges)
    return plan


def _at(c: Point, axes, su: float, sv: float, z: float) -> Point3:
    """A point `su` half-lengths along the ridge and `sv` half-widths across it from the block's centre."""
    ((ux, uy), length), ((vx, vy), width) = axes
    return c.x + su * length * ux + sv * width * vx, c.y + su * length * uy + sv * width * vy, z


def _roof_block(plan: RoofPlan, n: int, c: Point, axes, eave: float, slope: float, edges) -> None:
    """One rectangular block's roof, ridge along its first axis, added to the plan."""
    (_, length), (_, width) = axes
    top = eave + width * slope
    centre = (c.x, c.y, eave)
    inset = min(width, length) / length if plan.kind == "hip" else 0.0  # a hip's ridge stops short of the ends
    r1, r2 = _at(c, axes, -1 + inset, 0, top), _at(c, axes, 1 - inset, 0, top)
    plan.slopes += [_outward([_at(c, axes, -1, -1, eave), _at(c, axes, 1, -1, eave), r2, r1], centre),
                    _outward([_at(c, axes, 1, 1, eave), _at(c, axes, -1, 1, eave), r1, r2], centre)]
    for end, r in ((-1, r1), (1, r2)):
        a, b = _at(c, axes, end, 1, eave), _at(c, axes, end, -1, eave)
        tri = _outward([a, b, r], centre)
        if plan.kind == "hip":
            plan.slopes.append(tri)
        else:
            plan.gables.append(Gable(_on_wall(a[:2], b[:2], edges), tri, n))


ROOF_PROMPT = """Look at the roof of the main building in this photo (the one outlined as the building).

Answer in JSON only:
{"roof": "gable", "hip" or "flat", \
"ridge": "toward" if the main ridge runs away from the camera (you face a triangular gable end, or a hip's sloping \
end), or "across" if it runs from left to right, \
"pitch_deg": the roof's slope in degrees}"""


@dataclass
class RoofView:
    """What one photo shows of the roof."""

    kind: str | None
    ridge: str | None  # "toward" or "across" the camera
    pitch: float | None


def parse_roof(text: str) -> RoofView:
    """Muse's JSON, tolerating code fences and chatter; anything unreadable becomes None."""
    import json
    import re

    match = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        d = {}
    kind = str(d.get("roof", "")).lower()
    ridge = str(d.get("ridge", "")).lower()
    pitch = d.get("pitch_deg")
    return RoofView(kind if kind in KINDS else None, ridge if ridge in ("toward", "across") else None,
                    float(pitch) if isinstance(pitch, (int, float)) and pitch > 0 else None)


def combine(views: list[tuple[RoofView, float]]) -> tuple[str, float, float | None]:
    """Several photos' views (each with the compass heading it was taken at) as one roof: the kind most say, the
    middle pitch, and the ridges' compass bearing (directions averaged as lines, so 10 and 190 degrees agree)."""
    kinds = [v.kind for v, _ in views if v.kind]
    kind = max(KINDS, key=lambda k: (kinds.count(k), k == "gable")) if kinds else "flat"
    pitches = sorted(v.pitch for v, _ in views if v.pitch)
    pitch = pitches[len(pitches) // 2] if pitches else PITCH
    bearings = [heading + (90 if v.ridge == "across" else 0) for v, heading in views if v.ridge]
    if not bearings:
        return kind, pitch, None
    x = sum(math.cos(math.radians(2 * b)) for b in bearings)
    y = sum(math.sin(math.radians(2 * b)) for b in bearings)
    return kind, pitch, math.degrees(math.atan2(y, x)) / 2 % 180
