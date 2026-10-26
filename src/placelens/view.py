"""The view cone and which buildings in it you could actually see.

Rays are cast across the lens's width (plus a margin for compass error). Each ray stops at the first wall it
hits, so a building hidden behind another one gets no rays. A building's share of the rays is how much of the
photo it should fill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon

from .osm import Building

M_PER_DEG_LAT = 110_540


class LocalFrame:
    """Metres east/north of an origin. Flat-earth is fine at a few hundred metres."""

    def __init__(self, lat: float, lon: float):
        self.lat, self.lon = lat, lon
        self.m_per_deg_lon = 111_320 * math.cos(math.radians(lat))

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon - self.lon) * self.m_per_deg_lon, (lat - self.lat) * M_PER_DEG_LAT

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return self.lat + y / M_PER_DEG_LAT, self.lon + x / self.m_per_deg_lon


@dataclass
class Candidate:
    building: Building
    polygon: Polygon  # local metres
    share: float  # weighted share of the view's rays, 0..1
    distance: float  # metres to the nearest visible wall
    rank: int = 0


def direction(bearing: float) -> tuple[float, float]:
    rad = math.radians(bearing)
    return math.sin(rad), math.cos(rad)


def cone(heading: float, fov: float, reach: float, margin: float = 0) -> Polygon:
    half = fov / 2 + margin
    steps = max(2, int(2 * half))
    arc = [direction(heading - half + 2 * half * i / steps) for i in range(steps + 1)]
    return Polygon([(0, 0)] + [(dx * reach, dy * reach) for dx, dy in arc])


def find_candidates(
    buildings: list[Building],
    frame: LocalFrame,
    heading: float,
    fov: float,
    reach: float = 200,
    margin: float = 15,
    step: float = 0.5,
) -> list[Candidate]:
    """Buildings you could see, biggest share of the view first.

    Rays inside the lens's width count fully; rays in the compass-error margin count half.
    """
    origin = Point(0, 0)
    polys = []
    for b in buildings:
        poly = Polygon([frame.to_xy(lat, lon) for lat, lon in b.outline])
        if not poly.is_valid:
            poly = poly.buffer(0)
        # Standing "inside" a building is GPS error, not a view of it.
        if poly.is_empty or poly.contains(origin):
            continue
        polys.append((b, poly))

    view = cone(heading, fov, reach, margin)
    polys = [(b, p) for b, p in polys if p.intersects(view)]

    hits: dict[str, list] = {}
    half = fov / 2 + margin
    n = int(2 * half / step)
    total = 0.0
    for i in range(n + 1):
        offset = -half + i * step
        weight = 1.0 if abs(offset) <= fov / 2 else 0.5
        total += weight
        dx, dy = direction(heading + offset)
        ray = LineString([(0, 0), (dx * reach, dy * reach)])
        nearest = None
        for b, p in polys:
            if ray.intersects(p):
                d = origin.distance(ray.intersection(p))
                if nearest is None or d < nearest[0]:
                    nearest = (d, b, p)
        if nearest:
            d, b, p = nearest
            entry = hits.setdefault(b.osm_id, [b, p, 0.0, d])
            entry[2] += weight
            entry[3] = min(entry[3], d)

    candidates = [Candidate(b, p, w / total, d) for b, p, w, d in hits.values()]
    candidates.sort(key=lambda c: (-c.share, c.distance))
    for i, c in enumerate(candidates, 1):
        c.rank = i
    return candidates
