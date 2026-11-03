"""A building from phone photos: which building the photos show.

All the photos together, by triangulation. Each photo is a line on the map: from where it was taken (its GPS), in the
direction of the building's middle (its compass, and where SAM 3.1 found the building across the photo), out to about
how far away it looked (Muse, asked without any map, so its answer never changes with the map). The lines meet at the
building: a photo whose compass is off bends its own line, and the others still meet there. The building at the
meeting point is the one; photos whose line misses it are left out, each with why.

The old way, each photo voting on its own from a numbered list of nearby buildings, was tuned with the house nearly
alone on the map: with Overture's fuller map (Google- and Microsoft-traced buildings) it chose a neighbour, 6 photos
to 1 (1 Nov).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .progress import Counter, step

COMPASS_ERROR = 12.0  # degrees: a phone compass held still, outdoors
DISTANCE_ERROR = 0.35  # share of Muse's distance it may be out (its middle of 3 answers)
MISS = 3.0  # a photo whose line misses the meeting point by more than this many of its own errors is left out
SEARCH = 150  # metres round the photos for the map's buildings
CLOSE_CALL = 1.5  # the chosen building's support must beat the next one's this many times, or it's a close call
AT_ONCE = 4  # photos looked at together (SAM, Muse)
SAM_NOUNS = ("building", "house")  # asked in turn till SAM finds one (wide-lens photos answered only to "house")

HOW_FAR_PROMPT = """This photo shows a building. Answer in JSON only:
{"distance_m": how far the camera is from the building's nearest wall, in metres, \
"floors": how many floors the building has, \
"whole": true if the whole building's front is in the photo}"""


@dataclass
class Sight:
    """One photo's line to the building."""

    photo: Path
    small: Path
    east: float  # metres from the frame's origin
    north: float
    bearing: float  # degrees: from the camera to the building's middle
    distance: float | None  # metres, Muse's (None: not asked or no answer)
    floors: int | None
    fov: float
    heading: float
    box: tuple[int, int, int, int] | None = None  # SAM's outline's box in the small photo
    used: bool = True
    why: str = ""
    miss_m: float = 0.0  # how far its line passes from the meeting point


@dataclass
class Choice:
    building: object | None  # osm.Building
    at: tuple[float, float]  # the meeting point, metres east and north of the frame's origin
    frame: object  # view.LocalFrame
    sights: list[Sight]
    support: dict = field(default_factory=dict)  # building id -> support (sum over photos)
    close_call: bool = False
    why: str = ""

    def as_dict(self) -> dict:
        return {"building": self.building.osm_id if self.building else None, "close_call": self.close_call,
                "why": self.why, "meeting_point_m": [round(v, 1) for v in self.at],
                "photos": [{"photo": s.photo.name, "used": s.used, "why": s.why, "bearing": round(s.bearing, 1),
                            "distance_m": s.distance, "miss_m": round(s.miss_m, 1)} for s in self.sights],
                "support": {k: round(v, 2) for k, v in sorted(self.support.items(), key=lambda kv: -kv[1])[:5]}}


def sights(photos: list[Path], meta, replies: Path) -> tuple[list[Sight], object]:
    """Each photo's line to the building: SAM's outline (saved answers in `replies` reused), Muse's distance, the
    photo's GPS and compass. Photos without GPS or compass, or without a building SAM can find, are left out."""
    from PIL import Image

    from .meta import OPEN, segments_from_file
    from .muse import ask
    from .photo import read_camera
    from .pick import pixel_to_offset, prepare_photo
    from .view import LocalFrame

    cams = {p: read_camera(p) for p in photos}
    located = [p for p in photos if cams[p].lat is not None and cams[p].heading is not None]
    if not located:
        raise ValueError("no photo has both GPS and a compass direction")
    frame = LocalFrame(float(np.mean([cams[p].lat for p in located])), float(np.mean([cams[p].lon for p in located])))
    replies.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(HOW_FAR_PROMPT.encode()).hexdigest()[:8]
    looking = Counter("SAM 3.1 and Muse are looking at photo", len(located))

    def one(p: Path) -> Sight:
        cam = cams[p]
        small = prepare_photo(p, replies / f"{p.stem}.jpg")
        east, north = frame.to_xy(cam.lat, cam.lon)
        s = Sight(p, small, east, north, cam.heading, None, None, cam.fov, cam.heading)
        try:
            segments = []
            for noun in SAM_NOUNS:  # (named as pick saved them, so its answers count)
                reply = replies / (f"{p.stem}_sam.txt" if noun == SAM_NOUNS[0] else f"{p.stem}_sam_{noun}.txt")
                if reply.exists():
                    segments = segments_from_file(reply)
                else:
                    segments, raw = meta.segment(small, noun)
                    reply.write_text(raw, encoding="utf-8")
                if segments:
                    break
            if not segments:
                s.used, s.why = False, "SAM 3.1 found no building in it"
                return s
            s.box = segments[0].box
            width = Image.open(small).width
            s.bearing = (cam.heading + pixel_to_offset((s.box[0] + s.box[2]) / 2, width, cam.fov)) % 360
            found = {}
            for a in ask(meta, HOW_FAR_PROMPT, small, replies / f"{p.stem}_howfar_{tag}.txt", OPEN, votes=3):
                m = re.search(r"\{.*\}", a, re.S)
                try:
                    d = json.loads(m.group(0)) if m else {}
                except json.JSONDecodeError:
                    continue
                for k in ("distance_m", "floors"):
                    if isinstance(d.get(k), (int, float)) and d[k] > 0:
                        found.setdefault(k, []).append(float(d[k]))
            s.distance = float(np.median(found["distance_m"])) if "distance_m" in found else None
            s.floors = int(round(np.median(found["floors"]))) if "floors" in found else None
            return s
        except Exception as e:
            if type(e).__name__ == "PaidCallBlocked":
                raise
            s.used, s.why = False, f"couldn't look at it ({type(e).__name__})"
            return s
        finally:
            looking.tick()

    with ThreadPoolExecutor(AT_ONCE) as pool:
        out = list(pool.map(one, located))
    for p in photos:
        if p not in located:
            out.append(Sight(p, p, 0, 0, 0, None, None, cams[p].fov, 0, used=False,
                             why="no GPS or no compass direction in it"))
    return out, frame


def _unit(bearing: float) -> np.ndarray:
    b = math.radians(bearing)
    return np.array([math.sin(b), math.cos(b)])


def meet(lines: list[Sight]) -> tuple[float, float]:
    """The point the photos' lines meet at, by weighted least squares: across each line it's off by the compass's
    error at that distance, along it by Muse's distance's error. Photos that miss by far are left out, and it's
    worked out again without them."""
    use = [s for s in lines if s.used]
    # Start from the middle of where each photo alone puts it (its line, out to its distance): an outlier can't pull
    # the middle, so it's seen and left out before the fit, not after it has bent the fit towards itself.
    if len(use) >= 3:
        alone = np.array([[s.east, s.north] + _unit(s.bearing) * (s.distance or 15.0) for s in use])
        middle = np.median(alone, axis=0)
        for s in use:
            u = _unit(s.bearing)
            rel = middle - np.array([s.east, s.north])
            allowed = MISS * max((s.distance or 15.0) * math.tan(math.radians(COMPASS_ERROR)), 2.0)
            off = abs(float(np.array([-u[1], u[0]]) @ rel))
            if off > allowed or float(u @ rel) < 0:
                s.used, s.miss_m = False, off
                s.why = f"its compass or GPS is off: its line misses where the others meet by {off:.0f} m"
        use = [s for s in use if s.used]
    for _ in range(3):
        if not use:
            break
        a = np.zeros((2, 2))
        b = np.zeros(2)
        for s in use:
            u = _unit(s.bearing)
            n = np.array([-u[1], u[0]])
            p = np.array([s.east, s.north])
            d = s.distance or 15.0
            across = 1.0 / max(d * math.tan(math.radians(COMPASS_ERROR)), 1.0) ** 2
            a += across * np.outer(n, n)
            b += across * np.outer(n, n) @ p
            if s.distance:  # (its distance pins it along the line)
                along = 1.0 / max(DISTANCE_ERROR * d, 1.0) ** 2
                a += along * np.outer(u, u)
                b += along * (np.outer(u, u) @ p + u * d)
        x = np.linalg.solve(a + np.eye(2) * 1e-6, b)
        worst = None
        for s in use:
            u = _unit(s.bearing)
            n = np.array([-u[1], u[0]])
            rel = x - np.array([s.east, s.north])
            s.miss_m = abs(float(n @ rel))
            allowed = MISS * max((s.distance or 15.0) * math.tan(math.radians(COMPASS_ERROR)), 2.0)
            if s.miss_m > allowed and (worst is None or s.miss_m / allowed > worst[1]):
                worst = (s, s.miss_m / allowed)
            if float(u @ rel) < 0:  # (the point is behind the camera)
                worst = (s, 99.0)
        if worst is None:
            return float(x[0]), float(x[1])
        worst[0].used = False
        worst[0].why = (f"its compass or GPS is off: its line misses where the others meet by {worst[0].miss_m:.0f} m")
        use = [s for s in use if s.used]
    return (float(x[0]), float(x[1])) if use else (0.0, 0.0)


def choose(photos: list[str | Path], meta, replies: Path, use_cache: bool = True) -> Choice:
    """Which building the photos show: where their lines meet, and the building there."""
    from shapely.geometry import Point, Polygon

    from .footprints import overture_buildings

    photos = [Path(p) for p in photos]
    lines, frame = sights(photos, meta, replies)
    step("triangulating: where the photos point")
    at = meet(lines)
    lat, lon = frame.to_latlon(*at)
    step("Overture Maps: the buildings round there")
    try:
        buildings = overture_buildings(lat, lon, SEARCH, use_cache)
    except Exception as e:  # (Overture unreachable: OpenStreetMap)
        from .osm import fetch_buildings

        print(f"buildings: Overture failed ({type(e).__name__}); OpenStreetMap instead")
        buildings = fetch_buildings(lat, lon, SEARCH, use_cache)
    point = Point(at)
    polys = []
    for b in buildings:
        poly = Polygon([frame.to_xy(*p) for p in b.outline])
        polys.append((b, poly if poly.is_valid else poly.buffer(0)))
    # Each used photo supports the buildings near where its line meets the others: the nearer the building to its
    # line (within its compass's error), and the nearer the meeting point, the more.
    from shapely.geometry import LineString

    support: dict[str, float] = {}
    for s in [s for s in lines if s.used]:
        start = np.array([s.east, s.north])
        reach = 2.5 * (s.distance or 30.0) + 10.0
        line = LineString([tuple(start), tuple(start + _unit(s.bearing) * reach)])
        sigma = max((s.distance or 15.0) * math.tan(math.radians(COMPASS_ERROR)), 2.0)
        for b, poly in polys:
            if poly.distance(point) > 25.0:
                continue
            on_line = math.exp(-0.5 * (poly.distance(line) / sigma) ** 2)
            by_point = math.exp(-0.5 * (poly.distance(point) / 6.0) ** 2)
            support[b.osm_id] = support.get(b.osm_id, 0.0) + on_line * by_point
    if not support:
        return Choice(None, at, frame, lines, {}, False, "no building on the map where the photos point")
    ranked = sorted(support.items(), key=lambda kv: -kv[1])
    best = next(b for b, _ in polys if b.osm_id == ranked[0][0])
    close = len(ranked) > 1 and ranked[0][1] < CLOSE_CALL * ranked[1][1]
    inside = next((b for b, poly in polys if poly.contains(point)), None)
    why = (f"the photos' lines meet {'inside' if inside is best else 'by'} it; "
           f"{sum(s.used for s in lines)} of {len(lines)} photos agree")
    if close:
        why += f" (a close call with {ranked[1][0]})"
    return Choice(best, at, frame, lines, support, close, why)
