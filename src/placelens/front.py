"""The building's real front: each wall the camera saw, cut out of the photo and straightened.

First the camera is locked: the spot and compass direction that put the footprint's left and right edges exactly on
SAM's outline, as close as possible to what the phone said. Then every wall facing the camera has its four corners
(ground and roof height) projected into the photo, and a homography warps that quad into a flat, upright texture.

Assumes the phone was held level at eye height; the roof line can sit a little high or low when it was tilted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from .build import wall_edges
from .pick import COMPASS_ERROR, DISTANCE_ERROR, GPS_ERROR, pixel_to_offset

EYE = 1.5  # metres above the ground the photo was taken from
EDGE_ERROR = 1.0  # degrees: how closely the footprint's edges must meet the outline's
BORDER = 3  # pixels: an outline this close to the photo's edge is cut off by it, not the building's real edge
MAX_TEXTURE = 2048  # pixels, longest side of a wall's texture
MAX_TURN = 15.0  # degrees: the most a compass is believed to be off (pick allows the same)
MAX_DEPTH_SCALE = 2.0  # Muse's distance may stretch or shrink the walls' look up to this much
NEAR = 1.0  # metres: closer than this to the camera is out of the picture
IN_VIEW = 1.0  # metres of a wall that must be in the photo to get a texture
UNHIDDEN = 0.5  # share of that which mustn't be behind another part of the building
PEAK_WINDOW = 0.015  # share of the photo's width searched either side of where a gable's peak should be
ROWS_STRETCH = 4.0  # the outline may make a wall look this many times taller than the level-phone projection


@dataclass
class Camera:
    """A level pinhole camera on the map: metres east/north, compass heading, lens width, photo size."""

    x: float
    y: float
    heading: float
    fov: float
    width: int
    height: int
    eye: float = EYE
    # How much further away the building looks than the map puts it. The map's footprint can be narrower or wider
    # than the real one, and then the edges pull the camera too near or far. Across the photo the edges are right
    # anyway; up the photo, sizes follow how far away it looks (Muse's estimate).
    depth_scale: float = 1.0
    edges: int = 0  # how many of the building's edges (left, right) the photo shows; 0 means placed by GPS alone
    # Metres above the building's lowest ground that this photo's ground is: a house on a slope shows fewer floors
    # from uphill. The eye is this plus EYE, and the photo only shows walls from here up.
    ground: float = 0.0
    misfit: float = 0.0  # degrees: how far the footprint's edges still are from the outline's, on average

    @property
    def focal(self) -> float:
        return self.width / 2 / math.tan(math.radians(self.fov / 2))

    def depth(self, x: float, y: float) -> float:
        """Metres in front of the camera, along its view."""
        h = math.radians(self.heading)
        return (x - self.x) * math.sin(h) + (y - self.y) * math.cos(h)

    def project(self, x: float, y: float, z: float) -> tuple[float, float]:
        """Pixel (column, row) of a point on the map `z` metres above the ground. It must be in front."""
        h = math.radians(self.heading)
        dx, dy = x - self.x, y - self.y
        forward = dx * math.sin(h) + dy * math.cos(h)
        right = dx * math.cos(h) - dy * math.sin(h)
        return (self.width / 2 + self.focal * right / forward,
                self.height / 2 - self.focal * (z - self.eye) / (forward * self.depth_scale))


def _wrap(angle: float) -> float:
    return (angle + 180) % 360 - 180


def lock_camera(
    poly: Polygon,
    box: tuple[int, int, int, int],
    size: tuple[int, int],
    heading: float,
    fov: float,
    distance: float | None = None,
    max_turn: float = MAX_TURN,
    reach: float = 15.0,
    step: float = 0.5,
    walls: list[int] | None = None,
    around: tuple[float, float] = (0.0, 0.0),
    gps_error: float = GPS_ERROR,
) -> Camera:
    """The camera that puts the footprint's edges on the outline's (`box`, in pixels of a photo `size`), searching
    spots up to `reach` metres from the GPS. Moving and turning further from what the phone said costs more.

    For a spot, the best turn has a closed form: the edges pull it to line up, the compass prior pulls it to zero.
    It is held within `max_turn`: one edge can always be lined up by turning far enough, but a compass is rarely that
    wrong, and a spot that needs it is the wrong spot. `misfit` then says how well the best spot lines up.

    `walls` (wall_edges numbers) says which walls the photo shows, when someone has said so ("this is the back"):
    then the outline's edges are those walls' ends, only spots in front of them count, and the search is centred
    on `around` instead of the GPS (whose error, `gps_error`, may then be allowed to be larger).
    """
    width, height = size
    edges = []  # (which end of the span, photo angle) for each edge the photo really shows
    if box[0] > BORDER:
        edges.append((0, pixel_to_offset(box[0], width, fov)))
    if box[2] < width - BORDER:
        edges.append((1, pixel_to_offset(box[2], width, fov)))
    pull = 1 / EDGE_ERROR**2
    prior = 1 / COMPASS_ERROR**2
    if walls:
        shown = [wall_edges(poly)[w] for w in walls]
        corners = np.array([p for edge in shown for p in edge])
        target = MultiLineString(shown)
        outward = [((b[1] - a[1]) / math.dist(a, b), -(b[0] - a[0]) / math.dist(a, b), a) for a, b in shown]
    else:
        corners = np.array(poly.exterior.coords[:-1])
        target = poly.exterior
        outward = []

    best = (math.inf, Camera(0, 0, heading, fov, width, height))
    n = int(reach / step)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            dx, dy = i * step, j * step
            x, y = around[0] + dx, around[1] + dy
            if dx * dx + dy * dy > reach * reach or poly.intersects(Point(x, y)):
                continue
            if any(nx * (x - a[0]) + ny * (y - a[1]) < 0.5 for nx, ny, a in outward):  # behind a wall it shows
                continue
            bearings = np.degrees(np.arctan2(corners[:, 0] - x, corners[:, 1] - y))
            offsets = [_wrap(b - heading) for b in bearings]
            span = (min(offsets), max(offsets))
            if span[1] - span[0] >= 180:
                continue
            # The turn that best lines both edges up: offsets fall by `turn` when the camera turns by it.
            gaps = [span[end] - angle for end, angle in edges]
            turn = min(max(pull * sum(gaps) / (pull * len(gaps) + prior), -max_turn), max_turn)
            cost = pull * sum((g - turn) ** 2 for g in gaps) + prior * turn**2 + (x * x + y * y) / gps_error**2
            if distance:
                wall = max(target.distance(Point(x, y)), 1.0)
                cost += (math.log(wall / max(distance, 1.0)) / DISTANCE_ERROR) ** 2
            if cost < best[0]:
                misfit = math.sqrt(sum((g - turn) ** 2 for g in gaps) / len(gaps)) if gaps else 0.0
                best = (cost, Camera(x, y, heading + turn, fov, width, height, edges=len(edges), misfit=misfit))
    cam = best[1]
    if distance:
        nearest = max(target.distance(Point(cam.x, cam.y)), 1.0)
        cam.depth_scale = min(max(distance / nearest, 1 / MAX_DEPTH_SCALE), MAX_DEPTH_SCALE)
    return cam


@dataclass
class WallView:
    index: int  # in wall_edges order
    start: float  # the stretch of the wall in the photo, as fractions of its length from its left end (outside)
    end: float
    seen: float  # share of the whole wall in the photo and not hidden by the building itself, 0..1
    quad: list[tuple[float, float]]  # that stretch in photo pixels: bottom-left, bottom-right, top-right, top-left
    top: float  # metres: the quad's top edge (the wall's height, or its gable's peak)
    bottom: float = 0.0  # metres: the quad's bottom edge (the ground where the photo was taken)


def visible_walls(poly: Polygon, cam: Camera, height: float, tops: dict[int, float] | None = None,
                  samples: int = 64) -> list[WallView]:
    """Walls facing the camera, with the stretch of each that's in the photo, from the ground up to `height` or, where
    a gable stands on the wall, its `tops`. Most seen first.

    A close-up can show part of a wall that runs out of the frame or past the camera; only that part is used.
    """
    views = []
    for i, ((x0, y0), (x1, y1)) in enumerate(wall_edges(poly)):
        length = math.dist((x0, y0), (x1, y1))
        nx, ny = (y1 - y0) / length, -(x1 - x0) / length  # outward
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if nx * (cam.x - mx) + ny * (cam.y - my) <= 0:
            continue
        inside, unhidden = [], 0
        for k in range(samples):
            t = (k + 0.5) / samples
            px, py = x0 + t * (x1 - x0) + 0.05 * nx, y0 + t * (y1 - y0) + 0.05 * ny
            if cam.depth(px, py) < NEAR or not 0 <= cam.project(px, py, cam.eye)[0] <= cam.width:
                continue
            inside.append(t)
            unhidden += not poly.intersects(LineString([(cam.x, cam.y), (px, py)]))
        if not inside or len(inside) * length / samples < IN_VIEW or unhidden < UNHIDDEN * len(inside):
            continue
        start, end = max(inside[0] - 0.5 / samples, 0.0), min(inside[-1] + 0.5 / samples, 1.0)
        a = (x0 + start * (x1 - x0), y0 + start * (y1 - y0))
        b = (x0 + end * (x1 - x0), y0 + end * (y1 - y0))
        top = (tops or {}).get(i, height)
        g = cam.ground
        quad = [cam.project(*a, g), cam.project(*b, g), cam.project(*b, top), cam.project(*a, top)]
        views.append(WallView(i, start, end, unhidden / samples, quad, top, g))
    views.sort(key=lambda v: -v.seen)
    return views


def homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> np.ndarray:
    """The 3x3 matrix taking four `src` points to four `dst` points."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    return np.append(np.linalg.solve(np.array(rows, float), np.array(rhs, float)), 1).reshape(3, 3)


def texture_size(quad: list[tuple[float, float]], length: float, height: float) -> tuple[int, int]:
    """About as many pixels as the photo has for the wall, in the wall's own proportions."""
    bl, br, tr, tl = quad
    across = max(math.dist(bl, br), math.dist(tl, tr))
    up = max(math.dist(bl, tl), math.dist(br, tr))
    per_metre = max(across / length, up / height)
    scale = min(per_metre, MAX_TEXTURE / max(length, height))
    return max(8, round(length * scale)), max(8, round(height * scale))


def straighten(photo: Image.Image, quad: list[tuple[float, float]], size: tuple[int, int],
               fill: tuple[int, int, int] = (209, 204, 194)) -> Image.Image:
    """The quad warped flat and upright into a `size` texture. Parts outside the photo get `fill`."""
    w, h = size
    corners = [(0, h), (w, h), (w, 0), (0, 0)]  # bottom-left, bottom-right, top-right, top-left
    m = homography(corners, quad)  # PIL wants output pixel -> photo pixel
    coeffs = (m / m[2, 2]).flatten()[:8]
    return photo.transform(size, Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC, fillcolor=fill)


def gable_pitches(gables, cam: Camera, mask: np.ndarray, seen: set[int],
                  shift: tuple[float, float] = (0.0, 0.0)) -> list[tuple[int, float]]:
    """Each visible gable's pitch measured in the photo: the outline's highest point above the gable's peak, turned
    into metres through the camera. `gables` are roof.Gable in a frame `shift` metres from the camera's; `mask` is the
    building's outline over the whole photo; `seen` the walls this photo shows. Returns (block, pitch) pairs."""
    pitches = []
    for g in gables:
        if g.wall not in seen:
            continue
        apex = max(g.points, key=lambda p: p[2])
        base = [p for p in g.points if p is not apex]
        half = math.dist(base[0][:2], base[1][:2]) / 2
        eave = base[0][2]
        ax, ay = apex[0] + shift[0], apex[1] + shift[1]
        depth = cam.depth(ax, ay)
        if depth < NEAR or half <= 0:
            continue
        col = int(cam.project(ax, ay, eave)[0])
        w = max(3, int(PEAK_WINDOW * cam.width))
        tops = [int(np.argmax(mask[:, c])) for c in range(max(col - w, 0), min(col + w + 1, cam.width))
                if mask[:, c].any()]
        if not tops or min(tops) <= BORDER:  # the outline runs off the top of the photo: the peak isn't in it
            continue
        z = cam.eye + (cam.height / 2 - min(tops)) * depth * cam.depth_scale / cam.focal
        if z > eave:
            pitches.append((g.block, math.degrees(math.atan((z - eave) / half))))
    return pitches


def wall_span(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The outline's box with its left and right edges taken from the walls, not the roof: roofs overhang, and close
    up that's worth several degrees. Uses the rows from half to most of the way down the outline."""
    left, top, right, bottom = box
    rows = mask[top + (bottom - top) // 2: top + (bottom - top) * 17 // 20]
    filled = [np.flatnonzero(r) for r in rows if r.any()]
    if len(filled) < 5:
        return box
    return int(np.median([f[0] for f in filled])), top, int(np.median([f[-1] for f in filled])) + 1, bottom


def fit_rows(quad: list[tuple[float, float]], cam: Camera, mask: np.ndarray, bottom: float, eave: float,
             top: float, band: int = 3) -> list[tuple[float, float]]:
    """The quad's rows taken from the outline where it can see them: at each end of the wall, the outline's lowest
    row is the ground and its highest the eaves. The camera's own projection assumes a level phone at eye height,
    which close up and tilted is far out. It still decides where the outline can't (cut off by the photo's edge,
    or hidden behind something: then the outline stops short of the ground), and it bounds the change.

    `quad` is bottom-left, bottom-right, top-right, top-left from `bottom` up to `top` metres; `eave` is where the
    wall itself ends (below `top` when a gable stands on it).
    """
    h, w = mask.shape
    out = list(quad)
    for b, t in ((0, 3), (1, 2)):
        (xb, rb), (xt, rt) = quad[b], quad[t]
        col = int(round(xb))
        if not band <= col < w - band:
            continue
        cols = mask[:, col - band: col + band + 1].any(axis=1)
        rows = np.flatnonzero(cols)
        if len(rows) == 0:
            continue
        per_metre = (rb - rt) / (top - bottom)  # the projection's rows per metre up this end of the wall
        ground, eaves = rb, rb - per_metre * (eave - bottom)
        seen_low, seen_high = rows[-1], rows[0]
        if seen_low < h - BORDER:  # the ground is in the photo: the outline reaches lower than the projection
            ground = max(ground, seen_low)
        if seen_high > BORDER:
            eaves = min(eaves, seen_high)
        if ground - eaves > ROWS_STRETCH * per_metre * (eave - bottom) or ground <= eaves:
            continue  # the outline here isn't this wall (it runs into something else): keep the projection
        rows_per_metre = (ground - eaves) / (eave - bottom)
        out[b] = (xb, ground)
        out[t] = (xt, ground - rows_per_metre * (top - bottom))
    return out
