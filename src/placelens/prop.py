"""Props: photograph an object, get a game-ready 3D prop at its real size, standing where it really is.

Muse says what the object is, how big it is and how far away. SAM 3.1 cuts it out. The cut-out's outline is given
the object's real thickness, so it's solid from any angle, with the photo on its front and back. The photo's GPS and
compass, the object's place across the photo and its distance put it on the map, facing the way it did.
"""

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import shapely
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from .build import Part

MIN_DEPTH = 0.03  # metres: thinner than this and it's a sign or a sheet; still give it some body
OUTLINE_DETAIL = 0.004  # the outline is simplified to this share of the cut-out's size
MAX_TEXTURE = 1024  # pixels, longest side of a prop's picture

PROP_PROMPT = """Look at the main free-standing object in this photo: the thing it was taken of, not a building, \
the ground or the sky.

Answer in JSON only:
{"name": "what it is, a few words", \
"segment": "a short noun phrase that picks out just this object, for a segmentation model (e.g. "water tank", \
"bench", "potted plant")", \
"width_m": its width across the photo in metres, "depth_m": its depth front to back in metres, \
"height_m": its height in metres, \
"distance_m": distance from the camera to it in metres, \
"round": true if it looks the same from every side around an upright axis (a tank, pot, round bin, bottle, \
pole, tree trunk), false if not (a box, chair, bench, car, animal), \
"symmetric": true if its left and right sides are mirror images of each other (a teddy bear, a chair, a car), \
"fully_visible": true if the whole object is in the photo and nothing hides part of it, \
"hidden": "what hides or cuts off part of it, if anything", \
"confidence": 0.0 to 1.0}"""


@dataclass
class PropView:
    name: str | None
    segment: str | None
    width: float | None = None
    depth: float | None = None
    height: float | None = None
    distance: float | None = None
    confidence: float = 0.0
    fully_visible: bool | None = None
    hidden: str | None = None
    round: bool | None = None
    symmetric: bool | None = None
    raw: dict = field(default_factory=dict)


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def parse_prop(text: str) -> PropView:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        d = {}
    confidence = d.get("confidence")
    return PropView(
        str(d["name"]) if d.get("name") else None,
        str(d["segment"]) if d.get("segment") else None,
        _number(d.get("width_m")), _number(d.get("depth_m")), _number(d.get("height_m")), _number(d.get("distance_m")),
        float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        d["fully_visible"] if isinstance(d.get("fully_visible"), bool) else None,
        str(d["hidden"]) if d.get("hidden") else None,
        d["round"] if isinstance(d.get("round"), bool) else None,
        d["symmetric"] if isinstance(d.get("symmetric"), bool) else None,
        d,
    )


def outline(mask: np.ndarray) -> Polygon | None:
    """The mask's largest blob as a polygon, in mask pixels (x right, y down), simplified a little."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    padded = np.pad(mask.astype(float), 1)  # so blobs touching the edge still close
    fig = plt.figure()
    contours = plt.contour(padded, levels=[0.5]).allsegs[0]
    plt.close(fig)
    polys = [Polygon(c - 1) for c in contours if len(c) >= 4]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    polys = [g for p in polys for g in (p.geoms if hasattr(p, "geoms") else [p]) if g.area > 0]
    if not polys:
        return None
    best = max(polys, key=lambda p: p.area)
    size = max(mask.shape)
    simple = best.simplify(size * OUTLINE_DETAIL, preserve_topology=True)
    return Polygon(simple.exterior) if simple.is_valid and not simple.is_empty else Polygon(best.exterior)


def cutout_png(photo: Image.Image, mask: np.ndarray) -> bytes:
    """The object's pixels with everything else transparent, as a PNG at most MAX_TEXTURE on a side."""
    rgba = photo.convert("RGBA")
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(rgba.size, Image.BILINEAR)
    rgba.putalpha(alpha)
    rgba.thumbnail((MAX_TEXTURE, MAX_TEXTURE))
    buf = io.BytesIO()
    rgba.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def solid(shape: Polygon, size: tuple[int, int], height: float, depth: float, png: bytes) -> list[Part]:
    """The outline (pixels of a cut-out `size` wide and tall) made solid: `height` metres tall, `depth` thick, standing
    on the ground with its front facing +Z (glTF axes: X right, Y up). Front and back show the cut-out; the sides take
    the colour at the outline's edge."""
    w, h = size
    scale = height / max(shape.bounds[3] - shape.bounds[1], 1)
    cx = (shape.bounds[0] + shape.bounds[2]) / 2
    bottom = shape.bounds[3]
    half = max(depth, MIN_DEPTH) / 2

    def xy(px: float, py: float) -> tuple[float, float]:
        return (px - cx) * scale, (bottom - py) * scale

    def uv(px: float, py: float) -> tuple[float, float]:
        return px / w, py / h

    pos, nrm, uvs, idx = [], [], [], []

    def face(points, normal, coords):
        base = len(pos)
        pos.extend(points)
        nrm.extend([normal] * len(points))
        uvs.extend(coords)
        idx.extend(range(base, base + len(points)))

    # In pixels y runs down, so a polygon counter-clockwise on screen is clockwise in metres: orient in metres.
    for tri in shapely.constrained_delaunay_triangles(shape).geoms:
        corners = list(tri.exterior.coords)[:3]
        flat = [xy(*c) for c in corners]
        (ax, ay), (bx, by), (qx, qy) = flat
        if (bx - ax) * (qy - ay) - (by - ay) * (qx - ax) < 0:  # clockwise in metres: turn it round
            corners, flat = corners[::-1], flat[::-1]
        face([(x, y, half) for x, y in flat], (0.0, 0.0, 1.0), [uv(*c) for c in corners])  # front, facing +Z
        face([(x, y, -half) for x, y in flat[::-1]], (0.0, 0.0, -1.0), [uv(*c) for c in corners[::-1]])  # back

    ring = list(orient(Polygon([xy(*c) for c in shape.exterior.coords]), 1.0).exterior.coords)
    pixels = {xy(*c): c for c in shape.exterior.coords}
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:], strict=True):
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            continue
        normal = ((y1 - y0) / length, -(x1 - x0) / length, 0.0)  # outward from a counter-clockwise ring
        a, b = uv(*pixels[(x0, y0)]), uv(*pixels[(x1, y1)])
        base = len(pos)
        pos.extend([(x0, y0, half), (x0, y0, -half), (x1, y1, -half), (x1, y1, half)])
        nrm.extend([normal] * 4)
        uvs.extend([a, a, b, b])
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return [Part("prop", np.array(pos, np.float32), np.array(nrm, np.float32), np.array(uvs, np.float32),
                 np.array(idx, np.uint32), (1, 1, 1, 1), png, cutout=True)]


def place(heading: float, offset: float, distance: float) -> tuple[float, float, float]:
    """Metres east and north of the camera, and the compass bearing the prop's front faces (back at the camera), for
    an object `offset` degrees off the photo's centre line, `distance` metres away."""
    bearing = math.radians(heading + offset)
    return distance * math.sin(bearing), distance * math.cos(bearing), (heading + offset + 180) % 360


def lathe(mask: np.ndarray, jpeg: bytes, height: float, cut: tuple[bool, bool] = (False, False),
          max_radius_px: float | None = None, segments: int = 48, rows: int = 40) -> list[Part]:
    """A round object as a surface of revolution: the outline's half-width at each height, spun around an upright
    axis. The photo wraps the front; the back, out of sight, repeats it as a mirror image. `mask` is the cut-out
    (pixels, y down) and `jpeg` the photo of the same box; `cut` says whether it runs off the photo on the left or
    right, when the radius is measured from the other side (out to `max_radius_px` from it, if known).

    glTF axes: X right, Y up, standing on Y = 0, the photographed side facing +Z."""
    h, w = mask.shape
    filled = [y for y in range(h) if mask[y].any()]
    if len(filled) < 2:
        raise ValueError("the cut-out is too small to turn")
    top, bottom = filled[0], filled[-1]
    sample = np.linspace(top, bottom, rows).round().astype(int)
    edges = np.array([(np.flatnonzero(mask[y])[0], np.flatnonzero(mask[y])[-1]) for y in sample], float)
    widest = (edges[:, 1] - edges[:, 0]).max() / 2
    if cut[0] and not cut[1]:  # the left runs off the photo: measure from the right edge
        axis = np.median(edges[:, 1]) - (max_radius_px or widest)
        radii = edges[:, 1] - axis
    elif cut[1] and not cut[0]:
        axis = np.median(edges[:, 0]) + (max_radius_px or widest)
        radii = axis - edges[:, 0]
    else:
        axis = float(np.median(edges.mean(axis=1)))
        radii = (edges[:, 1] - edges[:, 0]) / 2
    radii = np.maximum(np.convolve(np.pad(radii, 1, mode="edge"), np.ones(3) / 3, mode="valid"), 0.5)
    scale = height / max(bottom - top, 1)
    inset = 0.97  # stay just inside the outline, off the background

    pos, uvs = [], []
    for y, r in zip(sample, radii, strict=True):
        for j in range(segments + 1):
            t = 2 * math.pi * j / segments
            pos.append((r * scale * math.sin(t), (bottom - y) * scale, r * scale * math.cos(t)))
            side = math.sin(t)
            if cut[0] != cut[1]:  # one side runs off the photo: the other side's picture goes all the way round
                side = abs(side) if cut[0] else -abs(side)
            uvs.append((min(max((axis + inset * r * side) / w, 0.0), 1.0), y / h))
    idx = []
    ring = segments + 1
    for k in range(rows - 1):
        for j in range(segments):
            a, b = k * ring + j, k * ring + j + 1  # upper ring
            c, d = a + ring, b + ring  # lower ring
            idx += [a, c, b, b, c, d]
    for y, row_start, up in ((sample[0], 0, True), (sample[-1], (rows - 1) * ring, False)):
        centre = len(pos)
        pos.append((0.0, (bottom - y) * scale, 0.0))
        uvs.append((min(max(axis / w, 0.0), 1.0), y / h))
        for j in range(segments):
            a, b = row_start + j, row_start + j + 1
            idx += [centre, a, b] if up else [centre, b, a]

    pos_a = np.array(pos, np.float32)
    idx_a = np.array(idx, np.uint32)
    tris = pos_a[idx_a].reshape(-1, 3, 3)
    face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = np.zeros_like(pos_a)
    for corner in range(3):
        np.add.at(nrm, idx_a[corner::3], face)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    return [Part("prop", pos_a, nrm.astype(np.float32), np.array(uvs, np.float32), idx_a, (1, 1, 1, 1), jpeg)]
