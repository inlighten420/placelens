"""The critique loop: Muse looks at the model drawn over each photo and says what's wrong, from a short menu of
edits (move a wall, notch one in or out, raise the eaves, change a roof block's pitch); each edit is tried at a few
sizes and kept only if the model's outline then matches SAM's outline of the real building better.

Muse is good at seeing *what* is off (a bay window sticking out, a gable too high) and rough at *how much*; the
outline is exact but can't say what anything is. So Muse proposes and the outline decides.

How well the model matches a photo is measured two ways. Along the top (`profile_error`): column by column, where
the model's silhouette starts against where SAM's starts, as a share of the photo's height; the bottom is left out,
as gates, walls and cars stand in front of it in most street photos. And at the corners (`corner_error`): a wall
corner or the edge of a recess is a long vertical line in the photo, so each of the model's corners the camera sees
should land on one. The top alone can't see a recess facing the camera: the roofline above it doesn't change.

Shapes are in the roof plan's frame (metres, the footprint's centre at the origin); `shift` moves them into a
photo's frame, where its locked Camera can project them.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from .build import wall_edges
from .front import NEAR, Camera
from .roof import MAX_PITCH, MIN_PITCH, RoofPlan, blocks, plan_roof

KINDS = ("move_wall", "notch", "eaves", "pitch")
MISS = 0.15  # error, as a share of the photo's height, for a column only one of the two outlines covers
TRIM = 0.2  # the worst share of columns left out (a tree or a lamp post in front of the roof)
TRIES = (0.5, 1.0, 1.5)  # sizes each edit is tried at, times what Muse said
MIN_GAIN = 0.02  # an edit must cut the error by this share to be kept
# ...and a change to the footprint by this much: the map's footprint is usually right (the house's was, bar a 1 m
# recess), while the outline's top can be pulled about by a tree or a camera locked to the old shape.
SHAPE_GAIN = 0.05
STEADY = 0.002  # a photo's error moving less than this counts as unchanged
CORNER_CAP = 0.03  # share of the photo's width: a model corner this far from any vertical line counts this much
CORNER_WEIGHT = 1.0  # corners against the top, in the mismatch
LINE_SHARE = 0.4  # share of a column's building rows on a vertical edge for it to count as a line (lower let in
# window frames, gate bars and a stair railing on the house)
LINE_GAP = 10  # pixels: lines closer than this are one line (the strongest kept)
OUTLIER = 2.0  # a photo whose mismatch is this many times the middle one is left out: its camera is off, not the shape
SUPPORT = 0.25  # share of answers that must propose an edit before it is tried
ROUNDS = 3
PARALLEL = 4  # photos critiqued at once
SMALLEST = 0.2  # metres: walls closer together than this, or a notch shallower, are merged away
MAX_MOVE = 4.0  # metres: no single edit moves a wall, or notches one, further than this
MAX_EAVES = 3.0  # metres: nor raises or lowers the eaves by more


@dataclass
class Shape:
    """What the critique can change: the footprint, the eaves' height and the roof."""

    poly: Polygon
    height: float
    kind: str
    pitch: float
    ridge: float | None = None
    pitches: dict[int, float] = field(default_factory=dict)

    def roof(self) -> RoofPlan:
        return plan_roof(self.poly, self.height, self.kind, self.pitch, self.ridge, self.pitches)


@dataclass
class Edit:
    kind: str
    target: int | None = None  # wall number (move_wall, notch) or block number (pitch)
    amount: float = 0.0  # metres (move_wall: + out; notch: + sticks out, - set in; eaves: + up); pitch: degrees
    span: tuple[float, float] = (0.0, 1.0)  # notch: from, to along the wall, from its left end seen from outside
    support: int = 1  # how many answers proposed it
    why: str = ""

    def label(self) -> str:
        if self.kind == "move_wall":
            return f"wall {self.target} {'out' if self.amount > 0 else 'in'} {abs(self.amount):.1f} m"
        if self.kind == "notch":
            what = "out" if self.amount > 0 else "in"
            return (f"wall {self.target} notched {what} {abs(self.amount):.1f} m "
                    f"from {self.span[0]:.0%} to {self.span[1]:.0%}")
        if self.kind == "eaves":
            return f"eaves {'up' if self.amount > 0 else 'down'} {abs(self.amount):.1f} m"
        return f"roof block {self.target} pitch {self.amount:.0f}°"


# --- seeing the model through a photo -----------------------------------------------------------------------------

def faces(shape: Shape, roof: RoofPlan, ground: float = 0.0) -> list[list[tuple[float, float, float]]]:
    """Every face of the model as 3D points: walls from `ground` to the eaves, the roof's slopes, gables and flats."""
    out = [[(*a, ground), (*b, ground), (*b, shape.height), (*a, shape.height)] for a, b in wall_edges(shape.poly)]
    out += [list(s) for s in roof.slopes] + [list(g.points) for g in roof.gables]
    out += [[(x, y, shape.height) for x, y in f.exterior.coords[:-1]] for f in roof.flats]
    return out


def _clip_near(points, cam: Camera):
    """The part of a face at least NEAR metres in front of the camera (Sutherland-Hodgman against that plane)."""
    out = []
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        dp, dq = cam.depth(p[0], p[1]) - NEAR, cam.depth(q[0], q[1]) - NEAR
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append(tuple(a + t * (b - a) for a, b in zip(p, q, strict=True)))
    return out


def _facing(face, cam: Camera) -> bool:
    """Whether a face (its points running counter-clockwise seen from outside) is turned toward the camera."""
    n = np.zeros(3)
    for i, p in enumerate(face):  # Newell's normal: fine for any flat polygon
        q = face[(i + 1) % len(face)]
        n += ((p[1] - q[1]) * (p[2] + q[2]), (p[2] - q[2]) * (p[0] + q[0]), (p[0] - q[0]) * (p[1] + q[1]))
    c = np.mean(face, axis=0)
    return float(np.dot(n, (cam.x - c[0], cam.y - c[1], cam.eye - c[2]))) > 0


def projected(shape: Shape, roof: RoofPlan, cam: Camera, shift: tuple[float, float],
              facing_only: bool = False) -> list[list[tuple[float, float]]]:
    """Each face as a polygon of photo pixels (faces wholly behind the camera left out; with `facing_only`, faces
    turned away from it too)."""
    polys = []
    for face in faces(shape, roof, cam.ground):
        moved = [(x + shift[0], y + shift[1], z) for x, y, z in face]
        if facing_only and not _facing(moved, cam):
            continue
        clipped = _clip_near(moved, cam)
        if len(clipped) >= 3:
            polys.append([cam.project(*p) for p in clipped])
    return polys


def silhouette(shape: Shape, roof: RoofPlan, cam: Camera, shift: tuple[float, float]) -> np.ndarray:
    """Where the model covers the photo (bool, height x width)."""
    img = Image.new("L", (cam.width, cam.height))
    draw = ImageDraw.Draw(img)
    for poly in projected(shape, roof, cam, shift):
        draw.polygon([(max(min(x, 1e5), -1e5), max(min(y, 1e5), -1e5)) for x, y in poly], fill=255)
    return np.asarray(img) > 0


def profile_error(model: np.ndarray, real: np.ndarray) -> float:
    """How far apart the two outlines' tops are: per column, the gap as a share of the photo's height (MISS where
    only one covers it), the worst TRIM of columns dropped, the rest averaged. 0 is a perfect match."""
    height = real.shape[0]
    mc, sc = model.any(0), real.any(0)
    either = mc | sc
    if not either.any():
        return 1.0
    gaps = np.full(model.shape[1], MISS)
    both = mc & sc
    gaps[both] = np.abs(model.argmax(0) - real.argmax(0))[both] / height
    gaps = np.sort(gaps[either])
    return float(gaps[: max(1, int(len(gaps) * (1 - TRIM)))].mean())


def vertical_lines(photo: Path, mask: np.ndarray) -> np.ndarray:
    """The photo's columns where a long vertical line runs down the building (a corner, a recess's edge), as a bool
    per column. Found locally, no API: pixels inside SAM's outline that are much stronger across than up and down,
    counted per column (a short window frame or a downpipe's bracket scores less than a full-height corner)."""
    gray = np.asarray(Image.open(photo).convert("L"), float)
    across = np.abs(np.diff(gray, axis=1))[:-1, :]
    down = np.abs(np.diff(gray, axis=0))[:, :-1]
    inside = mask[:-1, :-1]
    lines = np.zeros(gray.shape[1], bool)
    if not inside.any():
        return lines
    edge = inside & (across > np.percentile(across[inside], 90)) & (across > 2 * down)
    share = edge.sum(0) / np.maximum(inside.sum(0), 1)
    share = np.max([np.roll(share, k) for k in range(-2, 3)], axis=0)  # a line a few pixels wide or slanted
    peaks = (share >= LINE_SHARE) & (share >= np.roll(share, 1)) & (share >= np.roll(share, -1))
    for x in sorted(np.flatnonzero(peaks), key=lambda x: -share[x]):  # strongest first; neighbours merge into it
        if not lines[max(0, x - LINE_GAP): x + LINE_GAP + 1].any():
            lines[x] = True
    return lines


def corner_columns(shape: Shape, cam: Camera, shift: tuple[float, float]) -> list[float]:
    """Photo columns of the model's corners the camera sees (one of its two walls faces the camera). A level camera
    keeps vertical lines vertical, so a corner is one column from the ground to the eaves."""
    edges = wall_edges(shape.poly)

    def facing(edge) -> bool:
        (x0, y0), (x1, y1) = edge
        mx, my = (x0 + x1) / 2 + shift[0], (y0 + y1) / 2 + shift[1]
        return (y1 - y0) * (cam.x - mx) - (x1 - x0) * (cam.y - my) > 0

    columns = []
    for i, (_, (x, y)) in enumerate(edges):  # the end of wall i is the start of wall i + 1
        if not (facing(edges[i]) or facing(edges[(i + 1) % len(edges)])):
            continue
        x, y = x + shift[0], y + shift[1]
        if cam.depth(x, y) < NEAR:
            continue
        px = cam.project(x, y, cam.eye)[0]
        if 0 <= px < cam.width:
            columns.append(px)
    return columns


def corner_error(shape: Shape, cam: Camera, shift: tuple[float, float], lines: np.ndarray) -> float:
    """How far apart the model's corners and the photo's vertical lines are, both ways: each corner to its nearest
    line, and each line (across the model's width) to its nearest corner, so a recess the model lacks counts as well
    as a corner it has wrong. Distances are shares of the photo's width, each capped at CORNER_CAP; the two
    directions averaged. 0 if there's nothing to compare."""
    columns = np.array(corner_columns(shape, cam, shift))
    found = np.flatnonzero(lines)
    if not len(columns) or not len(found):
        return 0.0
    found = found[(found >= columns.min()) & (found <= columns.max())]
    to_line = [min(np.abs(np.flatnonzero(lines) - x).min() / cam.width, CORNER_CAP) for x in columns]
    to_corner = [min(np.abs(columns - x).min() / cam.width, CORNER_CAP) for x in found]
    return float(np.mean(to_line) + (np.mean(to_corner) if to_corner else 0.0)) / 2


# --- the edits ----------------------------------------------------------------------------------------------------

def _clean(poly) -> Polygon | None:
    """A valid, counter-clockwise polygon without slivers, or None."""
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon" or poly.area < 4:
        return None
    poly = poly.buffer(-SMALLEST / 2, join_style="mitre").buffer(SMALLEST / 2, join_style="mitre").simplify(0.05)
    if poly.geom_type != "Polygon" or poly.is_empty:
        return None
    return orient(Polygon(poly.exterior.coords), 1.0)


def apply(shape: Shape, edit: Edit, scale: float = 1.0) -> Shape | None:
    """The shape with `edit` made, its amount times `scale`. None if it can't be made (no such wall, nothing left)."""
    if edit.kind == "eaves":
        height = shape.height + max(min(edit.amount * scale, MAX_EAVES), -MAX_EAVES)
        return replace(shape, height=height) if height >= 2.0 else None
    if edit.kind == "pitch":
        current = shape.pitches.get(edit.target, shape.pitch)
        pitch = min(max(current + (edit.amount - current) * scale, MIN_PITCH), MAX_PITCH)
        return replace(shape, pitches={**shape.pitches, edit.target: pitch})
    edges = wall_edges(shape.poly)
    if edit.target is None or not 0 <= edit.target < len(edges):
        return None
    (x0, y0), (x1, y1) = edges[edit.target]
    length = math.dist((x0, y0), (x1, y1))
    nx, ny = (y1 - y0) / length, -(x1 - x0) / length  # outward
    d = max(min(edit.amount * scale, MAX_MOVE), -MAX_MOVE)
    if edit.kind == "move_wall":
        ring = list(shape.poly.exterior.coords)[:-1]
        moved = [(x + nx * d, y + ny * d) if (x, y) in ((x0, y0), (x1, y1)) else (x, y) for x, y in ring]
        poly = _clean(Polygon(moved))
    elif edit.kind == "notch":
        t0, t1 = sorted(min(max(t, 0.0), 1.0) for t in edit.span)
        if (t1 - t0) * length < SMALLEST or abs(d) < SMALLEST:
            return None
        a = (x0 + t0 * (x1 - x0), y0 + t0 * (y1 - y0))
        b = (x0 + t1 * (x1 - x0), y0 + t1 * (y1 - y0))
        # A little past the wall's line on the other side, so the cut or the bump meets the wall cleanly.
        lip = -0.05 if d > 0 else 0.05
        piece = Polygon([(a[0] + nx * lip, a[1] + ny * lip), (b[0] + nx * lip, b[1] + ny * lip),
                         (b[0] + nx * d, b[1] + ny * d), (a[0] + nx * d, a[1] + ny * d)])
        poly = _clean(shape.poly.union(piece) if d > 0 else shape.poly.difference(piece))
    else:
        return None
    if poly is None:
        return None
    # Measured pitches belong to blocks by number; if the cut into blocks changes, they no longer do.
    same = len(blocks(poly)) == len(blocks(shape.poly)) if shape.kind != "flat" else True
    return replace(shape, poly=poly, pitches=shape.pitches if same else {})


# --- asking Muse --------------------------------------------------------------------------------------------------

CRITIQUE_PROMPT = """Two pictures of the same photo of a building. In the second, our 3D model of the building is \
drawn over it: cyan lines are the model's outline and edges, white numbers are its wall numbers (on each wall the \
photo sees), yellow B-numbers are its roof blocks.

Compare the model with the real building and say what to change so it matches. Only these edits:
- {"edit": "move_wall", "wall": n, "metres": m} the whole wall n moves out (+) or in (-) by m metres
- {"edit": "notch", "wall": n, "from": f, "to": t, "metres": m} part of wall n, from f to t along it (0 = its left \
end as you look at it, 1 = its right end), sticks out (+) or is set back (-) by m metres: a bay window, a porch, \
a recess for steps
- {"edit": "eaves", "metres": m} the walls are m metres too low (+) or too high (-) where the roof starts
- {"edit": "pitch", "block": n, "degrees": d} roof block n should slope at d degrees

Look closely along each wall the photo sees: real walls often step back or stick out where the model's are flat \
(a recess for steps up to a door, a bay window, a porch, a garage set forward), and real eaves and gables often sit \
higher or lower than the model's. Judge sizes from things you know: a floor is about 3 m, a door about 2 m tall, \
a window about 1.2 m wide. Only change what this photo shows clearly; leave walls it doesn't see alone. Ignore \
cars, gates, plants and walls in front of the building.

Answer in JSON only:
{"fit": "good", "fair" or "poor", "edits": [{...}, ...] (at most 4, the biggest mismatch first; [] if it matches), \
"why": "one sentence"}"""


def parse_critique(text: str) -> list[Edit]:
    """Muse's edits, tolerating code fences and chatter; anything unreadable is left out."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return []
    edits = []
    for e in d.get("edits") or []:
        if not isinstance(e, dict) or e.get("edit") not in KINDS:
            continue
        kind = e["edit"]
        try:
            if kind == "pitch":
                edits.append(Edit(kind, int(e["block"]), float(e["degrees"])))
            elif kind == "eaves":
                edits.append(Edit(kind, None, float(e["metres"])))
            elif kind == "move_wall":
                edits.append(Edit(kind, int(e["wall"]), float(e["metres"])))
            else:
                edits.append(Edit(kind, int(e["wall"]), float(e["metres"]), (float(e["from"]), float(e["to"]))))
        except (KeyError, TypeError, ValueError):
            continue
    return [e for e in edits if e.amount != 0]


def proposals(answers: list[list[Edit]]) -> list[Edit]:
    """The edits enough answers agree on (same kind, same wall or block, same direction), each with the middle of
    their amounts and spans; most supported first."""
    groups: dict[tuple, list[Edit]] = defaultdict(list)
    for edits in answers:
        seen = set()
        for e in edits:
            key = (e.kind, e.target, e.amount > 0 if e.kind != "pitch" else None)
            if key not in seen:  # one vote per answer
                seen.add(key)
                groups[key].append(e)
    need = max(2, math.ceil(SUPPORT * len(answers)))
    out = []
    for (kind, target, _), es in groups.items():
        if len(es) < need:
            continue
        span = (statistics.median(e.span[0] for e in es), statistics.median(e.span[1] for e in es))
        out.append(Edit(kind, target, statistics.median(e.amount for e in es), span, len(es)))
    out.sort(key=lambda e: -e.support)
    return out


def draw_overlay(photo: Path, shape: Shape, roof: RoofPlan, cam: Camera, shift: tuple[float, float], out: Path) -> Path:
    """The photo with the model drawn over it: outline thick, the edges of faces turned toward the camera thin (cyan;
    a wireframe of the hidden back was hard to read), wall and roof block numbers."""
    img = Image.open(photo).convert("RGB")
    draw = ImageDraw.Draw(img)
    sil = Image.fromarray(silhouette(shape, roof, cam, shift).astype(np.uint8) * 255)
    img.paste((0, 230, 255), mask=ImageChops.subtract(sil.filter(ImageFilter.MaxFilter(7)), sil))  # thick outline
    for poly in projected(shape, roof, cam, shift, facing_only=True):
        draw.line([*poly, poly[0]], fill=(0, 230, 255), width=2)
    for i, ((x0, y0), (x1, y1)) in enumerate(wall_edges(shape.poly)):
        mx, my = (x0 + x1) / 2 + shift[0], (y0 + y1) / 2 + shift[1]
        nx, ny = (y1 - y0), -(x1 - x0)
        if nx * (cam.x - mx) + ny * (cam.y - my) <= 0 or cam.depth(mx, my) < NEAR:
            continue  # facing away, or behind the camera
        px, py = cam.project(mx, my, (cam.ground + shape.height) / 2)
        if 0 <= px < cam.width and 0 <= py < cam.height:
            draw.text((px, py), str(i), fill=(255, 255, 255), font_size=44, stroke_width=3, stroke_fill=(0, 0, 0),
                      anchor="mm")
    for n, block in enumerate(roof.blocks):
        c = block.centroid
        if cam.depth(c.x + shift[0], c.y + shift[1]) < NEAR:
            continue
        px, py = cam.project(c.x + shift[0], c.y + shift[1], shape.height + 1.0)
        if 0 <= px < cam.width and 0 <= py < cam.height:
            draw.text((px, py), f"B{n}", fill=(255, 220, 0), font_size=40, stroke_width=3, stroke_fill=(0, 0, 0),
                      anchor="mm")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=88)
    return out


# --- the loop -----------------------------------------------------------------------------------------------------

@dataclass
class View:
    """One photo the critique uses: its small picture, locked camera, SAM's outline and where the shape sits.
    `relock(shape)` locks the camera again onto a changed shape (its edges move with the walls), if given."""

    stem: str
    small: Path
    cam: Camera
    real: np.ndarray
    shift: tuple[float, float]
    relock: Callable[[Shape], Camera] | None = None
    lines: np.ndarray | None = None  # vertical_lines: the photo's columns with a long vertical line on the building

    def camera(self, shape: Shape) -> Camera:
        return self.relock(shape) if self.relock else self.cam


def errors(shape: Shape, views: list[View]) -> list[float]:
    """Each photo's mismatch (top, plus corners where the photo's lines are known), its camera locked onto this
    shape."""
    roof = shape.roof()
    out = []
    for v in views:
        cam = v.camera(shape)
        e = profile_error(silhouette(shape, roof, cam, v.shift), v.real)
        if v.lines is not None:
            e += CORNER_WEIGHT * corner_error(shape, cam, v.shift, v.lines)
        out.append(e)
    return out


def steady_views(shape: Shape, views: list[View], log=print) -> list[View]:
    """The photos whose mismatch isn't an OUTLIER (at least two kept): a photo placed only roughly (its side guessed,
    its GPS far off) can't line up with any shape, and would pull the edits toward fitting its error."""
    errs = errors(shape, views)
    middle = max(statistics.median(errs), 0.01)  # (perfect fits elsewhere don't make every gap an outlier)
    keep = [v for v, e in zip(views, errs, strict=True) if e <= OUTLIER * middle]
    if len(keep) < 2:
        return views
    for v, e in zip(views, errs, strict=True):
        if v not in keep:
            log(f"          {v.stem} left out of the critique: mismatch {e:.3f}, {e / middle:.1f}x the middle one")
    return keep


def error(shape: Shape, views: list[View]) -> float:
    return statistics.mean(errors(shape, views))


def better(new: list[float], old: list[float], gain: float) -> bool:
    """Whether `new` beats `old`: the mean down by `gain` (a share), and more photos better than worse. One photo
    improving a lot can't carry a change the rest disagree with."""
    up = sum(n > o + STEADY for n, o in zip(new, old, strict=True))
    down = sum(n < o - STEADY for n, o in zip(new, old, strict=True))
    return statistics.mean(new) < statistics.mean(old) * (1 - gain) and down > up


def try_edit(shape: Shape, edit: Edit, views: list[View], current: list[float]):
    """The best size of `edit` that's `better` than `current`: (new shape, scale, its per-photo errors), else None."""
    gain = SHAPE_GAIN if edit.kind in ("move_wall", "notch") else MIN_GAIN
    best = None
    for scale in TRIES:
        new = apply(shape, edit, scale)
        if new is None:
            continue
        e = errors(new, views)
        if better(e, current, gain) and (best is None or statistics.mean(e) < statistics.mean(best[2])):
            best = (new, scale, e)
    return best


def refine(shape: Shape, views: list[View], critique, out: Path, rounds: int = ROUNDS, log=print):
    """Rounds of: draw the model over every photo, `critique(view, overlay) -> list[list[Edit]]` (one list per
    answer), try what enough answers propose, keep what lowers the error. Topology-changing notches are tried last
    and at most one per round (they renumber the walls the other answers refer to). Returns the shape and a history
    of every round (for the log and the video)."""
    views = steady_views(shape, views, log)
    current = errors(shape, views)
    start = statistics.mean(current)
    history = [{"round": 0, "error": round(start, 4), "kept": [],
                "photos": {v.stem: round(e, 4) for v, e in zip(views, current, strict=True)}}]
    log(f"refine:   start, outline mismatch {start:.3f}")
    for n in range(1, rounds + 1):
        roof = shape.roof()
        folder = out / f"round_{n}"
        overlays = [draw_overlay(v.small, shape, roof, v.camera(shape), v.shift, folder / f"{v.stem}.jpg")
                    for v in views]
        with ThreadPoolExecutor(PARALLEL) as pool:  # each photo's critique is mostly waiting for Muse
            answers = [a for per_photo in pool.map(critique, views, overlays) for a in per_photo]
        props = proposals(answers)
        props.sort(key=lambda e: (e.kind == "notch", -e.support))
        kept = []
        for edit in props:
            if edit.kind == "notch" and any(k["kind"] == "notch" for k in kept):
                continue
            tried = try_edit(shape, edit, views, current)
            if tried is None:
                log(f"          round {n}: {edit.label()} ({edit.support} votes): no better, dropped")
                continue
            shape, scale, e = tried
            was, now = statistics.mean(current), statistics.mean(e)
            log(f"          round {n}: {edit.label()} x{scale:g} ({edit.support} votes): {was:.3f} -> {now:.3f}")
            kept.append({"kind": edit.kind, "edit": edit.label(), "scale": scale, "votes": edit.support,
                         "error": round(now, 4)})
            current = e
        history.append({"round": n, "error": round(statistics.mean(current), 4), "kept": kept,
                        "proposed": [{"edit": p.label(), "votes": p.support} for p in props],
                        "photos": {v.stem: round(e, 4) for v, e in zip(views, current, strict=True)}})
        if not kept:
            break
    for v in views:
        draw_overlay(v.small, shape, shape.roof(), v.camera(shape), v.shift, out / f"final_{v.stem}.jpg")
    log(f"refine:   end, outline mismatch {statistics.mean(current):.3f} ({start:.3f} before)")
    return shape, history

