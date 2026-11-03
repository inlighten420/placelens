"""A carving rebuilt piece by piece, each piece a clean shape where the photos earn one, else carved.

Carving keeps everything inside every photo's outline, so where no photo looked it leaves flat slabs and wedges (a
teddy seen from between two photos is all facets). But most things are made of simpler pieces. So Muse names the parts
that stick out ("head", "arm", "leg", "leaf"), SAM 3.1 finds each one in every photo, and each part's place on the
carving is voted by the photos that see it; what's left is the body. Then each piece, biggest first, tries a rounded
shape, a cylinder, a taper (a cylinder wider at one end) and a box, each fitted to the photos, and keeps one only if
the photos earn it (`Fitter.choose`): the whole still matches them, the piece matches its own outline, it's solid
enough for a shape with no dents or spikes, a flat face beats a rounded one only by a clear margin, and a piece on the
floor stands upright. A spiky piece (an aloe's leaves) stays carved. The pieces are blended with a smooth union and
meshed from that continuous field: no voxel steps where a shape was kept.

The prior only fills what the photos leave open. On the teddy (3 Nov), each photo in turn left out of the make was
predicted as well as by the carving (outline 0.744 against 0.747 IoU, colour 16.7 against 16.5 delta E), while
from every angle in between it looks like a teddy instead of a stack of slabs. It's kept only if it still matches
the photos nearly as well as the carving does (`KEEP_WITHIN`), so a thing that isn't made of parts keeps its carving.

The parts also make a rig: each a closed piece on a joint where it meets its parent, with a gentle idle loop (head
turns, arms sway; pieces resting on the ground carry the weight and stay still), written as `prop_moving.glb`.
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes

from . import carve as C
from .build import Part

PART_PROMPT = """This photo shows one object. Name the parts that stick out from its main body: each a short, common \
noun for one kind of part, the way a segmentation model would find it, e.g. "head", "arm", "leg", "ear", "tail", \
"handle", "lid". Biggest first, at most 5. Not the main body itself, and nothing small or painted on (eyes, \
nose, buttons, markings).

Answer in JSON only: {"parts": ["...", "..."]}"""
VOTES = 3
MAX_PARTS = 4  # kinds of part SAM is asked for in each photo (each a quarter of a US cent a photo)
BODY = 1  # in a label map: 0 outside the object, 1 the main body (what's left of its cut-out), 2.. each part
BODY_VOTE = 0.5  # "body" is what SAM didn't name, not something it found: a part it missed in one photo stays a part
MIN_PIECE = 0.004  # share of the carving: smaller pieces join what they touch most
OF_ITS_KIND = 0.25  # and so does a piece this much smaller than the biggest of its kind: the other arm is about as
# big as the first, while a sliver of shoulder a side photo saw over the arm isn't
PART_WEIGHT = 0.3  # how much each part matching SAM's mask for it counts, against the whole outline
FIT_ROUNDS = 5  # passes over every part's position, size and turn, the steps halving each pass
KEEP_WITHIN = 0.08  # the clean model is kept if it matches its own photos' outlines at most this much worse than
# the carving, which is built from those same photos and hugs every bump of them. On a photo left out of the make
# entirely (3 Nov), the teddy's smooth parts were 0.04 behind on its own photos and tied; the aloe with a cylinder pot
# 0.04 to 0.06 behind and ahead in every case (0.73 against 0.73, 0.69 against 0.67, 0.70 against 0.69). A wrong
# prior loses far more
SWING = 15.0  # degrees a hanging piece sways in the idle loop
TURN = 10.0  # degrees a piece standing straight up (a head) turns
LOOP = 2.4  # seconds


@dataclass
class Piece:
    name: str
    label: int
    mask: np.ndarray  # bool, the carving's grid
    parent: int | None = None
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))  # metres
    params: np.ndarray | None = None  # ellipsoid: centre (3), log radii (3), rotation vector (3)
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    swing: float = 0.0  # degrees


@dataclass
class Soft:
    part: Part  # the whole thing, smooth, painted from the photos
    pieces: list[Piece]
    rig: list[Part]  # each piece's own closed skin, about its pivot, for the moving version
    agreement: float
    carved_agreement: float


# --- Asking the models -------------------------------------------------------------------------------------------

def part_nouns(meta, photo: Path, out: Path) -> list[str]:
    """Muse's names for the parts (asked VOTES times; a name is used if most answers give it), biggest first."""
    from .meta import OPEN
    from .muse import ask

    answers = ask(meta, PART_PROMPT, photo, out / f"{photo.stem}_parts.txt", OPEN, VOTES)
    counts: dict[str, list[int]] = {}
    for a in answers:
        m = re.search(r"\{.*\}", a, re.S)
        try:
            names = json.loads(m.group(0))["parts"] if m else []
        except (ValueError, KeyError, TypeError):
            continue
        for rank, n in enumerate(names):
            n = str(n).strip().lower()
            if n and n not in ("body", "torso", "main body"):
                counts.setdefault(n, []).append(rank)
    need = len(answers) // 2 + 1
    kept = [n for n, ranks in counts.items() if len(ranks) >= need]
    kept.sort(key=lambda n: (np.mean(counts[n]), n))
    return kept[:MAX_PARTS]


def _sam_masks(meta, small: Path, noun: str, out: Path) -> list[np.ndarray]:
    """Every match SAM 3.1 finds for `noun` in the photo, as masks over it (the reply saved, so asked once; the same
    file objects._segment keeps, so a noun asked there isn't paid for twice)."""
    from .meta import segments_from_file

    slug = re.sub(r"[^a-z0-9]+", "-", noun.lower()).strip("-")
    reply = out / f"{small.stem}_sam_{slug}.txt"
    if reply.exists():
        segments = segments_from_file(reply)
    else:
        segments, raw = meta.segment(small, noun)
        reply.write_text(raw, encoding="utf-8")
    if not segments:
        return []
    with Image.open(small) as img:
        return [s.full_mask(img.size) for s in segments]


def label_map(cutout: np.ndarray, found: list[list[np.ndarray]]) -> np.ndarray:
    """One photo's labels: 0 outside, 1 body, 2.. the parts (found[k]: SAM's masks for part k). A match counts if
    most of it lies on the object and it isn't most of the object (that's SAM finding the whole thing again). Where
    two overlap, the smaller wins: it's the more specific."""
    lab = cutout.astype(np.uint8)
    pieces = []
    for k, masks in enumerate(found):
        for m in masks:
            area = int(m.sum())
            if area and (m & cutout).sum() > 0.6 * area and area < 0.6 * cutout.sum():
                pieces.append((area, k + 2, m & cutout))
    for _, k, m in sorted(pieces, key=lambda p: -p[0]):
        lab[m] = k
    return lab


# --- Parts on the carving ----------------------------------------------------------------------------------------

def _points(shape, corner, voxel) -> np.ndarray:
    return corner + (np.indices(shape).reshape(3, -1).T + 0.5) * voxel


def _seen(points: np.ndarray, s: C.Shot, height: float, voxel: float):
    """Which points are the nearest surface at their pixel in this photo (a point z-buffer, one voxel to about 1.5
    buffer pixels so the surface has no holes), and their pixels."""
    cols, rows, depth = s.project(points, height / 2)
    h, w = s.mask.shape
    scale = 1.5 / (s.focal * voxel / max(float(np.median(depth)), 1e-6))
    bw, bh = int(w * scale) + 1, int(h * scale) + 1
    c, r = (cols * scale).astype(int), (rows * scale).astype(int)
    ok = (depth > 0) & (c >= 0) & (c < bw) & (r >= 0) & (r < bh)
    buf = np.full(bh * bw, np.inf)
    np.minimum.at(buf, r[ok] * bw + c[ok], depth[ok])
    near = ndimage.minimum_filter(buf.reshape(bh, bw), size=3)
    seen = np.zeros(len(points), bool)
    seen[ok] = depth[ok] <= near[r[ok], c[ok]] + 0.75 * voxel
    return (np.flatnonzero(seen), np.clip(np.round(rows[seen]).astype(int), 0, h - 1),
            np.clip(np.round(cols[seen]).astype(int), 0, w - 1))


def vote_labels(kept, corner, voxel, shots, height, labels) -> np.ndarray:
    """A part label per kept voxel: the photos that see a surface voxel vote with their label there; inside takes
    the nearest surface's; then each voxel takes the commonest label round it, three times over."""
    count = max(int(lab.max()) for lab in labels) + 1
    idx = np.flatnonzero(kept.ravel())
    points = _points(kept.shape, corner, voxel)[idx]
    votes = np.zeros((len(idx), count))
    for s, lab in zip(shots, labels, strict=True):
        v, r, c = _seen(points, s, height, voxel)
        np.add.at(votes, (v, lab[r, c]), 1.0)
    votes[:, 0] = 0
    votes[:, BODY] *= BODY_VOTE
    best = np.where(votes.max(axis=1) > 0, votes.argmax(axis=1), 0)
    grid = np.zeros(kept.shape, np.int16)
    grid.ravel()[idx] = best
    if not (grid > 0).any():
        return grid
    _, near = ndimage.distance_transform_edt(grid == 0, return_indices=True)
    grid = np.where(kept, grid[tuple(near)], 0)
    for _ in range(3):
        counts = np.stack([ndimage.uniform_filter((grid == k).astype(float), 3) for k in range(count)])
        counts[0] = -1
        grid = np.where(kept, counts.argmax(axis=0), 0)
    return grid


def split(grid: np.ndarray, nouns: list[str]) -> list[Piece]:
    """Each connected piece of each part, the biggest piece of body first. Crumbs join what they touch most. A pair
    is named by side: facing the object, its own left is on your right (+X)."""
    total = int((grid > 0).sum())
    found = []
    for k in [int(k) for k in np.unique(grid) if k > 0]:
        comp, n = ndimage.label(grid == k)
        sizes = ndimage.sum(np.ones_like(comp), comp, range(1, n + 1))
        for j in range(1, n + 1):
            m = comp == j
            if m.sum() < MIN_PIECE * total or m.sum() < OF_ITS_KIND * sizes.max():
                ring = ndimage.binary_dilation(m) & ~m & (grid > 0) & (grid != k)
                if ring.any():
                    grid[m] = np.bincount(grid[ring]).argmax()
                continue
            found.append((k, m))
    found = [(k, m) for k, m in found if (grid[m] == k).all()]
    if not found:
        return []
    bodies = [f for f in found if f[0] == BODY] or found
    root = max(bodies, key=lambda f: f[1].sum())
    found.remove(root)
    pieces = [Piece("body" if root[0] == BODY else nouns[root[0] - 2], root[0], root[1])]
    for k, m in found:
        pieces.append(Piece("body" if k == BODY else nouns[k - 2], k, m))
    for noun in {p.name for p in pieces[1:]}:
        same = [p for p in pieces[1:] if p.name == noun]
        if len(same) == 2:
            a, b = sorted(same, key=lambda p: np.argwhere(p.mask)[:, 0].mean())
            a.name, b.name = f"right {noun}", f"left {noun}"
        elif len(same) > 2:
            for i, p in enumerate(same):
                p.name = f"{noun} {i + 1}"
    return pieces


def attach(pieces: list[Piece], corner: np.ndarray, voxel: float) -> None:
    """Each piece's parent (the attached piece it touches most: an ear goes on the head, not the body) and its pivot
    (the middle of where they touch); the root stands on the ground below its middle."""
    root = pieces[0]
    root.pivot = corner + (np.argwhere(root.mask).mean(axis=0) + 0.5) * voxel
    root.pivot[1] = corner[1]
    done, left = [0], list(range(1, len(pieces)))
    grown = {i: ndimage.binary_dilation(pieces[i].mask, iterations=2) for i in left}
    while left:
        touch, i, j = max((int((grown[i] & pieces[j].mask).sum()), i, j) for i in left for j in done)
        contact = grown[i] & pieces[j].mask
        where = np.argwhere(contact if touch else pieces[i].mask)
        pieces[i].parent, pieces[i].pivot = j, corner + (where.mean(axis=0) + 0.5) * voxel
        done.append(i)
        left.remove(i)
    lowest = np.argwhere(np.any([p.mask for p in pieces], axis=0))[:, 1]
    floor, tall = lowest.min(), max(lowest.max() - lowest.min(), 1)
    for p in pieces[1:]:
        cells = np.argwhere(p.mask)
        if cells[:, 1].min() - floor <= 0.08 * tall:  # resting on the ground: it carries the weight, so it's still
            continue
        d = corner + (cells.mean(axis=0) + 0.5) * voxel - p.pivot
        d /= max(np.linalg.norm(d), 1e-9)
        if abs(d[1]) > 0.8:  # standing up from its joint (a head): turns about the vertical
            p.axis, p.swing = np.array([0.0, 1.0, 0.0]), TURN
        else:  # hanging or sticking out (an arm): sways across its own direction
            a = np.cross(d, [0.0, 1.0, 0.0])
            p.axis, p.swing = a / np.linalg.norm(a), SWING


# --- Clean shapes ------------------------------------------------------------------------------------------------

SHAPES = ("ellipsoid", "cylinder", "taper", "box")  # what a piece may become (taper: a cylinder wider at one
# end, a plant pot or a cup); else it stays carved
PIECE_TOL = 0.03  # a piece becomes a clean shape if the whole thing then matches the photos at most this much worse
# than with the piece carved (the carving hugs every bump of its own photos; a wrong shape loses far more: an aloe's
# spiky leaves as a blob)
OWN_TOL = 0.03  # ...and the piece itself matches its own outline in the photos (SAM's mask for that part, or what's
# left for the body) at most this much worse than carved: thin spikes barely change the whole outline, but they're
# most of a leaf's own
EARNED = 0.01  # a flat face or a sharp edge has to be earned: shapes this close to the best count as a tie, and a tie
# goes to the roundest (SHAPES order). 3 Nov: a teddy's body scored 0.945 rounded, 0.947 as a cylinder (noise; it came
# out a flat-bottomed block); an aloe's pot 0.906 as a cylinder, 0.890 rounded (earned)
MIN_SOLID = 0.55  # a clean shape has no dents or spikes, so only a piece that fills at least this much of its own
# convex hull may become one (3 Nov: an aloe's spiky leaves 0.49, its pot 0.82; a teddy's body 0.59, its head 0.88)


def solidity(mask: np.ndarray) -> float:
    """How much of its own convex hull a piece fills (1: no dents or spikes)."""
    from scipy.spatial import ConvexHull, QhullError

    pts = np.argwhere(mask).astype(float)
    try:
        return min(1.0, len(pts) / ConvexHull(pts).volume) if len(pts) > 10 else 1.0
    except QhullError:  # (flat: all in one plane)
        return 1.0


def _moments(mask, corner, voxel):
    """A piece's middle, and its spread along its own three axes (variances, axes as columns), the axis nearest
    upright second (so a cylinder's axis starts standing up)."""
    pts = corner + (np.argwhere(mask) + 0.5) * voxel
    c = pts.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((pts - c).T) + np.eye(3) * (voxel / 2) ** 2)
    up = int(np.argmax(np.abs(v[1])))
    rest = [i for i in range(3) if i != up]
    order = [rest[0], up, rest[1]]
    w, v = w[order], v[:, order]
    if np.linalg.det(v) < 0:
        v[:, 0] *= -1
    return c, w, v


def _start(shape: str, mask, corner, voxel, upright: bool = False) -> np.ndarray:
    """A shape with the same middle and spread as a piece's voxels: centre (3), log radii (3), rotation vector (3).
    `upright`: it stands on the floor, so it's only turned about the vertical (a pot doesn't lean)."""
    c, w, v = _moments(mask, corner, voxel)
    if upright:  # (its spread measured along the vertical and two level axes)
        flat = corner + (np.argwhere(mask) + 0.5) * voxel - c
        lw, lv = np.linalg.eigh(np.cov(flat[:, [0, 2]].T) + np.eye(2) * (voxel / 2) ** 2)
        yaw = float(np.arctan2(lv[1, 0], lv[0, 0]))
        w = np.array([lw[0], np.var(flat[:, 1]) + (voxel / 2) ** 2, lw[1]])
        v = Rotation.from_rotvec([0.0, -yaw, 0.0]).as_matrix()
    if shape == "ellipsoid":  # (a solid ellipsoid's variance along an axis is r^2 / 5)
        r = np.sqrt(5 * w)
    elif shape == "cylinder":  # (a solid cylinder: r^2 / 4 across, h^2 / 3 along its half height h)
        r = np.array([2 * np.sqrt(w[0]), np.sqrt(3 * w[1]), 2 * np.sqrt(w[2])])
    else:  # box (half sides a: a^2 / 3)
        r = np.sqrt(3 * w)
    if shape == "taper":
        r = np.array([2 * np.sqrt(w[0]), np.sqrt(3 * w[1]), 2 * np.sqrt(w[2])])
    prm = np.concatenate([c, np.log(np.maximum(r, voxel / 2)), Rotation.from_matrix(v).as_rotvec()])
    return np.append(prm, 0.0) if shape == "taper" else prm  # (taper: how much wider the top is than the middle)


def shape_distance(shape: str, points: np.ndarray, prm: np.ndarray) -> np.ndarray:
    """About how far each point is outside the shape, in metres (negative inside)."""
    c, r, rot = prm[:3], np.exp(prm[3:6]), Rotation.from_rotvec(prm[6:9]).as_matrix()
    q = (points - c) @ rot
    if shape == "ellipsoid":  # (Quilez's estimate: exact on the surface, smooth everywhere)
        k0 = np.linalg.norm(q / r, axis=1)
        k1 = np.linalg.norm(q / (r * r), axis=1)
        return np.where(k1 > 1e-9, k0 * (k0 - 1.0) / np.maximum(k1, 1e-9), -float(r.min()))  # (at its very
        # middle the estimate is 0/0: that's as deep inside as it gets)
    if shape in ("cylinder", "taper"):  # (round across x and z, along y; capped; a taper's radius grows along y)
        grow = np.clip(1.0 + prm[9] * q[:, 1] / r[1], 0.05, None) if shape == "taper" else 1.0
        across = (np.linalg.norm(q[:, [0, 2]] / r[[0, 2]], axis=1) / grow - 1.0) * min(r[0], r[2]) * grow
        along = np.abs(q[:, 1]) - r[1]
        out = np.stack([np.maximum(across, 0), np.maximum(along, 0)], axis=1)
        return np.minimum(np.maximum(across, along), 0) + np.linalg.norm(out, axis=1)
    edge = np.abs(q) - r  # box
    return np.linalg.norm(np.maximum(edge, 0), axis=1) + np.minimum(edge.max(axis=1), 0)


def smooth_union(distances: list[np.ndarray], k: float) -> np.ndarray:
    """The parts blended into one: like taking the nearest, but rounded over a width `k` where they meet."""
    d = np.stack(distances)
    m = d.min(axis=0)
    return m - k * np.log(np.exp(-(d - m) / k).sum(axis=0))


def carved_distance(mask: np.ndarray, voxel: float) -> np.ndarray:
    """A carved piece as a distance field (metres, negative inside), softened a voxel like `carve.surface`."""
    sd = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)
    return ndimage.gaussian_filter(sd * voxel, C.SMOOTH)


class Fitter:
    """Each piece carved, or a clean shape fitted to the photos: its middle, size and turn stepped in turn (the steps
    halving each pass), keeping a step if the whole outline then matches every photo better, and each named part
    its own SAM mask (weighted PART_WEIGHT)."""

    def __init__(self, pieces, kept, corner, voxel, shots, height, labels):
        self.shots, self.height, self.k = shots, height, 1.5 * voxel
        self.pieces, self.full_voxel = pieces, voxel
        self.shape = tuple(int(n) for n in np.ceil(np.array(kept.shape) / 2))
        self.corner, self.voxel = corner, voxel * 2  # (scored on a half-size grid: 8 times fewer points)
        self.points = _points(self.shape, corner, self.voxel)
        self.carved = []  # each piece's carving as a distance field, on the full grid and on the half one
        half = np.indices(self.shape).reshape(3, -1) * 2 + 0.5
        for p in pieces:
            full = carved_distance(p.mask, voxel)
            self.carved.append((full, ndimage.map_coordinates(full, half, order=1, mode="nearest")))
        self.kinds = ["carved"] * len(pieces)
        rows = np.argwhere(kept)[:, 1]
        floor, tall = rows.min(), max(rows.max() - rows.min(), 1)
        self.upright = [np.argwhere(p.mask)[:, 1].min() - floor <= 0.08 * tall for p in pieces]  # (on the floor)
        self.params: list = [None] * len(pieces)
        self.labels = [p.label for p in pieces]
        self.per_part = {}  # label -> stand-in shots whose cut-out is SAM's mask for that part
        for lab in set(self.labels):
            stand_ins = []
            for s, lm in zip(shots, labels, strict=True):
                m = lm == lab
                if m.sum() > 200:
                    stand_ins.append(C.Shot(m, s.image, s.focal, s.angle, s.distance, s.centre, s.elevation,
                                            s.overhead, pose=s.pose))
            self.per_part[lab] = stand_ins

    def _each(self, kinds, params):
        return [self.carved[i][1] if kind == "carved" else shape_distance(kind, self.points, prm)
                for i, (kind, prm) in enumerate(zip(kinds, params, strict=True))]

    def score(self, kinds=None, params=None) -> float:
        kinds = self.kinds if kinds is None else kinds
        params = self.params if params is None else params
        each = self._each(kinds, params)
        whole = smooth_union(each, self.k).reshape(self.shape) < 0
        if not whole.any():
            return 0.0
        total = C.agreement(self.shots, whole, self.corner, self.voxel, self.height)
        parts = []
        for d, lab in zip(each, self.labels, strict=True):
            if lab != BODY and self.per_part.get(lab):  # (every named piece, carved or not: the same test)
                occ = (d < 0).reshape(self.shape)
                parts.append(C.agreement(self.per_part[lab], occ, self.corner, self.voxel, self.height)
                             if occ.any() else 0.0)
        return total + PART_WEIGHT * (float(np.mean(parts)) if parts else 0.0)

    def own(self, i: int, kind: str, prm) -> float:
        """How well piece i alone matches its own outline in the photos (SAM's mask for its part)."""
        stand = self.per_part.get(self.labels[i])
        if not stand:
            return 1.0
        d = self.carved[i][1] if kind == "carved" else shape_distance(kind, self.points, prm)
        occ = (d < 0).reshape(self.shape)
        return C.agreement(stand, occ, self.corner, self.voxel, self.height) if occ.any() else 0.0

    def _descend(self, which: list[int], kinds, params, rounds: int, best: float):
        reach = max(float(np.exp(params[i][3:6]).max()) for i in which)
        steps = [reach * 0.15] * 3 + [0.15] * 3 + [0.15] * 3 + [0.15]  # metres, log radius, radians, taper
        for r in range(rounds):
            for i in which:
                for j in range(len(params[i])):
                    if self.upright[i] and j in (6, 8):  # (on the floor: turned about the vertical only)
                        continue
                    for d in (-steps[j] * 0.5 ** r, steps[j] * 0.5 ** r):
                        trial = list(params)
                        trial[i] = params[i].copy()
                        trial[i][j] += d
                        got = self.score(kinds, trial)
                        if got > best + 1e-5:
                            best, params = got, trial
        return best, params

    def choose(self) -> float:
        """Biggest piece first, each tried as every clean shape; the best kept if the whole still matches the photos
        within PIECE_TOL of it carved. Then all the clean ones fitted together."""
        best = self.score()
        for i in sorted(range(len(self.pieces)), key=lambda i: -int(self.pieces[i].mask.sum())):
            if solidity(self.pieces[i].mask) < MIN_SOLID:  # (spiky or full of dents: it stays carved)
                continue
            tried = []
            for shape in SHAPES:
                kinds = list(self.kinds)
                kinds[i] = shape
                params = list(self.params)
                params[i] = _start(shape, self.pieces[i].mask, self.corner, self.full_voxel, self.upright[i])
                got, params = self._descend([i], kinds, params, 4, self.score(kinds, params))
                tried.append((got, shape, params))
            top = max(t[0] for t in tried)
            carved_own = self.own(i, "carved", None)
            for got, shape, params in tried:  # (the roundest of the best that the photos accept)
                if got >= top - EARNED and got >= best - PIECE_TOL and \
                        self.own(i, shape, params[i]) >= carved_own - OWN_TOL:
                    self.kinds[i], self.params, best = shape, params, got
                    break
        clean = [i for i, k in enumerate(self.kinds) if k != "carved"]
        if clean:
            best, self.params = self._descend(clean, self.kinds, self.params, FIT_ROUNDS, best)
        return best

    def extent(self, grid_corner: np.ndarray, grid_shape) -> tuple[np.ndarray, tuple[int, ...]]:
        """A grid (corner, shape) at the full voxel size holding the carving and every clean shape whole, with two
        voxels to spare: a shape grown past the carving's box would otherwise be cut open at its edge (3 Nov: a skull
        fitted as one rounded shape came out sliced, its inside showing from behind)."""
        v = self.full_voxel
        lo, hi = grid_corner.copy(), grid_corner + np.array(grid_shape) * v
        for kind, prm in zip(self.kinds, self.params, strict=True):
            if kind != "carved":
                reach = float(np.exp(prm[3:6]).max()) * (1 + abs(prm[9]) if kind == "taper" else 1.0)
                lo, hi = np.minimum(lo, prm[:3] - reach), np.maximum(hi, prm[:3] + reach)
        lo = grid_corner - np.ceil((grid_corner - lo) / v) * v - 2 * v  # (on the carving's own lattice)
        shape = tuple(int(n) for n in np.ceil((hi - lo) / v) + 2)
        return lo, shape

    def field(self, shape, corner, voxel, only: list[int] | None = None) -> np.ndarray:
        """The whole thing (or pieces `only`) as one distance field on a grid of `voxel` cells from `corner` (on the
        carving's lattice): a carved piece's field set in at its place, far outside elsewhere."""
        pts = _points(shape, corner, voxel)
        at = np.round((self.corner - corner) / voxel).astype(int)  # (the carving's corner, in this grid's cells)
        each = []
        for i in (only if only is not None else range(len(self.pieces))):
            if self.kinds[i] == "carved":
                full = np.full(shape, 4 * voxel)
                c = self.carved[i][0]
                full[at[0]:at[0] + c.shape[0], at[1]:at[1] + c.shape[1], at[2]:at[2] + c.shape[2]] = c
                each.append(full.ravel())
            else:
                each.append(shape_distance(self.kinds[i], pts, self.params[i]))
        return smooth_union(each, self.k).reshape(shape)


def _skin(field: np.ndarray, corner: np.ndarray, voxel: float):
    """The zero surface of a distance field sampled at voxel centres from `corner`, wound outward."""
    verts, faces, _, _ = marching_cubes(field, 0.0, spacing=(voxel,) * 3)
    verts = verts + corner + 0.5 * voxel
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    if np.einsum("ij,ij->i", n, tris.mean(axis=1) - verts.mean(axis=0)).sum() < 0:
        faces = faces[:, ::-1]
    return verts, faces


def _painted(name: str, verts, faces, shots, height, about: np.ndarray | None = None) -> Part:
    """A skin painted from the photos, its positions taken about `about` (a joint), else as they are."""
    about = np.zeros(3) if about is None else about
    normals = C.vertex_normals(verts, faces)
    colours = C.paint(verts, normals, shots, height)
    return Part(name, (verts - about).astype(np.float32), normals.astype(np.float32),
                np.zeros((len(verts), 2), np.float32), faces.astype(np.uint32).ravel(), (1, 1, 1, 1),
                colors=colours.astype(np.float32))


def rebuild(kept, corner, voxel, shots, height, labels, nouns, step=lambda _: None) -> tuple[Soft | None, list[str]]:
    """The carving rebuilt piece by piece: each piece a clean shape (rounded, a cylinder or a box) where one matches
    the photos, else carved; and its rig. None (with why) if no clean shape fits any piece."""
    grid = vote_labels(kept, corner, voxel, shots, height, labels)
    pieces = split(grid, nouns)
    if not pieces:
        return None, ["clean shapes: nothing to fit"]
    attach(pieces, corner, voxel)
    step(f"fitting clean shapes to its {len(pieces)} piece{'s' if len(pieces) > 1 else ''} "
         f"({', '.join(p.name for p in pieces)}) against the photos")
    fitter = Fitter(pieces, kept, corner, voxel, shots, height, labels)
    fitter.choose()
    if all(k == "carved" for k in fitter.kinds):
        return None, ["clean shapes: none matched the photos as well as the carving, so it's carved"]
    for p, prm in zip(pieces, fitter.params, strict=True):
        p.params = prm
    origin, shape = fitter.extent(corner, kept.shape)
    whole = fitter.field(shape, origin, voxel)
    at = np.round((corner - origin) / voxel).astype(int)
    solid = whole[at[0]:at[0] + kept.shape[0], at[1]:at[1] + kept.shape[1], at[2]:at[2] + kept.shape[2]] < 0
    smooth = C.agreement(shots, solid, corner, voxel, height)
    carved = C.agreement(shots, kept, corner, voxel, height)
    if smooth < carved - KEEP_WITHIN:
        return None, [f"clean shapes: matched the photos {smooth:.2f} against the carving's {carved:.2f}, so it's "
                      "carved"]
    part = _painted("prop", *_skin(whole, origin, voxel), shots, height)
    rig = []
    if len(pieces) > 1:
        for i, p in enumerate(pieces):
            mine = fitter.field(shape, origin, voxel, only=[i])
            if p.parent is not None:  # grown into its parent a little, so no gap opens when it turns
                mine = np.minimum(mine, np.maximum(mine - voxel, fitter.field(shape, origin, voxel,
                                                                              only=[p.parent])))
            rig.append(_painted(p.name, *_skin(mine, origin, voxel), shots, height, about=p.pivot))
    said = ", ".join(f"{p.name}: {k}" for p, k in zip(pieces, fitter.kinds, strict=True))
    lines = [f"clean shapes: {said}; they match the photos {smooth:.2f} (the carving {carved:.2f})"]
    return Soft(part, pieces, rig, smooth, carved), lines


def quaternion(axis: np.ndarray, degrees: float) -> list[float]:
    t = math.radians(degrees) / 2
    return [*(axis * math.sin(t)).tolist(), math.cos(t)]


def idle(pieces: list[Piece], keys: int = 25) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """The idle loop: times, and each moving piece's rotations (x, y, z, w) at them. A pair sways in turn."""
    times = np.linspace(0, LOOP, keys).astype(np.float32)
    turns = {}
    for i, p in enumerate(pieces):
        if p.swing:
            phase = 0.0 if p.name.startswith("left") else math.pi if p.name.startswith("right") else math.pi / 2
            turns[i] = np.array([quaternion(p.axis, p.swing * math.sin(2 * math.pi * t / LOOP + phase))
                                 for t in times], np.float32)
    return times, turns


def write_moving(path: Path, soft: Soft, name: str) -> Path:
    """The rig as glTF: a node per piece on its joint, and the idle loop."""
    from .build import write_glb

    pieces = soft.pieces
    joints = [(p.parent, (p.pivot - (pieces[p.parent].pivot if p.parent is not None else 0)).tolist())
              for p in pieces]
    return write_glb(path, soft.rig, name=name, joints=joints, animation=idle(pieces))


def soften(meta, photos, shots, grid, height: float, out: Path,
           step=lambda _: None) -> tuple[Soft | None, list[str]]:
    """The make's carving (`grid`: kept, corner, voxel; `shots` its photos' fitted cameras, in `photos` order;
    `height` what it was carved for) rebuilt from smooth parts. Asks Muse once (3 votes) and SAM 3.1 once per part
    per photo."""
    kept, corner, voxel = grid
    step("Muse names its parts")
    nouns = part_nouns(meta, photos[0].small, out)
    if nouns:  # (none: the whole thing is one piece, a box or a ball or a can, and SAM isn't asked anything)
        step(f"SAM 3.1 is finding its {', '.join(nouns)} in {len(photos)} photos")
    jobs = [(i, k) for i in range(len(photos)) for k in range(len(nouns))]
    with ThreadPoolExecutor(4) as pool:
        masks = list(pool.map(lambda j: _sam_masks(meta, photos[j[0]].small, nouns[j[1]], out), jobs))
    found = [[[] for _ in nouns] for _ in photos]
    for (i, k), m in zip(jobs, masks, strict=True):
        found[i][k] = m
    labels = [label_map(s.mask, f) for s, f in zip(shots, found, strict=True)]
    made, lines = rebuild(kept, corner, voxel, shots, height, labels, nouns, step)
    from . import seen

    for ph, s, lab in zip(photos, shots, labels, strict=True):  # (what SAM found, and the model over each photo)
        seen.parts(s.image, lab, out / "seen" / f"{ph.path.stem}_parts.jpg")
        if made is not None:
            faces = made.part.indices.reshape(-1, 3)
            model = seen.mesh_outline(made.part.positions.astype(float), faces, s, height / 2)
            seen.fit(s.image, s.mask, model, out / "seen" / f"{ph.path.stem}_fit.jpg")
    return made, lines
