"""Walk-around props: an object's 3D shape carved from its outlines in photos taken all round it.

Each photo's SAM 3.1 cut-out is the object's silhouette from one direction. A point in space that falls outside any
silhouette can't be part of the object, so carving away every such point from a block leaves the object's shape
(its "visual hull"). Each photo's direction comes from its compass, checked by Muse naming the side it shows. The
surface is then painted from the photos, each spot from the photo facing it most squarely.

Axes: metres, Y up, standing on Y = 0, the first photo's side facing +Z (as other props).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from .build import Part

RESOLUTION = 72  # voxels along the block's longest side (128 tried 31 Oct: lumpier, not more detail, on fur)
MARGIN = 1.3  # the block is this much bigger than Muse's size for the object
OVERHEAD = 75.0  # degrees above: how a photo looking down on the object from over it is taken to be
OVERHEAD_TILTS = (65.0, 75.0, 85.0)  # tried for it, with every turn
IMAGINED_SLACK = 0.03  # a drawing's outline is widened by this share of its height before it carves
IMAGINED_PAINT = 0.05  # a drawing's colours count this much against a photo's (they fill only what photos miss)
BORDER = 3  # pixels: a cut-out this close to the photo's edge runs off it
CLEAR = 0.02  # share of a photo's height: a point this far outside a cut-out is clearly not the object, and that one
# photo is enough to carve it (the forgiveness of `misses` is for points near an edge, where a compass a few degrees off
# would wrongly carve; given everywhere, it kept the gap between a teddy's legs and 40% of it in all)
SMOOTH = 1.0  # voxels: the carved block is blurred this much before its surface is taken, to round off the steps

# Walking round the object, which side each photo shows, and where that puts the camera: degrees round the object
# from the first photo's side, turning the way you'd walk to your left.
WALK = {"front": 0, "front-left": -45, "left": -90, "back-left": -135, "back": 180, "back-right": 135, "right": 90,
        "front-right": 45}

SIDE_PROMPT = """Both photos show the same object, photo 1 from its front. Walking round the object, which side of \
it does photo 2 show? Answer one of "front", "front-left", "left", "back-left", "back", "back-right", "right", \
"front-right", where left means walking round to your left from where photo 1 was taken; or "above" if photo 2 \
looks down on it from over the top. Use its shape and markings, and what's behind it, to tell.

Answer in JSON only: {"side": "...", "confidence": 0.0 to 1.0}"""


@dataclass
class Shot:
    """One photo of the object: its cut-out over the whole photo, the photo itself, and where the camera was."""

    mask: np.ndarray  # bool, rows x cols, the photo's size
    image: np.ndarray  # uint8 RGB, same size
    focal: float  # pixels
    angle: float  # degrees round the object from the first photo, positive turning right (see WALK)
    distance: float = 0.0  # metres from the object's middle; set by `place_shots`
    centre: tuple[float, float] = (0.0, 0.0)  # the pixel the object's middle shows at
    elevation: float = 0.0  # degrees the camera looks down at the object's middle from (standing over it)
    overhead: bool = False  # taken from above, looking down: its angle round is found by fit, not the compass
    imagined: bool = False  # drawn by Muse Image, not photographed: it carves only what's clearly outside, and
    # paints only where no photo can
    pose: object = None  # solve.Pose: the camera found by matching the photos; then angle, elevation, distance and
    # centre aren't used, and the photo's own centre is its middle

    def camera(self, mid: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Position, right, up and forward unit vectors of a camera looking at the object's middle (`mid` metres up)
        from `angle` round it and `elevation` above it; or where matching the photos put it."""
        if self.pose is not None:
            p = self.pose
            return p.position + np.array([0.0, mid, 0.0]), p.right, p.up, p.forward
        a, e = math.radians(self.angle), math.radians(self.elevation)
        position = np.array([self.distance * math.cos(e) * math.sin(a), mid + self.distance * math.sin(e),
                             self.distance * math.cos(e) * math.cos(a)])
        forward = np.array([-math.cos(e) * math.sin(a), -math.sin(e), -math.cos(e) * math.cos(a)])
        right = np.array([math.cos(a), 0.0, -math.sin(a)])
        return position, right, np.cross(right, forward), forward

    def project(self, points: np.ndarray, mid: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel columns, rows and depths of points (n x 3)."""
        position, right, up, forward = self.camera(mid)
        rel = points - position
        depth = rel @ forward
        safe = np.where(depth > 1e-6, depth, 1e-6)
        h, w = self.mask.shape
        cx, cy = (w / 2, h / 2) if self.pose is not None else self.centre  # (a solved camera is the real one)
        cols = cx + self.focal * (rel @ right) / safe
        rows = cy - self.focal * (rel @ up) / safe
        return cols, rows, depth


def place_shots(shots: list[Shot], height: float) -> None:
    """Each shot's distance from how tall the object looks in it, and where its middle shows."""
    for s in shots:
        rows = np.flatnonzero(s.mask.any(axis=1))
        cols = np.flatnonzero(s.mask.any(axis=0))
        s.distance = s.focal * height / max(rows[-1] - rows[0], 1)
        s.centre = ((cols[0] + cols[-1]) / 2, (rows[0] + rows[-1]) / 2)


def carve(shots: list[Shot], size: tuple[float, float, float], misses: int = 1,
          resolution: int = RESOLUTION) -> tuple[np.ndarray, np.ndarray, float]:
    """The block of space kept: points inside the silhouette in all but `misses` of the side photos (a photo whose
    angle is a little off shouldn't cut away a real part) and inside every overhead photo (the only one that sees
    into the hollows, so it mustn't be outvoted), then only its biggest connected piece. Beyond a photo's frame,
    a point is outside the object only if the object doesn't run off that edge; otherwise that photo can't say.
    `size` is the object's width, depth and height in metres. Returns the kept grid, its corner and voxel size."""
    width, depth, height = size
    half = max(width, depth) * MARGIN / 2
    voxel = max(2 * half, height * MARGIN) / resolution
    xs = np.arange(-half, half, voxel) + voxel / 2
    ys = np.arange(0, height * MARGIN, voxel) + voxel / 2
    gx, gy, gz = np.meshgrid(xs, ys, xs, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    votes = np.zeros(len(points), int)
    clear = np.zeros(len(points), bool)  # clearly outside some real photo's cut-out: carved whatever the others say
    firm = np.ones(len(points), bool)  # inside every overhead photo: the only view into hollows from above
    for s in shots:
        cols, rows, depth = s.project(points, height / 2)
        h, w = s.mask.shape
        c, r = np.round(cols).astype(int), np.round(rows).astype(int)
        framed = (depth > 0) & (c >= 0) & (c < w) & (r >= 0) & (r < h)
        inside = np.zeros(len(points), bool)
        mask = s.mask
        if s.imagined:  # a drawing is roughly right: let it cut away only what's clearly outside
            rows_ = np.flatnonzero(mask.any(axis=1))
            mask = ndimage.binary_dilation(mask, iterations=max(1, int(IMAGINED_SLACK * (rows_[-1] - rows_[0]))))
        inside[framed] = mask[r[framed], c[framed]]
        if not s.imagined and not s.overhead:
            away = ndimage.distance_transform_edt(~s.mask)
            clear[framed] |= away[r[framed], c[framed]] > CLEAR * h
        # Off an edge the object runs off, this photo can't tell: don't carve there.
        left, right = s.mask[:, :BORDER].any(), s.mask[:, -BORDER:].any()
        top, bottom = s.mask[:BORDER].any(), s.mask[-BORDER:].any()
        unknown = (left & (c < 0)) | (right & (c >= w)) | (top & (r < 0)) | (bottom & (r >= h))
        inside |= ~framed & (unknown | (depth <= 0))
        if s.overhead:
            firm &= inside
        else:
            votes += inside
    sides = sum(not s.overhead for s in shots)
    kept = ((votes >= sides - misses) & firm & ~clear).reshape(gx.shape)
    labels, count = ndimage.label(kept)
    if count > 1:  # the object is one piece; anything else is left over from carving
        sizes = ndimage.sum(kept, labels, range(1, count + 1))
        kept = labels == 1 + int(np.argmax(sizes))
    return kept, np.array([xs[0], ys[0], xs[0]]) - voxel / 2, voxel


def surface(kept: np.ndarray, corner: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """The kept block's skin as a smooth triangle mesh: vertices (metres) and faces, wound outward."""
    field = ndimage.gaussian_filter(np.pad(kept.astype(float), 2), SMOOTH)
    verts, faces, _, _ = marching_cubes(field, 0.5, spacing=(voxel, voxel, voxel))
    verts = verts + corner + (0.5 - 2) * voxel  # voxel centres, less the padding
    tris = verts[faces]
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    if np.einsum("ij,ij->i", normal, tris.mean(axis=1) - verts.mean(axis=0)).sum() < 0:
        faces = faces[:, ::-1]
    return verts, faces


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = verts[faces]
    face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, faces[:, k], face)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def paint(verts: np.ndarray, normals: np.ndarray, shots: list[Shot], height: float) -> np.ndarray:
    """Each vertex's colour from the photos that face it, the most squarely facing counting most."""
    total = np.zeros((len(verts), 3))
    weight = np.zeros(len(verts))
    for s in shots:
        position = s.camera(height / 2)[0]
        towards = position - verts
        towards /= np.linalg.norm(towards, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", normals, towards)
        cols, rows, depth = s.project(verts, height / 2)
        h, w = s.mask.shape
        c, r = np.round(cols).astype(int), np.round(rows).astype(int)
        ok = (facing > 0.1) & (depth > 0) & (c >= 0) & (c < w) & (r >= 0) & (r < h)
        ok[ok] &= s.mask[r[ok], c[ok]]  # only the object's own pixels, not what's behind it
        wgt = np.where(ok, facing, 0.0) ** 4 * (IMAGINED_PAINT if s.imagined else 1.0)
        total[ok] += wgt[ok, None] * s.image[r[ok], c[ok]]
        weight += wgt
    colours = np.full((len(verts), 3), 160.0)
    seen = weight > 0
    colours[seen] = total[seen] / weight[seen, None]
    if seen.any() and not seen.all():  # spots no picture faces: the colour of the nearest one that is
        from scipy.spatial import cKDTree

        _, nearest = cKDTree(verts[seen]).query(verts[~seen])
        colours[~seen] = colours[seen][nearest]
    return colours / 255


def agreement(shots: list[Shot], kept: np.ndarray, corner: np.ndarray, voxel: float, height: float,
              shrink: int = 8) -> float:
    """How well the carved shape, seen again from each photo's camera, matches that photo's cut-out: the average
    overlap (intersection over union), worked out on photos shrunk `shrink` times for speed."""
    idx = np.argwhere(kept)
    if len(idx) == 0:
        return 0.0
    centres = corner + (idx + 0.5) * voxel
    scores = []
    for s in shots:
        small = s.mask[::shrink, ::shrink]
        h, w = small.shape
        cols, rows, depth = s.project(centres, height / 2)
        seen = np.zeros_like(small)
        c, r = (cols / shrink).astype(int), (rows / shrink).astype(int)
        ok = (depth > 0) & (c >= 0) & (c < w) & (r >= 0) & (r < h)
        seen[r[ok], c[ok]] = True
        reach = max(1, int(round(s.focal * voxel / np.median(depth[ok]) / shrink / 2))) if ok.any() else 1
        seen = ndimage.binary_dilation(seen, iterations=reach)
        union = (seen | small).sum()
        scores.append((seen & small).sum() / union if union else 0.0)
    return float(np.mean(scores))


def find_elevation(shots: list[Shot], size: tuple[float, float, float], steps=range(0, 61, 5)) -> float:
    """How far above the object the photos were taken from (one angle for all: a person standing). Carved from the
    right angle, the shape seen again from every photo's camera matches every cut-out; from a wrong one it doesn't.
    So the angle whose carving matches best."""
    real = [s for s in shots if not s.imagined and not s.overhead and s.pose is None]  # (drawings are level;
    # overhead shots fixed; solved cameras known)
    if not real:
        return 0.0
    best = (-1.0, 0.0)
    for e in steps:
        for s in real:
            s.elevation = float(e)
        kept, corner, voxel = carve(shots, size, resolution=RESOLUTION // 2)
        best = max(best, (agreement(real, kept, corner, voxel, size[2]), float(e)))
    for s in real:
        s.elevation = best[1]
    return best[1]


def turn_overhead(shots: list[Shot], size: tuple[float, float, float], step: int = 15) -> None:
    """A photo from above has no useful compass (a phone pointing down hardly knows which way it faces). Turn it
    round in steps, at a few tilts, and keep the turn and tilt whose carving best matches all the photos."""
    photos = [t for t in shots if not t.imagined]
    for s in [s for s in shots if s.overhead and s.pose is None]:
        best = (-1.0, 0.0, OVERHEAD)
        for tilt in OVERHEAD_TILTS:
            s.elevation = tilt
            for a in range(0, 360, step):
                s.angle = float(a)
                kept, corner, voxel = carve(shots, size, resolution=RESOLUTION // 2)
                best = max(best, (agreement(photos, kept, corner, voxel, size[2]), float(a), tilt))
        _, s.angle, s.elevation = best


def _slide(s: Shot, pose, distance: float, k: float) -> None:
    """Shot `s` moved to `k` times its distance from the object's middle (from `pose`, or `distance` if it has none)."""
    from .solve import Pose

    if pose is not None:
        s.pose = Pose(pose.position * k, pose.right, pose.up, pose.forward)
    else:
        s.distance = distance * k


def fit_distances(shots: list[Shot], size: tuple[float, float, float], rounds: int = 3,
                  scales=(0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)) -> float:
    """How far each camera stood, fitted to the outlines. A distance from how tall the object looks is only
    rough (from the front, feet sticking out make it look taller), and every photo carves, so one camera set too near
    shaves everything. Each camera in turn is slid along its line of sight to the distance whose carving best matches
    all the cut-outs; a few rounds, as each move changes what fits the others. Returns the final agreement."""
    photos = [s for s in shots if not s.imagined]
    def score() -> float:
        kept, corner, voxel = carve(shots, size, resolution=RESOLUTION // 2)
        return agreement(photos, kept, corner, voxel, size[2])

    best = score()
    for _ in range(rounds):
        moved = False
        for s in [s for s in photos if s.pose is not None or not s.overhead]:  # (a compass-placed one too)
            start, distance = s.pose, s.distance
            tried = []
            for k in scales:
                _slide(s, start, distance, k)
                tried.append((score() if k != 1.0 else best, k))
            got, k = max(tried)
            _slide(s, start, distance, k)
            moved |= k != 1.0 and got > best
            best = max(best, got)
        if not moved:
            break
    return best


FIT_STEPS = {"angle": (-8.0, -4.0, -2.0, 2.0, 4.0, 8.0), "elevation": (-6.0, -3.0, 3.0, 6.0),
             "distance": (0.9, 0.95, 1.05, 1.1)}  # tried for each compass-placed camera (distance: times)
FIT_ROUNDS = 3  # passes over the cameras, at most (on the bear, 31 Oct: strict agreement 0.69, 0.77, 0.80, 0.81)


def fit_cameras(shots: list[Shot], size: tuple[float, float, float], rounds: int | None = None) -> float:
    """Each compass-placed camera nudged until the photos' outlines agree. A compass is 10 to 20 degrees off, one
    height fits everyone only roughly, and a distance from how tall the object looks is rough too; so the cut-outs
    don't line up, and a strict carving (every photo carves) shaves real parts while a forgiving one leaves junk where
    only one photo sees through (under a teddy bear, between its legs). In turn, each camera's angle round, height
    above and distance are stepped, keeping a step if the strict carving then matches all the cut-outs better. The
    first photo's angle stays put: it is the front (turning every camera alike would match just as well).
    Returns the final strict agreement."""
    photos = [s for s in shots if not s.imagined]
    movable = [s for s in photos if s.pose is None and not s.overhead]

    def score() -> float:
        kept, corner, voxel = carve(shots, size, misses=0, resolution=RESOLUTION // 2)
        return agreement(photos, kept, corner, voxel, size[2])

    best = score()
    for _ in range(FIT_ROUNDS if rounds is None else rounds):
        moved = False
        for s in movable:
            for field, steps in FIT_STEPS.items():
                if field == "angle" and s is photos[0]:
                    continue
                start = getattr(s, field)
                tried = []
                for step in steps:
                    setattr(s, field, start * step if field == "distance" else start + step)
                    tried.append((score(), step))
                got, step = max(tried)
                if got > best + 1e-4:
                    setattr(s, field, start * step if field == "distance" else start + step)
                    best, moved = got, True
                else:
                    setattr(s, field, start)
        if not moved:
            break
    return best


def hull_grid(shots: list[Shot], size: tuple[float, float, float],
              misses: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """The walk-around's carved block: find how far above it the photos were taken, fit the cameras to the
    outlines, carve (strictly by default: with fitted cameras every photo may carve). The kept grid, its corner and
    voxel size (as `carve`)."""
    find_elevation(shots, size)
    turn_overhead(shots, size)
    if any(s.pose is None and not s.overhead and not s.imagined for s in shots):
        fit_cameras(shots, size)
    if any(s.pose is not None for s in shots):
        fit_distances(shots, size)
    return carve(shots, size, misses=misses)


def hull_mesh(shots: list[Shot], size: tuple[float, float, float],
              misses: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The walk-around's shape, skinned: vertices, faces and normals."""
    verts, faces = surface(*hull_grid(shots, size, misses))
    return verts, faces, vertex_normals(verts, faces)


def skin(kept: np.ndarray, corner: np.ndarray, voxel: float, shots: list[Shot], height: float) -> Part:
    """A carved block as a part: its surface, painted from the photos (each vertex the photos' colour there)."""
    verts, faces = surface(kept, corner, voxel)
    normals = vertex_normals(verts, faces)
    colours = paint(verts, normals, shots, height)
    return Part("prop", verts.astype(np.float32), normals.astype(np.float32), np.zeros((len(verts), 2), np.float32),
                faces.astype(np.uint32).ravel(), (1, 1, 1, 1), colors=colours.astype(np.float32))


def hull_part(shots: list[Shot], size: tuple[float, float, float], misses: int = 0, judge=None) -> Part:
    """The shape painted from the photos: one part. `judge(kept, corner, voxel) -> kept` gets a look at the carved
    block before it's skinned (judge.py: Muse says what's wrong, and what's clearly extra is trimmed). (--clean
    builds it from parts and materials instead: dress.py.)"""
    kept, corner, voxel = hull_grid(shots, size, misses)
    if judge is not None:
        kept = judge(kept, corner, voxel)
    return skin(kept, corner, voxel, shots, size[2])
