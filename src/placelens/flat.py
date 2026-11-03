"""Flat things standing up (a steering wheel, a sign, a shield), built as 3D from their exact outline.

Carving a flat thing from photos mostly taken edge on leaves a slab, and Muse's design from boxes misses its outline.
SAM 3.1's outline in the photo that faces it is exact, holes and all. So the thing is built from it, as geometry:

- Its face is the outline on a fine grid, a few millimetres a cell, each cell a solid column the thing's depth (Muse's
  number) deep. Hand holes are real holes, and the side walls run round every edge, the holes' too.
- Each cell is painted from the photo, the way a carving is painted: the colour of the thing at that spot, on the 3D
  surface. Nothing is laid over it as a picture. Its back is painted the same way from the photo taken from behind
  (mirrored: from behind, left and right swap), if there is one.
- SAM 3.1 finds its buttons and knobs on each face, and each becomes its own raised cylinder, at its place and size,
  in its own colour.

Not pasted, built: on 3 Nov the standing steering wheel was first given its photo as a picture on a slab, which is the
look PlaceLens doesn't do, and was rebuilt this way. (Raising everything Muse names on each face as a relief, and
softening the paint, was tried the same day: SAM found nothing named on the wheel's back photo, low and far off, and
the softened face lost its light ring and its white buttons went grey. Kept as it is.)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .build import Part

CELLS = 10000  # about this many cells across the face (a 30 cm wheel: about 2.5 mm each)
CONTROLS = ("button", "knob")  # what SAM 3.1 is asked for on each face (a quarter of a US cent each)
CONTROL_AREA = (0.0004, 0.03)  # a control's share of the face: smaller is a speck, bigger is a panel, not a control
RAISE = 0.35  # a control stands this share of the thing's depth proud of its face


def _cells(mask: np.ndarray, image: np.ndarray, step: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """The face cut into square cells: which are the thing's (most of the cell inside SAM's outline), each cell's mean
    colour (0..1), and the cell size in pixels."""
    h, w = mask.shape
    step = step or max(1, int(math.ceil(math.sqrt(h * w / CELLS))))
    rows, cols = h // step, w // step
    m = mask[:rows * step, :cols * step].reshape(rows, step, cols, step).mean(axis=(1, 3))
    img = image[:rows * step, :cols * step].reshape(rows, step, cols, step, 3).mean(axis=(1, 3)) / 255
    return m >= 0.5, img, step


def _grid_mesh(inside: np.ndarray, colour: np.ndarray, back: np.ndarray | None, cell: float, depth: float):
    """The cells as one closed solid: front and back faces, and walls wherever a cell meets the outside. Positions in
    metres, the face's bottom on the ground, centred across, its front at +depth/2 (props face +Z). `back`: each
    cell's colour from a photo of the back (NaN where it doesn't show), else the face's own darker tone."""
    rows, cols = inside.shape
    half = depth / 2
    x0 = cols * cell / 2
    pos, nrm, col, idx = [], [], [], []
    plain = np.median(colour[inside], axis=0) * 0.85

    def quad(corners, normal, rgb):
        base = len(pos)
        pos.extend(corners)
        nrm.extend([normal] * 4)
        col.extend([rgb] * 4)
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    bottom = (inside.any(axis=1).nonzero()[0].max() + 1) * cell
    for r, c in zip(*np.nonzero(inside), strict=True):
        xa, xb = c * cell - x0, (c + 1) * cell - x0
        ya, yb = bottom - (r + 1) * cell, bottom - r * cell  # (rows run down the photo, metres run up)
        rgb = tuple(float(v) for v in colour[r, c])
        behind = back[r, c] if back is not None and not np.isnan(back[r, c]).any() else plain
        quad([(xa, ya, half), (xb, ya, half), (xb, yb, half), (xa, yb, half)], (0, 0, 1), rgb)
        quad([(xb, ya, -half), (xa, ya, -half), (xa, yb, -half), (xb, yb, -half)], (0, 0, -1),
             tuple(float(v) for v in behind))
        edge = tuple(float(v) * 0.8 for v in colour[r, c])
        if r == 0 or not inside[r - 1, c]:  # top
            quad([(xa, yb, half), (xb, yb, half), (xb, yb, -half), (xa, yb, -half)], (0, 1, 0), edge)
        if r == rows - 1 or not inside[r + 1, c]:  # bottom
            quad([(xa, ya, -half), (xb, ya, -half), (xb, ya, half), (xa, ya, half)], (0, -1, 0), edge)
        if c == 0 or not inside[r, c - 1]:  # left
            quad([(xa, ya, -half), (xa, ya, half), (xa, yb, half), (xa, yb, -half)], (-1, 0, 0), edge)
        if c == cols - 1 or not inside[r, c + 1]:  # right
            quad([(xb, ya, half), (xb, ya, -half), (xb, yb, -half), (xb, yb, half)], (1, 0, 0), edge)
    pos = np.array(pos, np.float32)
    return Part("face", pos, np.array(nrm, np.float32), np.zeros((len(pos), 2), np.float32),
                np.array(idx, np.uint32), (1, 1, 1, 1), colors=np.array(col, np.float32))


def squareness(ph) -> float:
    """How squarely a photo faces a flat thing: its outline's width for its height (side on, it's a sliver)."""
    rows, cols = np.flatnonzero(ph.mask.any(axis=1)), np.flatnonzero(ph.mask.any(axis=0))
    return (cols[-1] - cols[0] + 1) / max(rows[-1] - rows[0] + 1, 1)


def _round(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


class _View:
    """One photo of the flat thing, cropped to it and lined up with the front's crop (`size`, rows by columns): the
    back's photo mirrored, as seen from behind left and right swap."""

    def __init__(self, ph, size: tuple[int, int], mirrored: bool):
        rows, cols = np.flatnonzero(ph.mask.any(axis=1)), np.flatnonzero(ph.mask.any(axis=0))
        self.box = (rows[0], rows[-1], cols[0], cols[-1])
        self.size, self.mirrored, self.ph = size, mirrored, ph
        with Image.open(ph.small) as img:
            self.image = self.fit(np.asarray(img.convert("RGB")))
        self.mask = self.fit(ph.mask)

    def fit(self, a: np.ndarray) -> np.ndarray:
        """A whole-photo array (a mask, or the photo) cropped to the thing and lined up with the front's crop."""
        top, bottom, left, right = self.box
        crop = a[top:bottom + 1, left:right + 1]
        if crop.shape[:2] != self.size:
            boolean = crop.dtype == bool
            img = Image.fromarray(crop.astype(np.uint8) * (255 if boolean else 1))
            img = img.resize((self.size[1], self.size[0]), Image.NEAREST if boolean else Image.BILINEAR)
            crop = np.asarray(img) > 127 if boolean else np.asarray(img)
        return crop[:, ::-1] if self.mirrored else crop


def _controls(meta, view: _View, out: Path, scale: float, centre_x: float, bottom_px: float, depth: float,
              side: float) -> list[dict]:
    """Buttons and knobs SAM 3.1 finds in a view, as cylinders raised from that face (`side`: +1 the front, -1 the
    back), at their places and sizes, in their own colours."""
    from .soft import _sam_masks

    area = view.mask.sum()
    shapes, taken = [], np.zeros_like(view.mask)
    for noun in CONTROLS:
        for whole in _sam_masks(meta, view.ph.small, noun, out):
            m = view.fit(whole) & view.mask
            share = m.sum() / max(area, 1)
            if not CONTROL_AREA[0] <= share <= CONTROL_AREA[1] or (m & taken).sum() > 0.5 * m.sum():
                continue
            taken |= m
            ys, xs = np.nonzero(m)
            radius = math.sqrt(m.sum() / math.pi) * scale
            rgb = np.median(view.image[m], axis=0)
            shapes.append({"shape": "cylinder", "what": noun, "finish": "plastic",
                           "at": [(xs.mean() - centre_x) * scale, (bottom_px - ys.mean()) * scale,
                                  side * (depth / 2 + RAISE * depth / 2)],
                           "size": [2 * radius, RAISE * depth, 2 * radius], "turn": [90, 0, 0],
                           "colour": "#" + "".join(f"{int(v):02x}" for v in rgb)})
    return shapes


def built(meta, photos: list, wide: float, deep: float, tall: float, out: Path,
          step=lambda _: None) -> tuple[list[Part], str]:
    """A flat thing standing up, built from its outline in the photo that faces it most squarely, its back painted
    from the photo taken from behind (if there is one): its parts and a note."""
    from .parts import model_parts

    face = max(photos, key=squareness)
    behind = [ph for ph in photos if ph is not face and _round(ph.angle, face.angle) >= 135
              and squareness(ph) >= 0.6 * squareness(face)]
    back_ph = max(behind, key=squareness) if behind else None
    rows_, cols_ = np.flatnonzero(face.mask.any(axis=1)), np.flatnonzero(face.mask.any(axis=0))
    size = (rows_[-1] - rows_[0] + 1, cols_[-1] - cols_[0] + 1)
    front = _View(face, size, mirrored=False)
    inside, colour, px = _cells(front.mask, front.image)
    back_colour = None
    views = [(front, 1.0)]
    if back_ph is not None:
        back = _View(back_ph, size, mirrored=True)
        seen, painted, _ = _cells(back.mask, back.image, px)
        back_colour = np.where(seen[..., None], painted, np.nan)
        views.append((back, -1.0))
    scale = tall / size[0]  # metres a pixel
    depth = deep or 0.5 * wide
    body = _grid_mesh(inside, colour, back_colour, px * scale, depth)
    controls = []
    for view, side in views:
        step(f"SAM 3.1 is finding its buttons and knobs in {view.ph.path.name}")
        controls += _controls(meta, view, out, scale, inside.shape[1] * px / 2,
                              (inside.any(axis=1).nonzero()[0].max() + 1) * px, depth, side)
    said = f"; its back painted from {back_ph.path.name}" if back_ph is not None else "; no photo of its back"
    note = (f"flat: built from its outline in {face.path.name}, {depth:.2f} m deep, painted from that photo{said}; "
            f"{len(controls)} button(s) and knob(s) raised from its faces")
    return [body] + (model_parts(controls) if controls else []), note
