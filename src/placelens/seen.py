"""What the models saw, as pictures kept with each make (its `seen/` folder): the studio's "How it was made" shows them.

For each photo: SAM 3.1's outline of the thing; with "Find its parts", SAM 3.1's parts in colour; and the finished
model drawn over the photo (purple where model and photo agree, red where the model is and the thing isn't, blue
where the thing is and the model isn't). Small JPEGs, a few tens of KB each.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

LONG_SIDE = 480  # px: the pictures are for looking at, not measuring
PART_COLOURS = [(231, 76, 60), (52, 120, 230), (46, 204, 113), (241, 196, 15), (155, 89, 182), (26, 188, 156),
                (230, 126, 34)]
BODY_COLOUR = (150, 160, 165)  # what's left of the cut-out once the named parts are taken out
AGREE, MODEL_ONLY, THING_ONLY = (160, 80, 200), (230, 60, 60), (60, 140, 240)


def _save(rgb: np.ndarray, path: Path) -> Path:
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    img.thumbnail((LONG_SIDE, LONG_SIDE))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=82)
    return path


def outline(image: np.ndarray, mask: np.ndarray, path: Path) -> Path:
    """The photo dimmed outside SAM's cut-out, and the cut-out's edge drawn in yellow."""
    from scipy import ndimage

    out = image.astype(float) * np.where(mask[..., None], 1.0, 0.3)
    width = max(2, image.shape[0] // 250)
    edge = mask & ~ndimage.binary_erosion(mask, iterations=width)
    out[edge] = (255, 220, 0)
    return _save(out, path)


def parts(image: np.ndarray, labels: np.ndarray, path: Path) -> Path:
    """Each part in its own colour (labels: 0 outside, 1 the rest of the thing, 2.. the parts SAM found)."""
    out = image.astype(float) * 0.3
    for k in [int(k) for k in np.unique(labels) if k > 0]:
        m = labels == k
        colour = BODY_COLOUR if k == 1 else PART_COLOURS[(k - 2) % len(PART_COLOURS)]
        out[m] = 0.35 * image[m] + 0.65 * np.array(colour)
    return _save(out, path)


def fit(image: np.ndarray, mask: np.ndarray, model: np.ndarray, path: Path) -> Path:
    """The model's outline against the thing's: purple both, red model only, blue thing only."""
    out = image.astype(float) * 0.3
    for m, colour in ((model & mask, AGREE), (model & ~mask, MODEL_ONLY), (~model & mask, THING_ONLY)):
        out[m] = 0.35 * image[m] + 0.65 * np.array(colour)
    return _save(out, path)


def mesh_outline(verts: np.ndarray, faces: np.ndarray, shot, mid: float) -> np.ndarray:
    """Where a triangle mesh shows in a shot's photo, at the photo's size."""
    import cv2

    cols, rows, depth = shot.project(verts, mid)
    h, w = shot.mask.shape
    out = np.zeros((h, w), np.uint8)
    ok = (depth[faces] > 0).all(axis=1)
    pts = np.round(np.stack([cols, rows], axis=1) * 4).astype(np.int32)
    for tri in faces[ok]:
        cv2.fillConvexPoly(out, pts[tri], 1, shift=2)
    return out.astype(bool)


def shapes_outline(shapes: list[dict], shot, mid: float) -> np.ndarray:
    """Where a model of shapes (parts.py) shows in a shot's photo, at the photo's size."""
    from PIL import Image as Img

    from .sight import silhouette

    small = silhouette(shapes, shot, mid, 2)
    h, w = shot.mask.shape
    return np.asarray(Img.fromarray(small.astype(np.uint8) * 255).resize((w, h), Img.NEAREST)) > 0


def listed(folder: Path) -> list[str]:
    """The pictures a make kept, in the order to show them: photo by photo, outline then parts then fit."""
    seen = folder / "seen"
    if not seen.is_dir():
        return []
    order = {"outline": 0, "parts": 1, "fit": 2}
    names = [p.name for p in seen.glob("*.jpg")]
    return sorted(names, key=lambda n: (n.rsplit("_", 1)[0], order.get(n.rsplit("_", 1)[-1][:-4], 9)))
