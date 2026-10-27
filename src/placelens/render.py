"""A small map picture: every building, the view cone, and the numbered candidates."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from .pick import offset_to_pixel
from .view import Candidate, cone, direction

COLOURS = ["#e4572e", "#2e86ab", "#f3a712", "#4daa57", "#8e5cc2"]


def _fill(ax, poly: Polygon, **kw):
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, **kw)


def draw_map(
    out: Path,
    all_polygons: list[Polygon],
    candidates: list[Candidate],
    heading: float,
    fov: float,
    reach: float,
    margin: float,
    top: int = 5,
    corrected: tuple[float, float, float] | None = None,
    zoom: float | None = None,
) -> Path:
    """`corrected` is where the line-up says the camera really was: metres east, north and the true heading."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    _fill(ax, cone(heading, fov, reach, margin), color="#9ecae1", alpha=0.25, lw=0)
    _fill(ax, cone(heading, fov, reach), color="#3182bd", alpha=0.25, lw=0)
    for poly in all_polygons:
        _fill(ax, poly, color="#d0d0d0", ec="#999999", lw=0.5)
    for c in candidates[:top]:
        colour = COLOURS[(c.rank - 1) % len(COLOURS)]
        _fill(ax, c.polygon, color=colour, ec="black", lw=1)
        pt = c.polygon.representative_point()
        ax.annotate(str(c.rank), (pt.x, pt.y), ha="center", va="center", fontsize=12, weight="bold", color="white")
    ax.plot(0, 0, "o", color="black", ms=8, label="phone GPS")
    if corrected:
        x, y, true_heading = corrected
        dx, dy = direction(true_heading)
        ax.plot([0, x], [0, y], color="black", lw=1, ls=":")
        ax.plot(x, y, "o", mfc="white", mec="black", mew=2, ms=9, label="corrected")
        ax.annotate("", (x + dx * 12, y + dy * 12), (x, y), arrowprops={"arrowstyle": "->", "lw": 2})
        ax.legend(loc="lower left", fontsize=8)
    extent = zoom or reach
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("metres east")
    ax.set_ylabel("metres north")
    ax.set_title("PlaceLens: buildings in view (data © OpenStreetMap contributors)", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_check(out: Path, photo: Path, segment, span: tuple[float, float] | None, fov: float) -> Path:
    """The photo with SAM's outline tinted, and where the map says the chosen building's edges fall (dashed)."""
    img = Image.open(photo).convert("RGB")
    if segment is not None:
        alpha = Image.fromarray(segment.full_mask(img.size).astype(np.uint8) * 110)
        img.paste(Image.new("RGB", img.size, (46, 134, 171)), (0, 0), alpha)
    draw = ImageDraw.Draw(img)
    if span is not None:
        for offset in span:
            x = offset_to_pixel(offset, img.width, fov)
            for y in range(0, img.height, 24):
                draw.line([(x, y), (x, y + 12)], fill=(228, 87, 46), width=4)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=88)
    return out
