"""A small map picture: every building, the view cone, and the numbered candidates."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from .view import Candidate, cone

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
) -> Path:
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
    ax.plot(0, 0, "o", color="black", ms=8)
    ax.set_aspect("equal")
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_xlabel("metres east")
    ax.set_ylabel("metres north")
    ax.set_title("PlaceLens: buildings in view (data © OpenStreetMap contributors)", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
