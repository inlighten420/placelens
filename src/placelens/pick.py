"""Which candidate is the building in the photo? Two independent checks that must agree.

Geometry: SAM cuts the building out, its left and right edges in the photo become compass directions, and those
are lined up with each candidate's edges on the map. Sliding the line-up a few degrees also measures compass error.

Muse: the model looks at the photo and the candidates' map facts and names one, with a confidence and its reasons.

Agree and both are sure: picked. Anything else: ask the person, with the evidence.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps
from shapely.geometry import Point, Polygon

from .view import Candidate

MAX_SIDE = 1600  # long side sent to the API; plenty for SAM and Muse, and keeps uploads small
SURE = 0.7  # Muse confidence needed to pick without asking
# Geometry is sure when the best line-up covers most of the outline and clearly beats the runner-up.
GEO_FIT = 0.5
GEO_LEAD = 3.0  # times the runner-up's score


def prepare_photo(photo: str | Path, out: Path) -> Path:
    """An upright, smaller JPEG copy the API accepts (iPhone HEIC included)."""
    with Image.open(photo) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=90)
    return out


def pixel_to_offset(x: float, width: int, fov: float) -> float:
    """Degrees from the photo's centre line to pixel column x (pinhole camera)."""
    half = math.radians(fov / 2)
    return math.degrees(math.atan((2 * x / width - 1) * math.tan(half)))


def offset_to_pixel(offset: float, width: int, fov: float) -> float:
    """The inverse: pixel column of a direction `offset` degrees from the centre line."""
    return width / 2 * (1 + math.tan(math.radians(offset)) / math.tan(math.radians(fov / 2)))


def photo_span(box: tuple[int, int, int, int], width: int, fov: float) -> tuple[float, float]:
    left, _, right, _ = box
    return pixel_to_offset(left, width, fov), pixel_to_offset(right, width, fov)


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over union of two angle ranges."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


# Typical phone error in a street. A correction this size is believable; three times this size is not.
GPS_ERROR = 6.0  # metres
COMPASS_ERROR = 8.0  # degrees
DISTANCE_ERROR = 0.4  # Muse's eye for distance: within about 40%


@dataclass
class LineUp:
    candidate: Candidate
    fit: float  # overlap with the photo after the correction, 0..1
    score: float  # fit, discounted by how unlikely that correction is
    shift: float  # degrees the compass was off
    move: tuple[float, float]  # metres east, north the GPS was off
    span: tuple[float, float]  # where the map then puts the building across the photo

    @property
    def moved(self) -> float:
        return math.hypot(*self.move)


def map_span(poly: Polygon, at: tuple[float, float], heading: float) -> tuple[float, float] | None:
    """Left and right edges of a footprint seen from `at`, degrees from the heading. None if it wraps around."""
    offsets = [(math.degrees(math.atan2(x - at[0], y - at[1])) - heading + 180) % 360 - 180
               for x, y in poly.exterior.coords[:-1]]
    lo, hi = min(offsets), max(offsets)
    return (lo, hi) if hi - lo < 180 else None


def line_up(
    candidates: list[Candidate],
    span: tuple[float, float],
    heading: float,
    fov: float,
    margin: float,
    move: float = 10.0,
    distance: float | None = None,
    step: float = 1.0,
) -> list[LineUp]:
    """Each candidate's best line-up with the photo span, correcting the GPS up to `move` metres and the compass up
    to `margin` degrees. Bigger corrections count against it, so a wrong building can't win by moving the camera
    a long way. Best first.

    Two edges can't tell "nearer" from "turned"; `distance` (Muse's estimate, metres) settles it when given.
    """
    half = fov / 2
    shifts = [-margin + i * step for i in range(int(2 * margin / step) + 1)]
    n = int(move / step)
    results = []
    for c in candidates:
        best = LineUp(c, 0.0, 0.0, 0.0, (0.0, 0.0), (c.left, c.right))
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                at = (i * step, j * step)
                if c.polygon.contains(Point(at)):
                    continue
                seen = map_span(c.polygon, at, heading)
                if seen is None:
                    continue
                cost = (at[0] ** 2 + at[1] ** 2) / GPS_ERROR**2
                if distance:
                    wall = max(c.polygon.exterior.distance(Point(at)), 1.0)
                    cost += (math.log(wall / max(distance, 1.0)) / DISTANCE_ERROR) ** 2
                for sh in shifts:
                    lo, hi = max(seen[0] - sh, -half), min(seen[1] - sh, half)  # the photo's edges crop it
                    if hi <= lo:
                        continue
                    fit = overlap(span, (lo, hi))
                    score = fit * math.exp(-0.5 * (cost + (sh / COMPASS_ERROR) ** 2))
                    if score > best.score:
                        best = LineUp(c, fit, score, sh, at, (lo, hi))
        results.append(best)
    results.sort(key=lambda r: -r.score)
    return results


PROMPT = """You are checking which mapped building a street photo shows. The photographer stood at the map's \
origin; the main building in the photo has been outlined by a segmentation model.

Candidates from OpenStreetMap (distances are to the nearest visible wall):
{candidates}

Look at the main building in the photo: how far away it seems, how wide its visible front is, how many floors it \
has, and any names, numbers or signs. Compare with each candidate. Say "none" if no candidate fits.

Answer in JSON only:
{{"choice": candidate number or null, "confidence": 0.0 to 1.0, "reasons": [short reasons, for and against], \
"floors": floors you see (int), "distance_m": your estimate of the distance to it, "signs": [text you can read]}}"""


def describe(c: Candidate) -> str:
    t = c.building.tags
    rect = c.polygon.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    sides = sorted(math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(2))
    facts = [
        f"{c.distance:.0f} m away",
        f"footprint about {sides[1]:.0f} x {sides[0]:.0f} m",
        f"fills {c.share:.0%} of the view",
        f"visible from {c.left:+.0f} to {c.right:+.0f} degrees off centre",
    ]
    for key, label in (("name", "name"), ("building", "type"), ("building:levels", "floors"),
                       ("addr:housenumber", "house number"), ("addr:street", "street")):
        if t.get(key) and t[key] != "yes":
            facts.append(f"{label} {t[key]}")
    return f"{c.rank}. " + ", ".join(facts)


def build_prompt(candidates: list[Candidate]) -> str:
    return PROMPT.format(candidates="\n".join(describe(c) for c in candidates))


@dataclass
class Verdict:
    choice: int | None
    confidence: float
    reasons: list[str] = field(default_factory=list)
    floors: int | None = None
    distance_m: float | None = None
    signs: list[str] = field(default_factory=list)


def parse_verdict(text: str) -> Verdict:
    """The model's JSON, tolerating code fences and chatter around it."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return Verdict(None, 0.0, [f"unreadable answer: {text[:200]!r}"])
    try:
        d = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(None, 0.0, [f"unreadable answer: {text[:200]!r}"])
    choice = d.get("choice")
    return Verdict(
        choice=int(choice) if isinstance(choice, (int, float)) or str(choice).isdigit() else None,
        confidence=float(d.get("confidence") or 0),
        reasons=[str(r) for r in d.get("reasons") or []],
        floors=d.get("floors"),
        distance_m=d.get("distance_m"),
        signs=[str(s) for s in d.get("signs") or []],
    )


@dataclass
class Decision:
    status: str  # "picked" or "ask"
    rank: int | None  # the picked candidate, or the best guess when asking
    why: str


def geometry_sure(lineups: list[LineUp]) -> bool:
    if not lineups or lineups[0].fit < GEO_FIT or lineups[0].score < 0.05:
        return False
    runner_up = lineups[1].score if len(lineups) > 1 else 0.0
    return lineups[0].score >= GEO_LEAD * runner_up


def decide(lineups: list[LineUp], verdict: Verdict | None) -> Decision:
    geo = lineups[0] if lineups and lineups[0].score > 0 else None
    if verdict is None or verdict.choice is None:
        if geo and geometry_sure(lineups):
            return Decision("ask", geo.candidate.rank, "only the geometry has an answer")
        return Decision("ask", None, "neither check found a match")
    if geo is None:
        return Decision("ask", verdict.choice, "the building outline matches no candidate's position")
    if verdict.choice != geo.candidate.rank:
        return Decision("ask", verdict.choice, f"Muse says {verdict.choice}, geometry says {geo.candidate.rank}")
    if verdict.confidence < SURE or not geometry_sure(lineups):
        return Decision("ask", verdict.choice,
                        f"both say {verdict.choice}, but not sure enough "
                        f"(Muse {verdict.confidence:.0%}, outline fit {geo.fit:.0%})")
    return Decision("picked", verdict.choice,
                    f"Muse ({verdict.confidence:.0%}) and geometry (outline fit {geo.fit:.0%}) agree")
