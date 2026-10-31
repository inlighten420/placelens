"""Which side of the building a photo shows, asked of Muse by showing it next to a photo of the front.

GPS and compass close to a house are often too far off to tell the back from the side. Muse can: it matches roof
shapes, windows, doors and the garden between two photos. Its answer is then checked against the photo's compass,
which, however rough, rarely points the opposite way: an answer the compass contradicts isn't used.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

SIDES = {"front": 0, "left": 90, "back": 180, "right": -90}  # degrees from the front's outward bearing
CORNERS = {f"{a}-{b}" for a in ("front", "back") for b in ("left", "right")}
SURE_SIDE = 0.6  # Muse's confidence needed to use its answer
COMPASS_SLACK = 75  # degrees a photo may look away from straight at that side before the compass contradicts Muse

SIDE_PROMPT = """Both photos show buildings. Photo 1 shows the front of a house (the side facing the street). \
Is photo 2 of the same house? If so, which side of it does photo 2 mainly show?

- "front", "back", "left" or "right": left and right as seen by someone standing in the street, facing the front.
- A corner, if photo 2 shows two sides about equally: "front-left", "front-right", "back-left" or "back-right".
- "other" if photo 2 isn't this house.

Match roof shapes, gables, windows, doors, wall colours, the garden and the street between the photos.

Answer in JSON only: {"side": "...", "confidence": 0.0 to 1.0, "reasons": [short reasons]}"""


@dataclass
class SideView:
    side: str | None  # a side, a corner, "other", or None if unreadable
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)


def parse_side(text: str) -> SideView:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        d = {}
    side = str(d.get("side", "")).lower().strip().replace(" ", "-")
    if side not in SIDES and side not in CORNERS and side not in ("other", "above"):
        side = None
    confidence = d.get("confidence")
    return SideView(side, float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                    [str(r) for r in d.get("reasons") or []])


def side_offsets(side: str) -> list[float]:
    """Degrees from the front's outward bearing of the side, or of both sides of a corner."""
    return [SIDES[part] for part in side.split("-")] if side in CORNERS else [SIDES[side]]


def facing(side: str, front: float) -> float:
    """The compass bearing a side faces (a corner: halfway between its two sides)."""
    offsets = side_offsets(side)
    x = sum(math.sin(math.radians(front + o)) for o in offsets)
    y = sum(math.cos(math.radians(front + o)) for o in offsets)
    return math.degrees(math.atan2(x, y)) % 360


def compass_agrees(side: str, heading: float, front: float) -> bool:
    """Whether a photo taken looking along `heading` could show `side` of a building whose front faces `front`.
    Looking straight at a side means looking against the way it faces."""
    look = facing(side, front) + 180
    return abs((heading - look + 180) % 360 - 180) <= COMPASS_SLACK
