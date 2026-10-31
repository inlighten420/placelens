"""What a building is made of, to the eye: the colours of its walls, roof, window frames and doors, read by Muse from
the photos, so the model can be built clean (no trees, cars or neighbours stretched over its walls)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

LOOK_PROMPT = """Look at the main building in these photos (the house itself, not its neighbours, plants or cars).

Answer in JSON only, each colour as "#rrggbb" the way it looks in soft daylight:
{"walls": the main wall colour (paint, plaster or brick), \
"roof": the roof's colour, \
"frames": the window frames' colour, \
"door": the front door's colour, \
"garage_door": the garage door's colour, or null if there is none}"""


def _linear(hex_colour: str) -> tuple[float, float, float, float]:
    """A "#rrggbb" sRGB colour as glTF wants it: linear, with alpha."""
    r, g, b = (int(hex_colour[k:k + 2], 16) / 255 for k in (1, 3, 5))
    return (*(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)), 1.0)


@dataclass(frozen=True)
class Look:
    """Colours (linear RGBA) for a built building. The defaults are a plain light house."""

    walls: tuple = _linear("#e6e1d8")
    roof: tuple = _linear("#5f5853")
    frames: tuple = _linear("#f2f2ee")
    door: tuple = _linear("#6b4a32")
    garage_door: tuple = _linear("#e9e8e4")


def parse_look(text: str) -> Look:
    """Muse's JSON, tolerating code fences and chatter; a colour it didn't give (or gave wrongly) keeps the default."""
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        d = {}
    found = {k: _linear(v) for k, v in d.items()
             if k in Look.__dataclass_fields__ and isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v)}
    return Look(**found)
