"""An object built from parts: its own 3D version, not the photo pasted on.

Muse looks at the photo, with what the research found (what it is, its real size, how it sits), and describes the
object as simple shapes at real size: a carbon-black plate, two rubber grips, twelve round white buttons. PlaceLens
builds those shapes, coloured, as the model. Then the check: the model's outline, seen as the photo saw it, against
SAM's cut-out; if they don't match well, Muse sees both and corrects its parts once, and the better is kept.

For hard, made things (a wheel, a monitor, a bench). Soft ones (a teddy bear) are carved, with the photos' colours.

The parts use Open-Planet's own prop format (part_model.gd): shape, at, size, turn (degrees), colour, finish.
Axes: metres, Y up, standing on Y = 0, X right; its front (the side the photo saw) +Z, or up (+Y) if the photo
looked down on it lying flat.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .build import Part

SHAPES = ("box", "cylinder", "sphere", "cone", "wedge", "tube")
FINISHES = {"matte": (0.9, 0.0), "plastic": (0.5, 0.0), "rubber": (0.95, 0.0), "metal": (0.35, 0.8),
            "wood": (0.7, 0.0), "glass": (0.1, 0.0)}  # roughness, metallic
MAX_PARTS = 60
GOOD_MATCH = 0.8  # outline overlap (intersection over union) good enough not to ask for corrections
SEGMENTS = 24  # round shapes' sides
RASTER = 256  # px: the outline check's picture

PARTS_PROMPT = """You are building a simple 3D model of the object in this photo from basic shapes, so that it looks \
like the real thing: its own 3D version, not the photo. What's known about it: {research}

Its real size as it sits in the photo: {wide:.2f} m across, {deep:.2f} m front to back, {tall:.2f} m tall.
{pose}

Describe it as shapes, biggest first then details (grips, buttons, dials, legs, a screen, a lid), at real size. \
Coordinates in metres: x to the right across the photo, y up from what it stands or lies on (its bottom at y = 0), \
z towards the camera; the object centred on x = 0, z = 0. Each shape is 1 m across scaled by "size" [x, y, z], \
turned by "turn" [x, y, z] degrees, then moved so its centre is at "at". Shapes: "box", "cylinder" (round in x \
and z, along y), "sphere", "cone" (point up), "wedge" (a ramp: high at -z, low at +z), "tube" (a hollow cylinder). \
Colours as the real thing's, "#rrggbb". Finish: "matte", "plastic", "rubber", "metal", "wood" or "glass". At most \
{max_parts} shapes. Leave openings open: don't fill the gap between a wheel's grips, or under a table. Name each \
shape in "what" with the everyday word anyone would use to point at that part in a photo ("backrest", "seat", \
"armrest", "handle", "grip", "button", "leg", "wheel"), not what it's made of; add left, right, front or back when \
there are several.

Answer in JSON only: {{"parts": [{{"shape": "box", "at": [x, y, z], "size": [x, y, z], "turn": [0, 0, 0], \
"colour": "#222222", "finish": "plastic", "what": "left handle"}}, ...]}}"""

FIX_PROMPT = """These are the shapes you described for the object in photo 1:
{parts}

Photo 2 is the photo again with your model's outline drawn over it in red, seen the way the photo sees it. They \
overlap {score:.0%}. Correct the shapes so the outline matches the object: move, resize, add or remove shapes. Same \
rules and coordinates as before. Answer in JSON only: {{"parts": [...]}}"""

FLAT = "The photo looks down on it lying flat: its top, the face you see, is +y; the top of the photo is -z."
UPRIGHT = "The photo looks at it from the side: the face you see is +z."


def parse_parts(text: str) -> list[dict]:
    """Muse's shapes, checked: known shapes, numbers where numbers go, sizes above nothing; at most MAX_PARTS."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(m.group(0)).get("parts", []) if m else []
    except (json.JSONDecodeError, AttributeError):
        return []
    parts = []
    for p in raw if isinstance(raw, list) else []:
        try:
            shape = str(p.get("shape", "box")).lower()
            at = [float(v) for v in p.get("at", [0, 0, 0])][:3]
            size = [abs(float(v)) for v in p.get("size", [0.1, 0.1, 0.1])][:3]
            turn = [float(v) for v in p.get("turn", [0, 0, 0])][:3]
        except (TypeError, ValueError, AttributeError):
            continue
        if shape not in SHAPES or len(at) < 3 or len(size) < 3 or min(size) <= 0:
            continue
        colour = str(p.get("colour") or p.get("color") or "#808080")
        finish = str(p.get("finish", "matte")).lower()
        parts.append({"shape": shape, "at": at, "size": size, "turn": (turn + [0, 0, 0])[:3],
                      "colour": colour if re.fullmatch(r"#[0-9a-fA-F]{6}", colour) else "#808080",
                      "finish": finish if finish in FINISHES else "matte", "what": str(p.get("what", ""))})
    return parts[:MAX_PARTS]


def _rotation(turn_deg) -> np.ndarray:
    """Godot's Basis.from_euler (YXZ order, as part_model.gd turns its shapes)."""
    x, y, z = (math.radians(v) for v in turn_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return ry @ rx @ rz


def _unit(shape: str) -> tuple[np.ndarray, np.ndarray]:
    """A shape 1 m across, centred on the origin: vertices and triangles (counter-clockwise from outside)."""
    if shape in ("cylinder", "tube", "cone"):
        a = np.linspace(0, 2 * math.pi, SEGMENTS, endpoint=False)
        top_r = 0.0 if shape == "cone" else 0.5
        ring = np.stack([np.sin(a), np.zeros_like(a), np.cos(a)], axis=1) * 0.5
        top = ring * (top_r / 0.5) + [0, 0.5, 0]
        bottom = ring + [0, -0.5, 0]
        verts = np.vstack([top, bottom, [[0, 0.5, 0], [0, -0.5, 0]]])
        n, tris = SEGMENTS, []
        for i in range(n):
            j = (i + 1) % n
            tris += [[i, n + i, j], [j, n + i, n + j], [2 * n, i, j], [2 * n + 1, n + j, n + i]]
        return verts, np.array(tris)
    if shape == "sphere":
        rows, cols = SEGMENTS // 2, SEGMENTS
        verts = []
        for r in range(rows + 1):
            t = math.pi * r / rows
            for c in range(cols):
                p = 2 * math.pi * c / cols
                verts.append([0.5 * math.sin(t) * math.sin(p), 0.5 * math.cos(t), 0.5 * math.sin(t) * math.cos(p)])
        tris = []
        for r in range(rows):
            for c in range(cols):
                a, b = r * cols + c, r * cols + (c + 1) % cols
                tris += [[a, a + cols, b], [b, a + cols, b + cols]]
        return np.array(verts), np.array(tris)
    if shape == "wedge":  # high at -z, low at +z
        v = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, -.5, .5], [-.5, -.5, .5], [-.5, .5, -.5], [.5, .5, -.5]])
        t = [[0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4], [3, 4, 5], [3, 5, 2], [0, 4, 3], [1, 2, 5]]
        return v, np.array(t)
    v = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])  # box
    t = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
         [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return v, np.array(t)


def _placed(p: dict) -> tuple[np.ndarray, np.ndarray]:
    verts, tris = _unit(p["shape"])
    world = (verts * np.array(p["size"])) @ _rotation(p["turn"]).T + np.array(p["at"])
    return world, tris


def _hex(colour: str) -> tuple[float, float, float, float]:
    return (int(colour[1:3], 16) / 255, int(colour[3:5], 16) / 255, int(colour[5:7], 16) / 255, 1.0)


def model_parts(parts: list[dict]) -> list[Part]:
    """The shapes as model parts: one per colour and finish (fewer materials), each triangle its own corners so the
    edges stay sharp."""
    groups: dict[tuple[str, str], list] = {}
    for p in parts:
        groups.setdefault((p["colour"], p["finish"]), []).append(_placed(p))
    out = []
    for (colour, finish), pieces in groups.items():
        corners = np.concatenate([v[t].reshape(-1, 3) for v, t in pieces]).astype(np.float32)
        tri = corners.reshape(-1, 3, 3)
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        normals = np.repeat(normal, 3, axis=0).astype(np.float32)
        rough, metal = FINISHES[finish]
        out.append(Part(f"{finish} {colour}", corners, normals, np.zeros((len(corners), 2), np.float32),
                        np.arange(len(corners), dtype=np.uint32), _hex(colour), roughness=rough, metallic=metal))
    return out


def outline(parts: list[dict], flat: bool, size: int = RASTER) -> np.ndarray:
    """The model as the photo sees it, flattened: from above (flat) or the front, fitted to a square picture."""
    shots = []
    for p in parts:
        v, t = _placed(p)
        shots.append(v[:, [0, 2]] if flat else np.stack([v[:, 0], -v[:, 1]], axis=1))  # (x right, down the picture)
        shots[-1] = (shots[-1], t)
    every = np.concatenate([s for s, _ in shots])
    lo, hi = every.min(axis=0), every.max(axis=0)
    scale = (size - 1) / max(hi - lo)
    img = Image.new("1", (size, size))
    draw = ImageDraw.Draw(img)
    for pts, tris in shots:
        px = (pts - lo) * scale
        for a, b, c in tris:
            draw.polygon([tuple(px[a]), tuple(px[b]), tuple(px[c])], fill=1)
    return np.asarray(img)


def fitted(mask: np.ndarray, size: int = RASTER) -> np.ndarray:
    """A cut-out cropped to itself and fitted to the same square picture as `outline`."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    box = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    scale = (size - 1) / max(box.shape)
    w, h = max(1, round(box.shape[1] * scale)), max(1, round(box.shape[0] * scale))
    small = Image.fromarray(box.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    canvas = Image.new("L", (size, size))
    canvas.paste(small, (0, 0))
    return np.asarray(canvas) > 127


def match(parts: list[dict], mask: np.ndarray, flat: bool) -> float:
    """How well the model's outline matches the cut-out's (intersection over union, 0..1), shapes compared, not
    where they sit in the photo."""
    if not parts:
        return 0.0
    a, b = outline(parts, flat), fitted(mask)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def overlay(photo: Path, mask: np.ndarray, parts: list[dict], flat: bool, out: Path) -> Path:
    """The photo, cropped to the object, with the model's outline drawn over it in red (for Muse to correct)."""
    from scipy import ndimage

    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    with Image.open(photo) as img:
        crop = img.convert("RGB").crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    shape = outline(parts, flat, max(crop.size))[:crop.size[1], :crop.size[0]]
    edge = shape & ~ndimage.binary_erosion(shape, iterations=2)
    pixels = np.asarray(crop).copy()
    pixels[edge] = (255, 30, 30)
    Image.fromarray(pixels).save(out, quality=90)
    return out


def describe_parts(meta, photo: Path, research: str, size: tuple[float, float, float], flat: bool, out: Path,
                   step=lambda *_: None) -> tuple[list[dict], str]:
    """Muse's shapes for the object at its real size (one question, saved), and the question's tag."""
    from .meta import OPEN
    from .muse import ask

    wide, deep, tall = size
    prompt = PARTS_PROMPT.format(research=research, wide=wide, deep=deep, tall=tall, pose=FLAT if flat else UPRIGHT,
                                 max_parts=MAX_PARTS)
    tag = __import__("hashlib").sha1(prompt.encode()).hexdigest()[:8]
    step("Muse is designing it from parts, at its real size")
    shapes = []
    for a in ask(meta, prompt, photo, out / f"parts_{tag}.txt", OPEN, votes=1, effort="high"):
        shapes = parse_parts(a)
    return shapes, tag


def build_parts(meta, photo: Path, mask: np.ndarray, research: str, size: tuple[float, float, float], flat: bool,
                out: Path, step=lambda *_: None) -> tuple[list[Part], list[dict], float, list[str]]:
    """The object from parts: Muse's shapes, checked against the cut-out, corrected once if they don't match well.
    Returns the model parts, the shapes, how well they match (0..1) and notes."""
    from .meta import OPEN
    from .muse import ask

    shapes, tag = describe_parts(meta, photo, research, size, flat, out, step)
    if not shapes:
        return [], [], 0.0, ["Muse didn't describe it as parts"]
    score = match(shapes, mask, flat)
    notes = [f"built from {len(shapes)} parts; outline matches the photo {score:.0%}"]
    if score < GOOD_MATCH:
        step(f"checking it against the photo ({score:.0%} match): Muse corrects the parts")
        drawn = overlay(photo, mask, shapes, flat, out / f"parts_{tag}_check.jpg")
        fix = FIX_PROMPT.format(parts=json.dumps(shapes), score=score)
        fixed = []
        for a in ask(meta, fix, [photo, drawn], out / f"parts_{tag}_fix.txt", OPEN, votes=1, effort="high"):
            fixed = parse_parts(a)
        better = match(fixed, mask, flat) if fixed else 0.0
        if better > score:
            shapes, score = fixed, better
            notes.append(f"corrected after checking: now {score:.0%}")
        else:
            notes.append(f"a correction matched worse ({better:.0%}); kept the first")
    (out / "parts.json").write_text(json.dumps(shapes, indent=1), encoding="utf-8")
    return model_parts(shapes), shapes, score, notes
