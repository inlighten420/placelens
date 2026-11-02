"""The model sees what it built: Muse's parts drawn from each photo's own camera, shown to Muse over the photos, and
corrected in a few moves.

Each move is one Muse question with one picture: every photo's view side by side, the model's outline over the photo.
Muse answers with small edits by part number (move, resize, remove), not the whole model again. Each edit is then tried
and scored without Muse: how well the model's outline covers the cut-out in every photo together (intersection over
union, averaged), so fixing one view can't quietly break another. An edit is kept only if that score goes up.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MOVES = 3  # Muse questions at most
SCORE_PX = 192  # rows: each photo's cut-out is compared at about this height (fast, and plenty for an outline)
TILE_PX = 384  # rows: each view in the picture Muse sees
MAX_VIEWS = 4  # photos in that picture
MAX_EDITS = 8  # edits read from one answer
TRIES = (1.0, 0.5, 1.5)  # each edit is tried at this share of what Muse said
MIN_GAIN = 0.005  # an edit must raise the score this much to be kept
MAX_MOVE = 0.25  # share of the object's biggest size: no edit moves a part further than this
SCALE_RANGE = (0.5, 2.0)  # nor resizes it more than this
MAX_TURN = 30.0  # nor turns it more than this many degrees

PROMPT = """We are building a 3D model of {name} from simple shapes, and checking it against the photos. The picture \
shows {views} photo(s) of it side by side, labelled. Over each photo, the model as that photo's camera sees it: \
red is where the model is but the object isn't, blue is where the object is but the model isn't, purple where they \
agree. Now {score:.0%} agree.

The shapes (metres; x is to the right as seen in view 1, y is up from the floor, z is towards view 1's camera; "at" is \
the centre, "size" the full size):
{parts}

Say what to change so the model matches the object in every view at once: up to {max_edits} edits, the most \
important first. Fix whole parts (a leg too short, an armrest too low), not single pixels. Answer in JSON only:
{{"edits": [{{"part": 3, "move": [0, 0.05, 0]}}, {{"part": 5, "size": [1, 1.4, 1]}}, \
{{"part": 2, "turn": [-10, 0, 0]}}, {{"part": 7, "remove": true}}]}}
"move" is metres to add to "at"; "size" multiplies its size [x, y, z]; "turn" is degrees to add to its turn [x, y, z] \
(+x tips its top towards view 1's camera, -x away from it; +y turns it to the left seen from above)."""


def _parts_text(shapes: list[dict]) -> str:
    """The parts as short numbered lines: far fewer tokens than their JSON."""
    def r(v):
        return "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
    return "\n".join(f"{i}. {p.get('what') or p['shape']}: {p['shape']} at {r(p['at'])} size {r(p['size'])}"
                     + (f" turn {r(p['turn'])}" if any(abs(t) > 0.5 for t in p["turn"]) else "")
                     for i, p in enumerate(shapes))


def silhouette(shapes: list[dict], shot, mid: float, step: int) -> np.ndarray:
    """Where the model shows in `shot`'s photo, at 1/`step` of its size."""
    from .parts import _placed

    h, w = shot.mask.shape
    img = Image.new("1", ((w + step - 1) // step, (h + step - 1) // step))
    draw = ImageDraw.Draw(img)
    for p in shapes:
        verts, tris = _placed(p)
        cols, rows, depth = shot.project(verts, mid)
        if (depth <= 0).any():  # (behind the camera: not from a photo taken outside the object)
            continue
        px = np.stack([cols, rows], axis=1) / step
        for a, b, c in tris:
            draw.polygon([tuple(px[a]), tuple(px[b]), tuple(px[c])], fill=1)
    return np.asarray(img)


def _step(shot) -> int:
    return max(1, shot.mask.shape[0] // SCORE_PX)


def scores(shapes: list[dict], shots: list, mid: float) -> list[float]:
    """Each photo's agreement: the model's outline against the cut-out (intersection over union)."""
    out = []
    for s in shots:
        k = _step(s)
        real = s.mask[::k, ::k]
        model = silhouette(shapes, s, mid, k)[:real.shape[0], :real.shape[1]]
        union = (real | model).sum()
        out.append(float((real & model).sum() / union) if union else 0.0)
    return out


SLACK = 0.015  # share of a photo's height the outline is widened by before `seen` (cameras a little off)


def seen(part: dict, shots: list, mid: float) -> float:
    """How much of `part` falls inside the object's outline, in the photo that shows it worst (0..1). A part hidden
    behind another still falls inside the outline, so hidden is fine; a part in the wrong place falls outside it in
    some photo. Drawings (Muse Image) don't count: only photos."""
    from scipy import ndimage

    worst = None
    for s in shots:
        if s.imagined or s.overhead:
            continue
        k = _step(s)
        real = s.mask[::k, ::k]
        key = (id(s.mask), k)
        if getattr(s, "_slack_key", None) != key:  # (worked out once per photo: every part asks)
            s._slack = ndimage.binary_dilation(real, iterations=max(1, round(SLACK * real.shape[0])))
            s._slack_key = key
        model = silhouette([part], s, mid, k)[:real.shape[0], :real.shape[1]]
        if model.sum():
            share = float((model & s._slack).sum() / model.sum())
            worst = share if worst is None else min(worst, share)
    return worst or 0.0


def score(shapes: list[dict], shots: list, mid: float) -> float:
    return float(np.mean(scores(shapes, shots, mid))) if shots and shapes else 0.0


def picture(shapes: list[dict], shots: list, mid: float, out: Path, labels: list[str]) -> Path:
    """The views side by side, each the photo cropped round the object with the model's outline over it."""
    tiles = []
    for s, label in zip(shots, labels, strict=True):
        k = max(1, s.mask.shape[0] // (TILE_PX * 2))
        real = s.mask[::k, ::k]
        model = silhouette(shapes, s, mid, k)[:real.shape[0], :real.shape[1]]
        both = real | model
        rows, cols = np.flatnonzero(both.any(axis=1)), np.flatnonzero(both.any(axis=0))
        pad = max(4, (rows[-1] - rows[0]) // 12)
        top, bottom = max(0, rows[0] - pad), min(real.shape[0], rows[-1] + pad + 1)
        left, right = max(0, cols[0] - pad), min(real.shape[1], cols[-1] + pad + 1)
        photo = (s.image[::k, ::k][top:bottom, left:right] * 0.55).astype(np.float32)
        r, m = real[top:bottom, left:right], model[top:bottom, left:right]
        tint = np.zeros_like(photo)
        tint[m & ~r] = (230, 40, 40)
        tint[r & ~m] = (40, 110, 240)
        tint[r & m] = (150, 70, 200)
        mixed = np.where((r | m)[..., None], photo * 0.5 + tint * 0.5, photo).clip(0, 255).astype(np.uint8)
        tile = Image.fromarray(mixed)
        tile = tile.resize((max(1, round(tile.width * TILE_PX / tile.height)), TILE_PX))
        ImageDraw.Draw(tile).text((6, 4), label, fill=(255, 255, 255))
        tiles.append(tile)
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), TILE_PX), (0, 0, 0))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 6
    sheet.save(out, quality=85)
    return out


def parse_edits(text: str, count: int) -> list[dict]:
    """Muse's edits, checked: a real part number, numbers where numbers go."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(m.group(0)).get("edits", []) if m else []
    except (json.JSONDecodeError, AttributeError):
        return []
    edits = []
    for e in raw if isinstance(raw, list) else []:
        try:
            i = int(e.get("part"))
            if not 0 <= i < count:
                continue
            if e.get("remove") is True:
                edits.append({"part": i, "remove": True})
                continue
            move = [float(v) for v in e.get("move", [0, 0, 0])][:3]
            size = [float(v) for v in e.get("size", [1, 1, 1])][:3]
            turn = [float(v) for v in e.get("turn", [0, 0, 0])][:3]
        except (TypeError, ValueError, AttributeError):
            continue
        if len(move) == 3 and len(size) == 3 and len(turn) == 3 and min(size) > 0:
            edits.append({"part": i, "move": move, "size": size, "turn": turn})
    return edits[:MAX_EDITS]


def apply(shapes: list[dict], edit: dict, share: float, reach: float) -> list[dict]:
    """`shapes` with `edit` done at `share` of its strength (a removed part's place stays, as None, so the numbers
    Muse used still point at the same parts)."""
    new = copy.deepcopy(shapes)
    i = edit["part"]
    if new[i] is None:
        return new
    if edit.get("remove"):
        new[i] = None
        return new
    move = np.clip(np.array(edit["move"]) * share, -reach, reach)
    lo, hi = SCALE_RANGE
    size = np.clip(1 + (np.array(edit["size"]) - 1) * share, lo, hi)
    p = new[i]
    p["at"] = [float(v) for v in np.array(p["at"]) + move]
    p["size"] = [float(v) for v in np.array(p["size"]) * size]
    p["turn"] = [float(v) for v in np.array(p["turn"]) + np.clip(np.array(edit.get("turn", [0, 0, 0])) * share,
                                                                  -MAX_TURN, MAX_TURN)]
    p["at"][1] = max(p["at"][1], p["size"][1] / 2 * (abs(p["turn"][0]) < 1 and abs(p["turn"][2]) < 1))  # not into
    # the floor (upright parts; a turned part's bottom isn't its half-height)
    return new


def _candidates(edits: list[dict]) -> list[dict]:
    """What's tried, in turn: each edit as Muse said it; a turn also the other way (which way is +x is easy to get
    backwards, and the score decides); then each part's edits together (a tilt usually needs a move as well)."""
    out = []
    for e in edits:
        out.append(e)
        if any(e.get("turn", [0, 0, 0])):
            out.append({**e, "turn": [-v for v in e["turn"]]})
    by_part: dict[int, list[dict]] = {}
    for e in edits:
        if not e.get("remove"):
            by_part.setdefault(e["part"], []).append(e)
    for i, es in by_part.items():
        if len(es) > 1:
            out.append({"part": i, "move": list(np.sum([e["move"] for e in es], axis=0)),
                        "size": list(np.prod([e["size"] for e in es], axis=0)),
                        "turn": list(np.sum([e["turn"] for e in es], axis=0))})
    return out


def _live(shapes: list) -> list[dict]:
    return [p for p in shapes if p is not None]


def _views(shots: list, most: int = MAX_VIEWS) -> list:
    """The photos Muse sees: the front, then those furthest round from the ones already chosen."""
    real = [s for s in shots if not s.imagined and not s.overhead]  # (an overhead photo's camera is only
    # roughly known: it still shapes the carving, but drawn over it the model lands off, 2 Nov)
    chosen = real[:1]
    while len(chosen) < min(most, len(real)):
        def apart(s):
            return min(abs((s.angle - c.angle + 180) % 360 - 180) for c in chosen)
        chosen.append(max((s for s in real if all(s is not c for c in chosen)), key=apart))  # (by identity: a
        # Shot's == compares its photo arrays)
    return chosen


def refine(meta, shapes: list[dict], shots: list, mid: float, name: str, out: Path, moves: int = MOVES,
           step=lambda *_: None) -> tuple[list[dict], list[str]]:
    """`shapes` corrected in up to `moves` Muse questions, every edit kept only if all the photos agree better."""
    from .meta import OPEN
    from .muse import ask

    views = _views(shots)
    if not views or not shapes:
        return shapes, []
    labels = ["view 1 (front)"] + [f"view {i + 2} ({(s.angle + 180) % 360 - 180:+.0f}° round)"
                                   for i, s in enumerate(views[1:])]
    every = np.concatenate([np.abs(np.array(p["at"])) + np.array(p["size"]) / 2 for p in shapes])
    reach = MAX_MOVE * 2 * float(every.max())
    current = list(shapes)
    best = score(current, views, mid)
    notes = [f"the model over the photos: {best:.0%} agree"]
    for move in range(1, moves + 1):
        step(f"Muse looks at what it built (move {move} of {moves}: {best:.0%} agree)")
        live = _live(current)
        index = [i for i, p in enumerate(current) if p is not None]  # (Muse sees the live parts numbered 0..n-1)
        sheet = picture(live, views, mid, out / f"sight_{move}.jpg", labels)
        prompt = PROMPT.format(name=name, views=len(views), score=best, parts=_parts_text(live), max_edits=MAX_EDITS)
        tag = __import__("hashlib").sha1(prompt.encode()).hexdigest()[:8]
        answers = ask(meta, prompt, sheet, out / f"sight_{move}_{tag}.txt", OPEN, votes=1, effort="low")
        edits = parse_edits(answers[0] if answers else "", len(live))
        for e in edits:
            e["part"] = index[e["part"]]
        kept = 0
        for e in _candidates(edits):
            for share in TRIES:
                trial = apply(current, e, share, reach)
                s = score(_live(trial), views, mid)
                if s > best + MIN_GAIN:
                    current, best, kept = trial, s, kept + 1
                    break
        notes.append(f"move {move}: Muse proposed {len(edits)} edit(s), {kept} kept: {best:.0%} agree")
        if not kept:
            break
    picture(_live(current), views, mid, out / "sight_final.jpg", labels)
    return _live(current), notes
