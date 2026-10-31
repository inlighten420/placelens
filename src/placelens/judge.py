"""Muse judges a carved prop: the model seen from where a photo was taken, next to that photo. What's wrong?

Carving from outlines only knows what the photos rule out. A pocket no photo sees into (between a teddy's legs, under
its arms) stays filled, and Muse, which knows what a teddy looks like, can see that at a glance. So each of a few
photos (spread round the object) is shown to Muse with the model drawn from that photo's camera beside it. Muse
answers with the parts the model has that the real one doesn't ("a flap hanging between the feet"), boxed, and a
score out of 10. Several answers are asked for; a box counts only if at least two of them agree on it.

Muse points; the photos decide what they can. Inside an agreed box, what that photo shows as background is cut away
(the forgiveness the carving gives near an outline's edge is taken back there). Some pockets no photo sees into at all
(a teddy's crotch, from cameras looking down at it): there, if Muse is sure (SURE answers of 5), what's in the box that
no other photo saw as surface goes too: nothing backs it but the carving's leftovers. Then Muse looks again, and the
trim is kept only if its scores don't drop and the model still matches the cut-outs as well.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

VIEWS = 4  # photos shown to Muse, spread round the object
MIN_VOTES = 2  # answers that must agree on a box
SAME_BOX = 0.25  # boxes overlapping at least this much (intersection over union) are the same problem
RENDER = 480  # pixels, the long side of the drawings shown to Muse
SURE = 4  # answers (of 5) agreeing on a box for Muse to be trusted with what no photo saw (see `trim`)
AGREEMENT_SLACK = 0.01  # the trimmed model may match the cut-outs this much worse, no more

JUDGE_PROMPT = """The picture is in two halves. Left: a photo of {name}. Right: our 3D model of it, drawn from the \
same spot (only its shape matters here; the background is left out on purpose).

What does the model's shape get wrong? Look for parts the model has that the real {name} doesn't (lumps, flaps, \
filled-in gaps, things hanging down) and parts it's missing.

Answer in JSON only:
{{"score": 0 to 10 for how well the model's shape matches the real one, \
"problems": [{{"kind": "extra" or "missing", "what": "a few words", \
"box": [left, top, right, bottom] as fractions 0 to 1 of the RIGHT half}}]}}"""


@dataclass
class Verdict:
    score: float | None
    problems: list[dict] = field(default_factory=list)  # {kind, what, box}


def parse_verdict(text: str) -> Verdict:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return Verdict(None)
    if not isinstance(data, dict):
        return Verdict(None)
    problems = []
    for p in data.get("problems") or []:
        try:
            box = [float(v) for v in p["box"]]
            if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                problems.append({"kind": str(p.get("kind", "extra")).lower(), "what": str(p.get("what", "")),
                                 "box": [min(max(v, 0.0), 1.0) for v in box]})
        except (KeyError, TypeError, ValueError):
            continue
    score = data.get("score")
    return Verdict(float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None, problems)


def _iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def agreed(verdicts: list[Verdict], kind: str = "extra") -> list[dict]:
    """The problems of one kind that at least MIN_VOTES answers point at (boxes overlapping SAME_BOX), each with the
    boxes' middle and every answer's words."""
    boxes = [(n, p) for n, v in enumerate(verdicts) for p in v.problems if p["kind"] == kind]
    out, used = [], set()
    for i, (_, p) in enumerate(boxes):
        if i in used:
            continue
        members = [i] + [j for j, (_, q) in enumerate(boxes)
                         if j > i and j not in used and _iou(p["box"], q["box"]) >= SAME_BOX]
        voters = {boxes[j][0] for j in members}
        if len(voters) >= MIN_VOTES:
            used |= set(members)
            out.append({"box": np.median([boxes[j][1]["box"] for j in members], axis=0).tolist(),
                        "what": [boxes[j][1]["what"] for j in members], "votes": len(voters)})
    return out


def spread(shots, k: int = VIEWS) -> list[int]:
    """k real (not overhead, not drawn) photos as evenly spread round the object as there are."""
    real = [i for i, s in enumerate(shots) if not s.imagined and not s.overhead]
    chosen = [real[0]] if real else []
    for target in np.linspace(0, 360, k, endpoint=False)[1:]:
        rest = [i for i in real if i not in chosen]
        if rest:
            chosen.append(min(rest, key=lambda i: abs((shots[i].angle - target + 180) % 360 - 180)))
    return chosen


def draw(verts: np.ndarray, faces: np.ndarray, colours: np.ndarray, shot, height: float,
         long_side: int = RENDER) -> np.ndarray:
    """The model as seen from `shot`'s camera: RGB, the photo's shape scaled to `long_side`, white where
    there's nothing (a z-buffer, lit from the camera)."""
    h, w = shot.mask.shape
    k = long_side / max(h, w)
    H, W = round(h * k), round(w * k)
    cols, rows, depth = shot.project(verts, height / 2)
    x, y = cols * k, rows * k
    tris = faces
    normal = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    position = shot.camera(height / 2)[0]
    towards = position - verts[tris].mean(axis=1)
    towards /= np.linalg.norm(towards, axis=1, keepdims=True)
    light = np.clip(0.35 + 0.65 * np.abs(np.einsum("ij,ij->i", normal, towards)), 0, 1)
    img = np.full((H, W, 3), 255.0)
    zbuf = np.full((H, W), np.inf)
    for t in np.argsort(-depth[tris].mean(axis=1)):
        a, b, c = tris[t]
        xs, ys, zs = x[[a, b, c]], y[[a, b, c]], depth[[a, b, c]]
        if (zs <= 0).any():
            continue
        x0, x1 = max(0, int(xs.min())), min(W - 1, int(xs.max()) + 1)
        y0, y1 = max(0, int(ys.min())), min(H - 1, int(ys.max()) + 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-9:
            continue
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        zz = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        near = inside & (zz < sub)
        sub[near] = zz[near]
        colour = colours[[a, b, c]].mean(axis=0) * 255 * light[t]
        img[y0:y1 + 1, x0:x1 + 1][near] = colour
    return img.astype(np.uint8)


def side_by_side(photo: np.ndarray, render: np.ndarray, path) -> None:
    from PIL import Image

    left = Image.fromarray(photo).resize((render.shape[1], render.shape[0]))
    both = Image.new("RGB", (render.shape[1] * 2 + 8, render.shape[0]), "white")
    both.paste(left, (0, 0))
    both.paste(Image.fromarray(render), (render.shape[1] + 8, 0))
    both.save(path, quality=90)


def seen_surface(kept: np.ndarray, corner: np.ndarray, voxel: float, shots, height: float,
                 shrink: int = 4) -> np.ndarray:
    """For each kept voxel (in np.argwhere order), the photos (bit k = shots[k]) that see it as the nearest thing
    there, inside their cut-out: real surface someone saw."""
    idx = np.argwhere(kept)
    centres = corner + (idx + 0.5) * voxel
    seen = np.zeros(len(idx), np.int64)
    for k, s in enumerate(shots):
        if s.imagined:
            continue
        cols, rows, depth = s.project(centres, height / 2)
        h, w = s.mask.shape
        c, r = (cols / shrink).astype(int), (rows / shrink).astype(int)
        ok = (depth > 0) & (c >= 0) & (c < w // shrink) & (r >= 0) & (r < h // shrink)
        ok[ok] &= s.mask[np.minimum(r[ok] * shrink, h - 1), np.minimum(c[ok] * shrink, w - 1)]
        cell = r * (w // shrink + 1) + c
        nearest = np.full(cell.max() + 1 if len(cell) else 1, np.inf)
        np.minimum.at(nearest, cell[ok], depth[ok])
        front = ok & (depth <= nearest[np.where(ok, cell, 0)] + 1.5 * voxel)
        seen[front] |= 1 << k
    return seen


def trim(kept: np.ndarray, corner: np.ndarray, voxel: float, shot, height: float, box: list[float],
         unseen: np.ndarray | None = None) -> np.ndarray:
    """`kept` with what's inside `box` (fractions of the photo) and outside the photo's cut-out cut away; and, given
    `unseen` (per kept voxel: no other photo saw it as surface), that too."""
    idx = np.argwhere(kept)
    centres = corner + (idx + 0.5) * voxel
    cols, rows, depth = shot.project(centres, height / 2)
    h, w = shot.mask.shape
    c, r = np.round(cols).astype(int), np.round(rows).astype(int)
    inbox = (cols >= box[0] * w) & (cols <= box[2] * w) & (rows >= box[1] * h) & (rows <= box[3] * h) & (depth > 0)
    framed = inbox & (c >= 0) & (c < w) & (r >= 0) & (r < h)
    out = np.zeros(len(idx), bool)
    out[framed] = ~shot.mask[r[framed], c[framed]]
    if unseen is not None:
        out |= inbox & unseen
    kept = kept.copy()
    kept[tuple(idx[out].T)] = False
    return kept


SHEET_VIEWS = {  # the model drawn from these, round and over it: name -> (angle round, degrees above the middle)
    "FRONT": (0.0, 0.0), "LEFT": (-90.0, 0.0), "BACK": (180.0, 0.0), "RIGHT": (90.0, 0.0),
    "LOW FRONT": (0.0, None), "TOP": (0.0, 80.0)}  # (None: from just above the floor)
TILE = 320  # pixels, each view on the sheet
ROUNDS = 3  # sheets Muse looks at, at most (it stops sooner when it agrees on nothing more)
CUT_VOTES = 3  # answers (of 5) that must agree on a problem before anything is cut for it
LOSS = 0.015  # share of a photo's cut-out a cut may leave uncovered
THIN = 0.06  # metres: a part of the model thinner than this, where Muse sees something extra, is cut

SHEET_PROMPT = """Left: a photo of {name}. Right: six views of our 3D model of it, labelled {labels} (LOW FRONT is \
from just above the floor). Only the model's shape matters; its colours come from the photos.

What does the model's shape get wrong, compared with the real {name}? Look for parts the model has that the real one \
doesn't (lumps, flaps, filled-in gaps, things hanging down) and parts it's missing. Name the view each one is \
clearest in.

Answer in JSON only:
{{"score": 0 to 10 for how well the model's shape matches the real one, \
"problems": [{{"kind": "extra" or "missing", "what": "a few words", "view": one of the labels, \
"box": [left, top, right, bottom] as fractions 0 to 1 of that view}}]}}"""


def virtual_shots(size: tuple[float, float, float]) -> dict:
    """Cameras for the sheet's views: far enough back to show all of it, the same focal length for every view."""
    from .carve import Shot

    width, depth, height = size
    reach = max(width, depth, height) * 0.45  # (about half the object across: it fills its tile)
    distance = reach * 4
    focal = TILE / 2 * distance / reach * 0.9
    out = {}
    for name, (angle, up) in SHEET_VIEWS.items():
        if up is None:  # just above the floor: below the middle by most of the half height
            up = -float(np.degrees(np.arcsin(min(0.9, (height / 2 - 0.03) / distance))))
        out[name] = Shot(np.zeros((TILE, TILE), bool), np.zeros((TILE, TILE, 3), np.uint8), focal, angle,
                         distance=distance, centre=(TILE / 2, TILE / 2), elevation=up)
    return out


def sheet(photo: np.ndarray, verts, faces, colours, views: dict, height: float, path) -> None:
    """The photo on the left (two views tall), the model's views in a 3 x 2 grid on the right, labelled."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (TILE * 4 + 8, TILE * 2), "white")
    left = Image.fromarray(photo)
    left.thumbnail((TILE, TILE * 2))
    canvas.paste(left, ((TILE - left.width) // 2, (TILE * 2 - left.height) // 2))
    pen = ImageDraw.Draw(canvas)
    for n, (name, shot) in enumerate(views.items()):
        x, y = TILE + 8 + (n % 3) * TILE, (n // 3) * TILE
        canvas.paste(Image.fromarray(draw(verts, faces, colours, shot, height, TILE)), (x, y))
        pen.rectangle((x, y, x + TILE - 1, y + TILE - 1), outline=(200, 200, 200))
        pen.text((x + 6, y + 4), name, fill=(0, 0, 0))
    canvas.save(path, quality=90)


def parse_sheet(text: str, labels) -> Verdict:
    """A sheet answer: as parse_verdict, each problem also naming its view (unknown views dropped)."""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return Verdict(None)
    if not isinstance(data, dict):
        return Verdict(None)
    problems = []
    for p in data.get("problems") or []:
        try:
            box = [min(max(float(v), 0.0), 1.0) for v in p["box"]]
            view = str(p.get("view", "")).upper().strip()
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if len(box) == 4 and box[2] > box[0] and box[3] > box[1] and view in labels:
            problems.append({"kind": str(p.get("kind", "extra")).lower(), "what": str(p.get("what", "")),
                             "view": view, "box": box})
    score = data.get("score")
    return Verdict(float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None, problems)


def needed(kept: np.ndarray, corner: np.ndarray, voxel: float, shots, height: float, candidates: np.ndarray,
           shrink: int = 4) -> np.ndarray:
    """Of the `candidates` (a mask over np.argwhere(kept)), the ones some real photo's outline needs: without them a
    pixel of its cut-out would be left uncovered. The rest can go without contradicting any photo."""
    idx = np.argwhere(kept)
    centres = corner + (idx + 0.5) * voxel
    need = np.zeros(len(idx), bool)
    for s in shots:
        if s.imagined:
            continue
        cols, rows, depth = s.project(centres, height / 2)
        h, w = s.mask.shape
        hs, ws = h // shrink, w // shrink
        c, r = (cols / shrink).astype(int), (rows / shrink).astype(int)
        ok = (depth > 0) & (c >= 0) & (c < ws) & (r >= 0) & (r < hs)
        cell = np.where(ok, r * ws + c, 0)
        others = np.zeros(hs * ws, bool)
        others[cell[ok & ~candidates]] = True
        small = s.mask[: hs * shrink: shrink, : ws * shrink: shrink].ravel()
        alone = ok & candidates
        need[alone] |= small[cell[alone]] & ~others[cell[alone]]
    return need & candidates


def outline_loss(kept, corner, voxel, shots, height: float, go: np.ndarray, shrink: int = 4) -> float:
    """The most of any real photo's cut-out (share of its pixels) that would be left uncovered without `go` (a mask
    over np.argwhere(kept))."""
    idx = np.argwhere(kept)
    centres = corner + (idx + 0.5) * voxel
    worst = 0.0
    for s in shots:
        if s.imagined:
            continue
        cols, rows, depth = s.project(centres, height / 2)
        h, w = s.mask.shape
        hs, ws = h // shrink, w // shrink
        c, r = (cols / shrink).astype(int), (rows / shrink).astype(int)
        ok = (depth > 0) & (c >= 0) & (c < ws) & (r >= 0) & (r < hs)
        cell = np.where(ok, r * ws + c, 0)
        small = s.mask[: hs * shrink: shrink, : ws * shrink: shrink].ravel()
        before = np.zeros(hs * ws, bool)
        before[cell[ok]] = True
        after = np.zeros(hs * ws, bool)
        after[cell[ok & ~go]] = True
        worst = max(worst, float((small & before & ~after).sum()) / max(int(small.sum()), 1))
    return worst


def thin(kept: np.ndarray, voxel: float) -> np.ndarray:
    """The parts of the carved block thinner than THIN across: what shrinking it by half that and growing it back
    loses (flaps, spikes, strings), while legs and bodies come back whole."""
    from scipy import ndimage

    r = max(1, round(THIN / 2 / voxel))
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = x * x + y * y + z * z <= r * r
    return kept & ~ndimage.binary_opening(kept, structure=ball)


def cut_in_view(kept, corner, voxel, shots, height, view, box) -> np.ndarray:
    """`kept` less what's thin (see `thin`) inside `box` (fractions) of a sheet `view`: all of it if that leaves no
    photo's outline short by more than LOSS (the outlines don't quite agree with each other: cameras placed by
    compass are a few degrees off, and the carving keeps odd bits to satisfy them all); else only what no outline
    needs at all."""
    idx = np.argwhere(kept)
    centres = corner + (idx + 0.5) * voxel
    cols, rows, depth = view.project(centres, height / 2)
    inbox = (cols >= box[0] * TILE) & (cols <= box[2] * TILE) & (rows >= box[1] * TILE) & (rows <= box[3] * TILE)
    inbox &= depth > 0
    inbox &= thin(kept, voxel)[tuple(idx.T)]  # (Muse says where; what's thin there is what dangles)
    small_loss = outline_loss(kept, corner, voxel, shots, height, inbox) <= LOSS
    go = inbox if small_loss else inbox & ~needed(kept, corner, voxel, shots, height, inbox)
    kept = kept.copy()
    kept[tuple(idx[go].T)] = False
    return kept


COMPARE_PROMPT = """Left: a photo of {name}. Right: two 3D models of it, A (top row) and B (bottom row), each seen \
from the same three sides. Only the shape matters.

Which model's shape is closer to the real {name}?

Answer in JSON only: {{"closer": "A", "B" or "same", "why": "a few words"}}"""


def compare_sheet(photo: np.ndarray, a, b, path) -> None:
    """The photo, then the first row of sheet `a` (A) above the first row of sheet `b` (B)."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (TILE * 4 + 8, TILE * 2), "white")
    left = Image.fromarray(photo)
    left.thumbnail((TILE, TILE * 2))
    canvas.paste(left, ((TILE - left.width) // 2, (TILE * 2 - left.height) // 2))
    pen = ImageDraw.Draw(canvas)
    for row, (label, sheet_path) in enumerate((("A", a), ("B", b))):
        canvas.paste(Image.open(sheet_path).crop((TILE + 8, 0, TILE * 4 + 8, TILE)), (TILE + 8, row * TILE))
        pen.text((TILE + 14, row * TILE + 20), label, fill=(200, 0, 0))
    canvas.save(path, quality=90)


def parse_closer(text: str) -> str | None:
    match = re.search(r'"closer"\s*:\s*"(A|B|same)"', text, re.I)
    return match.group(1).upper() if match and match.group(1).lower() != "same" else ("SAME" if match else None)


def _by_view(verdicts: list[Verdict], view: str) -> list[Verdict]:
    return [Verdict(v.score, [q for q in v.problems if q["view"] == view]) for v in verdicts]


def judge(shots, kept, corner, voxel, height: float, ask, folder, name: str, size,
          log=print) -> tuple[np.ndarray, dict]:
    """Rounds of: Muse looks at a sheet (the photo, and the model from six sides); what it agrees is extra is cut, as
    far as the photos' outlines allow; the sheet is drawn again and the cut kept only if Muse, shown the old and new
    models together, doesn't mostly prefer the old one (its scores out of 10 are too coarse for a small fix: 3, 2, 2,
    2, 3 for one clearly better) and the model still matches the cut-outs. `ask(prompt, picture_path) -> [answers]`
    asks Muse (saved and voted). Returns the kept grid and a report."""
    from .carve import agreement, paint, surface, vertex_normals

    folder.mkdir(parents=True, exist_ok=True)
    photos = [s for s in shots if not s.imagined]
    views = virtual_shots(size)
    prompt = SHEET_PROMPT.format(name=name, labels=", ".join(views))
    front = next(i for i, s in enumerate(shots) if not s.imagined)

    def look(grid, tag: str) -> list[Verdict]:
        verts, faces = surface(grid, corner, voxel)
        colours = paint(verts, vertex_normals(verts, faces), shots, height)
        picture = folder / f"sheet_{tag}.jpg"
        sheet(shots[front].image, verts, faces, colours, views, height, picture)
        return [parse_sheet(a, views) for a in ask(prompt, picture)]

    def score(verdicts) -> float:
        given = [v.score for v in verdicts if v.score is not None]
        return float(np.median(given)) if given else 0.0

    def better(tag: str) -> tuple[bool, list[str]]:
        """Muse compares the last sheet (A) with the new one (B): kept if B wins or they're the same, more often than A
        wins."""
        picture = folder / f"compare_{tag}.jpg"
        compare_sheet(shots[front].image, folder / f"sheet_{previous}.jpg", folder / f"sheet_{tag}.jpg", picture)
        said = [parse_closer(a) for a in ask(COMPARE_PROMPT.format(name=name), picture)]
        return said.count("A") < len(said) / 2, [x or "?" for x in said]

    report = {"rounds": [], "missing": []}
    previous = "round0"
    verdicts = look(kept, "round0")
    for n in range(1, ROUNDS + 1):
        problems = [{**p, "view": view} for view in views for p in agreed(_by_view(verdicts, view))]
        report["missing"] = sorted({p["what"][0] for view in views for p in agreed(_by_view(verdicts, view),
                                                                                  "missing")})
        trimmed = kept
        for p in problems:
            if p["votes"] < CUT_VOTES:
                continue
            trimmed = cut_in_view(trimmed, corner, voxel, shots, height, views[p["view"]], p["box"])
            log(f"judge:    {p['view']}: {p['what'][0]!r} ({p['votes']} of {len(verdicts)} answers)")
        cut = int(kept.sum() - trimmed.sum())
        entry = {"score": score(verdicts), "cut_voxels": cut,
                 "problems": [{"view": p["view"], "what": p["what"][0], "votes": p["votes"]} for p in problems]}
        report["rounds"].append(entry)
        if not cut:
            break
        after = look(trimmed, f"round{n}")
        a_before = agreement(photos, kept, corner, voxel, height)
        a_after = agreement(photos, trimmed, corner, voxel, height)
        won, said = better(f"round{n}")
        entry.update(score_after=score(after), agreement=[round(a_before, 3), round(a_after, 3)], closer=said)
        entry["kept"] = won and a_after >= a_before - AGREEMENT_SLACK
        if not entry["kept"]:
            break
        kept, verdicts, previous = trimmed, after, f"round{n}"
    report["score"] = score(verdicts)
    return kept, report
