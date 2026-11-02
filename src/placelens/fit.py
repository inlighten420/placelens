"""Parts fitted to a carving: clean shapes where the carving is only lumpy.

A carving (carve.py) gets an object's size, outline and colours right from every photo, but it can't make a flat
seat or open a gap the outlines don't show: an office chair came out lumpy (2 Nov). Muse knows what such a thing is
made of (parts.py: a seat slab, a backrest, five legs on casters). Here each of Muse's parts is slid, sized and the
whole set turned until the parts sit inside the carving and fill it, then each part takes the photos' colour where
it sits. Used for hard, made things with see-through gaps; soft ones (a teddy bear) stay carved.

Frames: both in metres, Y up, standing on Y = 0, X right and Z towards the first photo's camera.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .parts import _rotation

BETA = 0.5  # the score weighs "the parts sit inside the carving" over "they fill all of it" (it fills the gaps)
GRID = 40  # voxels along the longest side while fitting (the carving is coarsened to this)
ROUNDS = 4  # passes over the parts
TURNS = (0, 45, -45, 90, -90, 180)  # degrees the whole set is tried turned about Y (Muse's front may be a corner)
SIZE_RANGE = (0.5, 2.0)  # a part may shrink or grow this much from Muse's size
MAX_SHIFT = 0.15  # a part may move this share of the object's biggest size from where Muse put it (once the whole
# set has moved together)
STRAY = 0.4  # a part with less than this share of itself inside the carving is dropped
LOST = 0.15  # ...and one of a kept set (a leg) with less than this
TOUCH = 2  # grid steps apart that parts may be and still count as one piece (each is slid on its own)
GOOD_INSIDE = 0.7  # at least this share of the parts must sit inside the carving...
GOOD_FILL = 0.45  # ...and fill this share of it, or the carving is kept
SEEN = 0.8  # share of a part that must fall inside the object's outline in every photo for the photos alone to keep it


def _coarse(kept: np.ndarray, corner: np.ndarray, voxel: float):
    """The carving at about GRID voxels a side, and its voxel centres (n x 3, metres)."""
    step = max(1, int(math.ceil(max(kept.shape) / GRID)))
    shape = tuple(s // step for s in kept.shape)
    k = kept[:shape[0] * step, :shape[1] * step, :shape[2] * step]
    k = k.reshape(shape[0], step, shape[1], step, shape[2], step).mean(axis=(1, 3, 5)) >= 0.5
    v = voxel * step
    axes = [corner[i] + (np.arange(shape[i]) + 0.5) * v for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return k.ravel(), np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def inside(p: dict, points: np.ndarray) -> np.ndarray:
    """Which points are inside the part (boxes, wedges as boxes; cylinders, cones and tubes as cylinders)."""
    local = (points - np.array(p["at"])) @ _rotation(p["turn"]) / np.maximum(np.array(p["size"]), 1e-6)
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    if p["shape"] == "sphere":
        return x * x + y * y + z * z <= 0.25
    if p["shape"] in ("cylinder", "cone", "tube"):
        return (np.abs(y) <= 0.5) & (x * x + z * z <= 0.25)
    return (np.abs(x) <= 0.5) & (np.abs(y) <= 0.5) & (np.abs(z) <= 0.5)


def score(count: np.ndarray, carved: np.ndarray) -> tuple[float, float, float]:
    """How well parts covering `count` (parts per voxel) match the carving: F-score, share inside, share filled."""
    union = count > 0
    both = np.count_nonzero(union & carved)
    if not both:
        return 0.0, 0.0, 0.0
    precision, recall = both / np.count_nonzero(union), both / max(np.count_nonzero(carved), 1)
    b2 = BETA * BETA
    return (1 + b2) * precision * recall / (b2 * precision + recall), precision, recall


def _turned(parts: list[dict], degrees: float) -> list[dict]:
    """The whole set turned about Y (through the origin)."""
    r = _rotation([0, degrees, 0])
    out = copy.deepcopy(parts)
    for p in out:
        p["at"] = list(np.array(p["at"]) @ r.T)
        p["turn"][1] += degrees
    return out


def fit(parts: list[dict], kept: np.ndarray, corner: np.ndarray, voxel: float,
        seen=None) -> tuple[list[dict], float, float]:
    """Muse's parts moved and sized to the carving. Returns the fitted parts, the share of them inside the carving and
    the share of the carving they fill. `seen(part)`: how much of the part the photos show inside the object's
    outline, from every side (0..1, sight.seen); a part the photos all show is kept though the carving lost it."""
    carved, points = _coarse(kept, corner, voxel)
    step = voxel * max(1, int(math.ceil(max(kept.shape) / GRID)))

    def total(ps):
        count = np.zeros(len(points), int)
        masks = [inside(p, points) for p in ps]
        for m in masks:
            count += m
        return count, masks

    best = None
    for degrees in TURNS:
        ps = _turned(parts, degrees)
        count, _ = total(ps)
        s = score(count, carved)[0]
        if best is None or s > best[0]:
            best = (s, degrees, ps)
    _, degrees, ps = best
    start = [list(p["size"]) for p in ps]
    twin = _twins(parts)
    normal = _rotation([0, degrees, 0]) @ np.array([1.0, 0.0, 0.0])  # the mirror plane's normal, turned with them

    # First the whole set together (where the object stands), then each part a little: a part may not wander more
    # than MAX_SHIFT of the object's size from where Muse put it (an armrest dragged into the seat, 2 Nov).
    count, masks = total(ps)
    current = score(count, carved)[0]
    for _ in range(ROUNDS):
        moved = False
        for axis in range(3):
            for change in (3 * step, -3 * step, step, -step):
                trial = copy.deepcopy(ps)
                for p in trial:
                    p["at"][axis] += change
                if min(_bottom(p) for p in trial) < -step:
                    continue
                c, m = total(trial)
                s = score(c, carved)[0]
                if s > current + 1e-4:
                    ps, count, masks, current, moved = trial, c, m, s, True
        # The whole set a little bigger or smaller, about its middle, each way across (Muse's proportions are right
        # but its overall size a few percent off: the outermost parts, armrests and casters, then land outside, 2 Nov)
        mid_x = np.mean([p["at"][0] for p in ps])
        mid_z = np.mean([p["at"][2] for p in ps])
        for axis, middle in ((0, mid_x), (2, mid_z)):
            for f in (0.9, 1.1, 0.95, 1.05):
                trial = copy.deepcopy(ps)
                for p in trial:
                    p["at"][axis] = middle + (p["at"][axis] - middle) * f
                    if abs(p["turn"][1]) < 1:  # (a part turned about y, a leg at 72°, keeps its size: its spread
                        p["size"][axis] *= f  # is what scales)
                c, m = total(trial)
                s = score(c, carved)[0]
                if s > current + 1e-4:
                    ps, count, masks, current, moved = trial, c, m, s, True
        if not moved:
            break
    origin = [np.array(p["at"]) for p in ps]
    lows, highs = zip(*(_box_of(p) for p in ps), strict=True)
    reach = MAX_SHIFT * float(np.max(np.max(highs, axis=0) - np.min(lows, axis=0)))  # (of the object's biggest size)
    for _ in range(ROUNDS):
        moved = False
        for i in range(len(ps)):
            j = twin[i]
            for key, axis, change in [("at", a, d * step * k) for a in range(3) for d in (1, -1) for k in (3, 1)] + \
                    [("size", a, f) for a in range(3) for f in (1.25, 0.8, 1.08, 0.93)]:
                trial = copy.deepcopy(ps[i])
                if key == "at":
                    trial["at"][axis] += change
                else:
                    trial["size"][axis] *= change
                    lo, hi = SIZE_RANGE
                    if not lo * start[i][axis] <= trial["size"][axis] <= hi * start[i][axis]:
                        continue
                if _bottom(trial) < -step or np.linalg.norm(np.array(trial["at"]) - origin[i]) > reach:
                    continue
                pair = [(i, trial)]
                if j is not None:  # its mirror twin moves the mirror way, sizes alike
                    other = copy.deepcopy(ps[j])
                    delta = np.array(trial["at"]) - np.array(ps[i]["at"])
                    other["at"] = list(np.array(other["at"]) + delta - 2 * (delta @ normal) * normal)
                    other["size"] = [a * b / c for a, b, c in zip(other["size"], trial["size"], ps[i]["size"],
                                                                  strict=True)]
                    pair.append((j, other))
                c = count.copy()
                new = {}
                for k, q in pair:
                    new[k] = inside(q, points)
                    c += new[k].astype(int) - masks[k].astype(int)
                s = score(c, carved)[0]
                if s > current + 1e-4:
                    for k, q in pair:
                        ps[k], masks[k] = q, new[k]
                    count, current, moved = c, s, True
        if not moved:
            break
    # A part still mostly outside the carving is in the wrong place (a chair's armrest posts left standing in the air
    # beside it, 2 Nov): dropped. Judged on points spread through the part against the full carving, so thin parts
    # (a post, a leg) are judged too.
    # Repeated parts (a chair's five legs, its casters) stand or fall together: a set is kept if its parts are
    # inside on average, so the one leg the carving's thin base misses isn't lost.
    share = [share_inside(p, kept, corner, voxel) for p in ps]
    photo_kept = []
    sets = _sets([dict(p, size=s) for p, s in zip(ps, start, strict=True)])  # (as Muse sized them)
    keep = []
    for i, (p, m) in enumerate(zip(ps, masks, strict=True)):
        in_set = np.mean([share[j] for j in sets[i]]) >= STRAY and share[i] >= LOST  # (one of a set clean off it)
        # A mirrored pair (a chair's two armrests) is kept if either side sits in the carving: the photos may only
        # show one side of a thin part well.
        paired = twin[i] is not None and max(share[i], share[twin[i]]) >= STRAY
        if in_set or paired:
            keep.append(p)
        elif seen is not None and (p := _nudged(p, seen, step, origin[i], reach)) is not None:  # (carving eats
            # thin parts, an armrest, when the cameras are a little off; the photos' own outlines don't: 2 Nov). Not
            # counted in the score against the carving.
            photo_kept.append(p)
        else:
            count = count - m
    keep = connected(keep, TOUCH * step)
    count = np.zeros(len(points), int)
    for p in keep:
        count += inside(p, points)
    _, precision, recall = score(count, carved)
    if photo_kept:
        keep = connected(keep + photo_kept, TOUCH * step)
    return keep, precision, recall


def _nudged(p: dict, seen, step: float, origin: np.ndarray, reach: float) -> dict | None:
    """`p` moved a little to where the photos agree it is (an armrest at the object's very edge lands half outside
    when the cameras are a few centimetres off), if they then all show it; else None. Never further than `reach`
    from where Muse put it."""
    best, share = p, seen(p)
    if share < SEEN / 3:  # (nowhere near: wrong, not a little off)
        return None
    for _ in range(ROUNDS):
        moved = False
        for axis in range(3):
            for change in (2 * step, -2 * step, step, -step):
                trial = copy.deepcopy(best)
                trial["at"][axis] += change
                if _bottom(trial) < -step or np.linalg.norm(np.array(trial["at"]) - origin) > reach:
                    continue
                s = seen(trial)
                if s > share + 1e-3:
                    best, share, moved = trial, s, True
        if not moved or share >= SEEN:
            break
    return best if share >= SEEN else None


def _box_of(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """The part's extent: lowest and highest corner (its box, turned)."""
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * np.array(p["size"])
    world = corners @ _rotation(p["turn"]).T + np.array(p["at"])
    return world.min(axis=0), world.max(axis=0)


def connected(parts: list[dict], gap: float) -> list[dict]:
    """Only the biggest group of parts that touch (within `gap`): an object is one piece, so a part left lying apart
    from it (a chair leg on the floor beside it, 2 Nov) is dropped."""
    if len(parts) < 2:
        return parts
    boxes = [_box_of(p) for p in parts]
    spots = [_spread_through(p) for p in parts]
    grown = [dict(p, size=[v + 2 * gap for v in p["size"]]) for p in parts]
    group = list(range(len(parts)))

    def root(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            (a0, a1), (b0, b1) = boxes[i], boxes[j]
            near = np.all(a0 - gap <= b1) and np.all(b0 - gap <= a1)
            if near and (inside(grown[j], spots[i]).any() or inside(grown[i], spots[j]).any()):  # (shapes, not boxes)
                group[root(i)] = root(j)
    volume: dict[int, float] = {}
    for i, p in enumerate(parts):
        volume[root(i)] = volume.get(root(i), 0.0) + float(np.prod(p["size"]))
    main = max(volume, key=volume.get)
    return [p for i, p in enumerate(parts) if root(i) == main]


def _bottom(p: dict) -> float:
    """The lowest point of the part (its box's corners, turned)."""
    return float(_box_of(p)[0][1])


def _twins(parts: list[dict], near: float = 0.03, within: float = 0.1) -> list[int | None]:
    """Each part's mirror twin across x = 0 as Muse described it (a left and a right armrest): same shape and colour,
    sizes within `within`, at (x, y, z) and (-x, y, z) within `near` metres. None for a part on the middle line."""
    out: list[int | None] = []
    for i, p in enumerate(parts):
        found = None
        if abs(p["at"][0]) > near:
            for j, q in enumerate(parts):
                if j != i and q["shape"] == p["shape"] and q["colour"] == p["colour"] \
                        and abs(p["at"][0] + q["at"][0]) <= near and abs(p["at"][1] - q["at"][1]) <= near \
                        and abs(p["at"][2] - q["at"][2]) <= near \
                        and all(abs(a - b) <= within * max(a, b) for a, b in zip(p["size"], q["size"], strict=True)):
                    found = j
                    break
        out.append(found)
    return out


def _sets(parts: list[dict], within: float = 0.2) -> list[list[int]]:
    """For each part, the parts that are copies of it as Muse described them: same shape and colour, sizes within
    `within` of each other (in any order: a leg turned 72° has its size in the same axes)."""
    out = []
    for p in parts:
        mine = sorted(p["size"])
        out.append([j for j, q in enumerate(parts) if q["shape"] == p["shape"] and q["colour"] == p["colour"]
                    and all(abs(a - b) <= within * max(a, b) for a, b in zip(mine, sorted(q["size"]), strict=True))])
    return out


def _spread_through(p: dict, n: int = 200) -> np.ndarray:
    """Points spread through the part (metres, n x 3), the same each time."""
    rng = np.random.default_rng(0)
    local = rng.uniform(-0.5, 0.5, (n * 2, 3))
    if p["shape"] in ("cylinder", "cone", "tube"):
        local = local[local[:, 0] ** 2 + local[:, 2] ** 2 <= 0.25]
    elif p["shape"] == "sphere":
        local = local[(local ** 2).sum(axis=1) <= 0.25]
    return (local[:n] * np.array(p["size"])) @ _rotation(p["turn"]).T + np.array(p["at"])


def share_inside(p: dict, kept: np.ndarray, corner: np.ndarray, voxel: float) -> float:
    """The share of points spread through the part that fall inside the carving."""
    world = _spread_through(p)
    idx = np.floor((world - corner) / voxel).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(kept.shape)), axis=1)
    hit = np.zeros(len(idx), bool)
    hit[ok] = kept[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return float(hit.mean()) if len(hit) else 0.0


def colour_from(parts: list[dict], verts: np.ndarray, colours: np.ndarray) -> None:
    """Each part takes the middle colour of the carving's surface where it sits (the photos' colour there); a part
    the surface doesn't reach keeps Muse's colour."""
    for p in parts:
        grown = dict(p, size=[s * 1.3 + 0.02 for s in p["size"]])  # (the surface is just outside a thin part)
        near = inside(grown, verts)
        if np.count_nonzero(near) >= 5:
            rgb = np.clip(np.median(colours[near], axis=0), 0, 1)
            p["colour"] = "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)
