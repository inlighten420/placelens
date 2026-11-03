"""Made things (a chair, a steering wheel): Muse's design kept whole, fitted to the parts SAM 3.1 finds in every photo.

Muse Spark Open describes a made thing well as shapes: an office chair as a backrest, a seat, armrests on struts, a
gas lift and a five-star base on casters. Fitting that design into the carving wrecked it (3 Nov): a chair's carving
is a blob, as thin legs and gaps can't be carved, so the legs were thrown out and the rest squashed. The photos
themselves say far more. SAM 3.1 outlines the thing to the pixel, legs and casters too, and it finds the parts Muse
names. So the design is fitted to those outlines instead:

1. SAM 3.1 is asked for the biggest kinds of part Muse named ("backrest", "seat cushion", "armrest pad") in every
   photo. What's inside the thing's outline but none of those is the rest (a chair's base, gas lift and struts),
   and the parts SAM couldn't find are fitted to that, the way a teddy's body is what's left of its cut-out.
2. The whole design moves, turns and scales to match every photo's outline, and each found part its own mask.
3. Then each assembly: a kind SAM found, with the parts it can't see riding along (struts with their armrest pad),
   and what stands on the floor (the star base, casters) as one group that slides and spins on the floor. Something
   standing up (a backrest) may lean; something flat (a seat) stays level.
4. The cameras are fitted again to the model, then the model to the cameras, twice. A photo from above, whose turn
   the carving had guessed 45 degrees out, gets a full turn round first.

Each kind SAM found takes the photos' colour inside its masks. On the office chair (3 Nov) the outline matched the
photos 0.67 against 0.59 for the design fitted to the carving, from the side 0.73 against 0.48; and it has its star
base, casters and armrests again. If SAM finds none of the parts in two photos, nothing pins the design, and the
old fit is used instead (a steering wheel whose parts were named "plate carbon" came out a slab).
"""

from __future__ import annotations

import copy
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import sight
from .parts import _rotation

MAX_KINDS = 6  # kinds of part SAM is asked for in every photo, biggest first (a quarter of a US cent each)
PART_WEIGHT = 0.3  # each found kind matching its own mask, against the whole outline
MIN_SEEN = 2  # photos a kind must be found in to pin the design
FLOOR_SHARE = 0.1  # of the height: parts whose bottom is this near the floor stand on it
REST, FLOOR = "(rest)", "(floor)"
SIDE_WORDS = r"\b(left|right|front|rear|back|forward|top|bottom|upper|lower|inner|outer|middle|centre|center|\d+)\b"


def kind(what: str | None) -> str:
    """A part's kind: its name without which one it is ("left armrest pad" -> "armrest pad")."""
    w = re.sub(SIDE_WORDS, " ", (what or "part").lower())
    return re.sub(r"\s+", " ", w).strip() or "part"


def volume(p: dict) -> float:
    return float(np.prod(p["size"])) * (math.pi / 4 if p["shape"] in ("cylinder", "cone", "tube") else 1.0)


def _box(p: dict) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * np.array(p["size"])
    world = corners @ _rotation(p["turn"]).T + np.array(p["at"])
    return world.min(axis=0), world.max(axis=0)


def moved(parts: list[dict], idx: list[int], prm: np.ndarray, pivot: np.ndarray) -> list[dict]:
    """Parts `idx` moved about `pivot` by prm: shift x, y, z; log scale; log stretch x, y, z; turn about the
    vertical; lean about x (degrees)."""
    out = list(parts)
    shift, scale = np.array(prm[:3]), math.exp(prm[3])
    stretch = np.exp(np.array(prm[4:7])) * scale
    yaw, lean = prm[7], prm[8]
    rot = _rotation([lean, yaw, 0.0])
    for i in idx:
        p = dict(parts[i])
        p["at"] = (rot @ ((np.array(p["at"]) - pivot) * stretch) + pivot + shift).tolist()
        turn = list(p.get("turn", [0, 0, 0]))
        size = np.array(p["size"], float)
        if abs(turn[0]) < 1e-6 and abs(turn[2]) < 1e-6:  # (turned only about the vertical: its height stays up)
            across = math.sqrt(stretch[0] * stretch[2])
            size = size * np.array([across, stretch[1], across])
        else:
            size = size * float(np.cbrt(np.prod(stretch)))
        p["size"] = size.tolist()
        p["turn"] = [turn[0] + lean, turn[1] + yaw, turn[2]]
        out[i] = p
    return out


class Aligner:
    """Muse's design fitted to the photos: the whole outline, and each found kind its own SAM masks."""

    def __init__(self, parts: list[dict], shots: list, mid: float, kind_masks: dict[str, list]):
        self.parts, self.shots, self.mid = parts, shots, mid
        self.kinds = [kind(p.get("what")) for p in parts]
        self.stand_ins: dict[str, list] = {}  # kind -> (photo index, a copy of its shot whose cut-out is the mask)
        for k, masks in kind_masks.items():
            stand = [(i, self._stand_in(i, m)) for i, m in enumerate(masks) if m is not None and m.sum() > 150]
            if len({i for i, _ in stand}) >= MIN_SEEN:
                self.stand_ins[k] = stand
        self.found = list(self.stand_ins)
        rest = []
        for i, s in enumerate(shots):
            named = np.zeros_like(s.mask)
            for k in self.found:
                if kind_masks[k][i] is not None:
                    named |= kind_masks[k][i]
            left = s.mask & ~named
            if named.any() and left.sum() > 150:
                rest.append((i, self._stand_in(i, left)))
        if rest and any(k not in self.found for k in self.kinds):
            self.stand_ins[REST] = rest
        self.scored_as = [k if k in self.found else REST for k in self.kinds]

    def _stand_in(self, i: int, mask: np.ndarray):
        t = copy.copy(self.shots[i])
        t.mask = mask
        return t

    def _sync(self) -> None:
        for stand in self.stand_ins.values():
            for i, t in stand:
                s = self.shots[i]
                t.angle, t.elevation, t.distance, t.centre = s.angle, s.elevation, s.distance, s.centre

    def score(self, parts: list[dict], only: int | None = None) -> float:
        shots = self.shots if only is None else [self.shots[only]]
        whole = sight.score(parts, shots, self.mid)
        each = []
        for k, stand in self.stand_ins.items():
            mine = [p for p, kk in zip(parts, self.scored_as, strict=True) if kk == k]
            views = [t for i, t in stand if only is None or i == only]
            if mine and views:
                each.append(sight.score(mine, views, self.mid))
        return whole + PART_WEIGHT * (float(np.mean(each)) if each else 0.0)

    def _descend(self, idx: list[int], pivot: np.ndarray, steps: list[float], rounds: int, best: float) -> float:
        prm = np.zeros(9)
        for r in range(rounds):
            for j, st in enumerate(steps):
                for d in ((-st * 0.5 ** r, st * 0.5 ** r) if st else ()):
                    trial = prm.copy()
                    trial[j] += d
                    got = self.score(moved(self.parts, idx, trial, pivot))
                    if got > best + 1e-5:
                        best, prm = got, trial
        self.parts = moved(self.parts, idx, prm, pivot)
        return best

    def assemblies(self) -> dict[str, list[int]]:
        """Each found kind with the parts SAM can't see that sit within its footprint (the smallest such); what
        stands on the floor as its own group."""
        floor = min(_box(p)[0][1] for p in self.parts)
        tall = max(_box(p)[1][1] for p in self.parts) - floor
        ground = [i for i, p in enumerate(self.parts)
                  if self.kinds[i] not in self.found and _box(p)[0][1] - floor < FLOOR_SHARE * tall]
        foot = {}
        for k in self.found:
            boxes = [_box(p) for p, kk in zip(self.parts, self.kinds, strict=True) if kk == k]
            lo = np.min([b[0][[0, 2]] for b in boxes], axis=0)
            hi = np.max([b[1][[0, 2]] for b in boxes], axis=0)
            pad = 0.2 * (hi - lo)
            foot[k] = (lo - pad, hi + pad)
        groups: dict[str, list[int]] = {k: [] for k in self.found}
        if ground:
            groups[FLOOR] = ground
        for i, (p, k) in enumerate(zip(self.parts, self.kinds, strict=True)):
            if i in ground:
                continue
            if k in groups:
                groups[k].append(i)
                continue
            xz = np.array(p["at"])[[0, 2]]
            under = [f for f in self.found if (foot[f][0] <= xz).all() and (xz <= foot[f][1]).all()]
            if under:
                groups[min(under, key=lambda f: float(np.prod(foot[f][1] - foot[f][0])))].append(i)
        return {k: v for k, v in groups.items() if v}

    def fit(self) -> float:
        self._sync()
        best = self.score(self.parts)
        reach = max(max(p["size"]) for p in self.parts)
        best = self._descend(list(range(len(self.parts))), np.zeros(3),
                             [0.1 * reach] * 3 + [0.1, 0.08, 0.08, 0.08, 10.0, 0.0], 5, best)
        groups = self.assemblies()
        floor = min(_box(p)[0][1] for p in self.parts)
        tall = max(_box(p)[1][1] for p in self.parts) - floor
        order = sorted(groups, key=lambda k: -sum(volume(self.parts[i]) for i in groups[k]))
        for _ in range(2):
            for k in order:
                idx = groups[k]
                grounded = any(_box(self.parts[i])[0][1] - floor < 0.05 * tall for i in idx)
                lo = np.min([_box(self.parts[i])[0] for i in idx], axis=0)
                hi = np.max([_box(self.parts[i])[1] for i in idx], axis=0)
                upright = (hi - lo)[1] > (hi - lo)[2]
                v = 0.0 if grounded else 1.0
                steps = [0.06 * reach, 0.06 * reach * v, 0.06 * reach, 0.12, 0.1, 0.1 * v, 0.1,
                         12.0 if k == FLOOR else 0.0, 8.0 * v if upright else 0.0]
                pivot = np.mean([self.parts[i]["at"] for i in idx], axis=0)
                best = self._descend(idx, pivot, steps, 4, best)
        return best

    def refit_camera(self, i: int) -> None:
        """Photo i's camera moved to match the model: its turn round (not the front's, which every other is measured
        from; a photo from above tries every 15 degrees first), its height above and its distance."""
        s = self.shots[i]

        def score() -> float:
            self._sync()
            return self.score(self.parts, only=i)

        best = score()
        if s.overhead:
            start = s.angle
            for a in range(0, 360, 15):
                s.angle = float(a)
                got = score()
                if got > best:
                    best, start = got, s.angle
            s.angle = start
        steps = {"angle": 6.0 if i else 0.0, "elevation": 6.0, "distance": 0.08}
        for r in range(5):
            for f, st in steps.items():
                if not st:
                    continue
                start = getattr(s, f)
                for d in (-st * 0.5 ** r, st * 0.5 ** r):
                    setattr(s, f, start * (1 + d) if f == "distance" else start + d)
                    got = score()
                    if got > best + 1e-5:
                        best, start = got, getattr(s, f)
                setattr(s, f, start)
        self._sync()


def kinds_to_find(design: list[dict]) -> list[str]:
    total: dict[str, float] = {}
    for p in design:
        total[kind(p.get("what"))] = total.get(kind(p.get("what")), 0.0) + volume(p)
    return sorted(total, key=lambda k: -total[k])[:MAX_KINDS]


def fit_design(meta, photos, shots: list, design: list[dict], height: float, out: Path,
               step=lambda _: None) -> tuple[list[dict] | None, list[str]]:
    """Muse's `design` fitted to the photos (`photos` and their fitted `shots`, in the same order), the cameras
    refitted with it. None (with why) if SAM found too little of it to pin it."""
    from . import seen
    from .soft import _sam_masks

    kinds = kinds_to_find(design)
    step(f"SAM 3.1 is finding its {', '.join(kinds)} in {len(photos)} photos")
    jobs = [(i, k) for i in range(len(photos)) for k in kinds]
    with ThreadPoolExecutor(4) as pool:
        found = list(pool.map(lambda j: _sam_masks(meta, photos[j[0]].small, j[1], out), jobs))
    masks: dict[str, list] = {k: [None] * len(photos) for k in kinds}
    for (i, k), ms in zip(jobs, found, strict=True):
        cut = shots[i].mask
        union = np.zeros_like(cut)
        for m in ms:  # (on the thing, and not the whole thing found again)
            if m.sum() and (m & cut).sum() > 0.5 * m.sum() and m.sum() < 0.9 * cut.sum():
                union |= m & cut
        masks[k][i] = union if union.any() else None
    real = [i for i, s in enumerate(shots) if not s.imagined]
    aligner = Aligner([dict(p) for p in design], [shots[i] for i in real], height / 2,
                      {k: [masks[k][i] for i in real] for k in kinds})
    if not aligner.found:
        return None, [f"SAM 3.1 didn't find its {', '.join(kinds)} in {MIN_SEEN}+ photos: fitted the old way"]
    step(f"fitting Muse's {len(design)} parts to the photos, and the cameras to them")
    before = sight.score(design, aligner.shots, height / 2)
    aligner.fit()
    for _ in range(2):
        for i in range(len(aligner.shots)):
            aligner.refit_camera(i)
        aligner.fit()
    parts = aligner.parts
    for k in aligner.found:  # each found kind takes the photos' colour inside its masks
        px = [shots[i].image[m] for i, m in enumerate(masks[k]) if m is not None]
        if px and sum(len(x) for x in px) > 500:
            c = np.median(np.concatenate(px), axis=0)
            for p, kk in zip(parts, aligner.kinds, strict=True):
                if kk == k:
                    p["colour"] = "#" + "".join(f"{int(v):02x}" for v in c)
    after = sight.score(parts, aligner.shots, height / 2)
    for n, i in enumerate(real):  # what SAM found, and the model over each photo
        s, ph = shots[i], photos[i]
        labels = s.mask.astype(np.uint8)
        for j, k in enumerate(kinds):
            if masks[k][i] is not None:
                labels[masks[k][i]] = j + 2
        seen.parts(s.image, labels, out / "seen" / f"{ph.path.stem}_parts.jpg")
        seen.fit(s.image, s.mask, seen.shapes_outline(parts, aligner.shots[n], height / 2),
                 out / "seen" / f"{ph.path.stem}_fit.jpg")
    names = ", ".join(aligner.found)
    return parts, [f"Muse's design fitted to the parts SAM 3.1 found ({names}): it matches the photos {after:.0%} "
                   f"(as designed {before:.0%})"]
