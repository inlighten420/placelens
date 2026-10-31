"""Asking Muse, saved and voted: every question goes through `ask`, which keeps each reply in a file (so asking again
is free unless `fresh`) and can ask several times at once. One answer's numbers wobble (Spark Open put the same house
4 m and 11 m away); the middle of five is steady. The merges below turn several parsed answers into one, with the
share that agreed folded into the confidence.

Vote 0 is saved at `reply` itself, so replies saved before voting existed still count as the first vote.
"""

from __future__ import annotations

import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .meta import OPEN

PARALLEL = 8  # Baseten took 17 s for 8 at once vs 13 s for one

NOTES = """

The person who took the photos adds: "{notes}"
Use that for what the photos can't show (a hidden side, a real size); where a photo clearly shows otherwise, go by \
the photo."""


def noted(prompt: str, notes: str | None) -> str:
    """`prompt` with the person's own notes after it. Without notes it's unchanged, so saved replies still count; with
    them it's a new question (the cache key is the prompt), asked afresh."""
    return prompt + NOTES.format(notes=notes.strip()) if notes and notes.strip() else prompt


def default_votes(model: str) -> int:
    """Five for Spark Open (on Baseten credits), one for the paid Meta API models."""
    return 5 if model == OPEN else 1


def vote_path(reply: Path, i: int) -> Path:
    return reply if i == 0 else reply.with_name(f"{reply.stem}_v{i}{reply.suffix}")


def ask(meta, prompt: str, photos, reply: Path, model: str, votes: int = 1, fresh: bool = False,
        effort: str | None = None) -> list[str]:
    """`votes` answers to one question, each saved next to `reply`; only the missing ones are asked, in parallel.
    `effort`: how long Spark Open thinks (part of what's asked, so give `reply` a name that says it)."""
    paths = [vote_path(reply, i) for i in range(max(1, votes))]
    missing = [p for p in paths if fresh or not p.exists()]

    def one(path: Path) -> None:
        extra = {"effort": effort} if effort else {}
        path.write_text(meta.ask(prompt, photos, model=model, **extra), encoding="utf-8")

    if missing:
        reply.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(min(PARALLEL, len(missing))) as pool:
            list(pool.map(one, missing))
    return [p.read_text(encoding="utf-8") for p in paths]


def majority(values: list) -> tuple[object, float]:
    """The most common value (None counts as a value) and the share of answers that gave it. Ties: the first seen."""
    if not values:
        return None, 0.0
    counts = Counter(values)
    best = max(counts.values())
    winner = next(v for v in values if counts[v] == best)
    return winner, best / len(values)


def middle(values: list) -> float | None:
    """The median of the usable numbers, or None."""
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.median(nums) if nums else None


def merge_verdicts(verdicts: list):
    """One pick verdict from several: the most chosen building; confidence = its share x their mean confidence;
    floors and distance the middle of those that chose it; reasons from the one nearest that distance."""
    if len(verdicts) == 1:
        return verdicts[0]
    choice, share = majority([v.choice for v in verdicts])
    agree = [v for v in verdicts if v.choice == choice]
    distance = middle([v.distance_m for v in agree])
    floors = middle([v.floors for v in agree])
    near = min(agree, key=lambda v: abs(v.distance_m - distance) if isinstance(v.distance_m, (int, float))
               and distance is not None else float("inf"))
    signs = list(dict.fromkeys(s for v in verdicts for s in v.signs))
    return replace(near, confidence=round(share * statistics.mean(v.confidence for v in agree), 3),
                   floors=round(floors) if floors is not None else None,
                   distance_m=round(distance, 1) if distance is not None else None, signs=signs,
                   reasons=[*near.reasons, f"{len(agree)} of {len(verdicts)} answers agreed"])


def merge_sides(views: list):
    """One side from several: the most named; confidence = its share x their mean confidence."""
    if len(views) == 1:
        return views[0]
    side, share = majority([v.side for v in views])
    agree = [v for v in views if v.side == side]
    best = max(agree, key=lambda v: v.confidence)
    return replace(best, confidence=round(share * statistics.mean(v.confidence for v in agree), 3))


def merge_props(views: list):
    """One object from several: the most given name, the middle of each size, the majority of each yes/no."""
    if len(views) == 1:
        return views[0]
    name, share = majority([v.name.lower() if v.name else None for v in views])
    agree = [v for v in views if (v.name.lower() if v.name else None) == name] or views
    fields = {f: middle([getattr(v, f) for v in agree]) for f in ("width", "depth", "height", "distance")}
    flags = {f: majority([getattr(v, f) for v in agree])[0] for f in ("fully_visible", "round", "symmetric")}
    return replace(agree[0], **fields, **flags,
                   confidence=round(share * statistics.mean(v.confidence for v in agree), 3))


def merge_looks(looks: list):
    """One set of colours from several: the middle of each channel."""
    if len(looks) == 1:
        return looks[0]
    first = looks[0]
    return replace(first, **{f: tuple(statistics.median(getattr(lk, f)[c] for lk in looks)
                                      for c in range(len(getattr(first, f))))
                             for f in first.__dataclass_fields__})
