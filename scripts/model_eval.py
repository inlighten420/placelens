"""Muse Spark vs Muse Spark Open, from the answers already saved by pick/build (no API calls).

    .venv/Scripts/python scripts/model_eval.py [replies folder]    # default: out/pick
    .venv/Scripts/python scripts/model_eval.py --markdown           # a table to paste

Per photo and model: the building picked, confidence, floors, distance; the side named; the roof read. For Spark Open
the answers are first merged as `pick`/`build` merge them, and "agree" says how many of its votes agreed.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

from placelens.muse import majority, merge_sides, merge_verdicts, middle
from placelens.pick import parse_verdict
from placelens.roof import parse_roof
from placelens.sides import parse_side

NAME = re.compile(r"^(?P<stem>.+?)_(?P<model>spark|wonder|open)_(?:(?P<kind>side|roof)_)?(?P<hash>[0-9a-f]{8})"
                  r"(?:_v(?P<vote>\d+))?\.txt$")


def latest_groups(folder: Path) -> dict:
    """{(stem, model, kind): [reply texts]}, keeping only the newest prompt's replies for each."""
    groups = defaultdict(lambda: defaultdict(list))
    for f in folder.glob("*.txt"):
        m = NAME.match(f.name)
        if m:
            key = (m["stem"], m["model"], m["kind"] or "pick")
            groups[key][m["hash"]].append(f)
    out = {}
    for key, by_hash in groups.items():
        files = max(by_hash.values(), key=lambda fs: max(f.stat().st_mtime for f in fs))
        out[key] = [f.read_text(encoding="utf-8") for f in sorted(files)]
    return out


def pick_cell(texts: list[str]) -> str:
    verdicts = [parse_verdict(t) for t in texts]
    v = merge_verdicts(verdicts)
    agree = f" {sum(x.choice == v.choice for x in verdicts)}/{len(verdicts)}" if len(verdicts) > 1 else ""
    dist = f"{v.distance_m:g} m" if isinstance(v.distance_m, (int, float)) else "?"
    return f"{v.choice or 'none'} {v.confidence:.0%}{agree}, {v.floors or '?'} fl, {dist}"


def side_cell(texts: list[str]) -> str:
    views = [parse_side(t) for t in texts]
    v = merge_sides(views)
    agree = f" {sum(x.side == v.side for x in views)}/{len(views)}" if len(views) > 1 else ""
    return f"{v.side or '?'} {v.confidence:.0%}{agree}"


def roof_cell(texts: list[str]) -> str:
    views = [parse_roof(t) for t in texts]
    kind, share = majority([v.kind for v in views])
    pitch = middle([v.pitch for v in views])
    return f"{kind or '?'} {pitch:.0f}°" if pitch is not None else f"{kind or '?'}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    groups = latest_groups(Path(args[0] if args else "out/pick"))
    cells = {"pick": pick_cell, "side": side_cell, "roof": roof_cell}
    stems = sorted({k[0] for k in groups})
    head = ["photo"] + [f"{kind} {model}" for kind in cells for model in ("spark", "open")]
    rows = [[stem] + [cells[kind](groups[(stem, model, kind)]) if (stem, model, kind) in groups else "-"
                      for kind in cells for model in ("spark", "open")] for stem in stems]
    if "--markdown" in sys.argv:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for r in rows:
            print("| " + " | ".join(r) + " |")
        return
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows, strict=True)]
    for r in [head, *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths, strict=True)))


if __name__ == "__main__":
    main()
