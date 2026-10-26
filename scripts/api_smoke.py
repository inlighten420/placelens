"""One call to each Meta model we depend on: does the key work, what comes back, what does it cost.

    .venv/Scripts/python scripts/api_smoke.py [photo]            # default: cache/samples/city_hall.jpg
    .venv/Scripts/python scripts/api_smoke.py [photo] --redraw   # re-draw the saved SAM reply, no API call

Reads MODEL_API_KEY from the environment or .env. Writes replies and an overlay picture to out/api_smoke/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from placelens.meta import SPARK, WONDER, MetaClient, Segment, parse_segments

OUT = Path("out/api_smoke")
COLOURS = [(255, 40, 40), (40, 120, 255), (250, 180, 0), (60, 170, 80)]

WONDER_PROMPT = (
    "This photo shows a building from the street. Answer in JSON only: "
    '{"name": name if you recognise it or null, "floors": visible floors of the main building (int), '
    '"signs": [text of street signs or name boards you can read], '
    '"occluders": [things standing in front of the facade, e.g. lamp post, tree, car]}'
)


def draw_segments(segments: list[Segment], photo: Path) -> Path:
    img = Image.open(photo).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(segments):
        colour = COLOURS[i % len(COLOURS)]
        alpha = Image.fromarray(s.full_mask(img.size).astype("uint8") * 110)
        img.paste(Image.new("RGB", img.size, colour), (0, 0), alpha)
        draw.rectangle([s.box[0], s.box[1], s.box[2] - 1, s.box[3] - 1], outline=colour, width=3)
        print(f"         #{s.object_id}: box {s.box}, {s.area / img.width / img.height:.0%} of the photo")
    img.save(OUT / "sam_overlay.jpg")
    print(f"         overlay: {OUT / 'sam_overlay.jpg'}")
    return OUT / "sam_overlay.jpg"


def timed(label: str, meta: MetaClient, fn):
    before, t = meta.spent, time.time()
    try:
        result = fn()
    except Exception as e:  # keep going: we want to know about every model
        print(f"[{label}] FAILED: {type(e).__name__}: {e}")
        return None
    print(f"[{label}] {time.time() - t:.1f}s  ${meta.spent - before:.5f}")
    return result


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    photo = Path(args[0] if args else "cache/samples/city_hall.jpg")
    OUT.mkdir(parents=True, exist_ok=True)
    if "--redraw" in sys.argv:
        draw_segments(parse_segments((OUT / "sam_raw.txt").read_text(encoding="utf-8")), photo)
        return

    meta = MetaClient()
    reply = timed("spark ", meta, lambda: meta.ask("Reply with exactly: PlaceLens OK", model=SPARK))
    if reply is not None:
        print(f"         {reply.strip()!r}")

    result = timed("sam   ", meta, lambda: meta.segment(photo, "building"))
    if result is not None:
        segments, raw = result
        (OUT / "sam_raw.txt").write_text(raw, encoding="utf-8")
        draw_segments(segments, photo)

    reply = timed("wonder", meta, lambda: meta.ask(WONDER_PROMPT, photo, model=WONDER))
    if reply is not None:
        (OUT / "wonder.txt").write_text(reply, encoding="utf-8")
        print("         " + reply.strip().replace("\n", "\n         "))

    print(f"total ~${meta.spent:.4f}")


if __name__ == "__main__":
    main()
