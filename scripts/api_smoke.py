"""One call to each Meta model we depend on: does the key work, what comes back, what does it cost.

    .venv/Scripts/python scripts/api_smoke.py cache/samples/city_hall.jpg

    .venv/Scripts/python scripts/api_smoke.py cache/samples/city_hall.jpg --redraw   # re-parse saved SAM reply

Reads MODEL_API_KEY from the environment or .env. Writes raw replies to out/api_smoke/.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format
from openai import OpenAI
from PIL import Image, ImageDraw

BASE_URL = "https://api.meta.ai/v1"
OUT = Path("out/api_smoke")
# USD per 1M tokens (input, output), from dev.meta.ai/docs/pricing-rate-limits, 26 Oct 2026
PRICES = {"muse-spark-1.3": (1.25, 4.25), "muse-wonder-1.0": (4.00, 20.00)}
SAM_PER_IMAGE = 2.50 / 1000


def load_key() -> str:
    if "MODEL_API_KEY" not in os.environ and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("MODEL_API_KEY="):
                os.environ["MODEL_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        sys.exit("No MODEL_API_KEY: put MODEL_API_KEY=... in .env (gitignored)")
    return key


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def cost(model: str, usage) -> float:
    pin, pout = PRICES[model]
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6


def test_spark(client: OpenAI) -> float:
    t = time.time()
    r = client.responses.create(model="muse-spark-1.3", input="Reply with exactly: PlaceLens OK")
    c = cost("muse-spark-1.3", r.usage)
    print(f"[spark]  {r.output_text.strip()!r}  {time.time() - t:.1f}s  ${c:.5f}")
    return c


def test_wonder(client: OpenAI, photo: Path) -> float:
    prompt = (
        "This photo shows a building from the street. Answer in JSON only: "
        '{"name": name if you recognise it or null, "floors": visible floors of the main building (int), '
        '"signs": [text of street signs or name boards you can read], '
        '"occluders": [things standing in front of the facade, e.g. lamp post, tree, car]}'
    )
    t = time.time()
    r = client.responses.create(
        model="muse-wonder-1.0",
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": data_url(photo)},
        ]}],
    )
    c = cost("muse-wonder-1.0", r.usage)
    (OUT / "wonder.txt").write_text(r.output_text, encoding="utf-8")
    print(f"[wonder] {time.time() - t:.1f}s  ${c:.5f}  in={r.usage.input_tokens} out={r.usage.output_tokens}")
    print("         " + r.output_text.strip().replace("\n", "\n         "))
    return c


def overlay_sam(raw: str, photo: Path) -> int:
    """Parse SAM's reply with Meta's own parser and draw each match on the photo. Returns the number of masks."""
    parser = image_segmentation_format().create_parser()
    parser.push(raw)
    result = parser.finish(CompletedOutcome()).result
    for d in result.diagnostics:
        print(f"         diagnostic: {d}")

    img = Image.open(photo).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    masks = 0
    for r in result.records:
        if r.kind == "box":
            print(f"         #{r.object_id}: box ({r.left},{r.top})-({r.right},{r.bottom})")
        elif r.kind == "mask":
            m, b = r.mask, r.bounds
            # A mask covers its own bounds, scaled down for big objects.
            raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
            alpha = Image.fromarray((raster > 0).astype(np.uint8) * 110).resize((b.right - b.left, b.bottom - b.top))
            colour = (255, 40, 40) if masks % 2 == 0 else (40, 120, 255)
            img.paste(Image.new("RGB", alpha.size, colour), (b.left, b.top), alpha)
            draw.rectangle([b.left, b.top, b.right - 1, b.bottom - 1], outline=colour + (255,), width=3)
            print(f"         #{r.object_id}: mask {m.width}x{m.height} ({m.encoding}), {raster.mean():.0%} of its box")
            masks += 1
    img.save(OUT / "sam_overlay.jpg")
    print(f"         overlay: {OUT / 'sam_overlay.jpg'}")
    return masks


def test_sam(client: OpenAI, photo: Path, prompt: str = "building") -> float:
    t = time.time()
    text = []
    stream = client.responses.create(
        model="sam-3.1",
        input=[{"role": "user", "content": [
            {"type": "input_image", "image_url": data_url(photo)},
            {"type": "input_text", "text": prompt},
        ]}],
        metadata={"mask_encoding": "one_bit"},
        stream=True,
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            text.append(event.delta)
    raw = "".join(text)
    (OUT / "sam_raw.txt").write_text(raw, encoding="utf-8")
    print(f"[sam]    {time.time() - t:.1f}s  ${SAM_PER_IMAGE:.5f}  {len(raw)} chars  (raw: {OUT / 'sam_raw.txt'})")
    overlay_sam(raw, photo)
    return SAM_PER_IMAGE


def main():
    photo = Path(sys.argv[1] if len(sys.argv) > 1 else "cache/samples/city_hall.jpg")
    OUT.mkdir(parents=True, exist_ok=True)
    if "--redraw" in sys.argv:  # re-parse the saved SAM reply, no API call
        overlay_sam((OUT / "sam_raw.txt").read_text(encoding="utf-8"), photo)
        return
    client = OpenAI(base_url=BASE_URL, api_key=load_key())
    total = 0.0
    for name, fn in (("spark", lambda: test_spark(client)),
                     ("sam", lambda: test_sam(client, photo)),
                     ("wonder", lambda: test_wonder(client, photo))):
        try:
            total += fn()
        except Exception as e:  # keep going: we want to know about every model
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
    print(f"total ~${total:.4f}")


if __name__ == "__main__":
    main()
