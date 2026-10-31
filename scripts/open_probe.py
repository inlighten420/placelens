"""What Muse Spark Open on Baseten supports: JSON mode, tool calls, reasoning effort, several photos, parallel calls.

    .venv/Scripts/python scripts/open_probe.py [photo]   # default: cache/samples/city_hall.jpg

Reads BASETEN_API_KEY from the environment or .env. Writes each raw reply to out/experiments/open_probe/.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from placelens.meta import OPEN, OPEN_URL, data_url, load_key

OUT = Path("out/experiments/open_probe")


def probe(label: str, fn):
    t = time.time()
    try:
        r = fn()
    except Exception as e:  # keep going: we want to know about every feature
        print(f"[{label}] NO  {type(e).__name__}: {str(e)[:200]}")
        return None
    (OUT / f"{label}.json").write_text(r.model_dump_json(indent=1), encoding="utf-8")
    m = r.choices[0].message
    extra = {k: v for k, v in (m.model_extra or {}).items() if v}
    print(f"[{label}] ok  {time.time() - t:.1f}s  usage {r.usage.prompt_tokens}/{r.usage.completion_tokens}")
    print(f"         content: {(m.content or '')[:160]!r}")
    if m.tool_calls:
        print(f"         tool_calls: {[(c.function.name, c.function.arguments) for c in m.tool_calls]}")
    for k, v in extra.items():
        print(f"         {k}: {str(v)[:160]!r}")
    return r


def main():
    photo = Path(sys.argv[1] if len(sys.argv) > 1 else "cache/samples/city_hall.jpg")
    OUT.mkdir(parents=True, exist_ok=True)
    c = OpenAI(base_url=OPEN_URL, api_key=load_key(name="BASETEN_API_KEY"))

    def chat(content, **kw):
        return c.chat.completions.create(model=OPEN, messages=[{"role": "user", "content": content}], **kw)

    img = {"type": "image_url", "image_url": {"url": data_url(photo)}}
    probe("plain", lambda: chat("Reply with exactly: PlaceLens OK"))
    probe("json_object", lambda: chat('Give {"floors": int} for a 2-storey house. JSON only.',
                                      response_format={"type": "json_object"}))
    schema = {"type": "object", "properties": {"floors": {"type": "integer"}}, "required": ["floors"]}
    probe("json_schema", lambda: chat("How many floors does a 2-storey house have?", response_format={
        "type": "json_schema", "json_schema": {"name": "floors", "schema": schema, "strict": True}}))
    tool = {"type": "function", "function": {"name": "pick_building", "description": "Pick a building by number.",
                                             "parameters": {"type": "object", "properties": {"n": {"type": "integer"}},
                                                            "required": ["n"]}}}
    probe("tools", lambda: chat("Building 3 is the one. Call pick_building for it.", tools=[tool]))
    for effort in ("low", "high"):
        probe(f"effort_{effort}", lambda e=effort: chat("Is 91 prime? One word.", reasoning_effort=e))
    probe("two_images", lambda: chat([{"type": "text", "text": "Are these two photos the same? One word."}, img, img]))

    t = time.time()
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda _: chat([{"type": "text", "text": "Floors of the main building? Number only."},
                                                img]), range(8)))
    answers = [r.choices[0].message.content.strip() for r in replies]
    print(f"[parallel_8] {time.time() - t:.1f}s for 8: {answers}")
    (OUT / "parallel_8.json").write_text(json.dumps(answers), encoding="utf-8")


if __name__ == "__main__":
    main()
