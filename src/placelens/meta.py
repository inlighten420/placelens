"""Meta Model API: SAM 3.1 cut-outs and Muse questions about a photo, with a running cost tally.

The API is OpenAI-compatible (https://dev.meta.ai/docs/quickstart). SAM's masks come back in Meta's own compact
format; we decode them with Meta's `meta-sam-parser` rather than by hand.

Muse Spark Open (the open-weights Spark) is served by Baseten, also OpenAI-compatible but chat completions only,
with its own key (BASETEN_API_KEY). It always reasons; only its final answer is returned.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format
from PIL import Image

BASE_URL = "https://api.meta.ai/v1"
SAM = "sam-3.1"
SPARK = "muse-spark-1.3"
WONDER = "muse-wonder-1.0"
IMAGE = "muse-image-1.0"
OPEN = "meta-models/muse-spark-open-1026"  # the hackathon's id; Baseten's docs say muse-spark-open
OPEN_URL = "https://inference.baseten.co/v1"
MODELS = {"spark": SPARK, "wonder": WONDER, "open": OPEN}  # --model choices
OPEN_USAGE = Path("out/open_usage.jsonl")  # one line per Spark Open call: what the Baseten credits went on

# USD per 1M tokens (input, output), from dev.meta.ai/docs/pricing-rate-limits, 26 Oct 2026.
# OPEN isn't listed: it runs on Baseten's hackathon credits ($100), so it counts as 0 here.
TOKEN_PRICES = {SPARK: (1.25, 4.25), WONDER: (4.00, 20.00)}
SAM_PER_IMAGE = 2.50 / 1000
IMAGE_PER_IMAGE = 0.01  # Muse Image, per picture made (developer.meta.com, Muse Image launch)


class MissingKeyError(RuntimeError):
    pass


class PaidCallBlocked(RuntimeError):
    """PLACELENS_META_OFF=1: a call to the paid Meta API was about to be made (only saved replies may be used)."""


def _paid(what: str) -> None:
    if os.environ.get("PLACELENS_META_OFF") == "1":
        raise PaidCallBlocked(f"{what} would call the paid Meta API (PLACELENS_META_OFF=1); only saved replies allowed")


def load_key(env_file: str | Path = ".env", name: str = "MODEL_API_KEY") -> str:
    """`name` from the environment, else from a .env file (gitignored)."""
    key = os.environ.get(name)
    if not key and Path(env_file).exists():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise MissingKeyError(f"no {name}: put {name}=... in .env (gitignored)")
    return key


def final_answer(text: str) -> str:
    """The answer without any <think>...</think> reasoning a model left in it."""
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


def data_url(path: str | Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


@dataclass
class Segment:
    """One SAM match. `mask` is full-resolution over `box` (left, top, right, bottom in photo pixels)."""

    object_id: str
    box: tuple[int, int, int, int]
    mask: np.ndarray  # bool, shape (bottom - top, right - left)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def full_mask(self, size: tuple[int, int]) -> np.ndarray:
        """The mask placed on a blank canvas the size of the photo (width, height)."""
        left, top, right, bottom = self.box
        canvas = np.zeros((size[1], size[0]), bool)
        canvas[top:bottom, left:right] = self.mask
        return canvas


def parse_segments(raw: str) -> list[Segment]:
    """SAM's output text to segments, biggest first."""
    parser = image_segmentation_format().create_parser()
    parser.push(raw)
    result = parser.finish(CompletedOutcome()).result
    segments = []
    for r in result.records:
        if r.kind != "mask":
            continue
        m, b = r.mask, r.bounds
        raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
        box = (b.left, b.top, b.right, b.bottom)
        size = (b.right - b.left, b.bottom - b.top)
        if (m.width, m.height) != size:  # big objects come back scaled down
            raster = np.asarray(Image.fromarray(raster * 255).resize(size, Image.NEAREST))
        segments.append(Segment(str(r.object_id), box, raster > 0))
    segments.sort(key=lambda s: -s.area)
    return segments


class MetaClient:
    def __init__(self, api_key: str | None = None):
        from openai import OpenAI

        self.key = api_key or load_key()
        self.client = OpenAI(base_url=BASE_URL, api_key=self.key)
        self.spent = 0.0  # USD, estimated from published prices
        self.open_tokens = 0  # Spark Open tokens (in + out), paid from Baseten credits
        self._open = None  # Baseten client, made on first use so its key is only needed then
        self._lock = threading.Lock()  # muse.ask votes in parallel

    def _add(self, usd: float = 0.0, tokens: int = 0) -> None:
        with self._lock:
            self.spent += usd
            self.open_tokens += tokens

    def open_client(self):
        with self._lock:
            if self._open is None:
                from openai import OpenAI

                self._open = OpenAI(base_url=OPEN_URL, api_key=load_key(name="BASETEN_API_KEY"))
        return self._open

    def segment(self, photo: str | Path, prompt: str = "building") -> tuple[list[Segment], str]:
        """Every match for a short noun phrase. Also returns SAM's raw text, worth keeping for debugging."""
        _paid(f"SAM 3.1 ({prompt!r})")
        stream = self.client.responses.create(
            model=SAM,
            input=[{"role": "user", "content": [
                {"type": "input_image", "image_url": data_url(photo)},
                {"type": "input_text", "text": prompt},
            ]}],
            metadata={"mask_encoding": "one_bit"},
            stream=True,
        )
        raw = "".join(e.delta for e in stream if e.type == "response.output_text.delta")
        self._add(SAM_PER_IMAGE)
        return parse_segments(raw), raw

    def draw(self, prompt: str, photos: list[str | Path]) -> bytes:
        """A new picture from Muse Image, made from up to 10 photos (called @Image1, @Image2... in the prompt).
        Returns the picture's bytes (WebP or PNG)."""
        _paid("Muse Image")
        import requests

        files = [(f"image[{i}]", (Path(p).name, Path(p).read_bytes(), mimetypes.guess_type(str(p))[0] or "image/jpeg"))
                 for i, p in enumerate(photos)]
        r = requests.post(f"{BASE_URL}/images/edits", headers={"Authorization": f"Bearer {self.key}"},
                          data={"model": IMAGE, "prompt": prompt}, files=files, timeout=300)
        r.raise_for_status()
        self._add(IMAGE_PER_IMAGE)
        return base64.b64decode(r.json()["data"][0]["b64_json"])

    def ask(self, prompt: str, photo: str | Path | list | None = None, model: str = SPARK,
            effort: str | None = None) -> str:
        """A question, optionally about a photo or several (in order). Returns the model's text. `effort` ("low",
        "medium", "high"): how long Spark Open thinks first; the Meta API models ignore it."""
        if model == OPEN:
            return self.ask_open(prompt, photo, effort)
        _paid(model)
        if photo is None:
            content = prompt
        else:
            photos = photo if isinstance(photo, list) else [photo]
            content = [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                *({"type": "input_image", "image_url": data_url(p)} for p in photos),
            ]}]
        r = self.client.responses.create(model=model, input=content)
        price_in, price_out = TOKEN_PRICES.get(model, (0, 0))
        self._add((r.usage.input_tokens * price_in + r.usage.output_tokens * price_out) / 1e6)
        return r.output_text

    def ask_open(self, prompt: str, photo: str | Path | list | None = None, effort: str | None = None) -> str:
        """`ask` for Muse Spark Open on Baseten (chat completions; images as image_url parts)."""
        photos = [] if photo is None else photo if isinstance(photo, list) else [photo]
        content = [{"type": "text", "text": prompt},
                   *({"type": "image_url", "image_url": {"url": data_url(p)}} for p in photos)]
        r = self.open_client().chat.completions.create(
            model=OPEN, messages=[{"role": "user", "content": content if photos else prompt}],
            **({"reasoning_effort": effort} if effort else {}))
        if r.usage:
            self._add(tokens=r.usage.total_tokens)
            self.log_usage(r.usage, len(photos), effort)
        return final_answer(r.choices[0].message.content or "")

    def log_usage(self, usage, photos: int, effort: str | None) -> None:
        """One line in OPEN_USAGE for a Spark Open call."""
        import json
        import time

        line = json.dumps({"time": round(time.time()), "in": usage.prompt_tokens, "out": usage.completion_tokens,
                           "photos": photos, "effort": effort})
        with self._lock:
            OPEN_USAGE.parent.mkdir(parents=True, exist_ok=True)
            with OPEN_USAGE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
