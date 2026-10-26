"""Meta Model API: SAM 3.1 cut-outs and Muse questions about a photo, with a running cost tally.

The API is OpenAI-compatible (https://dev.meta.ai/docs/quickstart). SAM's masks come back in Meta's own compact
format; we decode them with Meta's `meta-sam-parser` rather than by hand.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format
from PIL import Image

BASE_URL = "https://api.meta.ai/v1"
SAM = "sam-3.1"
SPARK = "muse-spark-1.3"
WONDER = "muse-wonder-1.0"

# USD per 1M tokens (input, output), from dev.meta.ai/docs/pricing-rate-limits, 26 Oct 2026
TOKEN_PRICES = {SPARK: (1.25, 4.25), WONDER: (4.00, 20.00)}
SAM_PER_IMAGE = 2.50 / 1000


class MissingKeyError(RuntimeError):
    pass


def load_key(env_file: str | Path = ".env") -> str:
    """MODEL_API_KEY from the environment, else from a .env file (gitignored)."""
    key = os.environ.get("MODEL_API_KEY")
    if not key and Path(env_file).exists():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            if line.startswith("MODEL_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise MissingKeyError("no MODEL_API_KEY: put MODEL_API_KEY=... in .env (gitignored)")
    return key


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

        self.client = OpenAI(base_url=BASE_URL, api_key=api_key or load_key())
        self.spent = 0.0  # USD, estimated from published prices

    def segment(self, photo: str | Path, prompt: str = "building") -> tuple[list[Segment], str]:
        """Every match for a short noun phrase. Also returns SAM's raw text, worth keeping for debugging."""
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
        self.spent += SAM_PER_IMAGE
        return parse_segments(raw), raw

    def ask(self, prompt: str, photo: str | Path | None = None, model: str = SPARK) -> str:
        """A question, optionally about a photo. Returns the model's text."""
        if photo is None:
            content = prompt
        else:
            content = [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url(photo)},
            ]}]
        r = self.client.responses.create(model=model, input=content)
        price_in, price_out = TOKEN_PRICES.get(model, (0, 0))
        self.spent += (r.usage.input_tokens * price_in + r.usage.output_tokens * price_out) / 1e6
        return r.output_text
