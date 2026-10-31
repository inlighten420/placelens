"""view.html: the built model in any browser, spin and zoom, with the model embedded so the one file is enough."""

from __future__ import annotations

import base64
import html
from pathlib import Path

MODEL_VIEWER = "https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script type="module" src="{script}"></script>
<style>
  :root {{ --bg: #eef0f2; --ink: #1d2329; --muted: #5b6670; --card: #ffffff; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #15191d; --ink: #e8ecef; --muted: #9aa5ae; --card: #1f252b; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
  model-viewer {{ width: 100%; height: 78vh; background: var(--bg); --poster-color: transparent; }}
  .card {{ max-width: 920px; margin: 0 auto; padding: 12px 16px 24px; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 2px 16px; margin: 8px 0; }}
  dt {{ color: var(--muted); }} dd {{ margin: 0; }}
  small {{ color: var(--muted); }}
</style>
</head>
<body>
<model-viewer src="{model}" alt="{title}" camera-controls touch-action="pan-y" shadow-intensity="1"
  exposure="1.05" camera-orbit="{orbit}" field-of-view="35deg" interaction-prompt="none"></model-viewer>
<div class="card">
  <h1>{title}</h1>
  <dl>{facts}</dl>
  <small>Drag to turn, scroll to zoom. {credit}</small>
</div>
</body>
</html>
"""


def write_viewer(out: Path, glb: Path, title: str, facts: dict[str, str], credit: str, facing: float = 0.0) -> Path:
    """`facing`: the compass bearing the main photo was taken looking along, so the view opens from that side."""
    model = "data:model/gltf-binary;base64," + base64.b64encode(glb.read_bytes()).decode()
    rows = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in facts.items())
    # model-viewer's orbit angle puts the camera at 0deg on +Z (south in our axes) and at 90deg on +X (east). A photo
    # looking along bearing b was taken from the opposite side: b = 0 (north) from the south, 0deg; b = 270 (west)
    # from the east, 90deg. So the angle is -b.
    orbit = f"{-facing % 360:.0f}deg 75deg auto"
    out.write_text(PAGE.format(title=html.escape(title), script=MODEL_VIEWER, model=model, orbit=orbit, facts=rows,
                               credit=html.escape(credit)), encoding="utf-8")
    return out
