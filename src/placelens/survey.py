"""`placelens survey FOLDER`: an agent that turns a folder of phone photos into a place pack by itself.

Muse Spark Open (on Baseten) runs the loop with tool calls. Its tools are PlaceLens's own commands, run as
subprocesses with --json so each step stays exactly what a person would run (and a crash in one can't take the
agent down), plus a quick look at a photo. It groups the photos (one building, one object walked round, ...), builds
each group, gathers the results into a pack, and asks the person only where it stays unsure: a building `pick`
couldn't decide, photos it can't place.

The person can add notes (`--notes`): what the photos can't show ("the back is the same as the front", "about 20 m
wide"). The agent turns what it can into the tools' settings and passes the notes on, so Muse reads them with every
photo.

Everything it does is saved in out/survey/<folder>/: transcript.json (every message and tool result) and
summary.md, written for a person. Its commands write into out/survey/<folder>/work/, never over other builds.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
MAX_STEPS = 15  # heavy tool calls (HEAVY) before the agent has to stop
MAX_CALLS = 80  # all tool calls, looks included: a backstop
HEAVY = {"pick", "build", "prop", "pack"}  # each runs SAM, Muse and geometry, and can take minutes
NEAR = 40.0  # metres: photos closer together than this, and ...
SOON = 2 * 3600  # ... taken within this many seconds of each other, are offered as one group (walking round a
# house took the user 20 minutes; a tighter window split it in two and the agent built two buildings)
LOOKS = 10  # describe calls allowed: the agent liked to look at every photo
RESULT_CHARS = 4000  # a tool's answer is cut to this before the agent sees it (the full one is in the transcript)

SYSTEM = """You are PlaceLens's survey agent. A person dropped a folder of phone photos; you turn it into a place pack \
(3D models of the real buildings and objects, placed on the map) using the tools, without asking them anything \
unless you must.

How to work:
1. list_photos first. Photos taken close together in place and time usually show one thing; the groups it offers \
are a starting point, not a rule.
2. describe one or two photos of each group to learn what it is (a building, part of one, or an object), and any \
photo that doesn't seem to fit its group. Not every photo: a group of the same object needs one look. You have a \
few looks, not one per photo.
3. Buildings: one build call with every photo of that building, even if list_photos put them in different groups \
(the front from the street and the back are one building). It picks the building in each photo itself and uses \
the ones that agree, so don't call pick first; pick is only for answering a question about one photo. Use \
refine=true when 3 or more photos see the building from the street.
4. Objects: one prop call per object, with every photo of it (a walk round it carves its real shape). Give `what` \
as describe named it.
5. When a tool fails or asks which building it is, don't retry it with small changes and don't guess: note it with \
ask_person and carry on with the rest.
6. If the person left notes, they know things the photos can't show. Turn what you can into the tools' arguments \
(a size, a floor count, "round", "symmetric" or "the other side is the same", what the object is) and pass the \
notes about each thing on in its call's `notes`. When a result disagrees with them (a building 12 m wide they said \
is 20 m), say so in the summary; don't fight the map.
7. pack everything built, then finish with a short summary for the person: what was built, what wasn't and why, \
and your questions. Only facts from the tools' results; give numbers with the units they have there.

Heavy tools (pick, build, prop, pack) take minutes each, and you have a budget of them; looks are cheap. Never run \
the same heavy call twice."""


TOOLS = [
    {"type": "function", "function": {
        "name": "list_photos",
        "description": "Every photo in the folder: GPS, compass, time, and suggested groups (close in place and time).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "describe",
        "description": "Look at one photo: is it a building, part of one (which part), or an object (what)?",
        "parameters": {"type": "object", "properties": {"photo": {"type": "string"}}, "required": ["photo"]}}},
    {"type": "function", "function": {
        "name": "pick",
        "description": "Which mapped building one photo shows (SAM 3.1 outline + Muse + map geometry).",
        "parameters": {"type": "object", "properties": {"photo": {"type": "string"}}, "required": ["photo"]}}},
    {"type": "function", "function": {
        "name": "build",
        "description": "Build one building from all its photos as a 3D model on the map. Takes minutes.",
        "parameters": {"type": "object", "properties": {
            "photos": {"type": "array", "items": {"type": "string"}},
            "refine": {"type": "boolean", "description": "critique loop: corrects the shape against the photos"},
            "building": {"type": "integer",
                         "description": "only with ONE photo: a candidate number from that photo's pick, if the "
                                        "person chose it"},
            "floors": {"type": "integer", "description": "only if the person said how many floors it has"},
            "height": {"type": "number", "description": "metres to the eaves, only if the person said"},
            "symmetric": {"type": "boolean",
                          "description": "the person said its far side matches the near side: walls no photo saw "
                                         "copy the opposite wall"},
            "notes": {"type": "string", "description": "the person's notes about this building, as they wrote them"}},
            "required": ["photos"]}}},
    {"type": "function", "function": {
        "name": "prop",
        "description": "Build an object (bench, tank, statue...) from one photo or a walk round it, at real size.",
        "parameters": {"type": "object", "properties": {
            "photos": {"type": "array", "items": {"type": "string"}},
            "what": {"type": "string", "description": "short name of the object, e.g. 'rain tank'"},
            "height": {"type": "number", "description": "its height in metres, only if the person said"},
            "width": {"type": "number", "description": "its width in metres, only if the person said"},
            "round": {"type": "boolean", "description": "the person said it's round (a tank, pot, pole)"},
            "symmetric": {"type": "boolean", "description": "the person said its left and right match"},
            "notes": {"type": "string", "description": "the person's notes about this object, as they wrote them"}},
            "required": ["photos"]}}},
    {"type": "function", "function": {
        "name": "pack",
        "description": "Gather the built folders (from build and prop results) into one place pack.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "folders": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "folders"]}}},
    {"type": "function", "function": {
        "name": "ask_person",
        "description": "A question only the person can answer (which building, which photos belong together). "
                       "Saved for them; carry on with everything else.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
]

DESCRIBE_PROMPT = """What is this photo of? Answer in JSON only:
{"kind": "building", "building part" or "object", \
"what": a short name (e.g. "two-storey house", "back of a house", "rain tank", "teddy bear"), \
"side": for a building or part, which side the camera faces if you can tell ("front", "back", "left", "right", \
"corner") else null, \
"whole": true if the whole building or object is in the photo}"""


def photo_time(path: Path) -> float | None:
    """When the photo was taken (seconds, from EXIF), or None."""
    from datetime import datetime

    from PIL import Image

    try:
        with Image.open(path) as img:
            stamp = img.getexif().get_ifd(0x8769).get(36867) or img.getexif().get(306)
        return datetime.strptime(str(stamp), "%Y:%m:%d %H:%M:%S").timestamp() if stamp else None
    except Exception:  # unreadable EXIF: the agent goes by place and its own look
        return None


def groups(photos: list[dict]) -> list[list[str]]:
    """Photos in runs that stay within NEAR metres and SOON seconds of the run's previous photo (in time order)."""
    ordered = sorted(photos, key=lambda p: (p["time"] is None, p["time"] or 0, p["photo"]))
    out: list[list[dict]] = []
    for p in ordered:
        last = out[-1][-1] if out else None
        if last and _near(last, p) and _soon(last, p):
            out[-1].append(p)
        else:
            out.append([p])
    return [[p["photo"] for p in g] for g in out]


def _near(a: dict, b: dict) -> bool:
    if None in (a["lat"], a["lon"], b["lat"], b["lon"]):
        return True
    dy = (a["lat"] - b["lat"]) * 111_320
    dx = (a["lon"] - b["lon"]) * 111_320 * math.cos(math.radians(a["lat"]))
    return math.hypot(dx, dy) <= NEAR


def _soon(a: dict, b: dict) -> bool:
    return a["time"] is None or b["time"] is None or abs(a["time"] - b["time"]) <= SOON


def list_photos(folder: Path) -> dict:
    from .photo import read_camera

    photos = []
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_TYPES):
        cam = read_camera(path)
        photos.append({"photo": str(path), "lat": cam.lat, "lon": cam.lon,
                       "compass": round(cam.heading) if cam.heading is not None else None,
                       "lens_deg": round(cam.fov), "time": photo_time(path)})
    return {"photos": photos, "groups": groups(photos)}


@dataclass
class Survey:
    """The agent loop. `chat(messages, tools) -> message` is the model; `run(argv) -> dict` runs a placelens command
    (--json); `describe(photo) -> str` looks at a photo. All three are swapped out in tests."""

    folder: Path
    chat: Callable
    run: Callable[[list[str]], dict]
    describe: Callable[[str], str]
    out: Path
    log: Callable = print
    messages: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    built: list = field(default_factory=list)  # (kind, folder, what)
    pack: str | None = None
    looks: int = 0
    replies: Path = Path("out/pick")
    notes: str | None = None  # the person's own words about the photos
    progress: Callable[[dict], None] = lambda event: None  # each tool call as it starts and ends (for a game)

    @property
    def work(self) -> Path:
        """Where the agent's commands write (--out): its own, so it can't overwrite a build made by hand."""
        return self.out / "work"

    def call(self, name: str, args: dict) -> dict:
        """One tool call. Errors come back as {"error": ...} for the agent to read, not as exceptions."""
        if name == "list_photos":
            return list_photos(self.folder)
        if name == "describe":
            if self.looks >= LOOKS:
                return {"error": f"no looks left ({LOOKS} used): decide from what you've seen"}
            self.looks += 1
            return {"photo": args["photo"], "answer": self.describe(args["photo"])}
        out = ["--out", str(self.work)]
        shared = ["--replies", str(self.replies)]  # SAM and Muse answers shared with hand-run commands
        if name == "pick":
            return _brief_pick(self.run(["pick", args["photo"], "--json", *out, *shared]))
        if name == "build":
            argv = ["build", *args["photos"], "--json", *out, *shared]
            if args.get("refine"):
                argv.append("--refine")
            if args.get("building") is not None:
                argv += ["--building", str(args["building"])]
            argv += _settings(args, {"floors": "--floors", "height": "--height", "notes": "--notes"},
                              {"symmetric": "--symmetric"})
            result = _brief_build(self.run(argv))
            if result.get("folder"):
                self.built.append(("building", result["folder"], result.get("building")))
            return result
        if name == "prop":
            argv = ["prop", *args["photos"], "--json", *out]
            argv += _settings(args, {"what": "--what", "height": "--height", "width": "--width", "notes": "--notes"},
                              {"symmetric": "--symmetric"})
            if args.get("round"):
                argv += ["--shape", "round"]
            result = _brief_prop(self.run(argv))
            if result.get("folder"):
                self.built.append(("prop", result["folder"], result.get("name")))
            return result
        if name == "pack":
            result = self.run(["pack", args["name"], *args["folders"], "--json", *out])
            if "error" not in result:
                self.pack = args["name"]
            return result
        if name == "ask_person":
            self.questions.append(args["question"])
            return {"saved": True, "note": "the person will see this; carry on with the rest"}
        return {"error": f"no tool {name!r}"}

    def go(self, max_steps: int = MAX_STEPS) -> str:
        """Run until the agent answers without a tool call (its summary) or runs out of steps."""
        brief = f"The folder: {self.folder}. Make its place pack."
        if self.notes:
            brief += f'\n\nThe person\'s notes: "{self.notes.strip()}"'
        self.messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": brief}]
        steps = calls_made = 0
        while True:
            reply = self.chat(self.messages, TOOLS)
            calls = reply.get("tool_calls") or []
            self.messages.append(reply)
            if not calls:
                self.pack_rest()
                self.save()
                return reply.get("content") or ""
            for c in calls:
                name = c["function"]["name"]
                steps += name in HEAVY
                calls_made += 1
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args, result = {}, {"error": "arguments weren't valid JSON"}
                else:
                    self.log(f"agent:    {name} {_short(args)}")
                    self.progress({"tool": name, "about": _short(args), "heavy": name in HEAVY})
                    try:
                        result = self.call(name, args)
                    except Exception as e:  # a tool bug is reported to the agent, which can try something else
                        result = {"error": f"{type(e).__name__}: {e}"}
                    self.progress({"tool": name, "ok": "error" not in result,
                                   **({"error": str(result["error"])[:300]} if "error" in result else {})})
                text = json.dumps(result)
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": text[:RESULT_CHARS]})
            self.save()
            if steps >= max_steps or calls_made >= MAX_CALLS:
                packed = self.pack_rest()
                note = f" What was built is packed: {json.dumps(packed)}." if packed else ""
                self.messages.append({"role": "user", "content": f"Out of steps.{note} Give your summary now, "
                                                                 "without calling tools."})
                reply = self.chat(self.messages, [])
                self.messages.append(reply)
                self.save()
                return reply.get("content") or ""

    def pack_rest(self) -> dict | None:
        """If things were built but the agent didn't pack them, pack them (named after the folder)."""
        if self.pack or not self.built:
            return None
        folders = list(dict.fromkeys(folder for _, folder, _ in self.built))
        result = self.run(["pack", self.folder.name, *folders, "--json", "--out", str(self.work)])
        if "error" not in result:
            self.pack = self.folder.name
        return result

    def save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "transcript.json").write_text(json.dumps(self.messages, indent=1), encoding="utf-8")
        lines = [f"# Survey of {self.folder.name}", ""]
        if self.notes:
            lines += [f"Your notes: {self.notes.strip()}", ""]
        final = self.messages[-1].get("content") if self.messages and self.messages[-1].get("role") == "assistant" \
            else None
        if final:
            lines += [final, ""]
        if self.built:
            lines += ["## Built", ""] + [f"- {kind}: {what or '?'} ({folder})" for kind, folder, what in self.built]
            lines.append("")
        if self.pack:
            lines += [f"Pack: `{self.pack}`", ""]
        if self.questions:
            lines += ["## Questions for you", ""] + [f"- {q}" for q in self.questions] + [""]
        (self.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _settings(args: dict, values: dict[str, str], flags: dict[str, str]) -> list[str]:
    """A tool call's optional arguments as command-line options: `values` take their value, `flags` are on if true."""
    argv = []
    for key, opt in values.items():
        if args.get(key) not in (None, ""):
            argv += [opt, str(args[key])]
    return argv + [opt for key, opt in flags.items() if args.get(key)]


def _short(args: dict) -> str:
    """A tool call's arguments for the log: file names, not paths; photos counted."""
    text = ", ".join(f"{k}={Path(v).name if isinstance(v, str) and len(Path(v).parts) > 1 else v}"
                     for k, v in args.items() if k not in ("photos", "notes"))
    if "photos" in args:
        text = f"{len(args['photos'])} photo(s)" + (f", {text}" if text else "")
    return text


def _brief_pick(r: dict) -> dict:
    if "error" in r:
        return r
    b = r.get("building") or {}
    m = r.get("muse") or {}
    return {"status": r.get("status"), "why": r.get("why"), "building": b.get("osm_id"), "name": b.get("name"),
            "muse": {"choice": m.get("choice"), "confidence": m.get("confidence"), "floors": m.get("floors")},
            "candidates": [{"rank": c.get("rank"), "osm_id": c.get("osm_id"), "distance_m": c.get("distance_m")}
                           for c in r.get("candidates", [])]}


def _brief_build(r: dict) -> dict:
    if "error" in r:
        return r
    photos = r.get("photos", [])
    refined = r.get("refined") or []
    return {"building": (r.get("building") or {}).get("osm_id"), "folder": str(Path(r["model"]).parent),
            "size_m": r.get("size_m"), "height_m": r.get("height_m"), "roof": (r.get("roof") or {}).get("kind"),
            "photos_used": [Path(p["photo"]).name for p in photos if p["used"]],
            "photos_not_used": {Path(p["photo"]).name: p["why"] for p in photos if not p["used"]},
            "symmetric": r.get("symmetric"),
            "refined": {"outline_mismatch_before_after": [refined[0]["error"], refined[-1]["error"]],
                        "mismatch_unit": "share of the photo's height, lower is better (not metres)",
                        "kept": [k["edit"] for h in refined for k in h.get("kept", [])]} if refined else None}


def _brief_prop(r: dict) -> dict:
    if "error" in r:
        return r
    return {"name": r.get("name"), "folder": str(Path(r["model"]).parent), "size_m": r.get("size_m"),
            "warnings": r.get("warnings"), "photos": r.get("photos")}


def run_cli(argv: list[str], timeout: int = 1800) -> dict:
    """A placelens command with --json, in its own process; its JSON (or the error)."""
    done = subprocess.run([sys.executable, "-m", "placelens.cli", *argv], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if lines:
        return json.loads(lines[-1])
    return {"error": (done.stderr.strip() or done.stdout.strip() or f"exit {done.returncode}")[-1500:]}


def open_chat(meta) -> Callable:
    """The agent's model: Muse Spark Open on Baseten, with tool calls (Baseten credits, free)."""
    from .meta import OPEN

    def chat(messages: list, tools: list) -> dict:
        r = meta.open_client().chat.completions.create(model=OPEN, messages=messages,
                                                       **({"tools": tools} if tools else {}))
        if r.usage:
            meta._add(tokens=r.usage.total_tokens)
            meta.log_usage(r.usage, 0, None)
        m = r.choices[0].message
        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                 for c in m.tool_calls]
        return msg

    return chat


def open_describe(meta, out: Path) -> Callable[[str], str]:
    """A quick look at a photo with Spark Open (3 votes, saved like every Muse answer)."""
    import hashlib

    from .meta import OPEN
    from .muse import ask
    from .pick import prepare_photo

    def describe(photo: str) -> str:
        stem = Path(photo).stem
        small = prepare_photo(photo, out / "photos" / f"{stem}.jpg")
        tag = hashlib.sha1(DESCRIBE_PROMPT.encode()).hexdigest()[:8]
        answers = ask(meta, DESCRIBE_PROMPT, small, out / "describe" / f"{stem}_open_{tag}.txt", OPEN, votes=3)
        return " | ".join(a.strip() for a in answers)

    return describe
