"""PlaceLens Studio: the object maker in your browser, no game needed.

`placelens studio` serves one page on this computer only (127.0.0.1): drop photos, say what it is if you like, Make,
then drag the 3D model round to see every side. Each make runs `placelens object` (the same maker the game uses)
in its own process, so it reads the same .env keys and reports the same progress and cost.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PHOTO_TYPES = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")
MAX_PHOTO = 40 * 1024 * 1024  # bytes: a phone photo is a few MB
MAX_PHOTOS = 16
SHOWN = 60  # things listed, newest first


def _safe(name: str) -> str:
    """A photo's file name with nothing that could leave its folder."""
    name = Path(name.replace("\\", "/")).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:80] or "photo.jpg"


class Studio:
    def __init__(self, out: Path):
        self.out = out.resolve()
        self.root = self.out / "studio"
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    # --- making ---------------------------------------------------------------------------------------------------

    def folder(self, job: str) -> Path:
        if not re.fullmatch(r"[a-z0-9]{4,20}", job):
            raise ValueError("bad job")
        return self.root / job

    def upload(self, job: str, name: str, data: bytes) -> str:
        name = _safe(name)
        if Path(name).suffix.lower() not in PHOTO_TYPES:
            raise ValueError(f"{name}: not a photo (JPEG, PNG, HEIC or WebP)")
        photos = self.folder(job) / "photos"
        photos.mkdir(parents=True, exist_ok=True)
        if len(list(photos.iterdir())) >= MAX_PHOTOS:
            raise ValueError(f"at most {MAX_PHOTOS} photos")
        (photos / name).write_bytes(data)
        try:  # (a small upright JPEG for the page: a browser can't show a phone's HEIC)
            _thumb(photos / name, self.folder(job) / "thumbs" / f"{Path(name).stem}.jpg")
        except Exception:
            pass
        return name

    def thumb(self, job: str, name: str) -> Path:
        path = self.folder(job) / "thumbs" / f"{Path(_safe(name)).stem}.jpg"
        if not path.is_file():
            raise ValueError("no such photo")
        return path

    def headings(self, job: str, order: list[str]) -> list[dict]:
        """Each photo's compass heading (degrees, or None), read from the photo itself: no AI, instant. The page
        places the photos round its circle from these."""
        from .objects import _camera

        folder = self.folder(job)
        out = []
        for n in order:
            path = folder / "photos" / _safe(n)
            if not path.is_file():
                raise ValueError("upload the photos first")
            try:
                heading = _camera(path).heading
            except Exception:
                heading = None
            out.append({"name": path.name, "heading": None if heading is None else round(float(heading), 1)})
        return out
    def make(self, job: str, order: list[str], what: str, notes: str, kind: str = "object", moves: int = 3,
             views: dict | None = None) -> None:
        """`kind`: "object" (`placelens object`) or "building" (`placelens build`: GPS, compass and the map)."""
        if kind not in ("object", "building"):
            raise ValueError("an object or a building")
        with self.lock:
            if any(j["running"] for j in self.jobs.values()):
                raise ValueError("one make at a time: wait for this one to finish")
            folder = self.folder(job)
            photos = [folder / "photos" / _safe(n) for n in order]
            if not photos or not all(p.is_file() for p in photos):
                raise ValueError("upload the photos first")
            progress = folder / "progress.jsonl"
            progress.unlink(missing_ok=True)
            command = "object" if kind == "object" else "build"
            cmd = [sys.executable, "-m", "placelens.cli", command, *map(str, photos), "--json", "--out", str(folder)]
            if what.strip() and kind == "object":
                cmd += ["--what", what.strip()[:120]]
            if notes.strip():
                cmd += ["--notes", notes.strip()[:600]]
            if kind == "building" and views:  # (the sides you placed them on: the maker doesn't guess those)
                cmd += [a for name, side in _building_sides(order[0], views).items()
                        for a in ("--side", f"{name}={side}")]
            if kind == "object":
                cmd += ["--moves", str(max(0, min(6, int(moves))))]
                clean = {_safe(k): v for k, v in (views or {}).items()
                         if v == "above" or isinstance(v, (int, float)) and not isinstance(v, bool)}
                if clean:
                    cmd += ["--views", json.dumps(clean)]
            env = {**os.environ, "PLACELENS_PROGRESS": str(progress), "PYTHONIOENCODING": "utf-8"}
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=os.getcwd(),
                                    text=True, encoding="utf-8", errors="replace")
            self.jobs[job] = {"running": True, "started": time.time(), "proc": proc, "progress": progress,
                              "kind": kind, "what": what.strip()[:120]}
            (folder / "request.json").write_text(json.dumps({"photos": [p.name for p in photos], "kind": kind,
                                                             "what": what, "notes": notes, "moves": moves,
                                                             "views": views}),
                                                       encoding="utf-8")
        threading.Thread(target=self._wait, args=(job,), daemon=True).start()

    def _wait(self, job: str) -> None:
        from .meta import usage_since

        j = self.jobs[job]
        stdout, stderr = j["proc"].communicate()
        result: dict = {}
        for line in reversed(stdout.strip().splitlines()):
            try:
                result = json.loads(line)
                break
            except ValueError:
                continue
        if not result:
            tail = [ln for ln in stderr.strip().splitlines() if ln.strip()][-3:]
            result = {"error": " ".join(tail) or f"the maker stopped (exit {j['proc'].returncode})"}
        if "model" in result:
            if j["kind"] == "building":  # (the page gets only what it shows: no map position, no address)
                result = _building(result, j["what"])
            result["id"] = self._id(Path(result["model"]))
            result["seconds"] = result.get("seconds", round(time.time() - j["started"]))
            result.setdefault("cost", usage_since(j["started"]))
            (self.folder(job) / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            if j["kind"] == "building":
                (Path(result["model"]).parent / "thing.json").write_text(json.dumps(result, indent=1),
                                                                         encoding="utf-8")
        elif "error" in result:
            result["error"] = _plain(result["error"])
        j.update(running=False, result=result)

    def status(self, job: str) -> dict:
        j = self.jobs.get(job)
        if not j:
            return {"error": "no such make"}
        events = []
        try:
            for line in j["progress"].read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
        except OSError:
            pass
        return {"running": j["running"], "seconds": round(time.time() - j["started"]), "events": events,
                "result": j.get("result")}

    def remake(self, id_: str, moves: int | None = None) -> str:
        """The make again, in its own folder with its own photos, by the code as it is now: every answer it saved is
        reused (free and quick), so only what the code now does differently is asked again. Returns the job."""
        rel = self.model(id_).relative_to(self.out).parts
        if len(rel) < 4 or rel[0] != "studio":
            raise ValueError("only things made in the studio can be remade here (it keeps their photos)")
        job = rel[1]
        folder = self.folder(job)
        asked = folder / "request.json"
        if asked.is_file():
            r = json.loads(asked.read_text(encoding="utf-8"))
        else:  # (made before the studio kept requests: its photos, the front first)
            names = sorted(p.name for p in (folder / "photos").iterdir() if p.suffix.lower() in PHOTO_TYPES)
            front = rel[3] if rel[2] == "objects" else ""
            names.sort(key=lambda n: Path(n).stem != front)
            r = {"photos": names, "kind": "building" if rel[2] == "build" else "object", "what": "", "notes": ""}
        self.make(job, r["photos"], r.get("what", ""), r.get("notes", ""), r.get("kind", "object"),
                  moves if moves is not None else r.get("moves", 3), r.get("views"))
        return job

    def stop(self) -> None:
        for j in self.jobs.values():
            if j["running"]:
                j["proc"].kill()

    # --- looking --------------------------------------------------------------------------------------------------

    def _id(self, model: Path) -> str:
        return model.resolve().relative_to(self.out).as_posix()

    def _timing(self, folder: Path) -> dict:
        """A studio make's time and cost (its result.json, in studio/<job>/)."""
        rel = folder.relative_to(self.out).parts
        result = self.root / rel[1] / "result.json" if len(rel) > 2 and rel[0] == "studio" else None
        if result and result.is_file():
            r = json.loads(result.read_text(encoding="utf-8"))
            story = result.parent / "story.json"  # (worked out once per make; a remake writes a new result)
            if not story.is_file() or story.stat().st_mtime < result.stat().st_mtime:
                made = _story(result.parent)
                if made:
                    story.write_text(json.dumps(made), encoding="utf-8")
            told = json.loads(story.read_text(encoding="utf-8")) if story.is_file() else None
            return {"cost": r.get("cost"), "seconds": r.get("seconds"), "story": told}
        return {}

    def things(self) -> list[dict]:
        """Everything made under out/, objects and buildings, newest first (no GPS, no address: the page may be on
        screen in a video)."""
        found = []
        for meta in self.out.rglob("object.json"):
            try:
                made = json.loads(meta.read_text(encoding="utf-8"))
                model = meta.parent / Path(made["model"]).name
                if not model.is_file():
                    continue
                found.append({"id": self._id(model), "kind": "object", "name": made.get("name", "object"),
                              "size_m": made.get("size_m"), "shape": made.get("shape"),
                              "photos": len(made.get("photos_used", [])), "notes": made.get("notes", []),
                              "warnings": made.get("warnings", []), "time": model.stat().st_mtime,
                              **self._timing(meta.parent)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        for model in self.out.rglob("building.glb"):
            try:
                said = model.parent / "thing.json"
                made = json.loads(said.read_text(encoding="utf-8")) if said.is_file() else {}
                if not made:  # (built before the studio, by the game or the command line)
                    place = json.loads((model.parent / "place.geojson").read_text(encoding="utf-8"))
                    props = place["features"][0]["properties"]
                    made = {"name": f"building, {props.get('roof') or 'flat'} roof",
                            "size_m": _glb_size(model)}
                found.append({"id": self._id(model), "kind": "building", "name": made.get("name", "building"),
                              "size_m": made.get("size_m"), "shape": "building", "photos": made.get("photos_used", 0),
                              "notes": made.get("notes", []), "warnings": [], "time": model.stat().st_mtime,
                              **self._timing(model.parent)})
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                continue
        for t in found:  # (the studio keeps a make's photos, so it can make it again)
            t["remake"] = t["id"].startswith("studio/")
        return sorted(found, key=lambda t: -t["time"])[:SHOWN]

    def model(self, id_: str) -> Path:
        path = (self.out / id_).resolve()
        if path.suffix != ".glb" or not path.is_relative_to(self.out) or not path.is_file():
            raise ValueError("no such model")
        return path

    def _made_folder(self, id_: str) -> Path:
        """What deleting a thing removes: a studio make's whole folder (its photos too), else the object's folder."""
        model = self.model(id_)
        folder = model.parent
        made = (folder / "object.json").is_file() or (model.name == "building.glb"
                                                      and (folder / "place.geojson").is_file())
        if not made:
            raise ValueError("not something PlaceLens made")
        rel = folder.relative_to(self.out).parts
        if len(rel) == 4 and rel[0] == "studio" and rel[2] in ("objects", "build"):  # studio/<job>/objects/<photo>
            job = rel[1]
            if self.jobs.get(job, {}).get("running"):
                raise ValueError("that one's still being made")
            return self.root / job
        return folder

    def delete(self, ids: list[str]) -> int:
        import shutil

        folders = {self._made_folder(i) for i in ids}  # (all checked before any is deleted)
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)
        return len(folders)

    def game_packs(self) -> Path | None:
        """Where the game reads packs (written by adapters/godot/install.py), or None: no game set up here."""
        try:
            packs = Path(json.loads((self.out / "game.json").read_text(encoding="utf-8"))["packs"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return packs if packs.parent.is_dir() else None

    def send(self, ids: list[str]) -> list[str]:
        """The chosen things into the game, each as its own pack (a running game picks it up in a few seconds, in
        My things). Sent again after a remake, it replaces its last copy there."""
        from .pack import write_pack

        packs = self.game_packs()
        if packs is None:
            raise ValueError("no game set up: run adapters/godot/install.py <your Open-Planet Godot folder> once")
        names = {t["id"]: t["name"] for t in self.things()}
        sent = []
        for i in ids:
            model = self.model(i)
            folder = model.parent
            key = re.sub(r"[^a-z0-9]+", "-", "-".join(model.relative_to(self.out).parts[:-1]).lower()).strip("-")
            for old in packs.glob(f"studio-{key}-*"):  # (its last copy: replaced, not doubled)
                shutil.rmtree(old, ignore_errors=True)
            dest = packs / f"studio-{key}-{int(model.stat().st_mtime)}"
            photos = sorted(p.name for p in (folder / "photos").glob("*")) if (folder / "photos").is_dir() else []
            write_pack(dest, names.get(i, folder.name), [folder], photos)
            sent.append(names.get(i, folder.name))
        return sent

    def download(self, ids: list[str]) -> bytes:
        """The chosen models in one zip, each named after what it is."""
        import io
        import zipfile

        names = {t["id"]: t["name"] for t in self.things()}
        buf, used = io.BytesIO(), set()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i in ids:
                path = self.model(i)
                base = re.sub(r"[^a-z0-9]+", "-", names.get(i, path.parent.name).lower()).strip("-") or "object"
                name, n = f"{base}.glb", 2
                while name in used:
                    name, n = f"{base}-{n}.glb", n + 1
                used.add(name)
                z.write(path, name)
        return buf.getvalue()


BUILDING_SIDES = {0: "front", 45: "front-right", 90: "right", 135: "back-right", 180: "back", -135: "back-left",
                  -90: "left", -45: "front-left"}  # (degrees round from the front, walking right: left and right as
# you face the front, as `placelens build --side` takes them)


def _building_sides(front: str, views: dict) -> dict[str, str]:
    """The circle's places as the sides of a building: the front photo is "front"; each other photo the nearest of
    the 8 sides and corners to where it was placed. Photos placed "above" say nothing about a side."""
    sides = {_safe(front): "front"}
    for name, v in views.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            a = (float(v) + 180) % 360 - 180
            nearest = min(BUILDING_SIDES, key=lambda s: abs((a - s + 180) % 360 - 180))
            sides[_safe(name)] = BUILDING_SIDES[nearest]
    return sides


def _who(detail: str) -> str | None:
    """The model doing a step: who it's about, at its start ("fitting Muse's parts" is geometry; "hiding brand
    names" is SAM finding them). None: PlaceLens's own work, no AI."""
    if "Muse Image" in detail:
        return "Muse Image"
    if detail.startswith(("SAM 3.1", "hiding brand")):
        return "SAM 3.1"
    if detail.startswith("Muse"):
        return "Muse Spark Open"
    return None
MODEL_NAMES = {"sam-3.1": "SAM 3.1", "muse-image-1.0": "Muse Image", "muse-spark-1.3": "Muse Spark (Meta API)",
               "muse-wonder-1.0": "Muse Wonder"}


def _story(job: Path) -> dict | None:
    """What a make did, step by step, from its progress log: each step's seconds from the start and the model that
    did it (None: PlaceLens's own geometry, no AI); and every model call in that time, from the usage logs."""
    try:
        events = [json.loads(line) for line in (job / "progress.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    except (OSError, ValueError):
        return None
    if not events:
        return None
    start = events[0]["time"]
    steps = []
    for e in events:
        detail = re.sub(r": photo \d+ of (\d+)$", r": \1 photos", e["detail"])  # (one line for a counted step)
        if any(s["detail"] == detail for s in steps):  # (the same step again: SAM's later photos, logged after
            continue  # a step that runs beside it)
        who = _who(e["detail"])
        steps.append({"t": round(e["time"] - start), "detail": detail, "model": who})
    end = events[-1]["time"]
    result = job / "result.json"
    if result.is_file():
        end = max(end, result.stat().st_mtime)
    calls: dict[str, int] = {}
    root = job.parent.parent  # (out/: where meta.py logs every call)
    for log, name in (("meta_usage.jsonl", None), ("open_usage.jsonl", "Muse Spark Open")):
        try:
            lines = (root / log).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if start - 1 <= c.get("time", 0) <= end + 2:
                what = name or MODEL_NAMES.get(str(c.get("what")), str(c.get("what", "Meta API")))
                calls[what] = calls.get(what, 0) + 1
    return {"steps": steps, "calls": calls, "seconds": round(end - start)}


def _thumb(photo: Path, out: Path, size: int = 360) -> Path:
    """A small upright JPEG of a photo (HEIC too) for the page."""
    import pillow_heif
    from PIL import Image, ImageOps

    pillow_heif.register_heif_opener()
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(photo) as img:
        small = ImageOps.exif_transpose(img).convert("RGB")
        small.thumbnail((size, size))
        small.save(out, quality=82)
    return out


def _building(built: dict, what: str) -> dict:
    """`placelens build --json`'s answer, cut down to what the page shows: no map position, address or map link."""
    wide, deep = (built.get("size_m") or [0, 0])[:2]
    roof = built.get("roof") or {}
    notes = [f"{Path(p['photo']).name}: " + (f"used ({p['side']} side)" if p.get("used") and p.get("side") else
                                             "used" if p.get("used") else f"left out: {p.get('why') or 'not needed'}")
             for p in built.get("photos", [])]
    notes.append(f"height {built.get('height_m')} m to the eaves ({built.get('height_source')}); "
                 f"{roof.get('kind', 'flat')} roof, {roof.get('pitch')}°")
    return {"name": what or f"building, {roof.get('kind') or 'flat'} roof", "kind": "building", "model": built["model"],
            "size_m": [wide, deep, round(float(built.get("height_m") or 0), 1)],
            "photos_used": sum(1 for p in built.get("photos", []) if p.get("used")), "notes": notes,
            "warnings": []}


PLAIN = (  # what the command line says, as a person using the page would want it said
    ("has no GPS", "These photos have no location. For a building, take them with your phone's own camera app with "
                   "location on (not sent through WhatsApp or similar, which strips it). Or make it as an object."),
    ("no compass direction", "These photos have no compass direction. Take them with your phone's own camera app "
                             "with location on. Or make it as an object."),
    ("not sure which building", "PlaceLens isn't sure which building this is. Add a photo taken straight at it from "
                                "across the street, and one from a corner."),
    ("no mapped buildings in view", "There's no building on the map where these photos point. The map may not "
                                    "have it yet, or the compass was off: try again from a little further back."),
)


def _plain(error: str) -> str:
    return next((said for clue, said in PLAIN if clue in error), error)


def _glb_size(path: Path) -> list[float] | None:
    """Wide (east), deep (south) and tall, in metres, from a .glb's own bounds (its POSITION accessors)."""
    import struct

    data = path.read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + length])
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            lo = [min(a, b) for a, b in zip(lo, acc["min"], strict=True)]
            hi = [max(a, b) for a, b in zip(hi, acc["max"], strict=True)]
    if lo[0] == float("inf"):
        return None
    x, y, z = (round(h - lo_, 1) for lo_, h in zip(lo, hi, strict=True))
    return [x, z, y]


KEY_NAMES = {"meta": "MODEL_API_KEY", "open": "BASETEN_API_KEY"}


def save_keys(keys: dict, env_file: Path = Path(".env")) -> None:
    """The keys typed into the page, into .env (gitignored), keeping its other lines. A key never leaves this computer
    except to its own service, and is never sent back to the page."""
    new = {}
    for field, name in KEY_NAMES.items():
        value = str(keys.get(field) or "").strip()
        if value:
            if not re.fullmatch(r"[\x21-\x7e]{8,400}", value):  # (one line of printable text, nothing to break .env)
                raise ValueError(f"that doesn't look like a key ({name})")
            new[name] = value
    if not new:
        raise ValueError("paste your Meta API key first")
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    lines = [ln for ln in lines if ln.split("=", 1)[0].strip() not in new]
    env_file.write_text("\n".join(lines + [f"{k}={v}" for k, v in new.items()]) + "\n", encoding="utf-8")


def _handler(studio: Studio, page: bytes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # quiet: the terminal shows only what matters
            pass

        def _send(self, body: bytes, kind: str = "application/json", code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200) -> None:
            self._send(json.dumps(data).encode(), code=code)

        def _ours(self) -> bool:
            """Only this studio's own page may use it: not another website open in the browser (which could reach
            127.0.0.1 too) and not a look-alike host name pointed here."""
            port = self.server.server_address[1]
            mine = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in mine:
                return False
            origin = self.headers.get("Origin")
            return origin is None or urlparse(origin).netloc in mine

        def do_GET(self):
            if not self._ours():
                return self._json({"error": "not from the studio page"}, 403)
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(page, "text/html; charset=utf-8")
                elif url.path == "/things":
                    self._json(studio.things())
                elif url.path == "/status":
                    self._json(studio.status(q.get("job", "")))
                elif url.path == "/thumb":
                    self._send(studio.thumb(q.get("job", ""), q.get("name", "")).read_bytes(), "image/jpeg")
                elif url.path == "/model":
                    self._send(studio.model(q.get("id", "")).read_bytes(), "model/gltf-binary")
                elif url.path == "/download":
                    ids = parse_qs(url.query).get("id", [])
                    if not ids:
                        raise ValueError("choose something to download")
                    data = studio.download(ids)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", 'attachment; filename="placelens-things.zip"')
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif url.path == "/keys":
                    from .meta import has_key

                    self._json({"meta": has_key("MODEL_API_KEY"), "open": has_key("BASETEN_API_KEY"),
                                "game": studio.game_packs() is not None})
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):
            if not self._ours() or self.headers.get("Origin") is None:  # (a browser always sends Origin on a POST)
                return self._json({"error": "not from the studio page"}, 403)
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            size = int(self.headers.get("Content-Length") or 0)
            if size > MAX_PHOTO:
                return self._json({"error": "that photo is too big"}, 413)
            body = self.rfile.read(size)
            try:
                if url.path == "/upload":
                    self._json({"name": studio.upload(q.get("job", ""), q.get("name", ""), body)})
                elif url.path == "/make":
                    ask = json.loads(body or b"{}")
                    studio.make(ask.get("job", ""), list(ask.get("photos", [])), str(ask.get("what", "")),
                                str(ask.get("notes", "")), str(ask.get("kind", "object")), int(ask.get("moves", 3)),
                                ask.get("views") if isinstance(ask.get("views"), dict) else None)
                    self._json({"ok": True})
                elif url.path == "/keys":
                    save_keys(json.loads(body or b"{}"))
                    self._json({"ok": True})
                elif url.path == "/headings":
                    ask = json.loads(body or b"{}")
                    self._json(studio.headings(str(ask.get("job", "")), list(ask.get("photos", []))))
                elif url.path == "/remake":
                    ask = json.loads(body or b"{}")
                    moves = ask.get("moves")
                    self._json({"job": studio.remake(str(ask.get("id", "")), None if moves is None else int(moves))})
                elif url.path == "/send":
                    self._json({"sent": studio.send(list(json.loads(body or b"{}").get("ids", [])))})
                elif url.path == "/delete":
                    self._json({"deleted": studio.delete(list(json.loads(body or b"{}").get("ids", [])))})
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)

    return Handler


class _Server(ThreadingHTTPServer):
    # (Python's default lets a second server bind a port that's in use on Windows, where both then half-work: a
    # taken port must fail so the next one is tried)
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True


def serve(out: Path, port: int = 8765, open_browser: bool = True) -> None:
    studio = Studio(out)
    page = (Path(__file__).parent / "studio.html").read_bytes()
    server = None
    for p in range(port, port + 20):  # the next free port if this one's taken
        try:
            server = _Server(("127.0.0.1", p), _handler(studio, page))
            break
        except OSError:
            continue
    if server is None:
        raise OSError(f"no free port from {port} to {port + 19}")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"PlaceLens Studio is open in your browser: {url}\n(only on this computer; close this window or press "
          "Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.stop()
        server.server_close()
