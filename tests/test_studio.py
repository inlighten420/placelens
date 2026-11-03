import json

import pytest

from placelens.studio import EXAMPLES, Studio, _safe, seed_examples


def test_photo_names_stay_in_their_folder():
    assert _safe("../../evil.jpg") == "evil.jpg"
    assert _safe("C:\\Users\\me\\IMG_0001.HEIC") == "IMG_0001.HEIC"
    assert "/" not in _safe("a/b<c>.jpg")


def test_upload_takes_only_photos(tmp_path):
    s = Studio(tmp_path)
    assert s.upload("abcd1", "IMG_1.HEIC", b"x") == "IMG_1.HEIC"
    assert (tmp_path / "studio" / "abcd1" / "photos" / "IMG_1.HEIC").read_bytes() == b"x"
    with pytest.raises(ValueError):
        s.upload("abcd1", "run.py", b"x")
    with pytest.raises(ValueError):
        s.upload("../up", "a.jpg", b"x")


def test_models_only_from_out(tmp_path):
    out = tmp_path / "out"
    made = out / "studio" / "abcd1" / "objects" / "IMG_1"
    made.mkdir(parents=True)
    (made / "prop.glb").write_bytes(b"glTF")
    (made / "object.json").write_text(json.dumps({"name": "chair", "model": "x/prop.glb", "size_m": [1, 1, 1],
                                                  "photos_used": ["IMG_1.HEIC"], "location": {"lat": 1, "lon": 2}}))
    (tmp_path / "secret.glb").write_bytes(b"no")
    s = Studio(out)
    things = s.things()
    assert [t["name"] for t in things] == ["chair"]
    assert "location" not in things[0] and "lat" not in json.dumps(things)
    assert s.model(things[0]["id"]).read_bytes() == b"glTF"
    for bad in ("../secret.glb", "studio/abcd1/objects/IMG_1/object.json"):
        with pytest.raises(ValueError):
            s.model(bad)


def test_name_loses_the_brand(tmp_path, monkeypatch):
    from placelens import muse, objects

    monkeypatch.setattr(muse, "ask", lambda *a, **kw: ["Sim racing steering wheel."])
    assert objects._unbranded_name(None, "MOZA KS steering wheel", tmp_path / "p.jpg", tmp_path) \
        == "sim racing steering wheel"
    monkeypatch.setattr(muse, "ask", lambda *a, **kw: ['{"name": "a long json answer that is not a name at all"}'])
    assert objects._unbranded_name(None, "MOZA wheel", tmp_path / "p.jpg", tmp_path) == "MOZA wheel"


def test_keys_go_in_env_keeping_the_rest(tmp_path):
    from placelens.studio import save_keys

    env = tmp_path / ".env"
    env.write_text("OTHER=1\nMODEL_API_KEY=old\n")
    save_keys({"meta": "  new-meta-key-123 ", "open": ""}, env)
    assert env.read_text().splitlines() == ["OTHER=1", "MODEL_API_KEY=new-meta-key-123"]
    for bad in ({"meta": ""}, {"meta": "short"}, {"meta": "abcdefgh\nEVIL=1"}):
        with pytest.raises(ValueError):
            save_keys(bad, env)


def test_only_the_studio_page_may_use_it(tmp_path):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from placelens.studio import _handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(Studio(tmp_path), b"page"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def status(path, method="GET", **headers):
        req = urllib.request.Request(base + path, data=b"{}" if method == "POST" else None, method=method,
                                     headers=headers)
        try:
            return urllib.request.urlopen(req).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert status("/") == 200
        assert status("/", Host="evil.example") == 403  # a look-alike host name pointed at this computer
        assert status("/keys", "POST", Origin="https://evil.example") == 403  # another website in the browser
        assert status("/keys", "POST") == 403  # no Origin: not a browser page
        assert status("/keys", "POST", Origin=base) == 400  # the page itself: allowed (no key given)
    finally:
        server.shutdown()


def _made(out, *where, name="chair"):
    folder = out.joinpath(*where)
    folder.mkdir(parents=True)
    (folder / "prop.glb").write_bytes(b"glTF" + name.encode())
    (folder / "object.json").write_text(json.dumps({"name": name, "model": "prop.glb", "photos_used": []}))
    return folder


def test_delete_and_download(tmp_path):
    import io
    import zipfile

    out = tmp_path / "out"
    _made(out, "studio", "job01", "objects", "IMG_1")
    (out / "studio" / "job01" / "photos").mkdir()
    _made(out, "objects", "IMG_2")
    _made(out, "objects", "IMG_3")
    s = Studio(out)
    ids = {t["id"] for t in s.things()}
    assert len(ids) == 3
    names = zipfile.ZipFile(io.BytesIO(s.download(sorted(ids)))).namelist()
    assert sorted(names) == ["chair-2.glb", "chair-3.glb", "chair.glb"]
    assert s.delete(["studio/job01/objects/IMG_1/prop.glb", "objects/IMG_2/prop.glb"]) == 2
    assert not (out / "studio" / "job01").exists()  # a studio make goes with its uploaded photos
    assert [t["id"] for t in s.things()] == ["objects/IMG_3/prop.glb"]
    (out / "loose").mkdir()
    (out / "loose" / "x.glb").write_bytes(b"x")  # not something PlaceLens made: never deleted
    for bad in ("loose/x.glb", "../out/objects/IMG_3/../../../x.glb"):
        with pytest.raises(ValueError):
            s.delete([bad])
    assert (out / "loose" / "x.glb").exists() and (out / "objects" / "IMG_3").exists()


def _glb(path, lo, hi):
    import struct

    doc = json.dumps({"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                      "accessors": [{"min": lo, "max": hi}]}).encode()
    doc += b" " * (-len(doc) % 4)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 20 + len(doc)) + struct.pack("<I", len(doc)) + b"JSON" + doc)


def test_buildings_are_things_too_without_where_they_are(tmp_path):
    from placelens.studio import _building

    out = tmp_path / "out"
    old = out / "build" / "way_1"  # built by the game, before the studio
    old.mkdir(parents=True)
    _glb(old / "building.glb", [-5, 0, -4], [5, 6.5, 4])
    (old / "place.geojson").write_text(json.dumps({"features": [{"properties": {"roof": "gable",
                                                                                "anchor": [18.42, -33.92]}}]}))
    new = out / "studio" / "job02" / "build" / "way_2"
    new.mkdir(parents=True)
    _glb(new / "building.glb", [0, 0, 0], [1, 1, 1])
    (new / "place.geojson").write_text("{}")
    built = {"model": str(new / "building.glb"), "size_m": [12.3, 8.1], "height_m": 2.9, "height_source": "muse",
             "roof": {"kind": "hip", "pitch": "25"}, "building": {"centre": [-33.92, 18.42], "name": "12 Some St"},
             "photos": [{"photo": "a/IMG_1.HEIC", "used": True, "side": "front"},
                        {"photo": "a/IMG_2.HEIC", "used": False, "why": "blurry"}]}
    thing = _building(built, "my house")
    (new / "thing.json").write_text(json.dumps(thing))
    assert thing["size_m"] == [12.3, 8.1, 2.9] and thing["photos_used"] == 1
    assert "-33.92" not in json.dumps(thing) and "Some St" not in json.dumps(thing)
    things = {t["name"]: t for t in Studio(out).things()}
    assert things["building, gable roof"]["size_m"] == [10, 8, 6.5]
    assert things["my house"]["kind"] == "building"
    assert "-33.92" not in json.dumps(list(things.values()))
    assert Studio(out).delete([things["my house"]["id"]]) == 1 and not (out / "studio" / "job02").exists()


def test_errors_said_plainly():
    from placelens.studio import _plain

    assert "location" in _plain("photo has no GPS; pass --lat and --lon")
    assert "--" not in _plain("not sure which building it is (x); run `placelens pick` and y")
    assert _plain("something else") == "something else"


def test_remake_uses_the_same_photos_front_first(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _made(out, "studio", "job03", "objects", "IMG_0202")
    photos = out / "studio" / "job03" / "photos"
    photos.mkdir()
    for n in ("IMG_0201.HEIC", "IMG_0202.HEIC", "IMG_0204.HEIC"):
        (photos / n).write_bytes(b"x")
    _made(out, "objects", "IMG_9")  # made from the command line: no photos kept here
    s = Studio(out)
    asked = []
    monkeypatch.setattr(s, "make", lambda *a: asked.append(a))
    assert s.remake("studio/job03/objects/IMG_0202/prop.glb") == "job03"
    assert asked[-1] == ("job03", ["IMG_0202.HEIC", "IMG_0201.HEIC", "IMG_0204.HEIC"], "", "", "object", 3, None, False)
    (out / "studio" / "job03" / "request.json").write_text(json.dumps(
        {"photos": ["IMG_0204.HEIC"], "kind": "object", "what": "skull", "notes": "n", "find_parts": True}))
    s.remake("studio/job03/objects/IMG_0202/prop.glb", moves=5)
    assert asked[-1] == ("job03", ["IMG_0204.HEIC"], "skull", "n", "object", 5, None, True)  # smooth parts again
    with pytest.raises(ValueError):
        s.remake("objects/IMG_9/prop.glb")
    flags = {t["id"].split("/")[0]: t["remake"] for t in s.things()}
    assert flags == {"studio": True, "objects": False}


def test_circle_places_become_sides_of_the_house():
    from placelens.studio import _building_sides

    sides = _building_sides("IMG_1.HEIC", {"IMG_2.HEIC": 85, "IMG_3.HEIC": -175, "IMG_4.HEIC": -50,
                                           "IMG_5.HEIC": "above", "IMG_6.HEIC": 140})
    assert sides == {"IMG_1.HEIC": "front", "IMG_2.HEIC": "right", "IMG_3.HEIC": "back",
                     "IMG_4.HEIC": "front-left", "IMG_6.HEIC": "back-right"}


def test_send_to_game_writes_one_pack_per_thing(tmp_path):
    out = tmp_path / "out"
    folder = _made(out, "studio", "job05", "objects", "IMG_7", name="aloe")
    (folder / "prop.geojson").write_text(json.dumps({"type": "Feature", "geometry": None, "properties": {
        "name": "aloe", "model": "prop.glb", "size_m": [0.1, 0.1, 0.2]}}))
    packs = tmp_path / "Saved" / "placelens"
    s = Studio(out)
    with pytest.raises(ValueError):  # (no game set up yet)
        s.send(["studio/job05/objects/IMG_7/prop.glb"])
    (out / "game.json").write_text(json.dumps({"packs": packs.as_posix()}))
    packs.parent.mkdir(parents=True)
    assert s.send(["studio/job05/objects/IMG_7/prop.glb"]) == ["aloe"]
    s.send(["studio/job05/objects/IMG_7/prop.glb"])  # (again: replaces, not doubles)
    made = list(packs.iterdir())
    assert len(made) == 1
    manifest = json.loads((made[0] / "manifest.json").read_text())
    assert manifest["items"][0]["kind"] == "prop" and manifest["items"][0]["name"] == "aloe"
    assert (made[0] / manifest["items"][0]["model"]).read_bytes().startswith(b"glTF")


def test_a_make_tells_its_steps_and_models(tmp_path):
    from placelens.studio import _story

    out = tmp_path / "out"
    job = out / "studio" / "job06"
    job.mkdir(parents=True)
    t = 1000.0
    events = [("getting 4 of 4 photos ready", 0), ("Muse is researching it", 0), ("SAM 3.1 is finding the pot", 17),
              ("SAM 3.1 is cutting out the pot: photo 0 of 4", 20),
              ("SAM 3.1 is cutting out the pot: photo 4 of 4", 24), ("carving the pot from 4 cut-outs", 33)]
    (job / "progress.jsonl").write_text("\n".join(json.dumps({"detail": d, "time": t + s}) for d, s in events))
    sam = [json.dumps({"time": t + s, "what": "sam-3.1"}) for s in (17, 20, 24, 500)]  # (500 s: another make)
    (out / "meta_usage.jsonl").write_text("\n".join(sam))
    (out / "open_usage.jsonl").write_text(json.dumps({"time": t + 5}))
    story = _story(job)
    assert [s["detail"] for s in story["steps"]] == [
        "getting 4 of 4 photos ready", "Muse is researching it", "SAM 3.1 is finding the pot",
        "SAM 3.1 is cutting out the pot: 4 photos", "carving the pot from 4 cut-outs"]
    assert [s["model"] for s in story["steps"]] == [None, "Muse Spark Open", "SAM 3.1", "SAM 3.1", None]
    assert story["calls"] == {"SAM 3.1": 3, "Muse Spark Open": 1} and story["steps"][-1]["t"] == 33


def test_each_step_is_put_down_to_who_did_it():
    from placelens.studio import _who

    assert _who("fitting Muse's 32 parts to the carving") is None
    assert _who("hiding brand names and logos") == "SAM 3.1"
    assert _who("Muse Image is drawing its other sides") == "Muse Image"
    assert _who("Muse looks at what it built (move 1 of 3)") == "Muse Spark Open"


def test_a_step_split_by_one_beside_it_stays_one(tmp_path):
    from placelens.studio import _story

    job = tmp_path / "out" / "studio" / "job07"
    job.mkdir(parents=True)
    events = [("SAM 3.1 is cutting out the pot: photo 0 of 4", 26),
              ("Muse checks which side of it each photo shows (3 photos), at the same time as SAM cuts them out", 26),
              ("SAM 3.1 is cutting out the pot: photo 4 of 4", 35), ("carving the pot from 4 cut-outs", 45)]
    (job / "progress.jsonl").write_text("\n".join(json.dumps({"detail": d, "time": 1000.0 + s}) for d, s in events))
    assert [s["detail"][:12] for s in _story(job)["steps"]] == ["SAM 3.1 is c", "Muse checks ", "carving the "]


def test_examples_are_copied_once_and_listed(tmp_path):
    import shutil

    ex = tmp_path / "examples" / "pot"
    ex.mkdir(parents=True)
    (ex / "object.json").write_text(json.dumps({"name": "pot", "model": "prop.glb", "size_m": [1, 1, 1]}))
    (ex / "prop.glb").write_bytes(b"glb")
    out = tmp_path / "out"
    assert seed_examples(out, tmp_path / "examples") == 1
    assert [t["name"] for t in Studio(out).things()] == ["pot"]
    shutil.rmtree(out / "examples" / "pot")  # the person deletes it: it stays deleted
    assert seed_examples(out, tmp_path / "examples") == 0
    assert Studio(out).things() == []


def test_the_shipped_examples_have_no_photos_or_places():
    for folder in (p for p in EXAMPLES.iterdir() if p.is_dir()):
        names = {f.name for f in folder.iterdir()}
        assert {"object.json", "prop.glb", "view.html"} <= names <= {"object.json", "prop.glb", "prop_moving.glb",
                                                                    "view.html"}  # (models only: no photos)
        made = json.loads((folder / "object.json").read_text(encoding="utf-8"))
        assert made["location"] is None and made["example"] is True
        assert all(p.startswith("photo ") for p in made["photos_used"])  # (no photo's own name)
        assert (made.get("moving") is not None) == ("prop_moving.glb" in names)


def test_what_the_models_saw_is_listed_and_served_and_nothing_else(tmp_path):
    out = tmp_path / "out"
    folder = _made(out, "studio", "job04", "objects", "IMG_1")
    (folder / "seen").mkdir()
    for n in ("IMG_1_fit.jpg", "IMG_1_outline.jpg", "IMG_1_parts.jpg"):
        (folder / "seen" / n).write_bytes(b"\xff\xd8" + n.encode())
    (folder / "secret.txt").write_text("not a picture")
    s = Studio(out)
    thing = next(t for t in s.things() if t["id"].endswith("IMG_1/prop.glb"))
    assert thing["seen"] == ["IMG_1_outline.jpg", "IMG_1_parts.jpg", "IMG_1_fit.jpg"]  # outline, parts, then fit
    assert s.seen(thing["id"], "IMG_1_parts.jpg").read_bytes().endswith(b"IMG_1_parts.jpg")
    for bad in ("../secret.txt", r"..\object.json", "nope.jpg"):
        with pytest.raises(ValueError):
            s.seen(thing["id"], bad)


def test_the_page_script_parses(tmp_path):
    """One stray quote in the page's script stops every button on it (3 Nov), and no other test runs the page."""
    import re
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node isn't installed")
    from pathlib import Path

    page = (Path(__file__).parent.parent / "src" / "placelens" / "studio.html").read_text(encoding="utf-8")
    script = tmp_path / "page.js"
    script.write_text("\n".join(re.findall(r"<script>(.*?)</script>", page, re.S)), encoding="utf-8")
    done = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
