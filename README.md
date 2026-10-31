# PlaceLens

**Take a photo of a real place. PlaceLens finds the building you're looking at, cuts out its real front, and builds
it at its real spot, real size, on the real map.** The map is the skeleton, the photo is the skin.

Built for the Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026), Indie track. Everything in this repo was
built during the hackathon.

> Status: day 3. Picking the building and building it in 3D with the photo on its walls work; game adapters are next.
> See [PLAN.md](PLAN.md) for the whole plan.

## What works today

```sh
placelens info photo.jpg      # what the photo knows: GPS, compass direction, lens width
placelens find photo.jpg      # which mapped buildings it's looking at, ranked, plus a map picture
placelens pick photo.jpg      # which one it is: SAM 3.1 outline + Muse, checked against each other
placelens build *.jpg         # the building in 3D from one or more photos: building.glb with the photos on
                              # its walls (each wall from the photo that sees it best), plus place.geojson
placelens prop photo.jpg      # an object in the photo as a 3D prop at its real size, standing where it
                              # really is (prop.glb + prop.geojson); several photos round it: carved 3D
placelens pack home out/build/... out/props/...   # one place pack (manifest.json) for any game
placelens survey folder/ --notes "the back is the same as the front"   # an agent does all of the above
```

Notes tell the AI what the photos can't show: "it's symmetrical, the other side looks the same", "about 20 m wide",
"a round tank, 1.8 m tall". `--notes` works on `build`, `prop` and `survey`; Muse reads it with every photo, and the
survey agent turns what it can into settings (`--symmetric`, `--floors`, `--height`, `--width`, `--shape round`).

Game adapters read the pack: `adapters/godot/` puts it into Open-Planet (Godot 4.7), replacing the
map's grey building with the real one, props at their spots: `python adapters/godot/install.py
<Open-Planet Godot folder> --pack out/pack/home` (`--uninstall` takes it out again).

In the game, **F7** opens PlaceLens: choose photos, write notes, choose where objects go (where the photos were
taken, or where you're looking), and press Make. The survey agent runs in the background, its steps show along the
top, and what it made appears in the world when it's done.

`find` reads where you stood and which way you faced, fetches the OpenStreetMap buildings around you, casts rays
across the lens's width and keeps the buildings you could actually see (one hidden behind another drops out).
Tested on Cape Town City Hall: it picks the hall first.

| Step | How | Status |
|---|---|---|
| Read the photo | EXIF GPS, compass, 35 mm focal length (JPEG and iPhone HEIC) | ✅ |
| Buildings nearby | OpenStreetMap via Overpass, cached | ✅ |
| Which ones are in view | View cone, ray casting, compass-error margin | ✅ |
| Cut out the building | **SAM 3.1** (Meta Model API) | ✅ tested |
| Pick the right one, count floors | **Muse Spark** / **Muse Wonder**, checked against the outline's geometry | ✅ |
| Straighten the front | Camera locked to the outline, each wall's corners projected, homography | ✅ |
| 3D building (`.glb`) | Footprint raised to its height (OSM, else Muse's floors), photo on the walls it shows | ✅ |
| Roof | Footprint cut into blocks; gable/hip/flat and ridge direction from **Muse**, pitch measured on SAM's outline | ✅ |
| Props | **Muse** names and sizes it, **SAM 3.1** cuts it out, the outline made solid, GPS + compass place it | ✅ |
| Clean up cars and people | **Muse Image** | planned |
| Game adapters | Place pack (manifest.json); Godot adapter for Open-Planet; web viewer | ✅ in progress |

## For people and AI agents

Every command takes `--json` and returns machine-readable output. When a photo is missing what it needs, the exit
code is `2` and the error says how to fix it (`--lat`/`--lon` for no GPS, `--heading` for no compass), so an agent
can recover without a human. When `build` isn't sure which building it is, the exit code is `3`: choose one with
`--building N` (the numbers are on `pick`'s map).

```sh
placelens find photo.jpg --json
placelens find photo.jpg --heading 225     # many Android phones don't save the compass direction
placelens find photo.jpg --help
```

## Setup

```sh
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # .venv/bin/pip on macOS/Linux
.venv/Scripts/python -m pytest
```

Meta Model API calls need `MODEL_API_KEY` in a `.env` file (gitignored). Try them with
`python scripts/api_smoke.py photo.jpg`: one call each to Muse Spark, SAM 3.1 and Muse Wonder, with costs.

## Privacy

Your own test photos go in `private/`, which is never committed. Maps and outputs go to `out/`, also gitignored.

## Credits

- Building data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, ODbL.
- SAM masks decoded with Meta's [`meta-sam-parser`](https://github.com/meta-models/meta-sam) (SAM License).
- Test fixture `tests/data/sam_city_hall.txt` is SAM 3.1's output on
  [Cape Town City Hall (6186284748).jpg](https://commons.wikimedia.org/wiki/File:Cape_Town_City_Hall_(6186284748).jpg),
  Wikimedia Commons, CC BY 2.0.

MIT licence: see [LICENSE](LICENSE).
