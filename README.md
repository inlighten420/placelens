# PlaceLens

**Take a photo of a real place. PlaceLens finds the building you're looking at, cuts out its real front, and builds
it at its real spot, real size, on the real map.** The map is the skeleton, the photo is the skin.

Built for the Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026), Indie track. Everything in this repo was
built during the hackathon.

> Status: day 1. Finding the building and cutting it out work; the 3D building and game adapters are next.
> See [PLAN.md](PLAN.md) for the whole plan.

## What works today

```sh
placelens info photo.jpg      # what the photo knows: GPS, compass direction, lens width
placelens find photo.jpg      # which mapped buildings it's looking at, ranked, plus a map picture
```

`find` reads where you stood and which way you faced, fetches the OpenStreetMap buildings around you, casts rays
across the lens's width and keeps the buildings you could actually see (one hidden behind another drops out).
Tested on Cape Town City Hall: it picks the hall first.

| Step | How | Status |
|---|---|---|
| Read the photo | EXIF GPS, compass, 35 mm focal length (JPEG and iPhone HEIC) | ✅ |
| Buildings nearby | OpenStreetMap via Overpass, cached | ✅ |
| Which ones are in view | View cone, ray casting, compass-error margin | ✅ |
| Cut out the building | **SAM 3.1** (Meta Model API) | ✅ tested |
| Pick the right one, count floors | **Muse Wonder** / **Muse Spark** | ✅ tested, being wired in |
| Straighten the front | Homography from its four corners | next |
| 3D building (`.glb`) | Footprint raised to its height, real front on its wall | next |
| Clean up cars and people | **Muse Image** | planned |
| Game adapters | Godot (Open-Planet), web viewer | planned |

## For people and AI agents

Every command takes `--json` and returns machine-readable output. When a photo is missing what it needs, the exit
code is `2` and the error says how to fix it (`--lat`/`--lon` for no GPS, `--heading` for no compass), so an agent
can recover without a human.

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
