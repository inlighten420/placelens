# PlaceLens

**Take photos of something real (your house, a chair, a teddy bear) and PlaceLens builds it in 3D at its real size.
A building goes on its real spot on the real map, in place of the map's grey box.** The phone's GPS, compass and lens
and the map do the free work; Meta's models answer only what sensors can't: what is it, how big, which floors, what
roof.

Built for the Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026), theme *Build for Your World*. Everything in
this repo was built during the hackathon. The game it plugs into, [Open-Planet](https://github.com/inlighten420/open-planet)
(a driving game of the real world), existed before and is not part of this entry; PlaceLens is an add-on for it,
and its place packs can be read by any engine.

## Try it

**The easy way:** download the repo (green **Code** button → **Download ZIP**), unzip it, and double-click
**`Start PlaceLens.bat`** (on macOS or Linux: `sh start.sh`). The first time, it sets itself up (a few minutes; it
needs [Python 3.10+](https://www.python.org/downloads/)). Then PlaceLens Studio opens in your browser: paste your Meta
API key once, drop in photos, press **Make it 3D**, and drag the model round to see every side. Everything runs on
your own computer; the key is saved in `.env` in that folder.

**The developer way:**

```sh
git clone https://github.com/inlighten420/placelens && cd placelens
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"        # .venv/bin/pip on macOS/Linux
.venv/Scripts/python -m pytest -q            # 157 tests, offline, no keys needed
.venv/Scripts/activate                       # source .venv/bin/activate on macOS/Linux: puts `placelens` on the path
```

**Keys.** PlaceLens needs a Meta Model API key; a Baseten key is optional. Any one of these works:
- paste it into the studio the first time it opens (it's saved for you);
- or put it in a `.env` file in the repo folder (gitignored);
- or set it as an environment variable (`setx MODEL_API_KEY ...` on Windows, `export MODEL_API_KEY=...` elsewhere).

```sh
MODEL_API_KEY=...       # Meta Model API: SAM 3.1, Muse Image (and Muse Spark if you have no Baseten key)
BASETEN_API_KEY=...     # optional: Muse Spark Open on Baseten does the reasoning; without it, Muse Spark does
```

Then make something. The easiest way is the studio, a page in your browser on your own computer:

```sh
placelens studio      # drop photos, press Make, then drag the 3D model round to see every side
```

In the studio:
- **The circle** places your photos round the object by each photo's compass; drag one if it's wrong.
- **How it was made** shows every step, which Meta model did it, and how long it took.
- **Remake** makes a thing again from the same photos; saved answers make it quick and nearly free.
- **Self-checks** sets how many times Muse looks at what it built and corrects it.
- **Send to game** puts it in Open-Planet in a few seconds.
- **Download** gives you the `.glb` for any engine.

Or from the command line:

```sh
placelens object chair_front.jpg chair_side.jpg chair_back.jpg   # an object: walk round it, or one good photo
placelens build house_*.jpg                                      # a building, from photos taken round it
placelens survey my_photos/                                      # an agent sorts the photos and makes it all
```

Each make writes a `.glb` model, a `view.html` to look at it in a browser, and says what it cost.

**Photos that work best:** a phone's own camera app with location on (iPhone HEIC or JPEG). For an object, **4 photos:
front, both sides and back**, the whole thing in each, phone at about its middle height (a 5th from above helps);
or one straight-on photo from the front. 3 photos leave its corners cut flat; 4 round them out. For a building, take
the street side from a few spots and the corners. Photos without GPS or compass still work for objects (Muse works
out which side each photo shows); a building needs at least one photo with GPS and compass.

## How it works

| Step | How | Meta model |
|---|---|---|
| Read the photo | EXIF: GPS, compass, lens width (HEIC and JPEG) | |
| Which building | Each photo is a line from where you stood; where the lines meet is the building. Footprints from Overture Maps (OpenStreetMap + Google + Microsoft), else OpenStreetMap | |
| Outline it, cut it out | Pixel masks of the building, the object, windows, doors, logos | **SAM 3.1** |
| What is it, how big | Research: what it is, how such things are made, its shape, real size, floors, roof, colours, brands | **Muse Spark Open** |
| Which side each photo shows | Compass first; Muse checks it, and wins when the compass is clearly off (indoors) or the object was turned in place | **Muse Spark Open** |
| Sides nobody photographed | Drawn from the one photo, each drawing checked by Muse | **Muse Image** |
| The shape | Buildings: footprint raised to its height, roof blocks, real windows and doors. Soft objects: carved from the photos' outlines, coloured from the photos. Hard, made things (a chair): Muse's parts fitted to the carving. Round things: turned on a lathe | |
| Brands | Logos found and painted over before anything goes in a game | **SAM 3.1** |
| The check | The model drawn over every photo from its own camera; Muse proposes fixes, and each is kept only if every photo agrees better (outline overlap) | **Muse Spark Open** |
| The plan | An agent groups the photos and calls the makers with tool calls | **Muse Spark Open** |

**Why it's cheap:** the sensors and the map are free and mostly right, so the models only fill the gaps. Real makes
in the studio (Meta credits; all the reasoning on Muse Spark Open, open weights, on Baseten):

| Thing | Photos | Time | Meta credits |
|---|---|---|---|
| Aloe plant | 4 | 47 s | 1¢ |
| Sugar skull | 3 | 31 s | 0.25¢ |
| Steering wheel | 4 | 2 min 38 s | 4.5¢ |
| A house | 12 | 16 min (85 s made again) | 9.75¢ |

## In the game

`python adapters/godot/install.py <Open-Planet Godot folder> --photos <your photo folder>` adds PlaceLens to
Open-Planet (Godot 4.7; `--uninstall` takes it out). In the game, **F7** opens PlaceLens: choose photos, say what it
is, Make. **I** opens your things: put what you made anywhere, or use a steering wheel as your car's. Buildings
replace the map's box where they really stand. PlaceLens's windows wear its own blue, so it's clear what's the add-on.
Things made in the studio go in with **Send to game**: the installer tells the studio where the game's packs are, and
the game picks new ones up while it runs.

## For agents

Every command takes `--json`. When a photo is missing something, the exit code is `2` and the error says how to fix
it (`--lat`/`--lon` for no GPS, `--heading` for no compass); `3` means it isn't sure which building it is.

## Honest limits

- Thin parts (chair legs, armrests) are hard to see in photos; a chair comes out blocky.
- A one-photo object looked down on from above can come out squat (the drawings are level).
- Buildings follow the map's footprint; a wrong footprint gives a wrong shape. Tested mostly on one house.
- Phone compasses are poor indoors and in narrow passages; Muse corrects objects (or drag them on the circle),
  buildings need open air.
- A building is slow the first time (5 to 15 minutes: Muse checks every photo several times).
- The first start installs everything, which can take up to 10 minutes; after that it starts in seconds.

## Privacy

Your own photos go in `private/`, which is never committed. Everything made goes to `out/`, also gitignored.
Brand logos are painted out of objects.

## Credits

- Building footprints: [Overture Maps](https://overturemaps.org/) and © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, ODbL.
- SAM masks decoded with Meta's [`meta-sam-parser`](https://github.com/meta-models/meta-sam) (SAM License).
- Test fixture `tests/data/sam_city_hall.txt` is SAM 3.1's output on
  [Cape Town City Hall (6186284748).jpg](https://commons.wikimedia.org/wiki/File:Cape_Town_City_Hall_(6186284748).jpg),
  Wikimedia Commons, CC BY 2.0.

MIT licence: see [LICENSE](LICENSE).
