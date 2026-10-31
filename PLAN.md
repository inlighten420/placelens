# PlaceLens: plan

Meta Global AI Developer Hackathon, Indie track, 26 Oct – 4 Nov 2026 (submit by 4 Nov 21:00 ET = 5 Nov 03:00 SAST).
Everything in this repo is built during the hackathon. Open-Planet (github.com/inlighten420/open-planet) existed before and is
only used to show the result.

## The idea

Take a photo of a real place. PlaceLens finds which building you're looking at, cuts out its real front, and builds
it at its real spot, real size, on the real map. Grey boxes become the real street.

**The map is the skeleton, the photo is the skin.**

| What | From |
|---|---|
| Where, shape, size | OpenStreetMap footprint |
| Height, floors | Muse Wonder counting floors in the photo |
| How it looks | The photo: cut out by SAM 3.1, straightened, cleaned by Muse Image |
| Free-standing things (trees, kiosks, statues) | SAM 3.1 cut-out, Muse for real size, Muse Image for the sides it can't see |

Not just for games: a cheap digital twin from phone photos. Mapping the unmapped, emergency services, disasters,
accessibility, training worlds for self-driving cars on African roads, town planning, heritage.

## Why this wins (judging: impact, use of Meta AI, technical depth, demo quality, equal weight)

- Impact: a cheap digital twin from phone photos; maps what Google's 3D tiles miss (much of Africa), open-licensed.
- Meta AI: SAM 3.1 (cut-out), Muse Wonder or Muse Spark (pick the building, count floors), Muse Image (clean-up),
  Muse Spark Open (open weights, via Baseten `meta-models/muse-spark-open-1026`).
- Not a "simple image analyzer": it builds 3D places, it doesn't just describe photos.
- Bonus prizes to aim for: Best Agentic Use-Case (an agent runs the CLI over a folder of street photos and only asks
  when unsure), Best Open Weight (building picker on Muse Spark Open), Most Viral (before/after clip on X),
  Best Creative.

## How it works

1. **Read the photo**: GPS (where you stood), compass (which way you faced), focal length (how wide you saw).
   GPS is stripped from anything saved.
2. **View cone**: from where you stood, the way you faced, the lens's width: a beam on the map.
3. **Candidates**: buildings in the beam, from OSM first, then Overture Maps (OSM + Google Open Buildings +
   Microsoft footprints, open-licensed, credited) where OSM is empty. None at all: a rough footprint estimated from
   the photo (marked "estimated"), corners draggable, and a button to open OSM's own editor there (a person adds
   it; no automatic uploads). Buildings (widened for phone error: GPS 5–15 m, compass 10–20°), nearest first.
4. **AI picks**: SAM 3.1 cuts the building out; Muse Wonder compares it with each candidate (floors, width of the
   front against the wall's length, names and signs in view, a landmark it knows).
5. **Unsure? Ask**: a small map with the candidates: "Is it this one?" One tap.
6. **Straighten**: the front warped flat (a homography from its four corners). Maths, not AI.
7. **Clean**: Muse Image takes out cars, people and trees in front.
8. **Build**: the footprint raised to its height, the real front on its wall.
9. (Stretch) **Lock the angle**: the building's corners in the photo matched to the footprint's, for the camera's
   exact spot. (Partly done in `pick`: the outline's edges plus Muse's distance correct the GPS and compass.)
10. (Stretch, after the place pipeline works) **`placelens item`**: photograph any object (a chair, a bin, a car)
   and get a game-ready .glb, no building or map needed. Same cut-out, then image to 3D. For filling a game with
   things as easily as places.

## Output: a place pack (standard files, any engine)

```
place.geojson    where: lat/lon, the OSM building, its walls
building.glb     the building, its real front (glTF: Godot, Unity, Unreal, Blender, the web)
objects/*.glb    trees, kiosks, statues
credits.json     every photo's source and licence
```

## Use

- CLI: `placelens add photo.jpg`, `placelens --help` (made for people and AI agents alike).
- Adapters: `adapters/godot/` (Open-Planet loads the pack as a decal/mesh on the building); others later.

## Rules we keep

- New work only; the demo shows only this repo's work, Open-Planet labelled as the existing base.
- Photos we have the right to: our own, Wikimedia Commons, Mapillary. Credits kept for every one. No logos close up.
- `private/` (own home tests) never committed.
- AI runs once, when a photo is added; the game only loads finished files (no slowdown).
- Spend: no API calls until the $150 credits show on Billing; the tool caps its own calls.

## Demo (3 min)

1. "This is Open-Planet, my world. It existed before." (10 s, labelled)
2. Turn on the plugin, drop in the photo. Before → after.
3. "Not just a game": a few real uses.
4. A famous public place (Beacon Isle, Plettenberg Bay, or a well-photographed Cape Town landmark).

## Days

| Day | Goal |
|---|---|
| Mon 26 | Repo, read a photo's GPS/compass, fetch OSM buildings, view cone, candidates (no AI needed) |
| Tue 27 | Meta API: SAM 3.1 outlines, Muse Wonder picks the building |
| Wed 28 | Straighten the front; build building.glb on the footprint |
| Thu 29 | Godot adapter: the first real building in Open-Planet ⭐ |
| Fri 30 | Muse Image clean-up; objects; credits |
| Sat 31 | Shoot the demo place |
| Sun 1 | Speed and polish |
| Mon 2 | Record the video |
| Tue 3 | Buffer, README, X post |
| Wed 4 | Submit early |

## Open

- Name (placelens for now).
- Demo place: Beacon Isle (only 2 Commons photos: own photos better) or a well-photographed Cape Town landmark.
- SAM 3D Objects / VGGT: not on the participant guide's model list (28 Oct), so they wouldn't score as Meta tooling.
  Props: SAM 3.1 cut-out + Muse size + Muse Image side/back views instead. Request Modal (first 1,000) and Baseten
  (Muse Spark Open, for Best Open Weight) credits.
- Muse Wonder vs Muse Spark for the pick: Wonder works on pay-as-you-go but took 53 s / $0.09 a photo. Compare.
