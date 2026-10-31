# Handoff

Where we are (28 Oct 2026, day 3 of the hackathon):

- Idea locked in: photo to real place (see PLAN.md). The user's idea. No more idea-hunting.
- Repo: `D:\Hackathon\placelens`, public on GitHub (github.com/inlighten420/placelens), MIT licence.
- **The user commits and pushes; Claude never does.** No Claude trailers in messages.
- Setup: `python -m venv .venv`, `.venv/Scripts/pip install -e .[dev]`. Key: `MODEL_API_KEY` in `.env` (gitignored).

Done:

- Day 1: `placelens info` / `placelens find` (GPS, compass, lens width; OSM buildings; view cone; map picture).
- Day 2 (commit aa31bbe): `placelens pick`. SAM 3.1 outlines the building; Muse Spark names a candidate with a
  confidence, reasons, floor count and distance; geometry lines the outline's edges up with each footprint while
  correcting the GPS (up to 10 m) and compass (up to 15°), using Muse's distance as the third measurement. Picks
  only when both agree and geometry has a clear winner; otherwise asks. Replies are saved in `out/pick/`, so
  re-runs are free (`--fresh` to call again). ~1.3 US cents a photo with Spark.
- Tested on the user's own house (4 iPhone photos in `private/`, all with GPS + compass): 3 picked correctly,
  1 correctly refused (that photo's GPS/compass put the house out of view; Muse said "none fit").
- Caveat: thresholds were tuned on those 4 photos of one house. Test on other places before trusting them.
- Private: the photos show the house number and number plates, and Muse reads them out. Keep them in `private/`
  and `out/`; blur before any video.
- Day 3: `placelens build photo...` (reuses pick's saved replies, so free). Writes
  `out/build/<osm id>/building.glb`, `place.geojson`, `wall_N.jpg` and a `check_<photo>.jpg` per photo (orange = the
  walls that photo painted). Several photos: each picks; the building most picked wins; floors = Muse's majority;
  each wall takes the photo that sees most of it, sharpest. A photo that fails is listed with why, not fatal.
  - `build.py`: height from OSM `height`, else `building:levels` x 3 m, else Muse's floors, else 2 floors; or
    `--floors` / `--height`. Footprint raised into walls + flat roof (shapely constrained triangulation); GLB written
    with numpy, no 3D library. Axes: metres, Y up, X east, Z south; origin = footprint centre at ground level,
    whose lat/lon is `anchor` in place.geojson. Walls numbered by `wall_edges` (counter-clockwise from above).
  - `front.py`: camera locked so the footprint's edges land on SAM's outline (search ±15 m, closed-form turn,
    priors on GPS/compass/Muse distance). Walls facing the camera get their in-view stretch projected (level phone,
    eye 1.5 m) and straightened by homography from the full-res photo into that wall's texture.
  - Finding: the OSM footprint can look narrower or wider than the real house, which pulls the locked camera too
    near (IMG_0166: 11 m vs Muse's 20 m). So across the photo the edges decide; up the photo, Muse's distance does
    (`depth_scale`, capped 0.5-2x). With that, IMG_0166's walls line up with the eaves.
  - IMG_0165: its GPS is fine (20 m from the house); its COMPASS is off (says east, ~33° beyond what fits). An
    earlier note here blamed the GPS; that was wrong.
  - 12-photo limits test (IMG_0169-0176 added 28 Oct: sides and back, tight passages; 0173-0175 on the 0.5x lens,
    0176 portrait looking down). Lens width is read from EXIF (0.5x = 14 mm = 102° across): handled.
    - SAM 3.1 returns an empty reply for "building" on the 0.5x photos but finds "house": `_pick` now tries
      "building", then "house" (`SAM_NOUNS`).
    - `build` treats its photos as one building: when some are sure, the rest are "rescued" if that building is
      within 30 m and Muse didn't confidently name another (`_rescue`). Rescued pictures count half.
    - First try let the lock turn the compass freely: it painted 0165's front onto the side wall and 0171's side
      onto the back. Now the lock turns at most 15° (`MAX_TURN`, as pick) and a photo is dropped if its outline still
      won't line up (`MISFIT` 3°), has no corner in view, or sits implausibly far/near (`_misplaced`).
    - Result: the tight back/side close-ups (0171, 0172, 0173, 0175, 0176) and 0165 are dropped with reasons;
      0169, 0170, 0174 line up but no wall picks them. In passages the GPS was 6-9 m off and compasses 7-98° off
      (0176 was pointing down; phone compasses are poor then). Placing such photos needs matching them against
      each other (shared features), not GPS + compass; that's the parked stitching work.
  - The house is on a hill: 2 storeys at the front, 1 at the back; the lower storey is smaller (user, 28 Oct).
    Each photo's camera now stands on the ground Muse's floor count implies (`_lock`: 2-floor house showing 1 floor
    -> ground 3 m up, eye 4.5 m), and paints walls only from that ground up (`WallPhoto.bottom`); below stays plain,
    as a foundation. Checked on IMG_0167: wall 2's eaves and ground line up.
  - Both edges in view: misfit allowed 5° (map footprint a bit off); one edge: 3° (`MISFIT`).
  - Gable pitch: an outline running off the top of the photo is skipped (was dragging 39° down to 22°).
  - `view.html` written by every build: model-viewer 3.5.0 (CDN) with the model embedded; opens facing the main
    photo's side. Local only (textures show the house number).
  - IMG_0173 (the back) now paints wall 1 with `--side IMG_0173=back`. Its GPS was ~20 m off, so it can't be placed
    alone. A told side looks for the camera in front of that side's walls (`_side_walls`, sides relative to the
    front = where the sure photos were taken from), with looser allowances (`TOLD_*`).
  - The lock's outline edges come from the walls' rows, not the roof overhang (`wall_span`). Each wall's rows come from
    the outline at its two ends (eaves above, ground below) where visible, else the level-phone projection
    (`fit_rows`); this fixes tilted close-ups like 0173.
  - A gable is painted only if the outline covers >= 60% of it (`GABLE_COVER`): the map's two back gables vs the
    real single wide one were showing sky. Now painted: the front-left gable only.
  - Current build: walls 1 (back, 0173), 2 (0167), 3/4/5 (0166) painted; wall 0 (east side) plain. 7 photos unused,
    each with a reason (compass beyond 15°, or no believable placement).
  - Muse names each unplaced photo's side automatically (`sides.py`, `_muse_sides`): shown next to the most
    confident front photo, it answers front/back/left/right or a corner; used if >= 60% sure and the photo's compass
    doesn't contradict it (within 75° of looking at that side). Muse-named sides free the GPS but keep the compass
    within 15°; `--side` (you) allows 35°. The user wants this fully automatic: no hand hints in the demo.
  - Walls are scored by how face-on and how wide they look (`FACE_ON`: more edge-on than 75° isn't used); a sliver
    from IMG_0169 had smeared the back wall.
  - Automatic build (no hints, 28 Oct): back wall from 0173 (Muse: back), left wall 2 from 0169 (stairs, checked),
    front from 0166. Unsure: 0170 puts the bay window on wall 3 and the brick panel on wall 4; 0167 had them on walls
    2 and 3. One of them is a wall off.
  - Muse critique probe (out/experiments/*_critique.txt): given the check overlays, Muse spots real misfits (the bay
    window sticking out ~1 m x 2.5 m on 0167, the garage gable too high, the back's single lower gable), numbers
    rough. Next: the critique loop (clearer overlays to the eaves; Muse critique -> wall moves / added parts /
    eave+gable height; verify by asking again; keep only changes that shrink the mismatch).
  - The user's house has 2 floors; all 4 photos together give 2 by majority (Muse said 2, 1, 2).

  - `roof.py`: footprint cut into rectangular blocks at inward corners (most rectangular, then longest pieces), each
    with a gable/hip/flat roof. Muse (one cached call per photo, `*_roof_*.txt`) votes kind, ridge toward/across the
    camera (turned into a compass bearing) and pitch. Pitch is then measured per block where a photo shows a gable's
    peak (SAM outline's top above the peak, through the locked camera); other blocks keep Muse's middle pitch.
    Gable ends standing on a painted wall are painted from the same picture (texture runs up to the peak).
    `--roof`, `--pitch` overrule. The user's house: 2 blocks (garage block, left block), two street-facing gables,
    39° measured on the left one, 20° (Muse) on the garage one (its peak is behind the tree).
  - Roof limit seen on the house: the real gables are narrower than the map's blocks and off-centre (the garage
    front has the two-storey part set back behind the braai terrace), so the drawn gables only approximate them.
  - Each Muse roof call cost ~3.5 US cents (more than pick's ~1.3).

- `placelens prop photo` (28 Oct): Muse names the object, its size and distance, and whether it's fully visible;
  SAM 3.1 cuts it out (asked for Muse's noun, or `--what`); the outline (matplotlib contour) is extruded to its
  real depth with the cut-out PNG front and back (alphaMode MASK, double-sided) (`prop.py`). GPS + compass + its
  place across the photo + distance put it on the map, front facing the camera (`prop.geojson`). Warns when the
  object runs off the photo or is hidden. Tested on the rain tank in IMG_0168: works, but that photo cuts it off and
  a bush hides it (both warned). Needs proper prop photos from the user. Next for props: Muse Image draws the back.
  - Round things (Muse's `round: true`, or `--shape round`) are turned on a lathe (`lathe`): the outline's half-width
    per height spun round an upright axis, the photo wrapped round; if the object runs off one side of the photo the
    clean side's picture goes all the way round. The rain tank is now a proper 1.82 m x 1.30 m cylinder.
  - Walk-around props (`placelens prop p1 p2 ...`, `carve.py`): SAM 3.1 cut-out per photo; angle round the object
    from the compass, checked by Muse naming the side (it wins if they differ by > 60°); camera height found by
    carving at 0-60° and keeping the angle whose carving, seen again from every camera, best matches the cut-outs
    (`agreement`; "most volume" was biased to steep angles). Visual hull at 72^3 voxels, outside-the-frame carved
    unless the object runs off that edge, biggest piece kept, marching cubes (scikit-image, new dependency) after a
    gaussian blur, vertex colours from the photos facing each spot (COLOR_0 in the GLB).
    Teddy bear (IMG_0177-0185, 8 photos, 28 Oct): Muse agreed with every compass angle; 20° above found; a
    recognisable 0.61 x 0.60 x 0.68 m bear (face, arms, legs). ~26 US cents.
  - Muse Image (API: POST /v1/images/edits, multipart `image[0]`.. numbered, model muse-image-1.0, ~US$0.01 an
    image, returns b64 WebP; `MetaClient.draw`). It won't draw a view from directly above (tried twice: drew front,
    then back), but draws back/left/right well. Muse Spark checks each drawing's side (`imagine.py`): wrong ones are
    dropped (it caught both top-view failures and a "right" that was a second left).
  - One photo, not round -> `placelens prop photo`: Muse Image draws back/left/right, Muse checks, SAM cuts each out,
    carved with the real photo (`misses=0`; drawings widened 3% so they only carve clear outside; they paint at 5%
    weight); a symmetric object's missing side is the other side mirrored. Unpainted spots take the nearest colour.
    Teddy from IMG_0177 alone: whole bear, all sides coloured, softer than the 8-photo carve. ~11 US cents.
  - The 8-photo teddy has a lump between the legs: a hollow no photo sees into. The user is taking a photo from
    above; Muse answers "above" for it, its turn (15° steps) and tilt (65/75/85°) found by fit (`turn_overhead`).
    It got a firm vote (`firm`: an overhead photo can't be outvoted, as `misses` let one view be ignored and so the
    top photo could never carve alone). IMG_0186 (top): fitted 105°, legs crisper, but the blob low between the
    legs barely changed (37 vs 38 surface points): fit a few cm off, or partly the bear's real seat. Parked.
  - Changing a Muse prompt re-asks every photo (the cache key is the prompt): the "above" option cost ~37 cents.
- User's call (28 Oct): the house shape is good enough for the demo; critique loop parked; props next.

- `placelens pack NAME folders...` (`pack.py`): manifest.json (format placelens-pack v1: items with kind, model,
  lat/lon, yaw_deg about Y in the X east / Z south frame, building `replaces` OSM id and footprint), credits.json,
  the .glb files. Pack "home": the house, the rain tank, the 9-photo bear.
- Godot adapter (`adapters/godot/placelens_packs.gd`, an autoload; `install.py` adds one autoload line to
  project.godot and copies the script, `--uninstall` restores it exactly). From a research pass on Open-Planet
  (D:\GeodrivePublic = the open-planet repo; D:\GeodriveBaseline a local copy): Godot axes metres, X east, Y up, Z
  south (same as PlaceLens); `world.from_lat_lon`, `ground_height`, `contains_lat_lon`, `is_ready`,
  `remove_building(osm way id)`, `get/set_removed_buildings`, signal `origin_shifted(offset)` (nodes do
  `global_position += offset`), `main.teleport(lat, lon, name)`, start with `-- --lat= --lon=`. No autoloads or
  plugin folder exist; packs folder is `GameSettings.saved_dir()/placelens/` (editor: <project>/../Saved).
  Open-Planet itself must not be committed into this repo; the adapter is new work here.

Known limits of the front: phone tilt unknown;
trees, walls and cars in front are baked into the texture (Fri: Muse Image); parts of an L-shape hidden behind
another wing get that wing's pixels.

User feedback (28 Oct): first thought the garage was missing; then confirmed the map is right. Walls 4-5 are a
two-storey garage block (closed braai area on top, under a gable roof); walls 2-3 are the side where steps go up to
the entrance. The only real shape error is a ~1 m stair recess there. The footprint is
`source=microsoft/BuildingFootprints` (AI-traced from satellite). The user wants it fully autonomous (no manual
tracing). The user is taking more photos from all round the house (street from 3-4 spots, front corners diagonally,
sides, back), into `private/`.

Corner probe (out/experiments/, ~9 US cents): SAM 3.1 found no "garage" (doors hidden behind the street wall). Muse
Spark's corner positions are off by up to ~10% of the photo width and its part names change between photos ("east
wing" vs "right wing"), but it describes parts, floors and how far they stick out consistently. So for stitching:
exact corners from SAM outlines / image edges, matching by geometry, Muse for description only. Stitching is parked
(a day's work for a 1 m recess here); it matters for badly mapped or unmapped buildings.

Next:

1. Place pack: `credits.json` (photo source and licence per texture). place.geojson is done.
2. Later: shape from photos (corners triangulated across photos, joint solve of corners + every photo's GPS/compass,
   map footprint as the prior); place photos whose compass/GPS are far off (IMG_0165, the passage close-ups) by matching photos to each other.
3. Thu: Godot adapter into Open-Planet. Fri: Muse Image clean-up (the tree in front of the house), objects.
4. Later: roof sections from the photos (offset or narrower gables); Muse Wonder vs Spark on more photos; Overture fallback; `placelens item` (PLAN.md 10).

User to-dos: check the $150 credits on Billing; claim Modal; ask @CV whether SAM 3D Objects counts as a Meta
open-weight model; pick the demo place (Beacon Isle with own photos, or a Cape Town landmark).

Wed 28 Oct, later: houses are now built, not wallpapered (the user prefers it: "much better").

- `placelens build` makes plain walls in colours Muse Spark reads from up to 3 photos at once (`look.py`, LOOK_PROMPT,
  one cached call, ~1.5c). Windows and doors SAM 3.1 finds in the straightened wall pictures are cut in and fitted
  (`build._built`, `_fittings`, `_box`): set-back dark glass, frames, glazing bars (PANE 0.9 m), a bar across tall
  windows, sills; doors a leaf and handle; garage doors ribbed panels (none found on the house: behind the brick wall).
  `--photo-walls` brings back the photos on the walls. 68 tests pass.
- Open-Planet segfaults (exit 139) on every quit, even `--quit-after`, with nothing in Godot's log. Not yet checked
  whether it happens without the PlaceLens autoload (one run with `install.py --uninstall`, then reinstall).

Thu 29 Oct: Muse Spark Open (Baseten) is the reasoning model.

- Budget rule (user, 29 Oct): **only free credits.** Baseten gave $100 + 2M tokens/min for
  `meta-models/muse-spark-open-1026` (key `BASETEN_API_KEY` in .env); Modal didn't come through. Every Muse question
  defaults to `--model open`; paid Meta API reasoning (Spark/Wonder) is opt-in only. SAM 3.1 and Muse Image exist
  only on the Meta API: user to confirm the Meta $150 credits are active (Billing) before leaning on them more.
- Baseten probe (`scripts/open_probe.py`): JSON mode, json_schema, tool calling, `reasoning_effort`, several images
  per call and 8 parallel calls all work; reasoning comes back in `reasoning_content`, never in the answer.
- `muse.py`: every Muse question goes through `ask()` (each vote saved as `<reply>_v<i>.txt`, vote 0 = the old
  single reply; missing votes asked in parallel) and a merge (`merge_verdicts/sides/props/looks`: majority choice,
  median numbers, confidence = agreement x mean). Default 5 votes for open, 1 for paid models (`--votes`).
- Pick with open x5 on 0165-0168: same as Spark (3 picked, 0165 "none"); 0168 went from ASK to picked.
  `scripts/model_eval.py` tabulates saved answers per model: Open matches Spark on picks/floors/distances, but is
  weak at naming a photo's side (splits votes, says "other"). `--second-opinion` (PAID) asks Spark then.
- `build` picks photos 4 at a time (`BUILD_PARALLEL`; matplotlib + OSM cache under `_SHARED` lock); side-naming too.
- `critique.py` + `build --refine`: the critique loop. The model's faces are projected through each locked camera;
  mismatch = per-column gap between the model silhouette's top and SAM's outline top (bottom ignored: gates, cars),
  worst 20% of columns dropped. Spark Open (5 votes, reasoning_effort high) sees photo + overlay and answers edits
  from a menu (move_wall, notch, eaves, pitch); edits >= 25% of answers propose are tried at 0.5/1/1.5x and kept only
  if the mean mismatch drops (2%; 5% for footprint edits) AND more photos improve than worsen. Cameras re-lock onto
  each trial shape (`_relock`); photos with > 2x the median mismatch are left out (`steady_views`). Rounds saved in
  out/build/<id>/critique/ (overlays, answers, history.json).
  - Results on the house: v1 (wireframe overlay, mean-only rule) moved wall 4 in 2.4 m: wrong (user: map is right).
    v2 (front faces only, majority rule): only eaves +0.5 m kept, 0.084 -> 0.078; Open's critiques contradict
    themselves (eaves up 17 vs down 18) and never found the stair recess; it wants flatter roofs than measured.
  - v3 (free only, 29 Oct): photos > 2x the median mismatch left out, cameras re-lock onto each trial shape,
    reasoning_effort high, and a corner measure (`vertical_lines`: long vertical edges inside SAM's outline, found
    locally; `corner_error` both ways, so a recess the model lacks counts). Open now proposes notches on wall 3 (the
    entrance side, where the real stair recess is) but at the wrong size; nothing kept. Diminishing returns on this
    house (tight close-ups, poor GPS/compass); parked in favour of the agent and the demo shoot.
- Side-naming with Open alone: 7 votes, effort high (`SIDE_VOTES`, `SIDE_EFFORT`): placed IMG_0165 only; the back
  (0173) stays "other". `--second-opinion` (PAID Spark) brings 0173 back.
- Props rebuilt free with Open (SAM cached; pin `--what` to the cached noun): teddy 9 photos 0.59 x 0.58 x 0.66 m
  (Open agreed with 6 compass angles, found the overhead shot); tank 1.52 x 1.20 m, 47% sure (photo cuts it off).
- `PLACELENS_META_OFF=1` blocks any uncached paid Meta call (`PaidCallBlocked`); a build then skips windows/doors.
  `out/open_usage.jsonl` logs every Spark Open call; `scripts/open_usage.py` totals it.
- `placelens survey FOLDER` (`survey.py`): Spark Open runs tool calls (list_photos, describe, pick, build, prop, pack,
  ask_person); commands run as subprocesses with --json into out/survey/<folder>/work (`--replies out/pick` shares
  saved answers). Budget: 15 heavy calls, 10 looks; what's built is packed even if it forgets. Run 3 on private/
  (free mode): 12 looks, ONE build of all 12 house photos (refine), one teddy prop (blocked only because free mode
  had no SAM reply saved for the noun "teddy bear"), pack; honest summary with every unused photo's reason. 3 min.
  Runs 1-2 were wasteful (described/picked every photo; split the house in two by time; overwrote out/build).

Next (from Thu 29 Oct evening):

1. User: claim/check the Meta $150 credits (Billing) before 4 Nov 21:00 ET; then the agent can run with SAM allowed
   (`placelens survey private` without PLACELENS_META_OFF) so the demo shows house + teddy together (~2 US cents).
2. User: commit today's work (muse.py, critique.py, survey.py, meta.py, cli.py, scripts, tests, HANDOFF.md).
3. Sat 31: shoot the demo place with the critique loop in mind: street side from 3-4 spots, front corners, fewer
   tight close-ups; a whole, unblocked photo of any prop.
4. Then: agent on the demo folder (transcript + summary for the video, Best Agentic); Godot adapter check;
   video Tue 3; submit Wed 4.

Sat 31 Oct: making things from inside the game, with notes.

- Notes: `--notes` on `pick`/`build`/`prop`/`survey` (`survey --notes-file` too). `muse.noted` adds them to every
  Muse question (pick, roof, look, prop, Muse Image drawings); without notes prompts are unchanged, so caches hold.
  New settings notes can turn into: `build --symmetric` (`build.mirror_walls`: a wall no photo saw copies the
  opposite wall's picture and openings, flipped; opposite = facing within 10° of the other way, length within 15%),
  `prop --symmetric`, `prop --width`. The survey agent gets the notes as its brief, its build/prop tools take
  floors/height/width/round/symmetric/notes, and it's told to report where results disagree with the notes.
- `placelens -C DIR`: run in DIR (where .env, cache/ and out/ are), for a game starting it from elsewhere.
- `survey --progress FILE` (a JSON line per tool start/end, last line `done` with summary, questions, pack) and
  `--pack-to DIR` (the pack copied there, named after the photos' folder so it can't overwrite another pack).
- Packs: a prop without GPS/compass is packed with lat/lon null (the game places it) instead of failing.
- Godot: `adapters/godot/placelens_maker.gd` (child of the packs autoload). F7 opens a panel: photos (native file
  picker), notes, objects where the photos were taken / where I'm looking, "allow paid Meta calls" (off = sets
  PLACELENS_META_OFF=1 for the child). The game pauses while it's open (as Open-Planet's menu does; not online),
  because the game polls keys every frame. Make copies the photos to out/game/<job>/, starts
  `python -m placelens.cli -C <repo> survey ...`, polls progress.jsonl every 0.5 s into a line along the top, then
  places props at the aim spot (manifest rewritten so they stay), adds the pack live (`add_pack`), and shows the
  agent's summary and questions in the panel with Go there. `install.py` copies both scripts and writes
  placelens.cfg (repo + .venv python). Installed in D:\GeodriveBaseline; headless run loads clean.
- Not yet tried by hand in the game window (panel layout, native picker, the pause). New photos need SAM 3.1
  (paid Meta API): with the box unticked only photos with saved SAM answers work.
