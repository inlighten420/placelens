"""placelens: command line for people and AI agents alike. Every command takes --json."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path

from . import __version__
from .osm import fetch_buildings
from .photo import read_camera
from .render import draw_map
from .view import LocalFrame, find_candidates

EXIT_MISSING = 2  # the photo lacks GPS or compass; pass them by hand
EXIT_ASK = 3  # not sure which building; choose one with --building N
SAM_NOUNS = ("building", "house")  # what SAM is asked to outline, in turn, until it finds something
RESCUE_REACH = 30.0  # metres: a photo that couldn't tell on its own may still show the building others agree on
BELIEVABLE = 2.5  # a locked camera this many times nearer or further than the building looks is wrongly placed
# Degrees a locked camera's footprint edges may stay off the outline's. With both edges in view, being off means the
# map's footprint is a little narrower or wider than the real one; with one, it means the photo can't be lined up.
MISFIT = {1: 3.0, 2: 5.0}
# A photo someone has said the side of ("this is the back") may have its GPS and compass further off.
TOLD_REACH, TOLD_GPS_ERROR, TOLD_TURN, TOLD_DISTANCE, TOLD_MISFIT = 12.0, 25.0, 35.0, 8.0, 2.0
SIDE_SPREAD = 45  # a wall belongs to a side if it faces within this many degrees of it
LOOK_PHOTOS = 3  # photos Muse looks at, together, for the building's colours
SIDE_VOTES, SIDE_EFFORT = 7, "high"  # Spark Open naming a photo's side
BUILD_PARALLEL = 4  # photos `build` picks at once (each holds a full-size photo in memory)
_SHARED = threading.Lock()  # matplotlib and the OSM cache aren't safe to use from several threads at once
OPENING_KINDS = ("garage door", "door", "window")  # what SAM 3.1 is asked to find in each wall, doors first
OPENING_MIN = 0.25  # metres: smaller is a vent or a light, not a window
OPENING_EDGE = 0.05  # metres kept between an opening and the wall's top or bottom
GABLE_COVER = 0.6  # share of a gable the outline must cover for its picture to be used
RESCUED_WEIGHT = 0.5  # a rescued photo's pictures count for less: they fill walls the sure photos don't show
FACE_ON = 0.26  # cos 75°: a wall seen more edge-on than this from a photo isn't painted from it
WALK_DISAGREE = 60  # degrees: a compass further than this from the side Muse sees is wrong (indoors, often)
BORDER_PX = 3  # an object this close to the photo's edge runs off it
DEFAULT_PROP_HEIGHT = 1.0  # metres, when Muse gives no height
DEFAULT_DEPTH_SHARE = 0.5  # depth as a share of width, when Muse gives no depth
SHARP_ENOUGH = 150  # pixels per metre of wall; a sharper picture of it isn't worth more
ATTRIBUTION = "Building data © OpenStreetMap contributors, ODbL (openstreetmap.org/copyright)"


class Failed(Exception):
    """A command can't go on. main() reports it; `build` skips a photo that fails and carries on with the rest."""

    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.msg, self.code = msg, code


def _fail(msg: str, as_json: bool = False, code: int = 1):
    raise Failed(msg, code)


def cmd_info(args) -> None:
    cam = read_camera(args.photo)
    info = {
        "photo": str(args.photo),
        "has_gps": cam.lat is not None,
        "has_compass": cam.heading is not None,
        "heading": cam.heading,
        "fov": round(cam.fov, 1),
        "fov_source": cam.fov_source,
        "size": [cam.width, cam.height],
    }
    if args.json:
        print(json.dumps(info))
        return
    print(f"photo:    {args.photo} ({cam.width}x{cam.height})")
    print(f"GPS:      {'yes' if info['has_gps'] else 'NO (pass --lat/--lon to find)'}")
    print(f"compass:  {f'{cam.heading:.0f}°' if cam.heading is not None else 'NO (pass --heading to find)'}")
    print(f"lens:     {cam.fov:.0f}° across ({cam.fov_source})")


def _locate(args):
    """GPS, compass and lens (photo or flags), the buildings around, and the ones in view."""
    cam = read_camera(args.photo)
    lat = args.lat if args.lat is not None else cam.lat
    lon = args.lon if args.lon is not None else cam.lon
    heading = args.heading if args.heading is not None else cam.heading
    fov = args.fov if args.fov is not None else cam.fov
    if lat is None or lon is None:
        _fail("photo has no GPS; pass --lat and --lon", args.json, EXIT_MISSING)
    if heading is None:
        _fail("photo has no compass direction; pass --heading (degrees, 0 = north, 90 = east)", args.json, EXIT_MISSING)

    try:
        buildings = fetch_buildings(lat, lon, args.radius, use_cache=not args.no_cache)
    except Exception as e:  # network, Overpass busy
        _fail(f"could not fetch OpenStreetMap buildings: {e}", args.json)

    frame = LocalFrame(lat, lon)
    candidates = find_candidates(buildings, frame, heading, fov, reach=args.radius, margin=args.margin)
    return cam, frame, buildings, candidates, heading, fov


def _row(c, frame) -> dict:
    centre = c.polygon.centroid
    t = c.building.tags
    return {
        "rank": c.rank,
        "osm_id": c.building.osm_id,
        "osm_url": c.building.url,
        "name": t.get("name"),
        "levels": t.get("building:levels"),
        "height": t.get("height"),
        "view_share": round(c.share, 3),
        "distance_m": round(c.distance, 1),
        "centre": [round(v, 6) for v in frame.to_latlon(centre.x, centre.y)],
    }


def cmd_find(args) -> None:
    cam, frame, buildings, candidates, heading, fov = _locate(args)

    map_path = None
    if not args.no_map:
        from shapely.geometry import Polygon

        polys = [Polygon([frame.to_xy(*p) for p in b.outline]) for b in buildings]
        map_path = draw_map(
            Path(args.out) / f"{Path(args.photo).stem}_map.png",
            polys, candidates, heading, fov, args.radius, args.margin, args.top,
        )

    rows = [_row(c, frame) for c in candidates[: args.top]]

    if args.json:
        print(json.dumps({
            "camera": {"heading": heading, "fov": round(fov, 1), "fov_source": cam.fov_source},
            "buildings_nearby": len(buildings),
            "candidates": rows,
            "map": str(map_path) if map_path else None,
            "attribution": ATTRIBUTION,
        }))
        return

    print(f"{len(buildings)} buildings within {args.radius} m; facing {heading:.0f}°, lens {fov:.0f}° across")
    if not rows:
        print("No mapped buildings in view. (Overture fallback and estimated footprints are coming.)")
    for r in rows:
        label = r["name"] or r["osm_id"]
        share, dist = r["view_share"], r["distance_m"]
        print(f"  {r['rank']}. {label:<32} {share:>5.0%} of view, {dist:>5.1f} m away  {r['osm_url']}")
    if map_path:
        print(f"map: {map_path}")


def _pick(args) -> dict:
    """Run the pick (reusing saved replies) and return its result, plus the chosen candidate and frame under "_"."""
    import hashlib

    from PIL import Image

    from .meta import MODELS, MetaClient, MissingKeyError, parse_segments
    from .muse import ask, merge_verdicts, noted
    from .pick import build_prompt, decide, line_up, parse_verdict, photo_span, prepare_photo
    from .render import draw_check

    with _SHARED:
        cam, frame, buildings, candidates, heading, fov = _locate(args)
    candidates = candidates[: args.top]
    if not candidates:
        _fail("no mapped buildings in view; check the compass with `placelens find` or widen --margin", args.json)
    try:
        meta = MetaClient()
    except MissingKeyError as e:
        _fail(str(e), args.json)

    # Replies are saved next to the outputs, so re-running costs nothing unless --fresh.
    out = Path(args.replies) if getattr(args, "replies", None) else Path(args.out) / "pick"
    stem = Path(args.photo).stem
    small = prepare_photo(args.photo, out / f"{stem}.jpg")
    width = Image.open(small).width

    segments = []
    for noun in SAM_NOUNS:  # some photos only answer to the second
        sam_file = out / (f"{stem}_sam.txt" if noun == SAM_NOUNS[0] else f"{stem}_sam_{noun}.txt")
        if sam_file.exists() and not args.fresh:
            segments = parse_segments(sam_file.read_text(encoding="utf-8"))
        else:
            segments, raw = meta.segment(small, noun)
            sam_file.write_text(raw, encoding="utf-8")
        if segments:
            break
    main = segments[0] if segments else None  # biggest: the building the photo is of
    span = photo_span(main.box, width, fov) if main else None

    model = MODELS[args.model]
    prompt = noted(build_prompt(candidates), getattr(args, "notes", None))
    muse_file = out / f"{stem}_{args.model}_{hashlib.sha1(prompt.encode()).hexdigest()[:8]}.txt"
    verdict = merge_verdicts([parse_verdict(a) for a in
                              ask(meta, prompt, small, muse_file, model, _votes(args, model), args.fresh)])
    distance = float(verdict.distance_m) if isinstance(verdict.distance_m, (int, float)) else None
    lineups = line_up(candidates, span, heading, fov, args.margin, args.move, distance) if span else []
    decision = decide(lineups, verdict)

    by_rank = {c.rank: c for c in candidates}
    chosen = by_rank.get(decision.rank)
    fix = next((lu for lu in lineups if lu.candidate is chosen), None)
    check_path = draw_check(out / f"{stem}_check.jpg", small, main, fix.span if fix else None, fov)
    map_path = None
    if not args.no_map:
        from shapely.geometry import Polygon

        polys = [Polygon([frame.to_xy(*p) for p in b.outline]) for b in buildings]
        corrected = (*fix.move, heading + fix.shift) if fix else None
        zoom = min(args.radius, max(40, 2 * chosen.distance)) if chosen else None
        with _SHARED:
            map_path = draw_map(
                out / f"{stem}_map.png", polys, candidates, heading, fov, args.radius, args.margin, args.top,
                corrected=corrected, zoom=zoom,
            )

    result = {
        "status": decision.status,
        "why": decision.why,
        "building": _row(chosen, frame) if chosen else None,
        "muse": {"model": model, **vars(verdict)},
        "geometry": {
            "photo_span": [round(v, 1) for v in span] if span else None,
            "line_up": [{"rank": lu.candidate.rank, "fit": round(lu.fit, 3), "score": round(lu.score, 3),
                         "compass_off": lu.shift, "gps_off_m": [*lu.move]}
                        for lu in lineups],
        },
        "candidates": [_row(c, frame) for c in candidates],
        "check": str(check_path),
        "map": str(map_path) if map_path else None,
        "cost_usd": round(meta.spent, 4),
        "attribution": ATTRIBUTION,
    }
    result["_"] = {"chosen": chosen, "by_rank": by_rank, "frame": frame, "verdict": verdict, "lineups": lineups,
                   "span": span, "model": model, "main": main, "small": small, "heading": heading, "fov": fov,
                   "meta": meta, "replies": out, "stem": stem, "fresh": args.fresh, "buildings": buildings,
                   "votes": _votes(args, model)}
    return result


def cmd_pick(args) -> None:
    result = _pick(args)
    ctx = result.pop("_")
    if args.json:
        print(json.dumps(result))
        return

    chosen, verdict, lineups, span, model = ctx["chosen"], ctx["verdict"], ctx["lineups"], ctx["span"], ctx["model"]
    decision_rank = result["building"]["rank"] if chosen else None
    check_path, map_path = result["check"], result["map"]
    if span:
        print(f"SAM:      building spans {span[0]:+.0f}° to {span[1]:+.0f}° across the photo")
    else:
        print("SAM:      no building outlined")
    for lu in lineups[:3]:
        fix_note = f"if GPS {lu.moved:.0f} m off, compass {lu.shift:+.0f}°"
        print(f"line-up:  {lu.candidate.rank}. {lu.candidate.building.osm_id:<22} "
              f"score {lu.score:>4.0%}  (fits {lu.fit:.0%} {fix_note})")
    choice = verdict.choice if verdict.choice is not None else "none"
    print(f"Muse:     picks {choice}, {verdict.confidence:.0%} sure ({model})")
    for r in verdict.reasons:
        print(f"          - {r}")
    if result["status"] == "picked":
        print(f"PICKED    {chosen.rank}. {chosen.building.osm_id}  {chosen.building.url}")
        print(f"          {result['why']}")
    else:
        guess = f" Best guess: {decision_rank}." if decision_rank else ""
        print(f"ASK       not sure: {result['why']}.{guess} Which numbered building is it? (see the map, "
              f"then `placelens build --building N`)")
    print(f"check:    {check_path}")
    if map_path:
        print(f"map:      {map_path}")
    print(f"cost:     ~${result['cost_usd']:.4f} (0 if replies were reused)")


def _lock(ctx, chosen, height: float, walls: list[int] | None = None, max_turn: float | None = None):
    """This photo's camera, locked so the footprint's edges land on SAM's outline (at the walls, below the roof's
    overhang), standing on the ground Muse's floor count implies: a 2-floor house photographed showing 1 floor was
    taken a floor uphill.

    `walls`: the walls someone said this photo shows. The camera is then looked for in front of them, allowing the
    GPS and compass to be further off, since what they'd otherwise have to decide is already known."""
    import math

    from PIL import Image

    from .build import FLOOR, wall_edges
    from .front import EYE, lock_camera, wall_span

    distance = ctx["verdict"].distance_m
    distance = float(distance) if isinstance(distance, (int, float)) and distance > 0 else None
    size = Image.open(ctx["small"]).size
    box = wall_span(ctx["main"].full_mask(size), ctx["main"].box)
    if walls:
        edges = [wall_edges(chosen.polygon)[w] for w in walls]
        mx = sum(a[0] + b[0] for a, b in edges) / (2 * len(edges))
        my = sum(a[1] + b[1] for a, b in edges) / (2 * len(edges))
        nx = sum((b[1] - a[1]) for a, b in edges)
        ny = sum(-(b[0] - a[0]) for a, b in edges)
        n = math.hypot(nx, ny) or 1.0
        away = distance or TOLD_DISTANCE
        around = (mx + nx / n * away, my + ny / n * away)
        cam = lock_camera(chosen.polygon, box, size, ctx["heading"], ctx["fov"], distance,
                          max_turn=max_turn or TOLD_TURN, reach=TOLD_REACH, walls=walls, around=around,
                          gps_error=TOLD_GPS_ERROR)
    else:
        cam = lock_camera(chosen.polygon, box, size, ctx["heading"], ctx["fov"], distance)
    floors = ctx["verdict"].floors
    if isinstance(floors, int) and floors > 0:
        cam.ground = min(max(height - floors * FLOOR, 0.0), height - FLOOR)
        cam.eye = cam.ground + EYE
    return cam


def _sides(args) -> dict[str, str]:
    """--side PHOTO=SIDE, by photo file name (with or without its extension)."""
    from .sides import CORNERS, SIDES

    sides = {}
    for item in args.side:
        photo, _, side = item.partition("=")
        if side.lower() not in SIDES and side.lower() not in CORNERS:
            _fail(f"--side {item}: the side must be one of {', '.join([*SIDES, *sorted(CORNERS)])}", args.json)
        sides[Path(photo).stem.lower()] = side.lower()
    return sides


def _front_bearing(used: list[dict]) -> float | None:
    """The compass bearing the building's front faces: back towards where the photos sure on their own were taken
    from (people photograph the front)."""
    import math

    sure = [r for r in used if not r.get("rescued") and not r.get("side")]
    if not sure:
        return None
    x = sum(math.sin(math.radians(r["ctx"]["heading"] + 180)) for r in sure)
    y = sum(math.cos(math.radians(r["ctx"]["heading"] + 180)) for r in sure)
    return math.degrees(math.atan2(x, y)) % 360


def _side_walls(poly, side: str, front: float | None) -> list[int] | None:
    """The walls on one side of the footprint (or both sides of a corner): those facing within SIDE_SPREAD of it."""
    import math

    from .build import wall_edges
    from .sides import side_offsets

    if front is None:
        return None
    wants = [(front + o) % 360 for o in side_offsets(side)]
    walls = []
    for i, (a, b) in enumerate(wall_edges(poly)):
        facing = math.degrees(math.atan2(b[1] - a[1], -(b[0] - a[0]))) % 360
        near = any(abs((facing - want + 180) % 360 - 180) <= SIDE_SPREAD for want in wants)
        if near and math.dist(a, b) >= 1.0:
            walls.append(i)
    return walls or None


def _muse_sides(args, runs: list[dict]) -> None:
    """For each photo that couldn't place itself, Muse says which side of the building it shows, by comparing it with
    the photo the others agree shows the front. Used only when Muse is fairly sure and the photo's compass doesn't
    contradict it. One small call per photo, saved like pick's."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from .meta import OPEN, SPARK
    from .muse import ask, merge_sides
    from .sides import SIDE_PROMPT, SURE_SIDE, compass_agrees, parse_side

    sure = [r for r in runs if r["used"]]
    front = _front_bearing(sure)
    if front is None:
        return
    ref = max(sure, key=lambda r: r["ctx"]["verdict"].confidence)["ctx"]
    asked = hashlib.sha1((SIDE_PROMPT + ref["stem"]).encode()).hexdigest()[:8]

    def unsure(view) -> bool:
        return view.side in (None, "other") or view.confidence < SURE_SIDE

    def side_of(r: dict):
        """Muse's side, and who said it. Spark Open's votes split on sides and backs (IMG_0173, the back: "other",
        3 of 5); with --second-opinion (paid, off by default) the Meta API Spark is then asked once (~1 US cent)."""
        ctx = r["ctx"]
        if ctx["model"] == OPEN:  # sides are Open's weak spot: it thinks longer and gets more votes (still free)
            reply = ctx["replies"] / f"{ctx['stem']}_{args.model}_side_{asked}_{SIDE_EFFORT}.txt"
            votes, effort = max(ctx["votes"], SIDE_VOTES), SIDE_EFFORT
        else:
            reply = ctx["replies"] / f"{ctx['stem']}_{args.model}_side_{asked}.txt"
            votes, effort = ctx["votes"], None
        view = merge_sides([parse_side(a) for a in ask(ctx["meta"], SIDE_PROMPT, [ref["small"], ctx["small"]], reply,
                                                       ctx["model"], votes, ctx["fresh"], effort)])
        if not unsure(view) or ctx["model"] != OPEN or not args.second_opinion:
            return view, "Muse"
        reply = ctx["replies"] / f"{ctx['stem']}_spark_side_{asked}.txt"
        second = parse_side(ask(ctx["meta"], SIDE_PROMPT, [ref["small"], ctx["small"]], reply, SPARK,
                                fresh=ctx["fresh"])[0])
        return (second, "Muse Spark, Open unsure") if not unsure(second) else (view, "Muse")

    todo = [r for r in runs if not r["used"] and not r["side"] and "ctx" in r]
    with ThreadPoolExecutor(BUILD_PARALLEL) as pool:
        views = list(pool.map(side_of, todo))
    for r, (view, who) in zip(todo, views, strict=True):
        ctx = r["ctx"]
        if unsure(view):
            r["side_note"] = f"Muse can't tell which side ({view.side or 'no answer'}, {view.confidence:.0%})"
        elif not compass_agrees(view.side, ctx["heading"], front):
            r["side_note"] = f"{who} says {view.side}, but the compass points elsewhere"
        else:
            r["side"], r["side_from"] = view.side, f"{who} {view.confidence:.0%}"


def _rescue(runs: list[dict]) -> None:
    """Photos given to `build` together show one building. When some of them are sure which it is, a photo that
    couldn't tell on its own (too close, or its compass off) is taken to show it too, if that building is within
    RESCUE_REACH of where it was taken and Muse didn't confidently name another one."""
    from shapely.geometry import Point, Polygon

    from .pick import SURE
    from .view import Candidate

    sure = [r for r in runs if r["used"]]
    if not sure:
        return
    votes = [r["chosen"].building.osm_id for r in sure]
    agreed = max(set(votes), key=votes.count)
    for r in runs:
        if r["used"] or "ctx" not in r:
            continue
        ctx = r["ctx"]
        verdict = ctx["verdict"]
        other = ctx["by_rank"].get(verdict.choice)
        told = r.get("side") is not None  # someone said which side it shows: that settles which building
        if not told and other is not None and other.building.osm_id != agreed and verdict.confidence >= SURE:
            continue
        building = next((b for b in ctx["buildings"] if b.osm_id == agreed), None)
        if building is None:
            continue
        poly = Polygon([ctx["frame"].to_xy(*q) for q in building.outline])
        distance = poly.exterior.distance(Point(0, 0))
        if distance > RESCUE_REACH and not told:
            continue
        r["chosen"] = next((c for c in ctx["by_rank"].values() if c.building.osm_id == agreed), None) or \
            Candidate(building, poly, 0.0, distance)
        r["used"], r["rescued"], r["why"] = True, True, f"on its own: {r['why']}; the other photos agree on {agreed}"


def _misfit_allowed(cam, side: str | None = None) -> float:
    return MISFIT.get(cam.edges, 0.0) + (TOLD_MISFIT if side else 0.0)


def _misplaced(cam, chosen, ctx, side: str | None = None) -> str | None:
    """Why this photo's locked camera can't be trusted to paint walls, or None if it can."""
    from shapely.geometry import Point

    from .front import MAX_TURN

    if cam.edges == 0:
        return "no corner of the building in the photo to line it up by"
    if cam.misfit > _misfit_allowed(cam, side):
        return f"its outline won't line up with the map within {MAX_TURN:.0f}° of its compass ({cam.misfit:.0f}° out)"
    distance = ctx["verdict"].distance_m
    if isinstance(distance, (int, float)) and distance > 0:
        nearest = max(chosen.polygon.exterior.distance(Point(cam.x, cam.y)), 1.0)
        if not 1 / BELIEVABLE <= distance / nearest <= BELIEVABLE:
            return f"lined up {nearest:.0f} m from the building, but it looks {distance:.0f} m away"
    return None


def _wall_views(photo_path, ctx, chosen, height: float, roof, cam, footprint) -> tuple[dict, list[dict]]:
    """Straighten every wall this photo shows. Returns the camera and, per wall, its picture with how well this
    photo sees it (`score`: share of the wall seen, times sharpness), and where each gable standing on it falls in
    the photo with how much of that the building's outline covers (a gable the real roof doesn't have is sky)."""
    import io
    import math

    from PIL import Image, ImageOps

    from .build import wall_edges
    from .front import fit_rows, straighten, texture_size, visible_walls

    with Image.open(photo_path) as img:  # full resolution for the textures
        photo = ImageOps.exif_transpose(img).convert("RGB")
    scale = photo.width / cam.width
    edges = wall_edges(chosen.polygon)
    views = []
    mask = ctx["main"].full_mask((cam.width, cam.height))
    shift = (chosen.polygon.centroid.x, chosen.polygon.centroid.y)  # roof plan frame -> this photo's frame
    for v in visible_walls(chosen.polygon, cam, height, roof.tops(), samples=64):
        v.quad = fit_rows(v.quad, cam, mask, v.bottom, height, v.top)
        gables = []
        for k, g in enumerate(roof.gables):
            if g.wall == v.index:
                tri = _on_photo(g.points, v, edges[v.index], shift)
                gables.append((k, tri, _coverage(mask, tri)))
        quad = [(x * scale, y * scale) for x, y in v.quad]
        (x0, y0), (x1, y1) = edges[v.index]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 * (v.end - v.start)
        # How squarely the photo faces the wall: 1 straight on, 0 edge-on. Nearly edge-on, a few pixels of photo
        # would be stretched over the whole wall.
        mx = x0 + (v.start + v.end) / 2 * (x1 - x0)
        my = y0 + (v.start + v.end) / 2 * (y1 - y0)
        to_cam = ((cam.x - mx) ** 2 + (cam.y - my) ** 2) ** 0.5
        face_on = ((y1 - y0) * (cam.x - mx) - (x1 - x0) * (cam.y - my)) / (length / (v.end - v.start) * to_cam)
        if face_on < FACE_ON:
            continue
        size = texture_size(quad, length, v.top - v.bottom)
        buf = io.BytesIO()
        straighten(photo, quad, size).save(buf, "JPEG", quality=88)
        across = max(math.dist(quad[0], quad[1]), math.dist(quad[3], quad[2]))
        sharp = min(across / length, SHARP_ENOUGH)  # photo pixels per metre along the wall; more doesn't help
        views.append({"wall": v.index, "start": v.start, "end": v.end, "top": v.top, "bottom": v.bottom,
                      "seen": v.seen, "quad": v.quad, "gables": gables,
                      "score": v.seen * sharp * face_on, "jpeg": buf.getvalue()})
    camera = {"gps_off_m": [round(cam.x, 1), round(cam.y, 1)], "compass_off": round(cam.heading - ctx["heading"], 1),
              "eye_m": cam.eye, "depth_scale": round(cam.depth_scale, 2)}
    return camera, views


def _on_photo(points, view, edge, shift) -> list[tuple[float, float]]:
    """Where points on a wall (roof plan frame, metres) fall in the photo: through the wall's own picture, so they
    agree with what gets painted."""
    from .front import homography

    (x0, y0), (x1, y1) = edge
    m = homography([(0, 1), (1, 1), (1, 0), (0, 0)], view.quad)
    pixels = []
    for x, y, z in points:
        x, y = x + shift[0], y + shift[1]
        t = ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2)
        u = (t - view.start) / (view.end - view.start)
        w = 1 - (z - view.bottom) / (view.top - view.bottom)
        px, py, pw = m @ [u, w, 1]
        pixels.append((px / pw, py / pw))
    return pixels


def _coverage(mask, tri) -> float:
    """Share of a triangle (photo pixels) the outline covers."""
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.new("1", (mask.shape[1], mask.shape[0]), 0)
    ImageDraw.Draw(canvas).polygon(tri, fill=1)
    inside = np.asarray(canvas, bool)
    return float(mask[inside].mean()) if inside.any() else 0.0


def _find_openings(meta, jpeg: bytes, view: dict, length: float, height: float, folder: Path, wall: int,
                   fresh: bool) -> tuple:
    """Windows and doors in a straightened wall picture, by SAM 3.1, as Openings in metres on the wall. The picture
    is already flat and upright, so a box in it is a box on the wall: no camera to work through."""
    import hashlib
    import io

    from PIL import Image

    from .build import Opening

    tag = hashlib.sha1(jpeg).hexdigest()[:8]
    small = folder / f"wall_{wall}_{tag}_small.jpg"
    if not small.exists():
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        img.thumbnail((1600, 1600))
        img.save(small, quality=90)
    w, h = Image.open(small).size
    top = view["top"] or height
    found = []
    for kind in OPENING_KINDS:  # doors first: where a door and a window overlap, it's the door
        segs = _cached_segment(meta, small, kind, folder / f"wall_{wall}_{tag}_{kind.replace(' ', '-')}.txt", fresh)
        for seg in segs:
            left, upper, right, lower = seg.box
            o = Opening(kind,
                        view["start"] + left / w * (view["end"] - view["start"]),
                        view["start"] + right / w * (view["end"] - view["start"]),
                        max(top - lower / h * (top - view["bottom"]), view["bottom"] + OPENING_EDGE),
                        min(top - upper / h * (top - view["bottom"]), height - OPENING_EDGE))
            wide = (o.end - o.start) * length
            if wide < OPENING_MIN or o.top - o.bottom < OPENING_MIN:
                continue
            if any(_overlaps(o, k, length) for k in found):
                continue
            found.append(o)
    return tuple(found)


def _overlaps(a, b, length: float) -> bool:
    """Whether two openings share more than a third of the smaller one."""
    wide = max(0.0, min(a.end, b.end) - max(a.start, b.start)) * length
    tall = max(0.0, min(a.top, b.top) - max(a.bottom, b.bottom))
    small = min((a.end - a.start) * length * (a.top - a.bottom), (b.end - b.start) * length * (b.top - b.bottom))
    return wide * tall > small / 3


def _draw_walls(small_path, views: list[dict], used: set[int], out: Path, gables=()) -> Path:
    """The photo with each wall it shows outlined: orange where it gave the wall its picture, grey where another
    photo did better. `gables` (photo pixels) are drawn yellow."""
    from PIL import Image, ImageDraw

    img = Image.open(small_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for v in views:
        colour = (228, 87, 46) if v["wall"] in used else (160, 160, 160)
        draw.polygon(v["quad"], outline=colour, width=4)
        cx, cy = sum(x for x, _ in v["quad"]) / 4, sum(y for _, y in v["quad"]) / 4
        draw.text((cx, cy), str(v["wall"]), fill=(255, 255, 255), font_size=48, stroke_width=3,
                  stroke_fill=(0, 0, 0), anchor="mm")
    for tri in gables:
        draw.polygon(tri, outline=(243, 167, 18), width=4)
    img.save(out, quality=88)
    return out


def _majority(values: list) -> int | None:
    """The most common whole number, the larger on a tie; None if there are none."""
    from collections import Counter

    counts = Counter(v for v in values if isinstance(v, int) and v > 0)
    return max(counts, key=lambda v: (counts[v], v)) if counts else None


def _roof(args, used: list[dict], footprint, height: float):
    """The roof plan: Muse looks at each photo's roof (one small call per photo, saved like pick's), the photos vote,
    and --roof / --pitch overrule. Returns the plan, the ridges' compass bearing and where the answer came from."""
    import hashlib

    from .muse import ask, noted
    from .roof import ROOF_PROMPT, combine, parse_roof, plan_roof

    views = []
    prompt = noted(ROOF_PROMPT, args.notes)
    asked = hashlib.sha1(prompt.encode()).hexdigest()[:8]
    if not (args.roof and args.pitch):
        for r in used:
            ctx = r["ctx"]
            reply = ctx["replies"] / f"{ctx['stem']}_{args.model}_roof_{asked}.txt"
            for answer in ask(ctx["meta"], prompt, ctx["small"], reply, ctx["model"], ctx["votes"], ctx["fresh"]):
                views.append((parse_roof(answer), ctx["heading"]))  # each vote is one more look at the roof
    kind, pitch, ridge = combine(views) if views else ("flat", 30.0, None)
    source = f"Muse, {len(used)} photo(s), {len(views)} answer(s)" if views else "you"
    if args.roof or args.pitch:
        source = "you" if args.roof and args.pitch else source + " and you"
    plan = plan_roof(footprint, height, args.roof or kind, args.pitch or pitch, ridge)
    return plan, ridge, source


def _look(args, used: list[dict]):
    """The building's colours: Muse Spark looks at the photos that see it best (up to LOOK_PHOTOS, in one call,
    saved like pick's). None with --photo-walls, where the photos themselves go on the walls."""
    import hashlib

    from .look import LOOK_PROMPT, parse_look
    from .muse import merge_looks, noted

    if args.photo_walls:
        return None
    seen = sorted((r for r in used if r.get("cam") is not None), key=lambda r: -len(r.get("views", ())))
    shots = (seen or used)[:LOOK_PHOTOS]
    ctx = shots[0]["ctx"]
    prompt = noted(LOOK_PROMPT, args.notes)
    tag = hashlib.sha1((prompt + "".join(r["ctx"]["stem"] for r in shots)).encode()).hexdigest()[:8]
    reply = ctx["replies"] / f"{ctx['stem']}_{args.model}_look_{tag}.txt"
    answers = _cached_ask(ctx["meta"], prompt, [r["ctx"]["small"] for r in shots], reply, ctx["fresh"],
                          ctx["model"], ctx["votes"])
    return merge_looks([parse_look(a) for a in answers])


def _pitches(roof) -> str:
    """The roof's pitch in degrees, or each block's when they differ."""
    import math

    found = []
    for g in roof.gables:
        apex = max(g.points, key=lambda p: p[2])
        base = [p for p in g.points if p is not apex]
        half = math.dist(base[0][:2], base[1][:2]) / 2
        found.append(round(math.degrees(math.atan((apex[2] - base[0][2]) / half))))
    found = sorted(set(found)) or [round(roof.pitch)]
    return "/".join(str(p) for p in found)


def cmd_build(args) -> None:
    import math
    from dataclasses import replace

    from shapely import affinity

    from .build import FLOOR, WallPhoto, building_height, extrude, mirror_walls, place_feature, wall_edges, write_glb
    from .front import MAX_TURN, gable_pitches, visible_walls
    from .meta import PaidCallBlocked
    from .roof import plan_roof
    from .viewer import write_viewer

    if args.building is not None and len(args.photo) > 1:
        _fail("--building numbers belong to one photo's candidates; use it with a single photo", args.json)

    # Pick the building in every photo, several at once (most of the time is waiting for Muse). A photo that can't
    # be used is noted, not fatal, unless it's the only one.
    from concurrent.futures import ThreadPoolExecutor

    def pick_one(photo: str):
        try:
            return _pick(argparse.Namespace(**{**vars(args), "photo": photo}))
        except Failed as e:
            if len(args.photo) == 1:
                raise
            return e

    with ThreadPoolExecutor(BUILD_PARALLEL) as pool:
        picked = list(pool.map(pick_one, args.photo))
    runs = []
    for photo, result in zip(args.photo, picked, strict=True):
        if isinstance(result, Failed):
            runs.append({"photo": photo, "used": False, "why": result.msg})
            continue
        ctx = result.pop("_")
        if args.building is not None:
            chosen = ctx["by_rank"].get(args.building)
            if chosen is None:
                _fail(f"no candidate {args.building}; candidates are {sorted(ctx['by_rank'])}", args.json)
        elif result["status"] == "picked":
            chosen = ctx["chosen"]
        else:
            chosen = None
        runs.append({"photo": photo, "used": chosen is not None, "why": result["why"], "ctx": ctx,
                     "chosen": chosen})

    sides = _sides(args)
    for r in runs:
        r["side"] = sides.get(Path(r["photo"]).stem.lower())
        r["side_from"] = "you" if r["side"] else None
    _muse_sides(args, runs)
    _rescue(runs)
    picked = [r for r in runs if r["used"]]
    if not picked:
        whys = "; ".join(f"{Path(r['photo']).name}: {r['why']}" for r in runs)
        hint = "choose one with --building N" if len(runs) == 1 else "build them one at a time with --building N"
        _fail(f"not sure which building it is ({whys}); run `placelens pick` and {hint}", args.json, EXIT_ASK)
    # The building most photos picked; photos of a different one are left out.
    votes = [r["chosen"].building.osm_id for r in picked]
    osm_id = max(set(votes), key=votes.count)
    for r in picked:
        if r["chosen"].building.osm_id != osm_id:
            r["used"], r["why"] = False, f"picked another building ({r['chosen'].building.osm_id})"
    used = [r for r in runs if r["used"]]
    chosen, frame = used[0]["chosen"], used[0]["ctx"]["frame"]

    tags = chosen.building.tags
    floors = _majority([r["ctx"]["verdict"].floors for r in used])
    if args.height:
        height, source = args.height, "you"
    elif args.floors:
        height, source = args.floors * FLOOR, "you (floors)"
    else:
        height, source = building_height(tags, floors)
    centre = chosen.polygon.centroid
    footprint = affinity.translate(chosen.polygon, -centre.x, -centre.y)
    anchor = frame.to_latlon(centre.x, centre.y)
    roof, ridge, roof_source = _roof(args, used, footprint, height)

    out = Path(args.out) / "build" / osm_id.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    # Every photo offers pictures of the walls it sees, and each wall takes the best offer. Wall numbers match across
    # photos: the footprint is the same, only its origin differs.
    front = _front_bearing(used)
    for r in used:
        walls = _side_walls(r["chosen"].polygon, r.get("side"), front) if r.get("side") else None
        # A side you named may need a big compass correction; a side Muse named only frees the GPS.
        turn = TOLD_TURN if r.get("side_from") == "you" else MAX_TURN
        r["turn"] = turn
        r["cam"] = _lock(r["ctx"], r["chosen"], height, walls, turn) if r["ctx"]["main"] is not None else None
        yours = r.get("side") if r.get("side_from") == "you" else None
        why = _misplaced(r["cam"], r["chosen"], r["ctx"], yours) if r["cam"] is not None else None
        if why and (r.get("rescued") or r["cam"].misfit > _misfit_allowed(r["cam"], yours)):
            # (a photo sure on its own keeps its lock)
            r["cam"], r["used"], r["why"] = None, False, why
    pitches: dict[int, float] = {}
    if not args.pitch and roof.gables:  # Muse guesses the pitch; the outline's peaks measure it
        measured = []
        for r in used:
            if r["cam"] is None:
                continue
            cam, chosen_here = r["cam"], r["chosen"]
            seen = {v.index for v in visible_walls(chosen_here.polygon, cam, height)}
            mask = r["ctx"]["main"].full_mask((cam.width, cam.height))
            c = chosen_here.polygon.centroid
            measured += gable_pitches(roof.gables, cam, mask, seen, (c.x, c.y))
        if measured:  # each block its own measured pitch (the middle one); the rest keep Muse's
            by_block: dict[int, list[float]] = {}
            for block, pitch in measured:
                by_block.setdefault(block, []).append(pitch)
            pitches = {b: sorted(v)[len(v) // 2] for b, v in by_block.items()}
            roof = plan_roof(footprint, height, roof.kind, roof.pitch, ridge, pitches)
            roof_source += f"; pitch measured on {len(pitches)} of {len(roof.blocks)} block(s)"
    refined = None
    if args.refine:  # Muse critiques the model over every photo; SAM's outline decides what's kept
        refined = _refine(args, used, footprint, height, roof, ridge, pitches, out, front)
        if refined:
            shape = refined["shape"]
            for r in used:  # every photo's copy of the footprint, where that photo's frame has it; cameras follow
                if r.get("cam") is not None:
                    r["cam"] = _relock(r, shape, front)
                c = r["chosen"].polygon.centroid
                r["chosen"] = replace(r["chosen"], polygon=affinity.translate(shape.poly, c.x, c.y))
            chosen = used[0]["chosen"]
            centre = chosen.polygon.centroid
            footprint = affinity.translate(chosen.polygon, -centre.x, -centre.y)
            anchor = frame.to_latlon(centre.x, centre.y)
            if shape.height != height:
                height, source = shape.height, source + ", refined"
            roof = plan_roof(footprint, height, shape.kind, shape.pitch, ridge, shape.pitches)
            roof_source += "; refined"
    best: dict[int, tuple[dict, dict]] = {}
    for r in used:
        r["views"] = []
        if args.no_front or r["cam"] is None:
            continue
        r["camera"], r["views"] = _wall_views(r["photo"], r["ctx"], r["chosen"], height, roof, r["cam"], footprint)
        for v in r["views"]:
            if r.get("rescued") and not r.get("side"):  # a side someone vouched for counts in full
                v["score"] *= RESCUED_WEIGHT
            if v["wall"] not in best or v["score"] > best[v["wall"]][0]["score"]:
                best[v["wall"]] = (v, r)
    photos = {}
    lengths = [math.dist(a, b) for a, b in wall_edges(footprint)]
    for wall, (v, r) in sorted(best.items()):
        (out / f"wall_{wall}.jpg").write_bytes(v["jpeg"])
        bare = frozenset(k for k, _, cover in v["gables"] if cover < GABLE_COVER)
        try:
            openings = () if args.no_openings else _find_openings(r["ctx"]["meta"], v["jpeg"], v, lengths[wall],
                                                                   height, out, wall, args.fresh)
        except PaidCallBlocked:  # windows and doors are a nicety: without them the wall is built plain
            openings = ()
        photos[wall] = WallPhoto(v["jpeg"], v["start"], v["end"], v["top"], v["bottom"], bare, openings)
    mirrored = mirror_walls(footprint, photos, roof.gables) if args.symmetric else {}
    photos.update({w: p for w, (_, p) in mirrored.items()})
    for r in used:
        if r["views"]:
            mine = {w for w, (_, owner) in best.items() if owner is r}
            gables = [tri for v in r["views"] for _, tri, _ in v["gables"]]
            r["check"] = _draw_walls(r["ctx"]["small"], r["views"], mine, out / f"check_{Path(r['photo']).stem}.jpg",
                                     gables)

    look = _look(args, used)
    glb = write_glb(out / "building.glb", extrude(footprint, height, photos, roof, look), name=osm_id)
    place = out / "place.geojson"
    outline = ([frame.to_latlon(x, y) for x, y in chosen.polygon.exterior.coords] if refined
               else chosen.building.outline)
    feature = place_feature(outline, anchor, height, source, osm_id, tags)
    feature["features"][0]["properties"].update(roof=roof.kind, roof_pitch=round(roof.pitch, 1),
                                                roof_ridge_bearing=round(ridge, 1) if ridge is not None else None)
    place.write_text(json.dumps(feature, indent=1), encoding="utf-8")

    rect = footprint.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    sides = sorted(((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2) ** 0.5 for i in range(2))
    painters = [r for r in used if any(owner is r for _, owner in best.values())]
    main = max(painters, key=lambda r: sum(owner is r for _, owner in best.values()), default=None)
    view = write_viewer(
        out / "view.html", glb, chosen.building.tags.get("name") or f"Building {osm_id}",
        {"Size": f"{sides[1]:.1f} x {sides[0]:.1f} m, {height:.1f} m to the eaves ({source})",
         "Roof": f"{roof.kind}, {_pitches(roof)}°, {len(roof.blocks)} block(s)",
         "Photos": f"{sum(r['used'] for r in runs)} of {len(runs)} used; {len(best)} wall(s) "
                   + ("measured" if look else "painted"),
         "Map": chosen.building.url},
        ATTRIBUTION, facing=main["cam"].heading if main else 0.0)
    built = {
        "building": _row(chosen, frame),
        "chosen_by": "you" if args.building is not None else "pick",
        "height_m": round(height, 2),
        "height_source": source,
        "size_m": [round(sides[1], 1), round(sides[0], 1)],
        "roof": {"kind": roof.kind, "pitch": _pitches(roof), "source": roof_source, "blocks": len(roof.blocks),
                 "ridge_bearing": round(ridge, 1) if ridge is not None else None,
                 "gables_painted": sum(1 for k, g in enumerate(roof.gables)
                                       if g.wall in photos and k not in photos[g.wall].bare_gables)},
        "openings": {str(w): [vars(o) for o in p.openings] for w, p in photos.items()},
        "mirrored": {str(w): s for w, (s, _) in mirrored.items()},
        "symmetric": (None if not args.symmetric else
                      f"walls {', '.join(f'{w} copies {s}' for w, (s, _) in sorted(mirrored.items()))}"
                      if mirrored else "asked, but no wall without a photo faces one with a photo at about its "
                                       "length (the footprint isn't symmetric), so nothing was mirrored"),
        "walls": {str(w): {"photo": r["photo"], "seen": round(v["seen"], 2),
                           "stretch": [round(v["start"], 3), round(v["end"], 3)],
                           "texture": str(out / f"wall_{w}.jpg")}
                  for w, (v, r) in sorted(best.items())},
        "photos": [{"photo": r["photo"], "used": r["used"], "why": r["why"], "camera": r.get("camera"),
                    "side": r.get("side"), "side_from": r.get("side_from"), "side_note": r.get("side_note"),
                    "muse_floors": r["ctx"]["verdict"].floors if "ctx" in r else None,
                    "check": str(r["check"]) if r.get("check") else None}
                   for r in runs],
        "refined": refined["history"] if refined else None,
        "model": str(glb),
        "place": str(place),
        "viewer": str(view),
        "cost_usd": round(sum(r["ctx"]["meta"].spent for r in runs if "ctx" in r), 4),
        "attribution": ATTRIBUTION,
    }
    if args.json:
        print(json.dumps(built))
        return
    print(f"BUILT     {osm_id} from {sum(r['used'] for r in runs)} of {len(runs)} photos "
          f"({built['chosen_by']} chose it)")
    print(f"size:     {sides[1]:.1f} x {sides[0]:.1f} m, {height:.1f} m tall ({source})")
    for r in runs:
        name = Path(r["photo"]).name
        if not r["used"]:
            side = f" [{r['side_note']}]" if r.get("side_note") else ""
            print(f"  {name:<14} not used: {r['why']}{side}")
            continue
        walls = ", ".join(str(w) for w, (_, owner) in sorted(best.items()) if owner is r) or "none"
        cam = r.get("camera")
        fix = (f"camera {cam['gps_off_m'][0]:+.1f} m E, {cam['gps_off_m'][1]:+.1f} m N, "
               f"compass {cam['compass_off']:+.1f}°" if cam else "no outline")
        side = f", {r['side']} ({r['side_from']})" if r.get("side") else ""
        print(f"  {name:<14} walls {walls:<10} {fix}  (Muse: {r['ctx']['verdict'].floors} floors{side})")
    r_ = built["roof"]
    print(f"roof:     {roof.kind}, {r_['pitch']}°, {r_['blocks']} block(s), {r_['gables_painted']} gable end(s) "
          f"painted ({roof_source})")
    if args.symmetric:
        print(f"symmetric: {built['symmetric']}")
    print(f"walls:    {len(best)} with a photo, {len(footprint.exterior.coords) - 1 - len(best) - len(mirrored)} plain"
          + ("; built in Muse's colours, not pictured" if look else ""))
    kinds = [o.kind for p in photos.values() for o in p.openings]
    if kinds:
        print("openings: " + ", ".join(f"{kinds.count(k)} {k}{'s' if kinds.count(k) > 1 else ''}"
                                        for k in OPENING_KINDS if k in kinds) + " made real (SAM 3.1)")
    for r in used:
        if r.get("check"):
            print(f"check:    {r['check']}")
    print(f"model:    {glb}")
    print(f"place:    {place}")
    print(f"view:     {view} (open in a browser)")
    print(f"cost:     ~${built['cost_usd']:.4f} (0 if replies were reused)")


def _relock(r: dict, shape, front: float | None):
    """Photo run `r`'s camera locked again onto a refined shape (critique.Shape, roof plan frame): same rules as
    the first lock, the walls a named side shows found again on the new footprint (a notch renumbers them)."""
    from dataclasses import replace

    from shapely import affinity

    c = r["chosen"].polygon.centroid
    poly = affinity.translate(shape.poly, c.x, c.y)
    walls = _side_walls(poly, r["side"], front) if r.get("side") else None
    return _lock(r["ctx"], replace(r["chosen"], polygon=poly), shape.height, walls, r.get("turn"))


CRITIQUE_EFFORT = "high"  # Spark Open thinks longer on critiques: free on Baseten credits, and they're the hard part


def _refine(args, used: list[dict], footprint, height: float, roof, ridge, pitches: dict, out: Path,
            front: float | None = None) -> dict | None:
    """The critique loop (critique.py) over every photo with a locked camera. Its rounds are saved in
    <out>/critique/: each photo with the model drawn over it, Muse's answers, and history.json. None if no photo
    can be used."""
    import hashlib

    from .critique import CRITIQUE_PROMPT, Shape, View, parse_critique, refine, vertical_lines
    from .meta import OPEN
    from .muse import ask

    views = []
    for r in used:
        cam, ctx = r.get("cam"), r["ctx"]
        if cam is None or ctx["main"] is None:
            continue
        c = r["chosen"].polygon.centroid
        mask = ctx["main"].full_mask((cam.width, cam.height))
        views.append(View(ctx["stem"], ctx["small"], cam, mask, (c.x, c.y),
                          relock=lambda shape, r=r: _relock(r, shape, front), lines=vertical_lines(ctx["small"], mask)))
    if not views:
        return None
    ctx = used[0]["ctx"]
    folder = out / "critique"

    def critique(view, overlay: Path):
        tag = hashlib.sha1(CRITIQUE_PROMPT.encode() + overlay.read_bytes()).hexdigest()[:8]
        effort = CRITIQUE_EFFORT if ctx["model"] == OPEN else None
        reply = folder / "answers" / f"{view.stem}_{args.model}_{tag}{'_' + effort if effort else ''}.txt"
        return [parse_critique(a) for a in ask(ctx["meta"], CRITIQUE_PROMPT, [view.small, overlay], reply,
                                               ctx["model"], ctx["votes"], args.fresh, effort)]

    shape = Shape(footprint, height, roof.kind, roof.pitch, ridge, dict(pitches))
    shape, history = refine(shape, views, critique, folder, log=(lambda *_: None) if args.json else print)
    (folder / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return {"shape": shape, "history": history}


def _cached_ask(meta, prompt: str, photos, reply: Path, fresh: bool, model: str, votes: int = 1) -> list[str]:
    """`votes` Muse answers, saved next to `reply` so asking again is free unless `fresh`."""
    from .muse import ask

    return ask(meta, prompt, photos, reply, model, votes, fresh)


def _votes(args, model: str) -> int:
    """--votes, else five for Spark Open (Baseten credits) and one for the paid models."""
    from .muse import default_votes

    return getattr(args, "votes", None) or default_votes(model)


def _cached_segment(meta, small: Path, noun: str, reply: Path, fresh: bool):
    from .meta import parse_segments

    if reply.exists() and not fresh:
        return parse_segments(reply.read_text(encoding="utf-8"))
    segments, raw = meta.segment(small, noun)
    reply.write_text(raw, encoding="utf-8")
    return segments


def cmd_prop_walkaround(args) -> None:
    """Several photos taken walking round one object: its shape carved from their cut-outs, painted from them."""
    import hashlib
    import math

    import numpy as np
    from PIL import Image

    from .build import write_glb
    from .carve import OVERHEAD, WALK, Shot, place_shots
    from .carve import SIDE_PROMPT as WALK_PROMPT
    from .meta import MODELS, MetaClient, MissingKeyError
    from .muse import merge_props, merge_sides, noted
    from .pick import pixel_to_offset, prepare_photo
    from .prop import PROP_PROMPT, parse_prop, place
    from .sides import parse_side
    from .viewer import write_viewer

    try:
        meta = MetaClient()
    except MissingKeyError as e:
        _fail(str(e), args.json)
    model = MODELS[args.model]
    first_stem = Path(args.photo[0]).stem
    out = Path(args.out) / "props" / f"{first_stem}_walk{len(args.photo)}"
    out.mkdir(parents=True, exist_ok=True)

    # What it is and how big, from the first photo (its front).
    smalls = [prepare_photo(p, out / f"{Path(p).stem}.jpg") for p in args.photo]
    prompt = noted(PROP_PROMPT, args.notes)
    asked = hashlib.sha1(prompt.encode()).hexdigest()[:8]
    votes = _votes(args, model)
    view = merge_props([parse_prop(a) for a in _cached_ask(
        meta, prompt, smalls[0], out / f"{first_stem}_{args.model}_prop_{asked}.txt", args.fresh, model, votes)])
    noun = args.what or view.segment or view.name or "object"
    height = args.height or view.height or DEFAULT_PROP_HEIGHT
    width = args.width or view.width or height
    depth = args.depth or view.depth or width

    # Each photo: its cut-out, and the angle round the object it was taken from.
    shots, notes, headings = [], [], []
    first_bearing = None
    slug = re.sub(r"[^a-z0-9]+", "-", noun.lower()).strip("-")
    walk_asked = hashlib.sha1(WALK_PROMPT.encode()).hexdigest()[:8]
    for i, (photo, small) in enumerate(zip(args.photo, smalls, strict=True)):
        stem = Path(photo).stem
        cam = read_camera(photo)
        segs = _cached_segment(meta, small, noun, out / f"{stem}_sam_{slug}.txt", args.fresh)
        if not segs:
            notes.append(f"{Path(photo).name}: SAM 3.1 found no {noun!r}; left out")
            continue
        seg = segs[0]
        img = Image.open(small).convert("RGB")
        fov = args.fov if args.fov is not None else cam.fov
        focal = img.width / 2 / math.tan(math.radians(fov / 2))
        left, top, right, bottom = seg.box
        if left <= BORDER_PX or right >= img.width - BORDER_PX or top <= BORDER_PX or bottom >= img.height - BORDER_PX:
            notes.append(f"{Path(photo).name}: the {noun} runs off the photo's edge; that side carves less")
        # Compass: walking round to your right turns your view to the left, so the angle round goes against it.
        bearing = (cam.heading or 0.0) + pixel_to_offset((left + right) / 2, img.width, fov)
        if first_bearing is None:
            first_bearing = bearing
        angle = -((bearing - first_bearing + 180) % 360 - 180)
        how = "compass"
        overhead = False
        if i > 0:  # Muse names the side it shows; if the compass disagrees badly, Muse wins
            side = merge_sides([parse_side(a) for a in _cached_ask(
                meta, WALK_PROMPT, [smalls[0], small], out / f"{stem}_{args.model}_walk_{walk_asked}.txt", args.fresh,
                model, votes)])
            if side.side == "above" and side.confidence >= 0.5:
                overhead, how = True, "Muse: from above; its turn found by fit"
            elif side.side in WALK and side.confidence >= 0.5:
                muse = WALK[side.side]
                if abs((angle - muse + 180) % 360 - 180) > WALK_DISAGREE:
                    angle, how = muse, f"Muse: {side.side} (compass said {angle:+.0f}°)"
                else:
                    how = f"compass, Muse agrees ({side.side})"
        if cam.heading is None and i > 0 and how == "compass":
            notes.append(f"{Path(photo).name}: no compass and Muse couldn't tell the side; left out")
            continue
        shots.append(Shot(seg.full_mask(img.size), np.asarray(img), focal, angle,
                          elevation=OVERHEAD if overhead else 0.0, overhead=overhead))
        headings.append(cam.heading)
        notes.append(f"{Path(photo).name}: " + ("from above" if overhead else f"{angle:+.0f}° round") + f" ({how})")
    if len(shots) < 3:
        _fail(f"only {len(shots)} usable photos of the {noun}; walk round it and take at least 6", args.json)

    place_shots(shots, height)
    if args.match:  # the cameras from the photos themselves, where they match each other
        from .solve import View, solve

        poses, _ = solve([View(s.image, s.focal, s.centre, s.distance, hd) for s, hd in zip(shots, headings,
                                                                                         strict=True)], first_bearing)
        for k, pose in poses.items():
            shots[k].pose = pose
        notes.append(f"cameras: {len(poses)} of {len(shots)} placed by matching the photos to each other, "
                     f"the rest by compass" if poses else "cameras: the photos didn't match each other; by compass")
    parts, made_of, judged = _dressed(meta, args, shots, (width, depth, height), view.name or noun, smalls[0], out,
                                      model)
    glb = write_glb(out / "prop.glb", parts, name=view.name or noun)
    every = np.concatenate([p.positions for p in parts])
    span = every.max(axis=0) - every.min(axis=0)

    location = None
    cam0 = read_camera(args.photo[0])
    if cam0.lat is not None and cam0.heading is not None:
        s0 = shots[0]
        offset = pixel_to_offset(s0.centre[0], s0.mask.shape[1], args.fov or cam0.fov)
        east, north, facing = place(cam0.heading, offset, s0.distance)
        lat, lon = LocalFrame(cam0.lat, cam0.lon).to_latlon(east, north)
        location = {"lat": round(lat, 7), "lon": round(lon, 7), "facing_bearing": round(facing, 1)}
    feature = {"type": "Feature",
               "geometry": {"type": "Point", "coordinates": [location["lon"], location["lat"]]} if location else None,
               "properties": {"name": view.name or noun, "model": "prop.glb",
                              "model_axes": "metres, Y up, standing on Y = 0, front (first photo's side) +Z",
                              "size_m": [round(float(v), 2) for v in (span[0], span[2], span[1])],
                              "facing_bearing": location["facing_bearing"] if location else None,
                              "photos": [Path(p).name for p in args.photo]}}
    geo = out / "prop.geojson"
    geo.write_text(json.dumps(feature, indent=1), encoding="utf-8")
    viewer = write_viewer(out / "view.html", glb, (view.name or noun).capitalize(),
                          {"Size": f"{span[0]:.2f} m wide, {span[2]:.2f} m deep, {span[1]:.2f} m tall",
                           "Photos": f"{len(shots)} of {len(args.photo)}, walked round it"},
                          "Shape carved from SAM 3.1 cut-outs; sides checked by Muse.", facing=0.0)
    result = {"name": view.name or noun, "segmented_as": noun, "shape": "walk-around",
              "size_m": feature["properties"]["size_m"], "photos": notes, "location": location,
              "made_of": made_of, "judged": judged, "model": str(glb), "place": str(geo), "viewer": str(viewer),
              "cost_usd": round(meta.spent, 4)}
    if args.json:
        print(json.dumps(result))
        return
    print(f"PROP      {result['name']} carved from {len(shots)} photos walked round it (SAM 3.1 cut out {noun!r})")
    print(f"size:     {span[0]:.2f} m wide, {span[2]:.2f} m deep, {span[1]:.2f} m tall")
    for n in notes:
        print(f"  {n}")
    if made_of:
        print("made of:  " + "; ".join(f"{part}: {what}" for part, what in made_of.items()))
    if judged:
        steps = [f"{r['score']:g}" + (f" -> {r['score_after']:g} ({'kept' if r.get('kept') else 'undone'}, "
                                      f"{r['cut_voxels']} voxels; closer: {' '.join(r['closer'])})"
                                      if "score_after" in r else "") for r in judged["rounds"]]
        print(f"judged:   Muse's score out of 10, round by round: {'; '.join(steps)}")
        for m in judged["missing"]:
            print(f"  missing (Muse): {m}")
    print(f"model:    {glb}")
    print(f"view:     {viewer} (open in a browser)")
    print(f"cost:     ~${meta.spent:.4f} (0 if replies were reused)")


def _imagined_shots(meta, args, name: str, noun: str, small: Path, out: Path, model: str, symmetric: bool = False):
    """Muse Image's drawings of the sides one photo doesn't show, each checked by Muse, as carving shots.
    Returns the shots that passed and a line about each side."""
    import hashlib
    import io

    import numpy as np
    from PIL import Image

    from .carve import SIDE_PROMPT as WALK_PROMPT
    from .carve import WALK, Shot
    from .imagine import DRAWN, DRAWN_FOCAL, draw_prompt
    from .muse import merge_sides, noted
    from .sides import parse_side

    shots, notes = [], []
    slug = re.sub(r"[^a-z0-9]+", "-", noun.lower()).strip("-")
    for side in DRAWN:
        prompt = noted(draw_prompt(name, side), args.notes)
        tag = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        picture = out / f"imagined_{side}_{tag}.jpg"
        if not picture.exists() or args.fresh:
            drawn = Image.open(io.BytesIO(meta.draw(prompt, [small]))).convert("RGB")
            drawn.thumbnail((1600, 1600))
            drawn.save(picture, quality=90)
        seen = merge_sides([parse_side(a) for a in _cached_ask(
            meta, WALK_PROMPT, [small, picture], out / f"imagined_{side}_{tag}_check.txt", args.fresh, model,
            _votes(args, model))])
        if seen.side != side:
            notes.append(f"{side}: Muse Image drew the {seen.side or 'wrong'} side instead; not used")
            continue
        segs = _cached_segment(meta, picture, noun, out / f"imagined_{side}_{tag}_sam_{slug}.txt", args.fresh)
        if not segs:
            notes.append(f"{side}: drawn, but SAM 3.1 found no {noun!r} in the drawing; not used")
            continue
        img = Image.open(picture).convert("RGB")
        shots.append(Shot(segs[0].full_mask(img.size), np.asarray(img), DRAWN_FOCAL * img.width, WALK[side],
                          imagined=True))
        notes.append(f"{side}: imagined by Muse Image, checked by Muse ({seen.confidence:.0%} sure it's the {side})")
    # A symmetric object's missing side is its other side in a mirror.
    have = {s.angle: s for s in shots}
    for side, other in (("left", "right"), ("right", "left")):
        if symmetric and WALK[side] not in have and WALK[other] in have:
            twin = have[WALK[other]]
            shots.append(Shot(twin.mask[:, ::-1].copy(), twin.image[:, ::-1].copy(), twin.focal, WALK[side],
                              imagined=True))
            notes.append(f"{side}: the {other} drawing mirrored ({'you' if args.symmetric else 'Muse'} said "
                         "it's symmetric)")
    return shots, notes


def _dressed(meta, args, shots, size: tuple[float, float, float], name: str, photo: Path, out: Path, model: str,
             misses: int = 0):
    """The carved shape built as its own graphic (dress.py): the photos' colour families on it, Muse naming each
    part and its material, clean colours, grain and features made real. Returns the parts, {part: material} and
    the judge's report (Muse comparing the shape with the photos; painted props only).
    Without --clean (the default): the photos' colours painted on (one part, no Muse call)."""
    import hashlib

    from .carve import hull_mesh, hull_part
    from .dress import (
        GRAINS,
        MATERIAL_PROMPT,
        dress,
        families,
        label_vertices,
        merge_materials,
        parse_materials,
        smooth_labels,
        swatch,
    )
    from .meta import PaidCallBlocked
    from .muse import noted

    if not args.clean:
        judged = {}

        def look(kept, corner, voxel):  # the last step: Muse judges the shape against the photos
            from .judge import judge

            def ask(prompt: str, picture: Path) -> list[str]:
                tag = hashlib.sha1(prompt.encode() + picture.read_bytes()).hexdigest()[:8]
                return _cached_ask(meta, noted(prompt, args.notes), [picture],
                                   picture.with_name(f"{picture.stem}_{args.model}_{tag}.txt"), args.fresh, model,
                                   _votes(args, model))

            kept, report = judge(shots, kept, corner, voxel, size[2], ask, out / "judge", name, size)
            judged.update(report)
            return kept

        part = hull_part(shots, size, misses, judge=None if args.no_judge else look)
        return [part], None, judged or None
    verts, faces, normals = hull_mesh(shots, size, misses)
    fams = families(shots)
    labels = smooth_labels(faces, label_vertices(verts, normals, shots, size[2], fams))
    picture = out / "families.png"
    swatch(fams, picture)
    prompt = noted(MATERIAL_PROMPT.format(name=name, numbers=", ".join(str(f.number) for f in fams),
                                          grains=", ".join(f'"{g}"' for g in GRAINS)), args.notes)
    tag = hashlib.sha1(prompt.encode() + picture.read_bytes()).hexdigest()[:8]
    try:
        materials = merge_materials([parse_materials(a) for a in _cached_ask(
            meta, prompt, [photo, picture], out / f"materials_{args.model}_{tag}.txt", args.fresh, model,
            _votes(args, model))])
    except PaidCallBlocked:  # (only with a paid model: the parts stay unnamed, still clean)
        materials = {}
    parts = dress(verts, faces, normals, labels, fams, materials)
    return parts, {m.part: m.material for m in materials.values()}, None


def cmd_prop(args) -> None:
    if len(args.photo) > 1:
        return cmd_prop_walkaround(args)
    args.photo = args.photo[0]
    import hashlib
    import io
    import math

    import numpy as np
    from PIL import Image, ImageOps

    from .build import write_glb
    from .meta import MODELS, MetaClient, MissingKeyError, parse_segments
    from .muse import merge_props, noted
    from .pick import pixel_to_offset, prepare_photo
    from .prop import PROP_PROMPT, cutout_png, lathe, outline, parse_prop, place, solid
    from .viewer import write_viewer

    cam = read_camera(args.photo)
    try:
        meta = MetaClient()
    except MissingKeyError as e:
        _fail(str(e), args.json)
    stem = Path(args.photo).stem
    out = Path(args.out) / "props" / stem
    small = prepare_photo(args.photo, out / f"{stem}.jpg")
    width = Image.open(small).width
    fov = args.fov if args.fov is not None else cam.fov

    # What it is and how big: Muse. Replies are saved, so re-running is free unless --fresh.
    model = MODELS[args.model]
    prompt = noted(PROP_PROMPT, args.notes)
    reply = out / f"{stem}_{args.model}_prop_{hashlib.sha1(prompt.encode()).hexdigest()[:8]}.txt"
    view = merge_props([parse_prop(a) for a in _cached_ask(meta, prompt, small, reply, args.fresh, model,
                                                           _votes(args, model))])
    if args.width:
        view.width = args.width
    noun = args.what or view.segment or view.name or "object"

    # Cut it out: SAM 3.1, asked for what Muse called it (or --what).
    sam_file = out / f"{stem}_sam_{re.sub(r'[^a-z0-9]+', '-', noun.lower()).strip('-')}.txt"
    if sam_file.exists() and not args.fresh:
        segments = parse_segments(sam_file.read_text(encoding="utf-8"))
    else:
        segments, raw = meta.segment(small, noun)
        sam_file.write_text(raw, encoding="utf-8")
    if not segments:
        _fail(f"SAM 3.1 found no {noun!r} in the photo; name it yourself with --what", args.json)
    seg = segments[0]  # the biggest: the thing the photo is of
    shape = outline(seg.mask)
    if shape is None:
        _fail("the cut-out has no usable outline", args.json)

    left, top, right, bottom = seg.box
    with Image.open(args.photo) as img:  # full resolution for the picture
        photo = ImageOps.exif_transpose(img).convert("RGB")
    k = photo.width / width
    crop = photo.crop((round(left * k), round(top * k), round(right * k), round(bottom * k)))
    png = cutout_png(crop, seg.mask)

    height = args.height or view.height or DEFAULT_PROP_HEIGHT
    depth = args.depth or view.depth or (view.width or height) * DEFAULT_DEPTH_SHARE
    tall = Image.open(small).height
    cut = (left <= BORDER_PX, right >= width - BORDER_PX)
    imagined_notes: list[str] = []
    made_of = None
    round_ = args.shape == "round" or (args.shape is None and view.round is True)
    if round_:  # the same all the way round: turn the outline's profile on a lathe
        rows = np.flatnonzero(seg.mask.any(axis=1))
        per_px = height / max(rows[-1] - rows[0], 1)
        buf = io.BytesIO()
        crop.resize((right - left, bottom - top)).save(buf, "JPEG", quality=90)
        parts = lathe(seg.mask, buf.getvalue(), height, cut,
                      max_radius_px=view.width / 2 / per_px if view.width else None)
        depth = across = float(2 * np.hypot(parts[0].positions[:, 0], parts[0].positions[:, 2]).max())
    elif args.shape == "cutout" or args.no_imagine:
        parts = solid(shape, (right - left, bottom - top), height, depth, png)
    else:  # imagine the sides it can't see, check them, and carve
        from .carve import Shot, place_shots

        drawn, imagined_notes = _imagined_shots(meta, args, view.name or noun, noun, small, out, model,
                                                symmetric=args.symmetric or bool(view.symmetric))
        if drawn:
            img = Image.open(small).convert("RGB")
            real = Shot(seg.full_mask(img.size), np.asarray(img), width / 2 / math.tan(math.radians(fov / 2)), 0.0)
            shots = [real, *drawn]
            place_shots(shots, height)
            parts, made_of, _ = _dressed(meta, args, shots, (view.width or height, depth, height),
                                         view.name or noun, small, out, model, misses=0)
            every = np.concatenate([p.positions for p in parts])
            span = every.max(axis=0) - every.min(axis=0)
            across, depth = float(span[0]), float(span[2])
        else:
            parts = solid(shape, (right - left, bottom - top), height, depth, png)
    glb = write_glb(out / "prop.glb", parts, name=view.name or noun)
    carved = not round_ and any("checked" in n for n in imagined_notes)
    if not round_ and not carved:
        across = (shape.bounds[2] - shape.bounds[0]) * height / max(shape.bounds[3] - shape.bounds[1], 1)

    # Where it stands: the photo's GPS and compass, its place across the photo, and how far away it is.
    location = None
    if cam.lat is not None and cam.heading is not None:
        focal = width / 2 / math.tan(math.radians(fov / 2))
        by_size = focal * height / max(bottom - top, 1)  # how far an object this tall must be to look this tall
        distance = view.distance or by_size
        offset = pixel_to_offset((left + right) / 2, width, fov)
        east, north, facing = place(cam.heading, offset, distance)
        lat, lon = LocalFrame(cam.lat, cam.lon).to_latlon(east, north)
        location = {"lat": round(lat, 7), "lon": round(lon, 7), "facing_bearing": round(facing, 1),
                    "distance_m": round(distance, 1), "distance_by_size_m": round(by_size, 1)}

    feature = {"type": "Feature",
               "geometry": {"type": "Point", "coordinates": [location["lon"], location["lat"]]} if location else None,
               "properties": {"name": view.name or noun, "model": "prop.glb",
                              "model_axes": "metres, Y up, standing on Y = 0, front +Z; turn it so the front faces "
                                            "facing_bearing",
                              "size_m": [round(across, 2), round(max(depth, 0.03), 2), round(height, 2)],
                              "facing_bearing": location["facing_bearing"] if location else None,
                              "muse_confidence": view.confidence, "photo": Path(args.photo).name}}
    geo = out / "prop.geojson"
    geo.write_text(json.dumps(feature, indent=1), encoding="utf-8")
    viewer = write_viewer(out / "view.html", glb, (view.name or noun).capitalize(),
                          {"Size": f"{across:.2f} m wide, {max(depth, 0.03):.2f} m deep, {height:.2f} m tall",
                           "Where": f"{location['lat']}, {location['lon']}, facing {location['facing_bearing']:.0f}°"
                                    if location else "no GPS or compass in the photo"},
                          "Cut out by SAM 3.1; size and kind by Muse.", facing=0.0)

    # Say what's wrong with the photo, so a person (or an agent) can take a better one.
    warnings = []
    if left <= BORDER_PX or right >= width - BORDER_PX or top <= BORDER_PX or bottom >= tall - BORDER_PX:
        warnings.append("it runs off the edge of the photo: step back and take it whole"
                        + ("; rebuilt from the side it shows, as it's round" if round_ else ", the prop is cut off"))
    if view.fully_visible is False:
        warnings.append(f"part of it is hidden ({view.hidden or 'something in front'})"
                        + ("" if round_ else ": the prop has a gap there"))
    result = {"name": view.name or noun, "segmented_as": noun,
              "shape": "round" if round_ else "imagined" if carved else "cutout", "imagined": imagined_notes,
              "size_m": feature["properties"]["size_m"],
              "location": location, "warnings": warnings, "made_of": made_of, "model": str(glb), "place": str(geo),
              "viewer": str(viewer), "cost_usd": round(meta.spent, 4)}
    if args.json:
        print(json.dumps(result))
        return
    how = ("turned on a lathe: round" if round_ else
           "carved with the sides Muse Image imagined" if carved else "cut-out made solid")
    print(f"PROP      {result['name']} (SAM 3.1 cut out {noun!r}; Muse {view.confidence:.0%} sure; {how})")
    print(f"size:     {across:.2f} m wide, {max(depth, 0.03):.2f} m deep, {height:.2f} m tall")
    if location:
        print(f"where:    {location['lat']}, {location['lon']}: {location['distance_m']} m from the camera, "
              f"front facing {location['facing_bearing']:.0f}°")
    else:
        print("where:    unknown (the photo has no GPS or compass)")
    for n in imagined_notes:
        print(f"  {n}")
    for w in warnings:
        print(f"WARNING   {w}")
    print(f"model:    {glb}")
    print(f"place:    {geo}")
    print(f"view:     {viewer} (open in a browser)")
    print(f"cost:     ~${meta.spent:.4f} (0 if replies were reused)")


def cmd_pack(args) -> None:
    from .pack import write_pack

    folders = [Path(f) for f in args.folders]
    photos = set()
    for f in folders:  # which photos went in, for the credits
        photos |= {p.stem.removeprefix("check_") for p in f.glob("check_*.jpg")}
        for name in ("prop.geojson",):
            if (f / name).exists():
                props = json.loads((f / name).read_text(encoding="utf-8")).get("properties", {})
                photos |= {Path(p).stem for p in [props.get("photo"), *props.get("photos", [])] if p}
    out = Path(args.out) / "pack" / args.name
    try:
        manifest = write_pack(out, args.name, folders, sorted(photos))
    except (ValueError, FileNotFoundError, KeyError) as e:
        _fail(str(e), args.json)
    items = json.loads(manifest.read_text(encoding="utf-8"))["items"]
    if args.json:
        print(json.dumps({"pack": str(out), "manifest": str(manifest), "items": len(items)}))
        return
    print(f"PACK      {args.name}: {len(items)} item(s) in {out}")
    for i in items:
        where = f"{i['lat']:.6f}, {i['lon']:.6f}" if i["lat"] is not None else "where the player puts it"
        extra = f"replaces {i['replaces']}" if i["kind"] == "building" else f"faces {i['facing_bearing']:.0f}°"
        print(f"  {i['kind']:<9} {i['name'] or i['id']:<32} {where}  {extra}")
    print(f"manifest: {manifest}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="placelens",
        description="Photo of a real place in, the real building on the real map out.",
    )
    p.add_argument("--version", action="version", version=f"placelens {__version__}")
    p.add_argument("-C", dest="home", metavar="DIR",
                   help="run in DIR (where .env, cache/ and out/ are), as when a game starts it")
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="what the photo knows: GPS, compass, lens width")
    info.add_argument("photo")
    info.add_argument("--json", action="store_true", help="machine-readable output")
    info.set_defaults(func=cmd_info)

    find = sub.add_parser("find", help="which mapped buildings the photo is looking at")
    pick = sub.add_parser("pick", help="which one it is: SAM 3.1 outline + Muse, checked against each other")
    build = sub.add_parser("build", help="the picked building as a 3D model (building.glb) and place.geojson")
    for cmd in (find, pick):
        cmd.add_argument("photo")
    build.add_argument("photo", nargs="+", help="one or more photos of the same building")
    for cmd in (find, pick, build):
        cmd.add_argument("--lat", type=float, help="override the photo's latitude")
        cmd.add_argument("--lon", type=float, help="override the photo's longitude")
        cmd.add_argument("--heading", type=float, help="override the compass direction (0 = north, 90 = east)")
        cmd.add_argument("--fov", type=float, help="override the lens width in degrees")
        cmd.add_argument("--radius", type=int, default=200, help="how far to look, metres (default 200)")
        cmd.add_argument("--margin", type=float, default=15, help="compass error allowance, degrees (default 15)")
        cmd.add_argument("--top", type=int, default=5, help="how many candidates to consider (default 5)")
        cmd.add_argument("--out", default="out", help="folder for pictures (default out/)")
        cmd.add_argument("--no-map", action="store_true", help="skip the map picture")
        cmd.add_argument("--no-cache", action="store_true", help="always fetch fresh OpenStreetMap data")
        cmd.add_argument("--json", action="store_true", help="machine-readable output")
    find.set_defaults(func=cmd_find)
    for cmd in (pick, build):
        cmd.add_argument("--model", choices=["open", "spark", "wonder"], default="open",
                         help="Muse Spark Open (open weights, default), Spark (fast, cheap) or Wonder (slower, deeper)")
        cmd.add_argument("--votes", type=int, help="ask Muse this many times and go with the majority "
                                                   "(default 5 for open, 1 for the paid models)")
        cmd.add_argument("--move", type=float, default=10, help="GPS error allowance, metres (default 10)")
        cmd.add_argument("--fresh", action="store_true", help="call the API again instead of reusing saved replies")
        cmd.add_argument("--replies", help="folder of saved SAM and Muse replies to reuse (default <out>/pick)")
        cmd.add_argument("--notes", help="what you know that the photos can't show (\"the back is the same\", "
                                         "\"about 20 m wide\"); Muse reads it with every photo")
    pick.set_defaults(func=cmd_pick)
    build.add_argument("--building", type=int, help="build this candidate number instead (when pick asked)")
    build.add_argument("--floors", type=int, help="how many floors it has, when the map and Muse don't know")
    build.add_argument("--height", type=float, help="how tall it is in metres (wins over --floors)")
    build.add_argument("--no-front", action="store_true", help="plain walls, without the photo on them")
    build.add_argument("--no-openings", action="store_true", help="windows and doors left painted, not cut in")
    build.add_argument("--photo-walls", action="store_true",
                       help="the photos themselves on the walls, instead of plain walls in Muse's colours")
    build.add_argument("--side", action="append", default=[], metavar="PHOTO=SIDE",
                       help="which side of the building a photo shows (front, back, left, right; left as you face "
                            "the front), e.g. --side IMG_0173=back; for photos it can't place by itself")
    build.add_argument("--refine", action="store_true",
                       help="critique loop: Muse says what's off in the model over each photo; changes are kept "
                            "only if the model then matches the building's outline better")
    build.add_argument("--second-opinion", action="store_true",
                       help="PAID: with --model open, ask Muse Spark (Meta API) when Open can't tell a photo's side")
    build.add_argument("--symmetric", action="store_true",
                       help="its far side mirrors its near side: walls no photo saw copy the opposite wall")
    build.add_argument("--roof", choices=["gable", "hip", "flat"], help="the roof's kind, instead of asking Muse")
    build.add_argument("--pitch", type=float, help="the roof's slope in degrees, instead of asking Muse")
    build.set_defaults(func=cmd_build)

    prop = sub.add_parser("prop", help="photo of an object to a 3D prop at its real size and spot (prop.glb)")
    prop.add_argument("photo", nargs="+",
                      help="a photo of the object; or several, walking round it, to carve its real 3D shape")
    prop.add_argument("--what", help="what to cut out, if Muse names the wrong thing (e.g. 'bench')")
    prop.add_argument("--height", type=float, help="its real height in metres, instead of Muse's estimate")
    prop.add_argument("--width", type=float, help="its width across the photo in metres, instead of Muse's")
    prop.add_argument("--depth", type=float, help="its depth front to back in metres, instead of Muse's estimate")
    prop.add_argument("--shape", choices=["round", "cutout"],
                      help="round: the same from every side (turned on a lathe); cutout: the outline made solid. "
                           "Default: Muse decides")
    prop.add_argument("--symmetric", action="store_true",
                      help="its left and right are mirror images (Muse decides otherwise)")
    prop.add_argument("--notes", help="what you know that the photos can't show (\"the back is plain wood\", "
                                      "\"1.2 m tall\"); Muse reads it with the photos")
    prop.add_argument("--fov", type=float, help="override the lens width in degrees")
    prop.add_argument("--model", choices=["open", "spark", "wonder"], default="open", help="which Muse looks at it")
    prop.add_argument("--votes", type=int, help="ask Muse this many times (default 5 for open, 1 for the paid ones)")
    prop.add_argument("--match", action="store_true",
                      help="walk-around: place the cameras by matching the photos to each other (needs photos about "
                           "30° apart on textured ground), not by compass alone")
    prop.add_argument("--no-judge", action="store_true",
                      help="carved props: skip the last step, where Muse compares the model with the photos and what's "
                           "clearly extra is trimmed")
    prop.add_argument("--clean", action="store_true",
                      help="carved props: build it clean from its parts and materials (Muse names them) instead of "
                           "painting the photos' colours on. Tried on a teddy 31 Oct: the photo colours looked better")
    prop.add_argument("--no-imagine", action="store_true",
                      help="one photo: don't have Muse Image draw the unseen sides (a solid cut-out instead)")
    prop.add_argument("--fresh", action="store_true", help="call the API again instead of reusing saved replies")
    prop.add_argument("--out", default="out", help="folder for outputs (default out/)")
    prop.add_argument("--json", action="store_true", help="machine-readable output")
    prop.set_defaults(func=cmd_prop)

    pack = sub.add_parser("pack", help="gather built buildings and props into one place pack for a game")
    pack.add_argument("name", help="the pack's name, e.g. home")
    pack.add_argument("folders", nargs="+", help="folders made by build (out/build/...) and prop (out/props/...)")
    pack.add_argument("--out", default="out", help="the pack goes in <out>/pack/<name> (default out/)")
    pack.add_argument("--json", action="store_true", help="machine-readable output")
    pack.set_defaults(func=cmd_pack)

    survey = sub.add_parser("survey", help="agent: a folder of photos to a place pack, asking only when unsure")
    survey.add_argument("folder", help="the photos (buildings and objects, as taken)")
    survey.add_argument("--max-steps", type=int, default=15,
                        help="most heavy tool calls (pick, build, prop, pack) the agent may make (default 15)")
    survey.add_argument("--out", default="out", help="the log goes in <out>/survey/<folder>/ (default out/)")
    survey.add_argument("--notes", help="what you know that the photos can't show, in your own words: the agent "
                                        "and Muse read it")
    survey.add_argument("--notes-file", help="the notes from a text file instead")
    survey.add_argument("--progress", help="write each step to this file as it happens, one JSON line each (for a "
                                           "game showing it); the last line has done: true")
    survey.add_argument("--pack-to", help="copy the finished pack into this folder (a game's packs folder), named "
                                          "after the photos' folder")
    survey.set_defaults(func=cmd_survey)
    return p


def cmd_survey(args) -> None:
    """The survey agent (survey.py): Muse Spark Open turns a folder of photos into a place pack by itself."""
    from .meta import MetaClient, MissingKeyError
    from .survey import Survey, open_chat, open_describe, run_cli

    folder = Path(args.folder)
    if not folder.is_dir():
        _fail(f"no folder {folder}")
    try:
        meta = MetaClient()
    except MissingKeyError as e:
        _fail(str(e))
    out = Path(args.out) / "survey" / folder.name
    notes = args.notes or (Path(args.notes_file).read_text(encoding="utf-8") if args.notes_file else None)

    def tell(event: dict) -> None:
        if args.progress:
            with open(args.progress, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    survey = Survey(folder, open_chat(meta), run_cli, open_describe(meta, out), out, notes=notes, progress=tell)
    try:
        summary = survey.go(args.max_steps)
    except Exception as e:  # the game waiting on --progress hears about it
        tell({"done": True, "error": f"{type(e).__name__}: {e}"})
        raise
    pack = survey.work / "pack" / survey.pack if survey.pack else None
    if pack and args.pack_to:
        import shutil

        # (named after the photos' folder, not what the agent called it, so it can't land on another pack)
        pack = shutil.copytree(pack, Path(args.pack_to) / folder.name, dirs_exist_ok=True)
    tell({"done": True, "summary": summary, "questions": survey.questions, "pack": str(pack) if pack else None,
          "built": [{"kind": k, "name": w} for k, _, w in survey.built]})
    print()
    print(summary)
    print()
    for q in survey.questions:
        print(f"QUESTION  {q}")
    if pack:
        print(f"pack:     {pack}")
    print(f"log:      {out / 'summary.md'} (every step: {out / 'transcript.json'})")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252, which mangles "°"
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.home:
        os.chdir(args.home)
    try:
        args.func(args)
    except Failed as e:
        if getattr(args, "json", False):
            print(json.dumps({"error": e.msg}))
        else:
            print(f"placelens: {e.msg}", file=sys.stderr)
        sys.exit(e.code)


if __name__ == "__main__":
    main()
