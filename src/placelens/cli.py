"""placelens: command line for people and AI agents alike. Every command takes --json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .osm import fetch_buildings
from .photo import read_camera
from .render import draw_map
from .view import LocalFrame, find_candidates

EXIT_MISSING = 2  # the photo lacks GPS or compass; pass them by hand
ATTRIBUTION = "Building data © OpenStreetMap contributors, ODbL (openstreetmap.org/copyright)"


def _fail(msg: str, as_json: bool, code: int = 1):
    if as_json:
        print(json.dumps({"error": msg}))
    else:
        print(f"placelens: {msg}", file=sys.stderr)
    sys.exit(code)


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


def cmd_pick(args) -> None:
    import hashlib

    from PIL import Image

    from .meta import SPARK, WONDER, MetaClient, MissingKeyError, parse_segments
    from .pick import build_prompt, decide, line_up, parse_verdict, photo_span, prepare_photo
    from .render import draw_check

    cam, frame, buildings, candidates, heading, fov = _locate(args)
    candidates = candidates[: args.top]
    if not candidates:
        _fail("no mapped buildings in view; check the compass with `placelens find` or widen --margin", args.json)
    try:
        meta = MetaClient()
    except MissingKeyError as e:
        _fail(str(e), args.json)

    # Replies are saved next to the outputs, so re-running costs nothing unless --fresh.
    out = Path(args.out) / "pick"
    stem = Path(args.photo).stem
    small = prepare_photo(args.photo, out / f"{stem}.jpg")
    width = Image.open(small).width

    sam_file = out / f"{stem}_sam.txt"
    if sam_file.exists() and not args.fresh:
        segments = parse_segments(sam_file.read_text(encoding="utf-8"))
    else:
        segments, raw = meta.segment(small, "building")
        sam_file.write_text(raw, encoding="utf-8")
    main = segments[0] if segments else None  # biggest: the building the photo is of
    span = photo_span(main.box, width, fov) if main else None

    model = WONDER if args.model == "wonder" else SPARK
    prompt = build_prompt(candidates)
    muse_file = out / f"{stem}_{args.model}_{hashlib.sha1(prompt.encode()).hexdigest()[:8]}.txt"
    if muse_file.exists() and not args.fresh:
        answer = muse_file.read_text(encoding="utf-8")
    else:
        answer = meta.ask(prompt, small, model=model)
        muse_file.write_text(answer, encoding="utf-8")
    verdict = parse_verdict(answer)
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
    if args.json:
        print(json.dumps(result))
        return

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
    if decision.status == "picked":
        print(f"PICKED    {chosen.rank}. {chosen.building.osm_id}  {chosen.building.url}")
        print(f"          {decision.why}")
    else:
        guess = f" Best guess: {decision.rank}." if decision.rank else ""
        print(f"ASK       not sure: {decision.why}.{guess} Which numbered building is it? (see the map)")
    print(f"check:    {check_path}")
    if map_path:
        print(f"map:      {map_path}")
    print(f"cost:     ~${meta.spent:.4f} (0 if replies were reused)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="placelens",
        description="Photo of a real place in, the real building on the real map out.",
    )
    p.add_argument("--version", action="version", version=f"placelens {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="what the photo knows: GPS, compass, lens width")
    info.add_argument("photo")
    info.add_argument("--json", action="store_true", help="machine-readable output")
    info.set_defaults(func=cmd_info)

    find = sub.add_parser("find", help="which mapped buildings the photo is looking at")
    pick = sub.add_parser("pick", help="which one it is: SAM 3.1 outline + Muse, checked against each other")
    for cmd in (find, pick):
        cmd.add_argument("photo")
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
    pick.add_argument("--model", choices=["spark", "wonder"], default="spark",
                      help="Muse Spark (fast, cheap) or Muse Wonder (slower, deeper)")
    pick.add_argument("--move", type=float, default=10, help="GPS error allowance, metres (default 10)")
    pick.add_argument("--fresh", action="store_true", help="call the API again instead of reusing saved replies")
    pick.set_defaults(func=cmd_pick)
    return p


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252, which mangles "°"
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
