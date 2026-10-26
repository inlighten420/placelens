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


def cmd_find(args) -> None:
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

    map_path = None
    if not args.no_map:
        from shapely.geometry import Polygon

        polys = [Polygon([frame.to_xy(*p) for p in b.outline]) for b in buildings]
        map_path = draw_map(
            Path(args.out) / f"{Path(args.photo).stem}_map.png",
            polys, candidates, heading, fov, args.radius, args.margin, args.top,
        )

    rows = []
    for c in candidates[: args.top]:
        centre = c.polygon.centroid
        t = c.building.tags
        rows.append({
            "rank": c.rank,
            "osm_id": c.building.osm_id,
            "osm_url": c.building.url,
            "name": t.get("name"),
            "levels": t.get("building:levels"),
            "height": t.get("height"),
            "view_share": round(c.share, 3),
            "distance_m": round(c.distance, 1),
            "centre": [round(v, 6) for v in frame.to_latlon(centre.x, centre.y)],
        })

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
    find.add_argument("photo")
    find.add_argument("--lat", type=float, help="override the photo's latitude")
    find.add_argument("--lon", type=float, help="override the photo's longitude")
    find.add_argument("--heading", type=float, help="override the compass direction (0 = north, 90 = east)")
    find.add_argument("--fov", type=float, help="override the lens width in degrees")
    find.add_argument("--radius", type=int, default=200, help="how far to look, metres (default 200)")
    find.add_argument("--margin", type=float, default=15, help="compass error allowance, degrees (default 15)")
    find.add_argument("--top", type=int, default=5, help="how many candidates to list (default 5)")
    find.add_argument("--out", default="out", help="folder for the map picture (default out/)")
    find.add_argument("--no-map", action="store_true", help="skip the map picture")
    find.add_argument("--no-cache", action="store_true", help="always fetch fresh OpenStreetMap data")
    find.add_argument("--json", action="store_true", help="machine-readable output")
    find.set_defaults(func=cmd_find)
    return p


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252, which mangles "°"
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
