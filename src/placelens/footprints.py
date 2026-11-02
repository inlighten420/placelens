"""Building footprints around a point: Overture Maps first, OpenStreetMap if Overture can't be reached.

Overture's buildings are OpenStreetMap's plus the ones Microsoft and Google traced from satellite pictures that nobody
has mapped (round the user's house: 157 from OpenStreetMap, 81 more from Google, 21 more from Microsoft, 1 Nov), so
far more buildings are there to be found. A building that came from OpenStreetMap keeps its OSM id ("way/123"), so a
game can still swap its own box for the model; the others are "overture/<id>".

Overture is read straight from its public files (the `overturemaps` package, its catalog narrowing which files to
read). Python's own certificate list rejected the catalog's certificate on 1 Nov while Windows accepted it, so the
system's trust store is used (`truststore`), with checking kept on.
"""

from __future__ import annotations

import json
import math
import re

from .osm import CACHE_DIR, Building

M_PER_DEG_LAT = 110_540


def fetch_buildings(lat: float, lon: float, radius: int = 200, use_cache: bool = True) -> list[Building]:
    """The buildings within about `radius` metres: Overture's, else (unreachable, or PLACELENS_FOOTPRINTS=osm)
    OpenStreetMap's.

"""
    import os

    if os.environ.get("PLACELENS_FOOTPRINTS", "overture") == "osm":
        from .osm import fetch_buildings as osm_buildings

        return osm_buildings(lat, lon, radius, use_cache)
    try:
        return overture_buildings(lat, lon, radius, use_cache)
    except Exception as e:  # (no network to Overture, its files moved...: the map we always had)
        from .osm import fetch_buildings as osm_buildings

        print(f"footprints: Overture failed ({type(e).__name__}: {e}); using OpenStreetMap")
        return osm_buildings(lat, lon, radius, use_cache)


def overture_buildings(lat: float, lon: float, radius: int = 200, use_cache: bool = True) -> list[Building]:
    cache = CACHE_DIR / f"overture_{lat:.4f}_{lon:.4f}_{radius}.json"
    if use_cache and cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        rows = _download(lat, lon, radius)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")
    return [b for b in (_building(r) for r in rows) if b is not None]


def _download(lat: float, lon: float, radius: int) -> list[dict]:
    """Overture's building rows in the square round the point: geometry (as GeoJSON), id, sources, height, floors,
    name, roof shape."""
    import truststore

    truststore.inject_into_ssl()
    from overturemaps import core
    from shapely import wkb
    from shapely.geometry import mapping

    dlat = radius / M_PER_DEG_LAT
    dlon = radius / (111_320 * math.cos(math.radians(lat)))
    reader = core.record_batch_reader("building", (lon - dlon, lat - dlat, lon + dlon, lat + dlat), stac=True)
    if reader is None:
        raise RuntimeError("Overture returned nothing for this area")
    rows = []
    for batch in reader:
        for r in batch.to_pylist():
            geometry = r.get("geometry")
            if geometry is None:
                continue
            names = r.get("names") or {}
            rows.append({"id": r.get("id"), "geometry": mapping(wkb.loads(geometry)),
                         "sources": [{"dataset": s.get("dataset"), "record_id": s.get("record_id")}
                                     for s in (r.get("sources") or [])],
                         "height": r.get("height"), "num_floors": r.get("num_floors"),
                         "roof_shape": r.get("roof_shape"), "name": names.get("primary")})
    return rows


def _building(row: dict) -> Building | None:
    """An Overture row as a Building: its outer ring as (lat, lon), its id (the OSM one if it came from OSM), and tags
    in OSM's words so the rest of PlaceLens reads them as it always has."""
    geometry = row["geometry"]
    rings = geometry["coordinates"] if geometry["type"] == "Polygon" else \
        max(geometry["coordinates"], key=lambda poly: len(poly[0])) if geometry["type"] == "MultiPolygon" else None
    if not rings or len(rings[0]) < 4:
        return None
    outline = [(float(y), float(x)) for x, y, *_ in rings[0]]
    tags = {"source": "overture: " + ", ".join(sorted({s["dataset"] for s in row["sources"] if s.get("dataset")}))}
    if row.get("height"):
        tags["height"] = str(row["height"])
    if row.get("num_floors"):
        tags["building:levels"] = str(row["num_floors"])
    if row.get("roof_shape"):
        tags["roof:shape"] = row["roof_shape"]
    if row.get("name"):
        tags["name"] = row["name"]
    return Building(_ident(row), outline, tags)


def _ident(row: dict) -> str:
    """'way/123456789' for a building OpenStreetMap gave Overture ('w123456789@1'), else 'overture/<id>'."""
    for s in row["sources"]:
        m = re.fullmatch(r"([wr])(\d+)(@\d+)?", str(s.get("record_id") or ""))
        if s.get("dataset") == "OpenStreetMap" and m:
            return f"{'way' if m.group(1) == 'w' else 'relation'}/{m.group(2)}"
    return f"overture/{row['id']}"
