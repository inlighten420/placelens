"""A place pack: buildings and props from PlaceLens in one folder, with a manifest any game can read.

    manifest.json   what's in it and where each thing goes (latitude, longitude, which way it turns)
    credits.json    where everything came from: map data, photos, models
    *.glb           the models

Every model is in metres with Y up. A prop whose photo had no GPS or compass has lat and lon null: the game puts
it where the player is. A building is already turned to the map (X east, Z south: "yaw_deg" 0). A prop's
front is its +Z; "yaw_deg" turns it about Y, in that same east/south frame, so its front faces "facing_bearing".
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "placelens-pack"
VERSION = 1
OSM = "Building data © OpenStreetMap contributors, ODbL (openstreetmap.org/copyright)"


def prop_yaw(facing_bearing: float) -> float:
    """Degrees to turn a prop about Y (X east, Z south) so its +Z front faces a compass bearing: +Z faces south
    (bearing 180) unturned, and turning by y swings it to bearing 180 - y."""
    return (180.0 - facing_bearing) % 360


def _feature(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["features"][0] if data.get("type") == "FeatureCollection" else data


def building_item(folder: Path) -> dict:
    f = _feature(folder / "place.geojson")
    p = f["properties"]
    ident = p["osm_id"].replace("/", "_")
    return {"kind": "building", "id": ident, "model": f"{ident}.glb", "source": folder / p.get("model", "building.glb"),
            "lon": p["anchor"][0], "lat": p["anchor"][1], "yaw_deg": 0.0, "replaces": p["osm_id"],
            "height_m": p.get("height_m"), "footprint": f["geometry"]["coordinates"][0],
            "roof": p.get("roof"), "name": p.get("name")}


def prop_item(folder: Path) -> dict:
    f = _feature(folder / "prop.geojson")
    p = f["properties"]
    # No GPS or compass in its photo: no lat/lon, and a game puts it where the player is
    lon, lat = f["geometry"]["coordinates"] if f.get("geometry") else (None, None)
    facing = p.get("facing_bearing") or 0.0
    return {"kind": "prop", "id": folder.name, "model": f"{folder.name}.glb",
            "source": folder / p.get("model", "prop.glb"), "lon": lon, "lat": lat,
            "yaw_deg": round(prop_yaw(facing), 1), "facing_bearing": facing, "name": p.get("name"),
            "size_m": p.get("size_m")}


def write_pack(out: Path, name: str, folders: list[Path], photos: list[str]) -> Path:
    """Gather the built buildings and props in `folders` into a pack at `out`. Returns the manifest's path."""
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for folder in folders:
        if (folder / "place.geojson").exists():
            item = building_item(folder)
        elif (folder / "prop.geojson").exists():
            item = prop_item(folder)
        else:
            raise ValueError(f"{folder}: neither a building (place.geojson) nor a prop (prop.geojson)")
        shutil.copyfile(item.pop("source"), out / item["model"])
        items.append(item)
    if not items:
        raise ValueError("nothing to pack")
    anchor = next((i for i in items if i["kind"] == "building"), None) or next(
        (i for i in items if i["lat"] is not None), items[0])
    manifest = {"format": FORMAT, "version": VERSION, "name": name,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "centre": {"lon": anchor["lon"], "lat": anchor["lat"]},
                "axes": "metres, Y up; buildings X east, Z south; props front +Z, turned by yaw_deg about Y",
                "items": items}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    credits = {"map": OSM if any(i["kind"] == "building" for i in items) else None,
               "photos": [{"file": p, "source": "taken by the pack's author"} for p in photos],
               "made_with": ["PlaceLens (github.com/inlighten420/placelens)", "Meta SAM 3.1", "Meta Muse Spark",
                             "Meta Muse Image (props: sides no photo saw, checked by Muse Spark)"]}
    (out / "credits.json").write_text(json.dumps(credits, indent=1), encoding="utf-8")
    return path
