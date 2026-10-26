"""Fetch building footprints around a point from OpenStreetMap (Overpass API), cached on disk."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Public Overpass servers are often busy; try each in turn.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",  # its 502s are usually momentary
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
USER_AGENT = "placelens/0.1 (github.com/inlighten420/placelens)"
CACHE_DIR = Path("cache")


@dataclass
class Building:
    osm_id: str  # e.g. "way/123456"
    outline: list[tuple[float, float]]  # (lat, lon), closed ring
    tags: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"https://www.openstreetmap.org/{self.osm_id}"


def _query(lat: float, lon: float, radius: int) -> str:
    around = f"around:{radius},{lat},{lon}"
    return f"""[out:json][timeout:25];
(way["building"]({around});relation["building"]({around}););
out tags geom;"""


def _ring(geometry) -> list[tuple[float, float]]:
    return [(p["lat"], p["lon"]) for p in geometry]


def parse(data: dict) -> list[Building]:
    buildings = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el:
            buildings.append(Building(f"way/{el['id']}", _ring(el["geometry"]), tags))
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and "geometry" in m:
                    buildings.append(Building(f"relation/{el['id']}", _ring(m["geometry"]), tags))
    return [b for b in buildings if len(b.outline) >= 4]


def _overpass(query: str) -> dict:
    errors = []
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            errors.append(f"{url.split('/')[2]}: {e}")
    raise RuntimeError("all Overpass servers failed: " + "; ".join(errors))


def fetch_buildings(lat: float, lon: float, radius: int = 200, use_cache: bool = True) -> list[Building]:
    cache = CACHE_DIR / f"osm_{lat:.4f}_{lon:.4f}_{radius}.json"
    if use_cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        data = _overpass(_query(lat, lon, radius))
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    return parse(data)
