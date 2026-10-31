import json
import math

from placelens.pack import prop_yaw, write_pack


def test_prop_yaw_turns_the_front_to_its_bearing():
    for bearing in (0, 90, 180, 257.7):
        yaw = math.radians(prop_yaw(bearing))
        x, z = math.sin(yaw), math.cos(yaw)  # +Z turned about Y; X is east, Z south
        assert abs((math.degrees(math.atan2(x, -z)) - bearing + 180) % 360 - 180) < 1e-9


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_pack_gathers_buildings_and_props(tmp_path):
    house, tank = tmp_path / "b", tmp_path / "p"
    _write(house / "place.geojson", {"type": "FeatureCollection", "features": [{
        "geometry": {"type": "Polygon", "coordinates": [[[18.4, -33.9], [18.5, -33.9], [18.5, -33.8], [18.4, -33.9]]]},
        "properties": {"osm_id": "way/7", "height_m": 6.0, "anchor": [18.45, -33.85], "model": "building.glb"}}]})
    (house / "building.glb").write_bytes(b"glb")
    _write(tank / "prop.geojson", {"type": "Feature", "geometry": {"type": "Point", "coordinates": [18.46, -33.86]},
                                   "properties": {"name": "tank", "model": "prop.glb", "facing_bearing": 90}})
    (tank / "prop.glb").write_bytes(b"glb")
    manifest = json.loads(write_pack(tmp_path / "pack", "home", [house, tank], ["IMG_1"]).read_text())
    assert manifest["format"] == "placelens-pack" and manifest["centre"] == {"lon": 18.45, "lat": -33.85}
    building, prop = manifest["items"]
    assert building["replaces"] == "way/7" and (tmp_path / "pack" / building["model"]).exists()
    assert prop["yaw_deg"] == 90 and prop["lat"] == -33.86


def test_a_prop_without_gps_is_packed_for_the_game_to_place(tmp_path):
    bear = tmp_path / "bear"
    _write(bear / "prop.geojson", {"type": "Feature", "geometry": None,
                                   "properties": {"name": "teddy bear", "model": "prop.glb", "facing_bearing": None}})
    (bear / "prop.glb").write_bytes(b"glb")
    manifest = json.loads(write_pack(tmp_path / "pack", "toys", [bear], []).read_text(encoding="utf-8"))
    item = manifest["items"][0]
    assert item["lat"] is None and item["lon"] is None and item["name"] == "teddy bear"
