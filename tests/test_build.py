import json
import struct

import numpy as np
from shapely.geometry import Polygon

from placelens.build import (
    Opening,
    WallPhoto,
    building_height,
    extrude,
    mirror_walls,
    parse_height,
    place_feature,
    write_glb,
)


def test_parse_height():
    assert parse_height("12") == 12
    assert parse_height("12.5 m") == 12.5
    assert abs(parse_height("40'") - 12.192) < 1e-9
    assert parse_height("tall") is None and parse_height(None) is None


def test_height_prefers_osm_then_muse_then_a_guess():
    assert building_height({"height": "9", "building:levels": "5"}, 2) == (9, "osm height")
    assert building_height({"building:levels": "2"}, 4) == (6, "osm floors")
    assert building_height({}, 3) == (9, "muse floors")
    assert building_height({}, None) == (6, "guess")


def _l_shape():
    return Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 8), (0, 8)])  # clockwise or not, it gets oriented


def test_extrude_walls_face_out_and_roof_covers_the_footprint():
    walls, roof = extrude(_l_shape(), 6.0)
    assert len(walls.positions) == 6 * 4 and len(walls.indices) == 6 * 6
    assert walls.positions[:, 1].min() == 0 and walls.positions[:, 1].max() == 6
    # Every wall's normal points away from the footprint's middle (map north is glTF -Z).
    for i in range(0, len(walls.positions), 4):
        mid = walls.positions[i:i + 4].mean(axis=0)
        probe = mid + walls.normals[i] * 0.1
        assert not _l_shape().contains_properly(Polygon(_square_around(probe[0], -probe[2])))
    # Roof triangles face up and add up to the footprint's area.
    tris = roof.positions[roof.indices].reshape(-1, 3, 3)
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert (cross[:, 1] > 0).all()
    assert abs(cross[:, 1].sum() / 2 - _l_shape().area) < 1e-4


def _square_around(x, y, r=0.01):
    return [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]


def test_wall_triangles_wind_counter_clockwise_from_outside():
    walls, _ = extrude(_l_shape(), 6.0)
    tris = walls.positions[walls.indices].reshape(-1, 3, 3)
    face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert (np.einsum("ij,ij->i", face, walls.normals[walls.indices[::3]]) > 0).all()


def _read_glb(data: bytes):
    magic, version, length = struct.unpack_from("<III", data)
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A and json_len % 4 == 0
    gltf = json.loads(data[20:20 + json_len])
    bin_len, bin_type = struct.unpack_from("<II", data, 20 + json_len)
    assert bin_type == 0x004E4942 and bin_len == gltf["buffers"][0]["byteLength"]
    return gltf, data[28 + json_len:28 + json_len + bin_len]


def test_glb_round_trip(tmp_path):
    parts = extrude(_l_shape(), 6.0)
    gltf, blob = _read_glb(write_glb(tmp_path / "b.glb", parts).read_bytes())
    prims = gltf["meshes"][0]["primitives"]
    assert [m["name"] for m in gltf["materials"]] == ["walls", "roof"]
    for prim, part in zip(prims, parts, strict=True):
        acc = gltf["accessors"][prim["attributes"]["POSITION"]]
        view = gltf["bufferViews"][acc["bufferView"]]
        assert view["byteOffset"] % 4 == 0
        got = np.frombuffer(blob, np.float32, acc["count"] * 3, view["byteOffset"]).reshape(-1, 3)
        assert np.array_equal(got, part.positions)
        assert acc["max"][1] == 6.0
        assert gltf["accessors"][prim["indices"]]["count"] == len(part.indices)


def test_place_feature_is_lon_lat():
    feature = place_feature([(-33.9, 18.4), (-33.9, 18.5), (-33.8, 18.5), (-33.9, 18.4)], (-33.85, 18.45),
                            6, "guess", "way/1", {})["features"][0]
    assert feature["geometry"]["coordinates"][0][0] == [18.4, -33.9]
    assert feature["properties"]["anchor"] == [18.45, -33.85]


def test_a_window_is_cut_in_and_set_back():
    from placelens.build import RECESS, Opening, WallPhoto

    box = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    window = Opening("window", 0.4, 0.6, 1.0, 2.0)  # 2 m wide, 1 m tall, in the middle of a 10 m wall
    parts = extrude(box, 3.0, {0: WallPhoto(b"jpeg", openings=(window,))})
    wall = next(p for p in parts if p.name == "wall 0")
    glass = next(p for p in parts if p.name == "glass 0")
    # The wall's face is 10 x 3 m less the window; the reveals are 2 x (2 + 1) m round x 0.12 m deep.
    tris = wall.positions[wall.indices].reshape(-1, 3, 3)
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2
    assert abs(area - (30 - 2 + 2 * 3 * RECESS["window"])) < 1e-4
    # The glass fills the hole, set back from the wall's face (the south wall's face is at z = 0: back is z < 0).
    assert len(glass.positions) == 4 and np.allclose(glass.positions[:, 2], -RECESS["window"], atol=1e-6)
    assert glass.roughness < 0.5


def test_a_built_building_has_no_pictures_and_real_windows():
    from placelens.build import FRAME, RECESS, Opening, WallPhoto
    from placelens.look import Look, parse_look

    box = Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    window = Opening("window", 0.4, 0.6, 1.0, 2.0)
    door = Opening("door", 0.1, 0.2, 0.05, 2.1)
    look = parse_look('```json\n{"walls": "#ffffff", "roof": "#000000", "door": "red"}\n```')
    assert look.walls == (1.0, 1.0, 1.0, 1.0) and look.roof[:3] == (0.0, 0.0, 0.0) and look.door == Look().door
    parts = {p.name: p for p in extrude(box, 3.0, {0: WallPhoto(b"jpeg", openings=(window, door))}, look=look)}
    assert set(parts) == {"walls", "glass", "door", "frames", "roof"}
    assert all(p.texture is None for p in parts.values())
    # The glass sits at the back of the hole; the frame stands in front of it, inside the hole.
    assert np.allclose(parts["glass"].positions[:, 2], -RECESS["window"], atol=1e-6)
    frames = parts["frames"].positions
    in_window = (frames[:, 0] > 4 - 0.1) & (frames[:, 0] < 6 + 0.1) & (frames[:, 1] > 1.0)
    assert np.isclose(frames[in_window, 2].min(), -RECESS["window"]) and np.isclose(
        frames[in_window, 2].max(), -RECESS["window"] + FRAME)
    # The sill stands out of the wall below the window.
    assert (frames[:, 2] > 0.05).any()
    # A 2 m window is split into two panes: a glazing bar down its middle.
    bar = (np.abs(frames[:, 0] - 5) < 0.03) & (frames[:, 1] > 1.2) & (frames[:, 1] < 1.8)
    assert not bar.any()  # (bars run the window's full height: their corners are at its top and bottom)
    assert ((np.abs(frames[:, 0] - 5) < 0.03) & np.isclose(frames[:, 1], 2.0)).any()


def _jpeg(w: int = 8, h: int = 4) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 100, 50)).save(buf, "JPEG")
    return buf.getvalue()


def test_a_symmetric_building_copies_the_opposite_wall_flipped():
    box = Polygon([(-10, -4), (10, -4), (10, 4), (-10, 4)])  # walls 0 south, 1 east, 2 north, 3 west
    door = Opening("door", 0.1, 0.2, 0.0, 2.1)
    seen = {0: WallPhoto(_jpeg(), 0.0, 0.8, None, 0.0, frozenset(), (door,))}
    mirrored = mirror_walls(box, seen)
    assert set(mirrored) == {2}  # the north wall; east and west have no opposite with a photo
    source, photo = mirrored[2]
    assert source == 0
    assert (round(photo.start, 6), round(photo.end, 6)) == (0.2, 1.0)
    assert [(o.kind, round(o.start, 6), round(o.end, 6)) for o in photo.openings] == [("door", 0.8, 0.9)]


def test_walls_of_a_different_length_are_not_mirrored():
    ell = Polygon([(0, 0), (20, 0), (20, 8), (12, 8), (12, 12), (0, 12)])  # wall 0 (20 m) faces walls 2 and 4 (8, 12 m)
    assert mirror_walls(ell, {0: WallPhoto(_jpeg())}) == {}
