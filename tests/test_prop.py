import math

import numpy as np
from PIL import Image

from placelens.build import write_glb
from placelens.prop import cutout_png, outline, parse_prop, place, solid

from .test_build import _read_glb


def test_parse_prop():
    v = parse_prop('```json\n{"name": "bin", "segment": "wheelie bin", "width_m": 0.6, "depth_m": 0.7, '
                   '"height_m": 1.1, "distance_m": 4, "fully_visible": false, "hidden": "a car", "confidence": 0.9}```')
    assert (v.name, v.segment, v.height, v.distance, v.fully_visible, v.hidden) == \
        ("bin", "wheelie bin", 1.1, 4.0, False, "a car")
    assert parse_prop("no idea").name is None


def _mask():
    mask = np.zeros((200, 100), bool)
    mask[20:180, 10:90] = True  # 80 wide, 160 tall
    return mask


def test_outline_follows_the_mask():
    shape = outline(_mask())
    assert abs(shape.area - 80 * 160) < 80 * 160 * 0.03
    minx, miny, maxx, maxy = shape.bounds
    assert abs(minx - 10) < 1.5 and abs(maxy - 180) < 1.5


def test_solid_prop_is_real_size_and_faces_forward(tmp_path):
    png = cutout_png(Image.new("RGB", (100, 200), "red"), _mask())
    [part] = solid(outline(_mask()), (100, 200), height=2.0, depth=0.5, png=png)
    p = part.positions
    assert abs(p[:, 1].min()) < 1e-6 and abs(p[:, 1].max() - 2.0) < 1e-6  # stands on the ground, 2 m tall
    assert abs(p[:, 0].max() - p[:, 0].min() - 1.0) < 0.03  # 80 x 160 px at 2 m tall: 1 m wide
    assert abs(p[:, 2].max() - 0.25) < 1e-6 and abs(p[:, 2].min() + 0.25) < 1e-6
    tris = p[part.indices].reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert (np.einsum("ij,ij->i", n, part.normals[part.indices[::3]]) > 0).all()  # every face winds outward
    gltf, _ = _read_glb(write_glb(tmp_path / "p.glb", [part]).read_bytes())
    assert gltf["images"][0]["mimeType"] == "image/png" and gltf["materials"][0]["alphaMode"] == "MASK"


def test_place():
    east, north, facing = place(heading=90, offset=0, distance=10)  # straight ahead, looking east
    assert math.isclose(east, 10) and abs(north) < 1e-9 and facing == 270  # its front faces back west


def test_lathe_turns_a_round_thing_all_the_way_round():
    import io

    from placelens.prop import lathe

    buf = io.BytesIO()
    Image.new("RGB", (100, 200), "red").save(buf, "JPEG")
    [part] = lathe(_mask(), buf.getvalue(), height=2.0)
    p = part.positions
    radius = np.hypot(p[:, 0], p[:, 2])
    assert abs(radius.max() - 0.5) < 0.01 and abs(p[:, 1].max() - 2.0) < 1e-6  # 1 m across, 2 m tall
    tris = p[part.indices].reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    mid = tris.mean(axis=1)
    size = np.linalg.norm(n, axis=1)
    lids = np.abs(n[:, 1]) > 0.99 * size  # flat faces: the top and bottom
    walls = ~lids & (size > 1e-9)
    assert (np.einsum("ij,ij->i", n[walls], mid[walls] * [1, 0, 1]) > 0).all()  # walls face out
    assert (np.sign(n[lids, 1]) == np.sign(mid[lids, 1] - 1.0)).all()  # the top faces up, the bottom down


def test_lathe_uses_the_clean_side_when_one_runs_off_the_photo():
    import io

    from placelens.prop import lathe

    buf = io.BytesIO()
    Image.new("RGB", (100, 200), "red").save(buf, "JPEG")
    [part] = lathe(_mask(), buf.getvalue(), height=2.0, cut=(True, False), max_radius_px=40)
    assert part.uvs[:, 0].min() >= 0.49  # only the right half of the photo is used
