import json
import struct

import numpy as np
import pytest
from scipy import ndimage

from placelens import soft
from placelens.carve import Shot, carve, place_shots

# A figure made of balls, sitting on the ground: body, head on top, an arm either side. Labels: 1 body, 2 head, 3 arm.
FIGURE = [((0.0, 0.25, 0.0), 0.25, 1), ((0.0, 0.62, 0.0), 0.15, 2), ((-0.3, 0.32, 0.0), 0.09, 3),
          ((0.3, 0.32, 0.0), 0.09, 3)]
HEIGHT = 0.77
SIZE = (0.8, 0.5, HEIGHT)
NOUNS = ["head", "arm"]


def _figure_photos(angles=(0, 90, 180, 270), elevation=20.0):
    """Photos of the figure from round it, each with its cut-out and its part labels (the nearest ball at each
    pixel)."""
    rng = np.random.default_rng(1)
    pts, labs = [], []
    for (cx, cy, cz), r, lab in FIGURE:
        v = rng.normal(size=(30000, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pts.append(v * r + [cx, cy, cz])
        labs.append(np.full(len(v), lab))
    pts, labs = np.concatenate(pts), np.concatenate(labs)
    shots, labels = [], []
    for a in angles:
        s = Shot(np.zeros((600, 800), bool), np.full((600, 800, 3), 128, np.uint8), 700.0, float(a), 1.8,
                 (400.0, 300.0), elevation)
        cols, rows, depth = s.project(pts, HEIGHT / 2)
        c, r = np.round(cols).astype(int), np.round(rows).astype(int)
        order = np.argsort(-depth)  # far first, so the nearest ball's label is written last
        lab = np.zeros(s.mask.shape, np.uint8)
        lab[r[order], c[order]] = labs[order]
        s.mask[:] = ndimage.binary_closing(lab > 0, iterations=3)
        _, (ir, ic) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        labels.append(np.where(s.mask, lab[ir, ic], 0).astype(np.uint8))
        shots.append(s)
    place_shots(shots, HEIGHT)
    for s in shots:
        s.elevation = elevation
    return shots, labels


def test_label_map_keeps_the_most_specific_part_and_ignores_the_whole_thing():
    cutout = np.zeros((100, 100), bool)
    cutout[10:90, 10:90] = True
    arm = np.zeros_like(cutout)
    arm[20:40, 10:30] = True
    hand = np.zeros_like(cutout)
    hand[20:28, 10:18] = True
    whole = cutout.copy()  # SAM finding the whole thing again as "head": not a part
    lab = soft.label_map(cutout, [[whole], [arm], [hand]])
    assert lab[50, 50] == soft.BODY and lab[0, 0] == 0
    assert lab[35, 25] == 3  # arm
    assert lab[22, 12] == 4  # the hand, smaller, wins over the arm it's on
    assert not (lab == 2).any()


def test_part_names_are_voted(tmp_path):
    photo = tmp_path / "front.jpg"
    replies = ['{"parts": ["head", "arm", "leg", "ear"]}', 'Sure: {"parts": ["head", "arm", "leg", "tail"]}',
               '{"parts": ["Body", "head", "leg", "arm"]}']
    for i, text in enumerate(replies):
        (tmp_path / ("front_parts.txt" if i == 0 else f"front_parts_v{i}.txt")).write_text(text, encoding="utf-8")
    nouns = soft.part_nouns(None, photo, tmp_path)  # (all three saved: nothing is asked)
    assert nouns == ["head", "arm", "leg"]  # ear and tail had one vote each; "body" is never a part


@pytest.fixture(scope="module")
def figure():
    shots, labels = _figure_photos()
    kept, corner, voxel = carve(shots, SIZE, misses=0, resolution=48)
    return shots, labels, (kept, corner, voxel)


def test_rebuilds_the_figure_from_smooth_parts_on_joints(figure, monkeypatch, tmp_path):
    monkeypatch.setattr(soft, "FIT_ROUNDS", 2)
    shots, labels, (kept, corner, voxel) = figure
    made, lines = soft.rebuild(kept, corner, voxel, shots, HEIGHT, labels, NOUNS)
    assert made is not None, lines
    names = [p.name for p in made.pieces]
    assert names[0] == "body" and made.pieces[0].parent is None
    assert sorted(names[1:]) == ["head", "left arm", "right arm"]
    for p in made.pieces[1:]:
        assert p.parent == 0  # all on the body
    head = next(p for p in made.pieces if p.name == "head")
    assert head.swing == soft.TURN and abs(head.axis[1]) == 1.0  # stands up: turns about the vertical
    arms = [p for p in made.pieces if "arm" in p.name]
    assert all(p.swing == soft.SWING for p in arms)
    left = next(p for p in arms if p.name == "left arm")
    assert left.pivot[0] > 0.15  # the figure's own left is +X, where the body meets that arm
    assert made.agreement >= made.carved_agreement - soft.KEEP_WITHIN
    top = made.part.positions[:, 1].max()
    assert abs(top - HEIGHT) < 0.08  # the head's top, about where it really is

    path = soft.write_moving(tmp_path / "moving.glb", made, "figure")
    data = path.read_bytes()
    size = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + size])
    assert [n["name"] for n in gltf["nodes"]] == names
    assert gltf["scenes"][0]["nodes"] == [0] and sorted(gltf["nodes"][0]["children"]) == [1, 2, 3]
    channels = gltf["animations"][0]["channels"]
    assert {c["target"]["node"] for c in channels} == {1, 2, 3}  # the body rests on the ground: it doesn't move
    assert all(c["target"]["path"] == "rotation" for c in channels)


def test_a_thing_the_parts_dont_describe_keeps_its_carving(figure, monkeypatch):
    monkeypatch.setattr(soft, "FIT_ROUNDS", 1)
    monkeypatch.setattr(soft, "KEEP_WITHIN", -1.0)  # (no smooth version could be that much better than its carving)
    shots, labels, (kept, corner, voxel) = figure
    made, lines = soft.rebuild(kept, corner, voxel, shots, HEIGHT, labels, NOUNS)
    assert made is None and "carved" in lines[0]


def test_shapes_know_inside_from_outside():
    prm = np.array([0.0, 0.5, 0.0, np.log(0.2), np.log(0.3), np.log(0.2), 0.0, 0.0, 0.0])
    pts = np.array([[0.0, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0], [0.15, 0.5, 0.0]])
    for shape in ("ellipsoid", "cylinder", "box"):
        d = soft.shape_distance(shape, pts, prm)
        assert d[0] < 0 and d[1] > 0 and d[2] > 0 and d[3] < 0, shape
    wide_top = np.append(prm, 0.5)  # (a taper half as wide again at the top)
    assert soft.shape_distance("taper", np.array([[0.25, 0.75, 0.0]]), wide_top)[0] < 0  # inside near the top
    assert soft.shape_distance("taper", np.array([[0.25, 0.25, 0.0]]), wide_top)[0] > 0  # outside near the bottom


def test_spiky_things_arent_solid():
    ball = np.linalg.norm(np.indices((21, 21, 21)) - 10, axis=0) <= 9
    spikes = np.zeros((21, 21, 21), bool)
    spikes[10, :, 10] = spikes[:, 10, 10] = spikes[10, 10, :] = True
    spikes[np.arange(21), np.arange(21), 10] = True
    assert soft.solidity(ball) > 0.9 > soft.MIN_SOLID > soft.solidity(spikes)


def _can_photos(angles=(0, 90, 180, 270), elevation=20.0):
    """Photos of a tin can 0.3 m tall, 0.2 m across, standing on the floor: no parts, the whole thing one piece."""
    rng = np.random.default_rng(2)
    n = 40000
    a, y = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 0.3, n)
    side = np.stack([0.1 * np.cos(a), y, 0.1 * np.sin(a)], axis=1)
    r = 0.1 * np.sqrt(rng.uniform(0, 1, n))
    top = np.stack([r * np.cos(a), np.full(n, 0.3), r * np.sin(a)], axis=1)
    pts = np.concatenate([side, top])
    shots = []
    for ang in angles:
        s = Shot(np.zeros((600, 800), bool), np.full((600, 800, 3), 128, np.uint8), 900.0, float(ang), 1.2,
                 (400.0, 300.0), elevation)
        cols, rows, _ = s.project(pts, 0.15)
        s.mask[np.round(rows).astype(int), np.round(cols).astype(int)] = True
        s.mask[:] = ndimage.binary_closing(s.mask, iterations=3)
        shots.append(s)
    place_shots(shots, 0.3)
    for s in shots:
        s.elevation = elevation
    return shots


def test_a_tin_can_comes_out_a_cylinder_standing_up(monkeypatch):
    monkeypatch.setattr(soft, "FIT_ROUNDS", 2)
    shots = _can_photos()
    kept, corner, voxel = carve(shots, (0.2, 0.2, 0.3), misses=0, resolution=40)
    labels = [s.mask.astype(np.uint8) for s in shots]  # (Muse named no parts: it's all body)
    made, lines = soft.rebuild(kept, corner, voxel, shots, 0.3, labels, [])
    assert made is not None, lines
    assert "body: cylinder" in lines[0] or "body: taper" in lines[0], lines[0]
    assert not made.rig  # (one piece: nothing to move)
    pos = made.part.positions
    assert abs(pos[:, 1].max() - 0.3) < 0.04 and pos[:, 1].min() < 0.03  # its real height, on the floor
    width = pos[:, [0, 2]].max(axis=0) - pos[:, [0, 2]].min(axis=0)
    assert np.all(np.abs(width - 0.2) < 0.04)  # round: as wide one way as the other


def test_a_shape_bigger_than_the_carving_isnt_cut_open(figure):
    """A clean shape fitted past the carving's box is meshed whole: the field is outside all round the grid's edge."""
    shots, labels, (kept, corner, voxel) = figure
    fitter = soft.Fitter([soft.Piece("body", soft.BODY, kept)], kept, corner, voxel, shots, HEIGHT, labels)
    fitter.kinds = ["ellipsoid"]
    big = soft._start("ellipsoid", kept, corner, voxel)
    big[3:6] += np.log(1.6)  # (60% bigger than the carving every way)
    fitter.params = [big]
    origin, shape = fitter.extent(corner, kept.shape)
    field = fitter.field(shape, origin, voxel)
    rim = np.concatenate([field[[0, -1]].ravel(), field[:, [0, -1]].ravel(), field[:, :, [0, -1]].ravel()])
    assert (rim > 0).all()  # closed: nothing reaches the grid's edge
    assert (field < 0).sum() > kept.sum()  # and bigger than the carving, as fitted
