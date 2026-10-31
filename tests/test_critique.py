from shapely.geometry import box

from placelens.build import wall_edges
from placelens.critique import (
    Edit,
    Shape,
    View,
    apply,
    draw_overlay,
    error,
    parse_critique,
    profile_error,
    proposals,
    refine,
    silhouette,
)
from placelens.front import Camera

# A camera 20 m south of a 12 x 8 m building, looking north.
CAM = Camera(0, -20, 0, 60, 800, 600)
REAL = Shape(box(-6, -4, 6, 4), 6.0, "gable", 30.0)


def _south(shape: Shape) -> int:
    """The number of the wall facing the camera."""
    return next(i for i, (a, b) in enumerate(wall_edges(shape.poly)) if a[1] == b[1] == min(a[1], b[1], -3.9))


def _views(real: Shape) -> list[View]:
    return [View("synthetic", None, CAM, silhouette(real, real.roof(), CAM, (0, 0)), (0, 0))]


def test_the_real_shape_matches_itself_and_a_wrong_one_doesnt():
    views = _views(REAL)
    assert error(REAL, views) == 0
    low = Shape(REAL.poly, 4.5, "gable", 30.0)
    assert error(low, views) > 0.02


def test_profile_error_counts_columns_only_one_covers():
    real = silhouette(REAL, REAL.roof(), CAM, (0, 0))
    narrow = Shape(box(-3, -4, 3, 4), 6.0, "gable", 30.0)
    assert profile_error(silhouette(narrow, narrow.roof(), CAM, (0, 0)), real) > 0.05


def test_move_wall_and_notch():
    south = _south(REAL)
    moved = apply(REAL, Edit("move_wall", south, 1.0))
    assert moved.poly.bounds == (-6, -5, 6, 4)
    recess = apply(REAL, Edit("notch", south, -1.0, (0.25, 0.5)))
    assert abs(recess.poly.area - (96 - 3)) < 0.1  # 3 m of wall set back 1 m
    bay = apply(REAL, Edit("notch", south, 1.0, (0.25, 0.5)))
    assert abs(bay.poly.area - (96 + 3)) < 0.1
    assert apply(REAL, Edit("move_wall", 99, 1.0)) is None
    assert apply(REAL, Edit("eaves", None, 0.5)).height == 6.5
    assert apply(REAL, Edit("pitch", 0, 40.0), 0.5).pitches == {0: 35.0}


def test_parse_critique_keeps_only_the_menu():
    text = ('```json\n{"fit": "fair", "edits": [{"edit": "eaves", "metres": 1.2}, {"edit": "notch", "wall": 2, '
            '"from": 0.2, "to": 0.4, "metres": -1}, {"edit": "paint", "wall": 1}, {"edit": "move_wall", "wall": "x"}]}'
            '\n```')
    edits = parse_critique(text)
    assert [(e.kind, e.target, e.amount) for e in edits] == [("eaves", None, 1.2), ("notch", 2, -1.0)]
    assert edits[1].span == (0.2, 0.4)
    assert parse_critique("no json") == []


def test_proposals_need_agreement():
    answers = [[Edit("eaves", None, 1.0)], [Edit("eaves", None, 2.0), Edit("move_wall", 1, 1.0)],
               [Edit("eaves", None, 1.5)], [Edit("move_wall", 1, -1.0)]]
    props = proposals(answers)
    assert [(p.kind, p.amount, p.support) for p in props] == [("eaves", 1.5, 3)]


def test_refine_keeps_what_helps_and_drops_what_doesnt(tmp_path):
    from PIL import Image

    photo = tmp_path / "p.jpg"
    Image.new("RGB", (800, 600), "grey").save(photo)
    views = [View("p", photo, CAM, silhouette(REAL, REAL.roof(), CAM, (0, 0)), (0, 0))]
    start = Shape(REAL.poly, 4.5, "gable", 30.0)
    south = _south(REAL)

    def critique(view, overlay):  # three answers: the eaves are low; one also wants the front wall moved
        assert overlay.exists()
        return [[Edit("eaves", None, 1.5)], [Edit("eaves", None, 1.5)],
                [Edit("eaves", None, 1.0), Edit("move_wall", south, 2.0)]]

    shape, history = refine(start, views, critique, tmp_path / "critique", log=lambda *_: None)
    assert abs(shape.height - 6.0) < 0.01
    assert shape.poly.equals(REAL.poly)  # one vote for moving the wall: not enough to try
    assert history[1]["kept"][0]["kind"] == "eaves"
    assert history[-1]["error"] < history[0]["error"]
    assert (tmp_path / "critique" / "round_1" / "p.jpg").exists()


def test_overlay_draws(tmp_path):
    from PIL import Image

    photo = tmp_path / "p.jpg"
    Image.new("RGB", (800, 600), "grey").save(photo)
    out = draw_overlay(photo, REAL, REAL.roof(), CAM, (0, 0), tmp_path / "o.jpg")
    img = Image.open(out)
    assert img.size == (800, 600)
    import numpy as np

    a = np.asarray(img).astype(int)
    cyan = (a[..., 2] > 200) & (a[..., 1] > 180) & (a[..., 0] < 80)
    assert cyan.sum() > 2000  # the model's outline and edges


def test_only_faces_toward_the_camera_are_outlined():
    from placelens.critique import _facing, faces

    walls = faces(REAL, REAL.roof())[:4]
    facing = [_facing(w, CAM) for w in walls]
    assert facing.count(True) == 1 and facing[_south(REAL)]


def test_a_change_most_photos_dislike_is_refused():
    from placelens.critique import better

    assert better([0.05, 0.05, 0.05], [0.06, 0.06, 0.06], 0.02)
    assert not better([0.01, 0.07, 0.07], [0.06, 0.06, 0.06], 0.02)  # one photo much better, two worse
    assert not better([0.059, 0.059], [0.06, 0.06], 0.05)  # better, but not by enough for the footprint


def test_a_photo_that_cant_line_up_is_left_out():
    from placelens.critique import steady_views

    good = _views(REAL)[0]
    off = View("off", None, Camera(4, -20, 8, 60, 800, 600), good.real, (0, 0))  # its camera 4 m and 8° out
    kept = steady_views(REAL, [good, good, off], log=lambda *_: None)
    assert off not in kept and len(kept) == 2


def test_cameras_follow_the_shape():
    calls = []

    def relock(shape):
        calls.append(shape.height)
        return CAM

    view = View("v", None, Camera(0, -30, 0, 60, 800, 600), silhouette(REAL, REAL.roof(), CAM, (0, 0)), (0, 0), relock)
    assert error(REAL, [view]) == 0  # the relocked camera, not the stale one, is used
    assert calls == [6.0]


def _photo_with_corners(tmp_path, shape: Shape):
    """A grey photo of `shape` (lighter than the sky) with a dark full-height line at each of its seen corners."""
    import numpy as np
    from PIL import Image

    from placelens.critique import corner_columns

    mask = silhouette(shape, shape.roof(), CAM, (0, 0))
    img = np.where(mask, 170, 230).astype(np.uint8)
    for x in corner_columns(shape, CAM, (0, 0)):
        for dx in (-1, 0, 1):  # a few pixels wide, as a real corner is
            img[mask[:, int(x) + dx], int(x) + dx] = 40
    path = tmp_path / "corners.png"
    Image.fromarray(img).save(path)
    return path, mask


def test_vertical_lines_find_the_corners(tmp_path):
    from placelens.critique import corner_columns, vertical_lines

    path, mask = _photo_with_corners(tmp_path, REAL)
    lines = vertical_lines(path, mask)
    for x in corner_columns(REAL, CAM, (0, 0)):
        assert lines[max(0, int(x) - 6): int(x) + 7].any()  # within a few pixels (lines are pooled over 5)


def test_a_recess_is_seen_by_its_corners(tmp_path):
    from placelens.critique import corner_error, vertical_lines

    recessed = apply(REAL, Edit("notch", _south(REAL), -1.0, (0.3, 0.6)))
    path, mask = _photo_with_corners(tmp_path, recessed)
    lines = vertical_lines(path, mask)
    # The roofline can't tell them apart; the recess's corner lines can.
    assert corner_error(REAL, CAM, (0, 0), lines) > 2 * corner_error(recessed, CAM, (0, 0), lines)
