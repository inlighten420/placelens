import math

import numpy as np
from shapely.geometry import Polygon

from placelens.build import WallPhoto, extrude, wall_edges
from placelens.front import Camera, gable_pitches
from placelens.roof import RoofView, blocks, combine, parse_roof, plan_roof

L_SHAPE = Polygon([(0, 0), (5, 0), (5, 4), (12, 4), (12, 16), (0, 16)])  # 12 x 16 with a 7 x 4 notch out of a corner
BOX = Polygon([(-5, 20), (5, 20), (5, 26), (-5, 26)])  # 10 wide (east-west), 6 deep, 20 m north of the origin


def test_an_l_splits_into_two_long_blocks():
    parts = sorted(blocks(L_SHAPE), key=lambda p: p.area)
    assert len(parts) == 2 and abs(sum(p.area for p in parts) - L_SHAPE.area) < 1e-6
    # Side by side, both running north-south.
    assert {round(p.area) for p in parts} == {80, 84}  # 5 x 16 and 7 x 12, not 12 x 12 plus a 5 x 4 stub
    assert blocks(BOX) == [BOX]


def _faces_point_out(plan, centre):
    for face in plan.slopes + [g.points for g in plan.gables]:
        a, b, c = (np.array(p) for p in face[:3])
        n = np.cross(b - a, c - a)
        assert n @ (np.mean(face, axis=0) - centre) > 0


def test_gable_roof_on_a_box():
    plan = plan_roof(BOX, 6.0, "gable", 30)
    assert len(plan.slopes) == 2 and len(plan.gables) == 2 and not plan.flats
    peak = 6 + 3 * math.tan(math.radians(30))  # ridge along the 10 m length, rising over half the 6 m width
    assert all(abs(max(z for *_, z in g.points) - peak) < 1e-9 for g in plan.gables)
    walls = sorted(g.wall for g in plan.gables)
    edges = wall_edges(BOX)
    assert all(math.dist(*edges[w]) == 6 for w in walls)  # the gables stand on the short end walls
    assert plan.tops() == {w: peak for w in walls}
    _faces_point_out(plan, np.array([0, 23, 6.0]))


def test_ridge_direction_from_muse_wins_over_the_long_axis():
    plan = plan_roof(BOX, 6.0, "gable", 30, ridge=0)  # ridge running north-south, across the 6 m depth
    assert all(math.dist(*wall_edges(BOX)[g.wall]) == 10 for g in plan.gables)


def test_hip_roof_has_no_gables_and_flat_has_no_slopes():
    hip = plan_roof(BOX, 6.0, "hip", 30)
    assert not hip.gables and len(hip.slopes) == 4 and not hip.tops()
    _faces_point_out(hip, np.array([0, 23, 6.0]))
    flat = plan_roof(BOX, 6.0, "flat")
    assert not flat.slopes and flat.flats == [BOX]


def test_pitches_per_block():
    plan = plan_roof(L_SHAPE, 6.0, "gable", 20, pitches={0: 40})
    peaks = {g.block: max(z for *_, z in g.points) for g in plan.gables}
    halves = {n: _frame_half_width(b) for n, b in enumerate(plan.blocks)}
    assert abs(peaks[0] - 6 - halves[0] * math.tan(math.radians(40))) < 1e-6
    assert abs(peaks[1] - 6 - halves[1] * math.tan(math.radians(20))) < 1e-6


def _frame_half_width(p):
    xs, ys = p.exterior.xy
    return min(max(xs) - min(xs), max(ys) - min(ys)) / 2


def test_parse_and_combine_roof_views():
    v = parse_roof('```json\n{"roof": "Gable", "ridge": "toward", "pitch_deg": 35}\n```')
    assert v == RoofView("gable", "toward", 35.0)
    assert parse_roof("no idea") == RoofView(None, None, None)
    # Two photos facing north see the ridge end-on, one facing east sees it side-on: all say north-south.
    kind, pitch, ridge = combine([(v, 10), (RoofView("gable", "toward", 25), 190), (RoofView("hip", "across", 30), 80)])
    assert kind == "gable" and pitch == 30 and min(ridge, 180 - ridge) < 7


def test_painted_gable_takes_its_wall_photo():
    plan = plan_roof(BOX, 6.0, "gable", 30)
    wall = plan.gables[0].wall
    parts = extrude(BOX, 6.0, {wall: WallPhoto(b"jpeg", top=plan.tops()[wall])}, plan)
    painted = next(p for p in parts if p.name == f"wall {wall}")
    assert len(painted.positions) == 4 + 3  # the wall and its gable
    assert painted.uvs[:, 1].min() == 0 and painted.uvs[:, 1].max() == 1  # the picture runs up to the peak
    assert abs(painted.uvs[4:, 0].mean() - 0.5) < 1e-6  # the gable sits in the middle of it


def test_gable_pitch_measured_from_the_outline():
    cam = Camera(0, 0, 0, 60, 1600, 1200)
    true = plan_roof(BOX, 6.0, "gable", 35, ridge=0)  # gables face the camera, on the 10 m walls
    guess = plan_roof(BOX, 6.0, "gable", 20, ridge=0)
    near = next(g for g in true.gables if min(y for _, y, _ in g.points) < 21)
    peak = max(near.points, key=lambda p: p[2])
    col, row = cam.project(peak[0], peak[1], peak[2])
    mask = np.zeros((1200, 1600), bool)
    mask[int(round(row)):, int(col) - 40:int(col) + 40] = True  # the outline reaches up to the real peak
    found = gable_pitches(guess.gables, cam, mask, {near.wall})
    assert len(found) == 1 and abs(found[0][1] - 35) < 1


def test_no_pitch_from_an_outline_cut_off_by_the_top_of_the_photo():
    cam = Camera(0, 0, 0, 60, 1600, 1200)
    plan = plan_roof(BOX, 6.0, "gable", 20, ridge=0)
    near = next(g for g in plan.gables if min(y for _, y, _ in g.points) < 21)
    mask = np.zeros((1200, 1600), bool)
    mask[0:, 700:900] = True  # the building fills the photo up to its top edge
    assert gable_pitches(plan.gables, cam, mask, {near.wall}) == []
