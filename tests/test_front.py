import math

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from placelens.build import WallPhoto, extrude, wall_edges
from placelens.front import Camera, homography, lock_camera, straighten, visible_walls, wall_span
from placelens.pick import pixel_to_offset

BOX = Polygon([(-5, 20), (5, 20), (5, 30), (-5, 30)])  # 10 x 10 m, its south wall 20 m north of the camera


def _cam(**kw):
    return Camera(**{"x": 0, "y": 0, "heading": 0, "fov": 60, "width": 1600, "height": 1200, **kw})


def test_projection_agrees_with_pick_and_puts_the_horizon_at_eye_height():
    cam = _cam()
    col, row = cam.project(5, 20, cam.eye)
    assert abs(pixel_to_offset(col, 1600, 60) - math.degrees(math.atan2(5, 20))) < 1e-9
    assert row == 600
    assert cam.project(0, 20, 0)[1] > 600 > cam.project(0, 20, 6)[1]  # ground below, roof above


def test_lock_camera_finds_where_the_photo_was_taken():
    true = _cam(x=2, y=-3, heading=4)
    cols = [true.project(x, y, 0)[0] for x, y in BOX.exterior.coords[:-1]]
    box = (round(min(cols)), 300, round(max(cols)), 900)
    # The phone said (0, 0) facing 0. Two edges can't tell a move from a turn, so it needn't find the exact spot,
    # but it must put the footprint's edges on the photo's, nearby.
    cam = lock_camera(BOX, box, (1600, 1200), heading=0, fov=60, distance=23)
    got = [cam.project(x, y, 0)[0] for x, y in BOX.exterior.coords[:-1]]
    assert abs(min(got) - box[0]) < 15 and abs(max(got) - box[2]) < 15
    assert math.dist((cam.x, cam.y), (true.x, true.y)) <= 2 and abs(cam.heading - true.heading) < 4


def test_depth_scale_follows_muse_distance():
    cam = lock_camera(BOX, (0, 0, 1600, 1200), (1600, 1200), heading=0, fov=60, distance=40)  # edges cut off
    nearest = BOX.exterior.distance(Point(cam.x, cam.y))
    assert abs(cam.depth_scale - 40 / nearest) < 1e-9 and cam.depth_scale > 1  # looks further than the map says
    far = lock_camera(BOX, (0, 0, 1600, 1200), (1600, 1200), heading=0, fov=60, distance=200)
    assert far.depth_scale == 2.0  # capped


def test_only_walls_facing_the_camera_and_the_part_in_view():
    [view] = visible_walls(BOX, _cam(), height=6)
    (x0, y0), (x1, y1) = wall_edges(BOX)[view.index]
    assert y0 == y1 == 20 and x0 < x1  # the south wall, left end first as seen from outside
    assert view.start == 0 and view.end == 1 and view.seen == 1
    bl, br, tr, tl = view.quad
    assert bl[0] < br[0] and tl[1] < bl[1]
    # Close up, only the middle of the wall is in the photo.
    [near] = visible_walls(BOX, _cam(y=17), height=6)
    assert 0.1 < near.start < 0.4 and 0.6 < near.end < 0.9


def test_homography_and_straighten():
    quad = [(100, 900), (700, 800), (700, 200), (100, 100)]
    m = homography([(0, 1), (1, 1), (1, 0), (0, 0)], quad)
    for (u, v), (x, y) in zip([(0, 1), (1, 1), (1, 0), (0, 0)], quad, strict=True):
        p = m @ [u, v, 1]
        assert np.allclose(p[:2] / p[2], (x, y))
    # A red quad in a white photo comes out as an all-red texture.
    photo = Image.new("RGB", (800, 1000), "white")
    from PIL import ImageDraw

    ImageDraw.Draw(photo).polygon(quad, fill="red")
    flat = np.asarray(straighten(photo, quad, (60, 80)))[2:-2, 2:-2]
    assert (flat[..., 0] > 200).all() and (flat[..., 1] < 60).all()


def test_a_wall_photo_covers_only_its_stretch():
    parts = extrude(BOX, 6.0, {0: WallPhoto(b"jpeg", 0.25, 0.75)})
    assert [p.name for p in parts] == ["walls", "wall 0", "roof"]
    wall = parts[1]
    (x0, _), (x1, _) = wall_edges(BOX)[0]
    xs = sorted({float(v) for v in wall.positions[:, 0]})
    assert np.allclose(xs, sorted([x0 + 0.25 * (x1 - x0), x0 + 0.75 * (x1 - x0)]))
    assert len(parts[0].positions) == 4 * 5  # three plain walls and the two leftover ends of wall 0


def test_lock_wont_swing_the_compass_to_fit_one_edge():
    true = _cam(heading=25)  # the compass said 0, but the photo was taken facing 25 degrees
    right = max(true.project(x, y, 0)[0] for x, y in BOX.exterior.coords[:-1])
    box = (0, 300, round(right), 900)  # the building runs off the left of the photo: only its right edge shows
    cam = lock_camera(BOX, box, (1600, 1200), heading=0, fov=60, distance=20)
    assert abs(cam.heading) <= 15 + 1e-9 and cam.edges == 1
    assert cam.misfit > 3 or math.dist((cam.x, cam.y), (0, 0)) > 5  # it either can't line up, or had to move far
    # Allowed to turn that far, the same photo lines up cleanly.
    assert lock_camera(BOX, box, (1600, 1200), heading=0, fov=60, distance=20, max_turn=45).misfit < 0.5


def test_wall_span_ignores_the_roof_overhang():
    mask = np.zeros((1200, 1600), bool)
    mask[300:500, 100:1500] = True  # roof, overhanging both sides
    mask[500:1100, 250:1350] = True  # the walls under it
    assert wall_span(mask, (100, 300, 1500, 1100)) == (250, 300, 1350, 1100)


def test_fit_rows_takes_ground_and_eaves_from_the_outline():
    from placelens.front import fit_rows

    quad = [(400, 700), (1200, 700), (1200, 500), (400, 500)]  # level-phone guess: 3 m wall, 200 px tall
    mask = np.zeros((1200, 1600), bool)
    mask[420:1000, 380:1220] = True  # the photo shows it taller (tilted phone, close up)
    cam = _cam()
    fitted = fit_rows(quad, cam, mask, 0.0, 3.0, 3.0)
    assert fitted[0] == (400, 999) and fitted[3] == (400, 420)
    # Hidden behind something: the outline stops short of the ground, so the guess stands for the bottom.
    mask[700:] = False
    assert fit_rows(quad, cam, mask, 0.0, 3.0, 3.0)[0] == (400, 700)
