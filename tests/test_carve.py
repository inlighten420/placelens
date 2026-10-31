import math

import numpy as np
from scipy import ndimage

from placelens.carve import Shot, carve, paint, place_shots, surface, vertex_normals

BALLS = [((0.0, 0.25, 0.0), 0.25), ((0.0, 0.62, 0.0), 0.15)]  # a snowman, 0.77 m tall
HEIGHT = 0.77


def _snowman_points(n=40000):
    """Points on the snowman's skin, and which way the skin faces at each."""
    rng = np.random.default_rng(0)
    pts, normals = [], []
    for (cx, cy, cz), r in BALLS:
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pts.append(v * r + [cx, cy, cz])
        normals.append(v)
    return np.concatenate(pts), np.concatenate(normals)


def _photos(angles, focal=700.0, distance=1.6, size=(600, 800), elevation=0.0):
    """Photos of the snowman from round it: red front (+Z), blue back."""
    points, normals = _snowman_points()
    colours = np.where(points[:, 2:3] > 0, [[220, 30, 30]], [[30, 30, 220]]).astype(np.uint8)
    shots = []
    for a in angles:
        s = Shot(np.zeros(size, bool), np.zeros((*size, 3), np.uint8), focal, a, distance, (400.0, 300.0),
                 elevation)
        cols, rows, depth = s.project(points, HEIGHT / 2)
        c, r = np.round(cols).astype(int), np.round(rows).astype(int)
        s.mask[r, c] = True
        position = s.camera(HEIGHT / 2)[0]
        seen = np.einsum("ij,ij->i", normals, position - points) > 0  # only skin facing the camera shows
        s.image[r[seen], c[seen]] = colours[seen]
        painted = s.image.any(axis=2)
        s.mask[:] = ndimage.binary_closing(s.mask, iterations=3)
        _, (ir, ic) = ndimage.distance_transform_edt(~painted, return_indices=True)
        s.image[:] = s.image[ir, ic]  # fill between the dots from the nearest one
        s.distance, s.centre, s.elevation = 0.0, (0.0, 0.0), 0.0  # forget them: they must be found again
        shots.append(s)
    return shots


def test_walk_around_carves_the_shape_back():
    shots = _photos(range(0, 360, 45))
    place_shots(shots, HEIGHT)
    assert all(abs(s.distance - 1.6) < 0.25 for s in shots)
    kept, corner, voxel = carve(shots, (0.5, 0.5, HEIGHT))
    volume = kept.sum() * voxel**3
    true = sum(4 / 3 * math.pi * r**3 for _, r in BALLS)
    assert true * 0.9 < volume < true * 1.5  # a visual hull is a little bigger than the thing, never smaller
    verts, faces = surface(kept, corner, voxel)
    assert abs(verts[:, 1].max() - HEIGHT) < 0.06 and verts[:, 1].min() < 0.05
    colours = paint(verts, vertex_normals(verts, faces), shots, HEIGHT)
    facing = np.abs(verts[:, 0]) < 0.1  # straight in front and behind, away from where red meets blue
    front, back = facing & (verts[:, 2] > 0.1), facing & (verts[:, 2] < -0.1)
    assert (colours[front, 0] > colours[front, 2]).mean() > 0.9  # the front came out red
    assert (colours[back, 2] > colours[back, 0]).mean() > 0.9  # and the back blue


def test_one_wrong_photo_doesnt_cut_the_object_away():
    shots = _photos(range(0, 360, 45))
    shots[3].angle += 25  # its compass was off
    place_shots(shots, HEIGHT)
    kept, _, voxel = carve(shots, (0.5, 0.5, HEIGHT), misses=1)
    true = sum(4 / 3 * math.pi * r**3 for _, r in BALLS)
    assert kept.sum() * voxel**3 > true * 0.9


def test_finds_how_far_above_the_photos_were_taken():
    from placelens.carve import find_elevation

    shots = _photos(range(0, 360, 45), elevation=30)
    place_shots(shots, HEIGHT)
    assert abs(find_elevation(shots, (0.5, 0.5, HEIGHT)) - 30) <= 10


def test_fitting_the_cameras_lines_up_a_compass_that_was_off(monkeypatch):
    from placelens.carve import agreement, fit_cameras

    lopsided = [((-0.18, 0.2, 0.0), 0.2), ((0.18, 0.2, 0.0), 0.2), ((0.0, 0.5, 0.0), 0.16), ((0.0, 0.5, 0.16), 0.06)]
    monkeypatch.setitem(globals(), "BALLS", lopsided)  # wide, with a nose: its outline changes as you walk round
    size = (0.8, 0.45, HEIGHT)
    shots = _photos(range(0, 360, 60), elevation=20)
    place_shots(shots, HEIGHT)
    for s in shots:
        s.elevation = 20.0
    shots[2].angle += 12  # its compass was off

    def strict() -> float:
        kept, corner, voxel = carve(shots, size, misses=0)
        return agreement(shots, kept, corner, voxel, HEIGHT)

    before = strict()
    fit_cameras(shots, size, rounds=2)
    assert shots[0].angle == 0  # the front stays the front
    assert abs(shots[2].angle - 120) <= 6  # turned back to about where it stood
    assert strict() > before
