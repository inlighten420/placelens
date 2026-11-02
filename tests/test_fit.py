import numpy as np

from placelens.fit import _sets, colour_from, fit


def _block(lo, hi, voxel=0.02, span=(1.0, 1.0, 1.0)):
    """A carving: a solid box from lo to hi (metres) in a grid starting at (-0.5, 0, -0.5)."""
    corner = np.array([-0.5, 0.0, -0.5])
    shape = tuple(int(s / voxel) for s in span)
    centres = [corner[i] + (np.arange(shape[i]) + 0.5) * voxel for i in range(3)]
    gx, gy, gz = np.meshgrid(*centres, indexing="ij")
    kept = (gx >= lo[0]) & (gx <= hi[0]) & (gy >= lo[1]) & (gy <= hi[1]) & (gz >= lo[2]) & (gz <= hi[2])
    return kept, corner, voxel


def _box(at, size, colour="#808080", what=""):
    return {"shape": "box", "at": list(at), "size": list(size), "turn": [0, 0, 0], "colour": colour,
            "finish": "matte", "what": what}


def test_a_part_slightly_off_is_pulled_onto_the_carving_and_a_stray_one_is_dropped():
    kept, corner, voxel = _block((-0.2, 0.0, -0.2), (0.2, 0.4, 0.2))  # a 40 cm cube on the floor
    seat = _box((0.06, 0.24, 0.0), (0.4, 0.4, 0.4), what="seat")  # 6 cm right, 4 cm high
    stray = _box((0.4, 0.2, 0.4), (0.03, 0.3, 0.03), what="post in the air")
    parts, inside, fill = fit([seat, stray], kept, corner, voxel)
    assert [p["what"] for p in parts] == ["seat"]
    assert abs(parts[0]["at"][0]) < 0.03 and abs(parts[0]["at"][1] - 0.2) < 0.03
    assert inside > 0.9 and fill > 0.8


def test_copies_of_a_part_are_one_set():
    legs = [_box((0.1 * i, 0, 0), (0.28, 0.04, 0.04)) for i in range(5)]
    seat = _box((0, 0.4, 0), (0.5, 0.1, 0.5))
    sets = _sets(legs + [seat])
    assert sets[0] == [0, 1, 2, 3, 4] and sets[5] == [5]


def test_each_part_takes_the_colour_where_it_sits():
    seat, base = _box((0, 0.4, 0), (0.5, 0.1, 0.5)), _box((0, 0.05, 0), (0.1, 0.1, 0.1))
    verts = np.array([[x, 0.45, z] for x in (-0.2, 0, 0.2) for z in (-0.2, 0, 0.2)] +
                     [[x, 0.05, z] for x in (-0.04, 0.04) for z in (-0.04, 0, 0.04)])
    colours = np.array([[0.5, 0.5, 0.52]] * 9 + [[0.05, 0.05, 0.05]] * 6)
    colour_from([seat, base], verts, colours)
    assert seat["colour"] == "#808085" and base["colour"] == "#0d0d0d"


def test_a_part_lying_apart_from_the_object_is_dropped():
    from placelens.fit import connected

    seat, post, leg = _box((0, 0.45, 0), (0.5, 0.1, 0.5)), _box((0, 0.2, 0), (0.06, 0.4, 0.06)), \
        _box((0.15, 0.2, 0), (0.3, 0.04, 0.04))
    apart = _box((0.9, 0.02, 0.9), (0.3, 0.04, 0.04))
    assert connected([seat, post, leg, apart], gap=0.02) == [seat, post, leg]


def test_mirrored_parts_are_twins_and_a_pair_survives_if_one_side_fits():
    from placelens.fit import _twins

    seat = _box((0, 0.45, 0), (0.5, 0.1, 0.5), what="seat")
    left, right = _box((-0.3, 0.62, 0), (0.07, 0.04, 0.33), what="left arm"), \
        _box((0.3, 0.62, 0), (0.07, 0.04, 0.33), what="right arm")
    assert _twins([seat, left, right]) == [None, 2, 1]
    # The carving: the seat, and only the left armrest (the right one too thin to show in the photos).
    posts = [_box((-0.3, 0.55, 0), (0.04, 0.12, 0.04), what="left post"),
             _box((0.3, 0.55, 0), (0.04, 0.12, 0.04), what="right post")]  # (holding the armrests to the seat)
    kept, corner, voxel = _block((-0.34, 0.0, -0.25), (0.25, 0.65, 0.25))
    parts, _, _ = fit([seat, left, right] + posts, kept, corner, voxel)
    names = [p["what"] for p in parts]
    assert "left arm" in names and "right arm" in names
    arm = next(p for p in parts if p["what"] == "left arm")
    assert arm["at"][0] < -0.2  # not dragged into the seat
