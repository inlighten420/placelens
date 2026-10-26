from placelens.osm import Building
from placelens.view import LocalFrame, find_candidates

ORIGIN = (-33.9255, 18.4239)


def square(frame, cx, cy, half):
    """A square building centred cx, cy metres east/north of the origin."""
    corners = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    ring = [frame.to_latlon(x, y) for x, y in corners]
    return ring + ring[:1]


def test_front_building_hides_the_one_behind_it():
    frame = LocalFrame(*ORIGIN)
    front = Building("way/1", square(frame, 0, 30, 10))
    behind = Building("way/2", square(frame, 0, 60, 8))  # same direction, fully hidden
    side = Building("way/3", square(frame, 40, 40, 8))  # 45 degrees right, outside a 60 degree lens
    ids = [c.building.osm_id for c in find_candidates([front, behind, side], frame, heading=0, fov=60, margin=0)]
    assert ids == ["way/1"]


def test_bigger_share_of_view_ranks_first():
    frame = LocalFrame(*ORIGIN)
    small = Building("way/1", square(frame, -20, 30, 4))
    big = Building("way/2", square(frame, 10, 40, 15))
    cands = find_candidates([small, big], frame, heading=0, fov=70, margin=0)
    assert [c.building.osm_id for c in cands] == ["way/2", "way/1"]
    assert cands[0].share > cands[1].share


def test_standing_inside_a_footprint_is_ignored():
    frame = LocalFrame(*ORIGIN)
    around_me = Building("way/1", square(frame, 0, 0, 5))
    ahead = Building("way/2", square(frame, 0, 30, 10))
    ids = [c.building.osm_id for c in find_candidates([around_me, ahead], frame, heading=0, fov=60, margin=0)]
    assert ids == ["way/2"]


def test_frame_round_trip():
    frame = LocalFrame(*ORIGIN)
    lat, lon = frame.to_latlon(123.0, -45.0)
    x, y = frame.to_xy(lat, lon)
    assert abs(x - 123.0) < 1e-6 and abs(y + 45.0) < 1e-6
