from pathlib import Path

from placelens.buildings import Sight, meet


def _sight(east, north, bearing, distance):
    return Sight(Path("p.jpg"), Path("p.jpg"), east, north, bearing, distance, None, 65.0, bearing)


def test_photos_lines_meet_at_the_building_and_a_bad_compass_is_left_out():
    # a building at (0, 20): three photos from the south, west and east point at it; one compass is 60 degrees off
    lines = [_sight(0, 0, 0, 20), _sight(-20, 20, 90, 20), _sight(20, 20, 270, 20), _sight(5, 0, 300, 20)]
    x, y = meet(lines)
    assert abs(x) < 2 and abs(y - 20) < 2
    assert [s.used for s in lines] == [True, True, True, False]
    assert "compass or GPS is off" in lines[3].why
