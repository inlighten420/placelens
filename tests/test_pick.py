import math

from placelens.osm import Building
from placelens.pick import (
    LineUp,
    Verdict,
    decide,
    line_up,
    map_span,
    offset_to_pixel,
    overlap,
    parse_verdict,
    pixel_to_offset,
)
from placelens.view import LocalFrame, find_candidates

from .test_view import ORIGIN, square


def test_pixels_and_directions_round_trip():
    assert abs(pixel_to_offset(0, 1600, 60) + 30) < 1e-9
    assert pixel_to_offset(800, 1600, 60) == 0
    assert abs(offset_to_pixel(pixel_to_offset(1234, 1600, 67), 1600, 67) - 1234) < 1e-6


def test_overlap():
    assert overlap((0, 10), (0, 10)) == 1
    assert overlap((0, 10), (5, 15)) == 5 / 15
    assert overlap((0, 10), (20, 30)) == 0


def test_candidates_know_their_edges():
    frame = LocalFrame(*ORIGIN)
    ahead = Building("way/1", square(frame, 0, 30, 10))  # spans atan(10/20) = 26.6 degrees each side
    [c] = find_candidates([ahead], frame, heading=0, fov=80, margin=0)
    assert -27.5 < c.left < -25.5 and 25.5 < c.right < 27.5


def _two_houses():
    frame = LocalFrame(*ORIGIN)
    left = Building("way/1", square(frame, -12, 30, 6))
    right = Building("way/2", square(frame, 12, 30, 6))
    cands = find_candidates([left, right], frame, heading=0, fov=80, margin=0)
    return cands, next(c for c in cands if c.building.osm_id == "way/2")


def test_line_up_finds_the_compass_error():
    cands, right = _two_houses()
    seen = (right.left - 5, right.right - 5)  # the photo shows it 5 degrees further left than the map
    best = line_up(cands, seen, heading=0, fov=80, margin=10, move=0)[0]
    assert best.candidate is right and best.fit > 0.95 and best.shift == 5


def test_line_up_finds_the_gps_error():
    cands, right = _two_houses()
    true_spot = (4.0, 0.0)  # the phone said (0, 0); the photo was taken 4 m east
    seen = map_span(right.polygon, true_spot, heading=0)
    # With the compass trusted, the move is the only way to fit, and it lands near the real spot.
    best = line_up(cands, seen, heading=0, fov=80, margin=0, move=6)[0]
    assert best.candidate is right and best.fit > 0.9 and math.dist(best.move, true_spot) <= 1.5
    # Two edges can't tell a move from a move plus a turn; it settles on the smallest believable correction.
    best = line_up(cands, seen, heading=0, fov=80, margin=5, move=6)[0]
    assert best.candidate is right and best.fit > 0.9 and best.moved <= 4


def test_a_far_fetched_correction_scores_low():
    cands, right = _two_houses()
    left = next(c for c in cands if c is not right)
    seen = (right.left, right.right)
    [far] = [lu for lu in line_up(cands, seen, heading=0, fov=80, margin=15, move=10) if lu.candidate is left]
    [near] = [lu for lu in line_up(cands, seen, heading=0, fov=80, margin=15, move=10) if lu.candidate is right]
    assert near.score > 0.95 and far.score < 0.3


def test_parse_verdict_tolerates_fences():
    v = parse_verdict('Sure!\n```json\n{"choice": 2, "confidence": 0.8, "reasons": ["two floors"]}\n```')
    assert v.choice == 2 and v.confidence == 0.8 and v.reasons == ["two floors"]
    assert parse_verdict('{"choice": null, "confidence": 0.9}').choice is None
    assert parse_verdict("no idea").choice is None


def _lineup(rank, score):
    frame = LocalFrame(*ORIGIN)
    [c] = find_candidates([Building(f"way/{rank}", square(frame, 0, 30, 10))], frame, heading=0, fov=80, margin=0)
    c.rank = rank
    return LineUp(c, score, score, 0.0, (0.0, 0.0), (c.left, c.right))


def test_decide_needs_agreement_and_confidence():
    assert decide([_lineup(1, 0.8)], Verdict(1, 0.9)).status == "picked"
    assert decide([_lineup(1, 0.8)], Verdict(2, 0.9)).status == "ask"  # they disagree
    assert decide([_lineup(1, 0.3)], Verdict(1, 0.9)).status == "ask"  # outline barely fits the map
    assert decide([_lineup(1, 0.8)], Verdict(None, 0.9)).status == "ask"  # Muse says none fit
    assert decide([_lineup(1, 0.8)], Verdict(1, 0.5)).status == "ask"  # Muse unsure


def test_geometry_needs_a_clear_winner():
    assert decide([_lineup(1, 0.8), _lineup(2, 0.1)], Verdict(1, 0.9)).status == "picked"
    assert decide([_lineup(1, 0.8), _lineup(2, 0.5)], Verdict(1, 0.9)).status == "ask"  # two buildings fit
