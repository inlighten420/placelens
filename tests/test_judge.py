import numpy as np

from placelens.carve import Shot
from placelens.judge import agreed, parse_verdict, trim


def test_a_verdict_is_read_and_bad_boxes_dropped():
    v = parse_verdict('ok {"score": 4, "problems": [{"kind": "extra", "what": "flap", "box": [0.4, 0.6, 0.7, 0.9]},'
                      '{"kind": "extra", "what": "upside down", "box": [0.7, 0.6, 0.4, 0.9]}]}')
    assert v.score == 4 and [p["what"] for p in v.problems] == ["flap"]
    assert parse_verdict("no json here").score is None


def test_only_boxes_two_answers_agree_on_count():
    flap = '{"score": 3, "problems": [{"kind": "extra", "what": "flap", "box": [0.40, 0.60, 0.70, 0.90]}]}'
    near = '{"score": 4, "problems": [{"kind": "extra", "what": "dangling bit", "box": [0.45, 0.62, 0.72, 0.88]}]}'
    lone = '{"score": 4, "problems": [{"kind": "extra", "what": "horn", "box": [0.0, 0.0, 0.1, 0.1]}]}'
    found = agreed([parse_verdict(t) for t in (flap, near, lone)])
    assert len(found) == 1 and found[0]["votes"] == 2 and found[0]["what"] == ["flap", "dangling bit"]


def test_trim_cuts_only_what_the_photo_shows_as_background():
    mask = np.zeros((100, 100), bool)
    mask[:, :50] = True  # the object fills the left half of the photo
    shot = Shot(mask, np.zeros((100, 100, 3), np.uint8), focal=100.0, angle=0.0, distance=5.0, centre=(50.0, 50.0))
    kept = np.ones((10, 10, 10), bool)
    corner, voxel = np.array([-0.5, 0.0, -0.5]), 0.1
    out = trim(kept, corner, voxel, shot, 1.0, [0.0, 0.0, 1.0, 1.0])
    xs = corner[0] + (np.argwhere(out)[:, 0] + 0.5) * voxel
    assert out.sum() < kept.sum() and (xs < 0.05).all()  # only the left (in the cut-out) is left
