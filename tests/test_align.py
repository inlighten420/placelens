import numpy as np

from placelens import align, sight
from placelens.carve import Shot

HEIGHT = 1.0
MID = HEIGHT / 2


def _chair():
    """A simple chair: a seat, a backrest standing up behind it, four legs on the floor."""
    parts = [{"shape": "box", "at": [0, 0.45, 0], "size": [0.5, 0.08, 0.5], "turn": [0, 0, 0], "what": "seat"},
             {"shape": "box", "at": [0, 0.75, -0.22], "size": [0.5, 0.55, 0.06], "turn": [0, 0, 0],
              "what": "backrest"}]
    for x in (-0.2, 0.2):
        for z in (-0.2, 0.2):
            parts.append({"shape": "box", "at": [x, 0.2, z], "size": [0.05, 0.4, 0.05], "turn": [0, 0, 0],
                          "what": f"{'front' if z > 0 else 'back'} {'left' if x < 0 else 'right'} leg"})
    return parts


def _photos(parts, angles=(0, 60, 120, 180, 240, 300)):
    """Photos of the chair from round it: its outline, and each kind's outline (what SAM would find)."""
    shots, kinds = [], {"seat": [], "backrest": []}
    for a in angles:
        s = Shot(np.zeros((600, 800), bool), np.full((600, 800, 3), 128, np.uint8), 700.0, float(a), 2.6,
                 (400.0, 300.0), 20.0)
        s.mask = sight.silhouette(parts, s, MID, 1)[:600, :800]
        for k in kinds:
            kinds[k].append(sight.silhouette([p for p in parts if align.kind(p["what"]) == k], s, MID, 1)[:600, :800])
        shots.append(s)
    return shots, kinds


def test_kinds_drop_which_one_it_is():
    assert align.kind("front left leg") == "leg"
    assert align.kind("Right armrest pad 2") == "armrest pad"
    assert align.kind(None) == "part"


def test_a_design_drifted_off_is_pulled_back_onto_the_photos():
    truth = _chair()
    shots, kinds = _photos(truth)
    drifted = align.moved(truth, list(range(len(truth))), np.array([0.08, 0.0, -0.06, 0.12, 0, 0, 0, 0, 0]),
                          np.zeros(3))
    aligner = align.Aligner(drifted, shots, MID, kinds)
    assert set(aligner.found) == {"seat", "backrest"}
    groups = aligner.assemblies()
    assert align.FLOOR in groups  # the legs stand on the floor: one group, moved together
    assert sorted(aligner.kinds[i] for i in groups[align.FLOOR]) == ["leg"] * 4
    before = sight.score(drifted, shots, MID)
    aligner.fit()
    after = sight.score(aligner.parts, shots, MID)
    assert after > before + 0.1 and after > 0.85
    seat = next(p for p in aligner.parts if p["what"] == "seat")
    assert abs(seat["at"][0]) < 0.03 and abs(seat["at"][2]) < 0.03  # back where it is
    legs = [p for p in aligner.parts if align.kind(p["what"]) == "leg"]
    assert all(abs(p["size"][1] - legs[0]["size"][1]) < 1e-9 for p in legs)  # still a set


def test_a_camera_turned_wrong_is_turned_back_by_the_model():
    truth = _chair()
    shots, kinds = _photos(truth)
    shots[2].angle += 14  # its compass was off
    aligner = align.Aligner([dict(p) for p in truth], shots, MID, kinds)
    aligner.refit_camera(2)
    assert abs(shots[2].angle - 120) <= 4  # from 14 degrees out (a symmetric chair barely changes over a few)


def test_nothing_found_means_the_old_fit(monkeypatch, tmp_path):
    from placelens import soft

    truth = _chair()
    shots, _ = _photos(truth)

    class Photo:
        def __init__(self, i):
            self.small = tmp_path / f"p{i}.jpg"
            self.path = tmp_path / f"p{i}.HEIC"

    monkeypatch.setattr(soft, "_sam_masks", lambda meta, small, noun, out: [])
    parts, lines = align.fit_design(None, [Photo(i) for i in range(len(shots))], shots, truth, HEIGHT, tmp_path)
    assert parts is None and "old way" in lines[0]
