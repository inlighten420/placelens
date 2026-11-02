import numpy as np

from placelens import muse, sight
from placelens.carve import Shot


def _box(at, size, what=""):
    return {"shape": "box", "at": list(at), "size": list(size), "turn": [0, 0, 0], "colour": "#808080",
            "finish": "matte", "what": what}


TRUE = [_box((0, 0.45, 0), (0.5, 0.1, 0.5), "seat"), _box((0, 0.2, 0), (0.08, 0.4, 0.08), "post")]


def _shots(angles=(0, 90)):
    """Photos of the true model from these angles: its own outline is the cut-out."""
    shots = []
    for a in angles:
        s = Shot(mask=np.zeros((400, 400), bool), image=np.full((400, 400, 3), 128, np.uint8), focal=500.0,
                 angle=a, distance=3.0, centre=(200.0, 200.0))
        s.mask = sight.silhouette(TRUE, s, 0.25, 1)[:400, :400]
        shots.append(s)
    return shots


def test_a_better_model_scores_higher():
    shots = _shots()
    short = [TRUE[0], _box((0, 0.1, 0), (0.08, 0.2, 0.08), "post")]
    assert sight.score(TRUE, shots, 0.25) > 0.95
    assert sight.score(short, shots, 0.25) < sight.score(TRUE, shots, 0.25)


def test_only_edits_that_help_every_view_are_kept(tmp_path, monkeypatch):
    shots = _shots()
    wrong = [_box((0, 0.45, 0), (0.5, 0.1, 0.5), "seat"), _box((0, 0.1, 0), (0.08, 0.2, 0.08), "post")]
    said = ['{"edits": [{"part": 0, "move": [0.2, 0, 0]}, {"part": 1, "size": [1, 2, 1], "move": [0, 0.1, 0]}]}']
    monkeypatch.setattr(muse, "ask", lambda *a, **kw: said)
    before = sight.score(wrong, shots, 0.25)
    fixed, notes = sight.refine(None, wrong, shots, 0.25, "stool", tmp_path, moves=1)
    assert fixed[0]["at"] == [0, 0.45, 0]  # moving the seat sideways made it worse: not kept
    assert abs(fixed[1]["size"][1] - 0.4) < 1e-6 and abs(fixed[1]["at"][1] - 0.2) < 1e-6
    assert sight.score(fixed, shots, 0.25) > before
    assert "1 kept" in notes[-1] and (tmp_path / "sight_1.jpg").exists()


def test_removed_parts_keep_the_numbers_right(tmp_path, monkeypatch):
    shots = _shots()
    extra = [*TRUE, _box((0.6, 0.5, 0), (0.1, 0.1, 0.1), "stray")]
    answers = iter(['{"edits": [{"part": 2, "remove": true}]}', '{"edits": [{"part": 1, "move": [0, 0, 0]}]}'])
    monkeypatch.setattr(muse, "ask", lambda *a, **kw: [next(answers)])
    fixed, _ = sight.refine(None, extra, shots, 0.25, "stool", tmp_path, moves=2)
    assert [p["what"] for p in fixed] == ["seat", "post"]


def test_bad_answers_are_ignored():
    assert sight.parse_edits("no json here", 3) == []
    assert sight.parse_edits('{"edits": [{"part": 9, "move": [0, 0, 0]}, {"part": "x"}]}', 3) == []
    assert sight.parse_edits('{"edits": [{"part": 1, "size": [1, 0, 1]}]}', 3) == []


def test_seen_keeps_what_every_photo_shows():
    shots = _shots()
    assert sight.seen(TRUE[1], shots, 0.25) > 0.95  # the post: inside the outline from both sides
    beside = _box((0.6, 0.45, 0), (0.08, 0.3, 0.08), "post in the air beside it")
    assert sight.seen(beside, shots, 0.25) < 0.5  # the front photo shows it outside
    behind = _box((0, 0.45, -0.2), (0.06, 0.06, 0.06), "hidden behind the seat")
    assert sight.seen(behind, shots, 0.25) > 0.95  # hidden is fine: still inside the outline
