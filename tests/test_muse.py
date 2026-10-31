from pathlib import Path

from placelens.look import Look, parse_look
from placelens.meta import OPEN, SPARK
from placelens.muse import (
    ask,
    default_votes,
    majority,
    merge_looks,
    merge_props,
    merge_sides,
    merge_verdicts,
    middle,
    noted,
)
from placelens.pick import Verdict
from placelens.prop import PropView
from placelens.sides import SideView


class FakeMeta:
    def __init__(self):
        self.calls = 0

    def ask(self, prompt, photos, model):
        self.calls += 1
        return f'{{"n": {self.calls}}}'


def test_votes_are_saved_and_the_old_reply_is_vote_zero(tmp_path):
    reply = tmp_path / "IMG_1_open_abc.txt"
    reply.write_text("saved before voting", encoding="utf-8")
    meta = FakeMeta()
    answers = ask(meta, "q", None, reply, OPEN, votes=3)
    assert answers[0] == "saved before voting"
    assert meta.calls == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["IMG_1_open_abc.txt", "IMG_1_open_abc_v1.txt",
                                                          "IMG_1_open_abc_v2.txt"]
    assert ask(meta, "q", None, reply, OPEN, votes=3) == answers  # all saved: no new calls
    assert meta.calls == 2
    ask(meta, "q", None, reply, OPEN, votes=3, fresh=True)
    assert meta.calls == 5


def test_default_votes():
    assert default_votes(OPEN) == 5
    assert default_votes(SPARK) == 1


def test_majority_and_middle():
    assert majority([1, 2, 1, None, 1]) == (1, 0.6)
    assert majority([]) == (None, 0.0)
    assert middle([4, "x", 11, 12, None, True]) == 11
    assert middle([None]) is None


def test_merge_verdicts_steadies_distance_and_floors():
    # Spark Open on IMG_0167: one answer said a bungalow 5 m away, the rest about 11 m and 2 floors.
    votes = [Verdict(1, 0.9, ["near"], 1, 5), Verdict(1, 0.8, ["mid"], 2, 11), Verdict(1, 0.9, ["far"], 2, 12),
             Verdict(1, 0.7, ["x"], 2, 10), Verdict(None, 0.6, ["none fit"], None, None)]
    v = merge_verdicts(votes)
    assert v.choice == 1
    assert v.distance_m == 10.5 and v.floors == 2
    assert v.confidence == round(0.8 * 0.825, 3)
    assert v.reasons[0] in ("mid", "x") and v.reasons[-1] == "4 of 5 answers agreed"


def test_one_vote_is_left_alone():
    v = Verdict(3, 0.5, ["only"], "two", 7)
    assert merge_verdicts([v]) is v


def test_merge_sides():
    v = merge_sides([SideView("back", 0.9), SideView("back", 0.7), SideView("left", 0.9)])
    assert v.side == "back" and v.confidence == round(2 / 3 * 0.8, 3)


def test_merge_props():
    views = [PropView("Rain tank", "tank", 1.8, 1.8, 1.3, 6, 0.9, True, round=True),
             PropView("rain tank", "tank", 2.0, 2.0, 1.4, 7, 0.8, False, round=True),
             PropView("bin", "bin", 0.6, 0.6, 1.0, 5, 0.5, True, round=False)]
    v = merge_props(views)
    assert v.name == "Rain tank" and v.width == 1.9 and v.height == 1.35 and v.round is True
    assert v.confidence == round(2 / 3 * 0.85, 3)


def test_merge_looks_takes_the_middle_colour():
    looks = [parse_look('{"walls": "#ffffff"}'), parse_look('{"walls": "#000000"}'), parse_look('{"walls": "#ffffff"}')]
    assert merge_looks(looks).walls == parse_look('{"walls": "#ffffff"}').walls
    assert merge_looks(looks).roof == Look().roof


def test_vote_files_sit_next_to_the_reply():
    from placelens.muse import vote_path

    assert vote_path(Path("a/b_roof_1.txt"), 0) == Path("a/b_roof_1.txt")
    assert vote_path(Path("a/b_roof_1.txt"), 4) == Path("a/b_roof_1_v4.txt")


def test_notes_follow_the_prompt_and_no_notes_leave_it_alone():
    assert noted("Q?", None) == "Q?" and noted("Q?", "  ") == "Q?"
    asked = noted("Q?", " the back matches the front ")
    assert asked.startswith("Q?") and '"the back matches the front"' in asked
