from placelens.objects import spread


def test_spread_keeps_the_front_and_goes_evenly_round():
    headings = [0, 10, 20, 90, 100, 180, 190, 270, 280, 350]
    chosen = spread(headings, keep=4)
    assert chosen[0] == 0 and len(chosen) == 4
    picked = sorted(headings[i] for i in chosen)
    gaps = [(b - a) for a, b in zip(picked, picked[1:], strict=False)] + [360 - picked[-1] + picked[0]]
    assert max(gaps) <= 100  # four photos round the whole object, not bunched on one side


def test_spread_keeps_every_photo_when_there_are_few():
    assert spread([0, None, 50], keep=6) == [0, 1, 2]


def test_photos_without_a_compass_fill_any_room_left():
    assert spread([0, 180, None, None, 90], keep=4) == [0, 1, 2, 4]


def test_sam_is_tried_with_other_names_when_the_first_finds_nothing():
    from placelens.objects import _names

    answers = ['{"name": "racing steering wheel", "other_names": ["game controller", "Steering Wheel", "device"]}']
    assert _names("racing steering wheel", "racing steering wheel", answers) == [
        "racing steering wheel", "game controller", "steering wheel", "device", "wheel"]


def test_a_photo_looking_down_on_a_flat_thing_is_told_from_a_side_view():
    import numpy as np

    from placelens.objects import _looking_down

    wheel = np.ones((50, 100), bool)  # looks half as tall as wide, from above
    assert _looking_down(wheel, wide=0.31, tall=0.04)  # but Muse says it's 4 cm tall: it's lying flat
    bear = np.ones((110, 100), bool)
    assert not _looking_down(bear, wide=0.6, tall=0.66)  # seen from the side: they agree


def test_laid_flat_its_front_faces_up_and_it_stands_on_the_ground():
    import numpy as np

    from placelens.build import Part
    from placelens.objects import _lay_flat

    standing = Part("p", np.array([[0, 0, 0.02], [0.3, 0.15, 0.02], [0, 0.15, -0.02]], np.float32),
                    np.array([[0, 0, 1]] * 3, np.float32), np.zeros((3, 2), np.float32), np.arange(3, dtype=np.uint32),
                    (1, 1, 1, 1))
    (flat,) = _lay_flat([standing])
    assert np.allclose(flat.normals[0], [0, 1, 0])  # the photographed front now faces the sky
    assert flat.positions[:, 1].min() == 0 and np.ptp(flat.positions[:, 1]) <= 0.04 + 1e-6  # thickness = height
    assert np.isclose(np.ptp(flat.positions[:, 2]), 0.15)  # its length runs along the ground



def _side_photos(monkeypatch, tmp_path, said):
    import json

    import numpy as np

    from placelens import muse, objects

    mask = np.zeros((40, 60), bool)
    mask[10:30, 25:35] = True  # the object in the middle of each photo
    monkeypatch.setattr(muse, "ask", lambda meta, prompt, images, reply, model, votes, **kw:
                        [json.dumps(said[images[1].stem])])
    return lambda name, heading: objects.Photo(tmp_path / f"{name}.HEIC", tmp_path / f"{name}.jpg", heading, None,
                                               None, 60.0, mask=mask)


def test_an_object_turned_in_place_takes_its_angles_from_muse(monkeypatch, tmp_path):
    from placelens import objects

    # The office chair, 2 Nov: turned round in place, so the compass barely moved (179, 138, 135).
    photo = _side_photos(monkeypatch, tmp_path, {"side": {"side": "right", "confidence": 0.9},
                                                  "back": {"side": "back", "confidence": 0.8}})
    photos = [photo("front", 179.0), photo("side", 138.0), photo("back", 135.0)]
    objects._angles(None, photos, tmp_path)
    assert [p.angle for p in photos[1:]] == [90, 180]
    assert all("turned in place" in p.how for p in photos[1:])


def test_walking_round_keeps_the_compass_unless_it_is_far_out(monkeypatch, tmp_path):
    from placelens import objects

    photo = _side_photos(monkeypatch, tmp_path, {"right": {"side": "right", "confidence": 0.9},
                                                  "back": {"side": "back", "confidence": 0.9},
                                                  "left": {"side": "left", "confidence": 0.9}})
    # Walked round: right and left by compass as Muse sees them; the back's compass is 100 degrees out.
    photos = [photo("front", 0.0), photo("right", 275.0), photo("back", 80.0), photo("left", 85.0)]
    objects._angles(None, photos, tmp_path)
    right, back, left = photos[1:]
    assert right.angle == 85 and right.how.startswith("compass")
    assert back.angle == 180 and "Muse" in back.how
    assert left.angle == -85 and left.how.startswith("compass")


def test_muse_naming_the_mirror_side_does_not_overrule_the_compass(monkeypatch, tmp_path):
    from placelens import objects

    # A chair walked round (2 Nov): its left side by compass; Muse, unable to tell a symmetric thing's left from its
    # right, says right. The compass stays.
    photo = _side_photos(monkeypatch, tmp_path, {"side": {"side": "right", "confidence": 0.9},
                                                  "back": {"side": "back", "confidence": 0.9}})
    photos = [photo("front", 135.0), photo("side", 226.0), photo("back", 317.0)]
    objects._angles(None, photos, tmp_path)
    assert photos[1].angle == -91 and photos[1].how.startswith("compass")


def test_a_photo_from_above_carves_as_overhead_not_as_a_side(monkeypatch, tmp_path):
    from placelens import objects

    # The office chair's top photo, 2 Nov: Muse said "above", sure; it was carved as if from behind at eye level.
    photo = _side_photos(monkeypatch, tmp_path, {"side": {"side": "right", "confidence": 0.9},
                                                  "back": {"side": "back", "confidence": 0.9},
                                                  "top": {"side": "above", "confidence": 1.0}})
    photos = [photo("front", 0.0), photo("side", 270.0), photo("back", 180.0), photo("top", 175.0)]
    objects._angles(None, photos, tmp_path)
    side, back, top = photos[1:]
    assert top.overhead and "above" in top.how and len(photos) == 4
    assert not side.overhead and side.angle == 90 and back.angle == 180


def test_a_description_gets_shorter_for_sam():
    from placelens.objects import _shorter

    assert _shorter("small round white vase with a white candle") == [
        "small round white vase with a white candle", "small round white vase", "white vase", "vase"]
    assert _shorter("black box") == ["black box", "box"]
    assert _shorter(None) == []


def test_photos_placed_by_hand_are_used_as_placed(tmp_path):
    import numpy as np

    from placelens import objects

    def ph(name):
        return objects.Photo(tmp_path / name, tmp_path / f"{name}.jpg", 10.0, None, None, 60.0,
                             mask=np.ones((4, 4), bool))

    photos = [ph("front.HEIC"), ph("side.HEIC"), ph("top.HEIC"), ph("lost.HEIC")]
    objects._place(None, photos, tmp_path, {"side.HEIC": 270, "top.HEIC": "above"})  # (no Muse: none is asked)
    assert [p.path.name for p in photos] == ["front.HEIC", "side.HEIC", "top.HEIC"]
    assert photos[1].angle == -90 and photos[1].how == "where you placed it"
    assert photos[2].overhead and photos[0].angle == 0


def test_sides_asked_while_cutting_out_are_used_not_asked_again(monkeypatch, tmp_path):
    from placelens import muse, objects
    from placelens.muse import merge_sides
    from placelens.sides import parse_side

    photo = _side_photos(monkeypatch, tmp_path, {})
    photos = [photo("front", None), photo("side", None), photo("back", None)]

    def asked(*a, **kw):
        raise AssertionError("asked again")

    monkeypatch.setattr(muse, "ask", asked)
    early = {"side.HEIC": merge_sides([parse_side('{"side": "right", "confidence": 0.9}')]),
             "back.HEIC": merge_sides([parse_side('{"side": "back", "confidence": 0.9}')])}
    objects._angles(None, photos, tmp_path, early)
    assert [p.angle for p in photos[1:]] == [90, 180]
