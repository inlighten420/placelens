from placelens.sides import SideView, compass_agrees, facing, parse_side


def test_parse_side():
    assert parse_side('```json\n{"side": "Back Left", "confidence": 0.8, "reasons": ["gable"]}\n```') == \
        SideView("back-left", 0.8, ["gable"])
    assert parse_side('{"side": "upstairs", "confidence": 0.9}').side is None
    assert parse_side("no idea") == SideView(None)


def test_facing_and_compass():
    front = 168  # the front faces south-south-east
    assert round(facing("back", front)) == 348 and round(facing("left", front)) == 258
    assert round(facing("back-left", front)) == 303  # halfway between back and left
    assert compass_agrees("back", 140, front)  # looking roughly at the back: south-east
    assert not compass_agrees("back", 348, front)  # looking north: that's the front's view
    assert compass_agrees("left", 100, front) and not compass_agrees("left", 250, front)
