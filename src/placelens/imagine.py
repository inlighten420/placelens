"""One photo to a whole 3D prop: Muse Image draws the sides the photo doesn't show, Muse checks each drawing.

Muse Image is good at drawing an object from its standard sides (the back, the left, the right) in the same pose and
colours. It doesn't always draw what's asked, so Muse Spark looks at each drawing next to the photo and says which
side it shows: a drawing of the wrong side isn't used. The drawings that pass are carved with the photo like a
walk-around, but gently: they only cut away what's clearly outside, and only paint what the photo can't see.
"""

from __future__ import annotations

DRAWN = {"back": 180, "left": -90, "right": 90}  # the sides drawn, at their angles round the object (carve.WALK)

WHERE = {
    "back": "from directly behind it: its back, the side facing away from the camera in @Image1",
    "left": "from the side you'd see after walking a quarter of the way round it to your left, starting from where "
            "@Image1 was taken",
    "right": "from the side you'd see after walking a quarter of the way round it to your right, starting from where "
             "@Image1 was taken",
}

DRAW_PROMPT = """@Image1 is a photo of {name}. Make a photo of this same {name} seen {where}. Keep its pose, shape, \
proportions, colours and markings exactly as they are. The camera is level with its middle and far enough back to \
show all of it. Plain white background, no shadow, nothing else in the picture."""

DRAWN_FOCAL = 4.0  # drawings look like long-lens product shots: a focal length this many times their width


def draw_prompt(name: str, side: str) -> str:
    return DRAW_PROMPT.format(name=name, where=WHERE[side])
