"""An object (a teddy bear, a water tank, a bench) from phone photos: a 3D model at its real size, on the map.

The rules: the phone's own readings first, the AI only for what they can't tell, each
question asked once, nothing slow waited on for long, and no one failure losing the make.

1. Photos: up to MAX_PHOTOS of them, chosen to be spread evenly round the object by their compass.
2. Muse, once (VOTES answers to one question, for steady numbers): what it is, what to call it for SAM, its size,
   whether it's round.
3. SAM 3.1 cuts it out of each photo, several at once.
4. Where each photo was taken from, round the object: its compass. Muse is asked only for a photo without one.
5. The shape: walked round, it's carved from the cut-outs (visual hull) and painted with the photos' own colours.
   One photo: a round thing is turned on a lathe; anything else is carved with the sides Muse Image draws (if paid
   calls are allowed), else made a solid cut-out.
6. Where it stands: the first photo's GPS and compass, its place across the photo and how far away it is.

Every step reports progress (progress.py) and every answer is saved, so making it again costs nothing.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .progress import Counter, step

MAX_PHOTOS = 6  # a walk-round: more than this carves about the same shape, slower
VOTES = 3  # answers to the one question about it: the middle of 3 numbers is steady, 5 was no better
SAM_AT_ONCE = 4  # cut-outs asked for at once
MIN_CARVE = 3  # photos round it needed to carve its shape
MAX_NAMES = 5  # names SAM is tried with on the first photo, at most (each a quarter of a US cent)
PHOTO_COVERS = 50  # degrees: a real photo this close round to a side shows it, and that side isn't drawn
SAM_NAMES = 4  # names SAM tries on each photo after the first, till one finds it (a quarter of a US cent each)
LOGO_NOUNS = ("logo", "brand name text")  # what SAM is asked for to paint over (each a quarter of a US cent)
LOGO_MARGIN = 8  # pixels painted over round what SAM found (its outline of letters runs tight)
LOGO_MOST = 0.25  # a "logo" bigger than this share of the object is the object itself, not a logo on it
PARTS_AT_LEAST = 0.5  # parts whose outline matches the photo less than this aren't used
SIZE_OFF = 1.6  # built this many times bigger or smaller than the research's size: said so
LOOKING_DOWN = 2.5  # the cut-out this many times taller (for its width) than Muse's size says: seen from above
CARRY = 1.0  # metres: anything smaller (its biggest side) can be picked up and moved, so it isn't pinned to the map
SHRUNK = 0.5  # a carving with drawn sides smaller than this share of Muse's size didn't work: a cut-out instead
BORDER_PX = 3  # an object this close to the photo's edge runs off it
DEFAULT_HEIGHT = 1.0  # metres, if Muse gives none
DEPTH_SHARE = 0.5  # depth as a share of width, for a one-photo cut-out with no depth from Muse


class ObjectFailed(RuntimeError):
    """The object couldn't be made; the message says why, in words for the person."""


@dataclass
class Photo:
    path: Path
    small: Path  # upright, smaller JPEG copy (what the models see)
    heading: float | None
    lat: float | None
    lon: float | None
    fov: float
    angle: float = 0.0  # degrees round the object from the first photo (carve.WALK: positive walking right)
    how: str = "compass"
    mask: np.ndarray | None = None  # SAM's cut-out over the small photo
    note: str = ""
    overhead: bool = False  # looks down on it from over the top (Muse said): its turn is found by carving


@dataclass
class Made:
    name: str
    folder: Path
    model: Path
    size_m: list[float]  # wide, deep, tall
    shape: str  # "carved", "round", "imagined", "cutout"
    location: dict | None
    photos_used: list[str]
    notes: list[str] = field(default_factory=list)  # one line per photo or step worth knowing
    warnings: list[str] = field(default_factory=list)  # what to do better next time

    def as_dict(self) -> dict:
        return {"name": self.name, "folder": str(self.folder), "model": str(self.model), "size_m": self.size_m,
                "shape": self.shape, "location": self.location, "photos_used": self.photos_used,
                "notes": self.notes, "warnings": self.warnings}


def spread(headings: list[float | None], keep: int = MAX_PHOTOS) -> list[int]:
    """Which photos to use: the first (the object's front) and then, in turn, the one whose compass is furthest from
    all those chosen, so the chosen ones go evenly round. Photos without a compass come after, as room allows."""
    if len(headings) <= keep:
        return list(range(len(headings)))
    known = [i for i, h in enumerate(headings) if h is not None]
    chosen = [0]
    while len(chosen) < keep and len(chosen) < len(known) + (0 if headings[0] is not None else 1):
        def gap(i: int) -> float:
            return min(abs((headings[i] - headings[c] + 180) % 360 - 180) for c in chosen if headings[c] is not None) \
                if any(headings[c] is not None for c in chosen) else 180.0
        rest = [i for i in known if i not in chosen]
        if not rest:
            break
        chosen.append(max(rest, key=gap))
    chosen += [i for i, h in enumerate(headings) if h is None and i not in chosen][:keep - len(chosen)]
    return sorted(chosen)


def make_object(photos: list[str | Path], out: Path, meta, notes: str | None = None, what: str | None = None,
                height: float | None = None, width: float | None = None, round_: bool | None = None,
                imagine: bool = True, symmetric: bool | None = None, from_parts: bool = False,
                views: dict[str, float | str] | None = None) -> Made:
    """The object in `photos` as a 3D model in `out`. `what`, `height`, `width`, `round_`: what the person said
    (else Muse decides). `views`: each photo's place round it, as the person arranged them (file name: degrees
    round from the front, or "above"); the first photo is the front. `meta` is a MetaClient. Raises ObjectFailed
    with a reason a person can act on."""
    from .meta import OPEN
    from .muse import ask, merge_props, noted
    from .pick import prepare_photo
    from .prop import parse_prop

    out.mkdir(parents=True, exist_ok=True)
    all_photos = [Path(p) for p in photos]
    if not all_photos:
        raise ObjectFailed("no photos")
    cams = [_camera(p) for p in all_photos]
    use = list(range(len(all_photos))) if views else spread([c.heading for c in cams])  # (arranged: all of them)
    chosen = []
    for i in use:
        p, c = all_photos[i], cams[i]
        chosen.append(Photo(p, out / "photos" / f"{p.stem}.jpg", c.heading, c.lat, c.lon, c.fov))
    step(f"getting {len(chosen)} of {len(all_photos)} photos ready")
    with ThreadPoolExecutor(SAM_AT_ONCE) as pool:
        list(pool.map(lambda ph: prepare_photo(ph.path, ph.small), chosen))
    left_out = [p.name for i, p in enumerate(all_photos) if i not in use]

    # Research, before anything's built: what it is, how such things are made, their real size (from what Muse knows
    # of them, checked against the photo), how it sits in the photo and where the camera was. One question, about
    # the first photo (its front), answered VOTES times.
    step("Muse is researching it: what it is, how such things are made, their real size")
    prompt = noted(RESEARCH_PROMPT, notes)
    answers = ask(meta, prompt, chosen[0].small, out / f"{chosen[0].path.stem}_research_{_tag(prompt)}.txt", OPEN,
                  VOTES)
    view = merge_props([parse_prop(a) for a in answers])
    found = _research(answers)
    noun = what or view.segment or view.name or "object"
    names = _names(noun, view.name, answers)  # (what SAM is asked for, in turn, till it finds it)
    name = what or view.name or noun
    if not what and found["logos"] is True:  # (the logo's painted over; its name shouldn't carry the brand either)
        name = _unbranded_name(meta, name, chosen[0].small, out)
    tall = height or view.height or DEFAULT_HEIGHT
    wide = width or view.width or tall
    deep = view.depth or wide
    is_round = round_ if round_ is not None else (found["shape"] == "round" if found["shape"] else bool(view.round))
    flat_from_above = found["pose"] == "lying flat" and found["camera"] == "from above"

    # Cut it out of each photo. SAM knows things by some names and not others (it found a racing wheel as "game
    # controller", not "steering wheel" or "wheel", 1 Nov): the first photo tries the names in turn, the rest use
    # the one that worked first.
    step(f"SAM 3.1 is finding the {noun}")
    first_mask, found_as = None, None
    for name_ in names:
        try:
            first_mask = _segment(meta, chosen[0].small, name_, out)
        except Exception as e:
            if type(e).__name__ == "PaidCallBlocked":
                raise
            continue
        if first_mask is not None:
            found_as = name_
            break
    chosen[0].mask = first_mask
    if found_as is None:
        chosen[0].note = f"SAM 3.1 found nothing as {', '.join(repr(n) for n in names)}"
    else:
        names = [found_as] + [n for n in names if n != found_as]
        noun = found_as
    cutting = Counter(f"SAM 3.1 is cutting out the {noun}: photo", len(chosen))
    cutting.tick()

    def cut(ph: Photo) -> None:
        try:
            for name_ in names[:SAM_NAMES]:  # (the name that worked first, then the others: from behind, a skull
                # candle holder is just a "candle holder", 2 Nov)
                ph.mask = _segment(meta, ph.small, name_, out)
                if ph.mask is not None:
                    break
            if ph.mask is None:  # (seen from another side it may not look like its name at all: Muse says what
                # this photo shows, and SAM tries that: from behind, a skull candle holder looks like a vase, 2 Nov)
                for said in _shorter(_seen_as(meta, ph.small, name, out)):
                    if said not in names:
                        ph.mask = _segment(meta, ph.small, said, out)
                        if ph.mask is not None:
                            break
            if ph.mask is None:
                ph.note = f"SAM 3.1 found no {noun!r}"
        except Exception as e:  # (a paid call switched off is the whole make's problem; anything else, this photo's)
            if type(e).__name__ == "PaidCallBlocked":
                raise
            ph.note = f"cut-out failed ({type(e).__name__})"
        finally:
            cutting.tick()

    # Which side each photo shows needs no cut-outs: Muse is asked while SAM cuts them out (it was after, 2 Nov:
    # 10-25 s a make waiting). Not when the person placed the photos, or the front wasn't found (another photo
    # becomes the front, and the sides would be against the wrong one).
    early = None
    side_pool = ThreadPoolExecutor(1)
    if not views and len(chosen) > 1 and first_mask is not None:
        step(f"Muse checks which side of it each photo shows ({len(chosen) - 1} photo"
             f"{'s' if len(chosen) > 2 else ''}), at the same time as SAM cuts them out")
        early = side_pool.submit(_side_answers, meta, chosen[0], chosen[1:], out)
    with ThreadPoolExecutor(SAM_AT_ONCE) as pool:
        list(pool.map(cut, chosen[1:]))
    try:
        answers = early.result() if early is not None else None
    except Exception:  # (asked again where needed, after)
        answers = None
    side_pool.shutdown(wait=False)
    cut_out = [ph for ph in chosen if ph.mask is not None]
    lines = [f"{ph.path.name}: {ph.note}; left out" for ph in chosen if ph.mask is None]
    if not cut_out:
        raise ObjectFailed(f"SAM 3.1 couldn't find the {noun} in any photo; say what it is in the notes")

    front = cut_out[0]
    warnings = _warnings(front, view)
    if found["logos"] is True:  # (brands aren't ours to put in a game: SAM finds them, they're painted over)
        step("hiding brand names and logos")
        hidden = sum(_unbrand(meta, ph, out) for ph in cut_out)
        lines.append(f"brand names and logos painted over in {hidden} photo(s)" if hidden
                     else "it shows a brand, but SAM 3.1 couldn't find it to paint over")
    built = None
    if from_parts:  # its own 3D version from parts, not the photo on a shape (parts.py)
        from .parts import build_parts

        about = f"{name}. {found['made_of'] or ''} Shape: {found['shape']}." + (
            " It has openings you can see through." if found["holes"] else "")
        flat = flat_from_above or _looking_down(_crop(front.mask), wide, tall)
        parts_, shapes, score, extra = build_parts(meta, front.small, front.mask, about, (wide, deep, tall), flat, out,
                                                   step)
        lines += extra
        if parts_ and score >= PARTS_AT_LEAST:
            every = np.concatenate([p.positions for p in parts_])
            span = every.max(axis=0) - every.min(axis=0)
            built = (parts_, "parts", (float(span[0]), float(span[2]), float(span[1])))
        else:
            lines.append(f"the parts didn't fit the photo ({score:.0%}): made the other way instead")
    if built:
        parts, shape, size = built
    elif len(cut_out) < MIN_CARVE:  # (two outlines crossed carve a block: one photo and drawn sides do better)
        others = []
        if len(cut_out) > 1:  # (the other photos carve and paint too: Muse Image draws only the sides they don't
            # show; a skull's real side photo was left out for a drawing that put its face on the side, 2 Nov)
            _place(meta, cut_out, out, views, answers if cut_out[0] is chosen[0] else None)
            others = [ph for ph in cut_out if ph is not front]
            lines += [f"{ph.path.name}: {ph.angle:+.0f}° round ({ph.how})" for ph in others]
            lines.append(f"only {len(cut_out)} photos: the sides they don't show are drawn; walk round it and take "
                         f"{MIN_CARVE}+ to carve it from photos alone")
        parts, shape, size, extra = _one_photo(meta, front, noun, name, (wide, deep, tall), is_round, imagine,
                                               view.symmetric if symmetric is None else symmetric, notes, out,
                                               flat_from_above, others)
        lines += extra
        grid = getattr(parts, "grid", None)
        made = found["holes"] is True and found["shape"] in MADE_SHAPES and not flat_from_above
        if grid and shape == "imagined" and made:
            fitted, extra = _fit_parts(meta, front, name, found, (wide, deep, tall), parts[0], grid, out,
                                          getattr(parts, "shots", None))
            lines += extra
            if fitted:
                parts, shape = fitted, "parts"
    else:
        before = list(cut_out)
        _place(meta, cut_out, out, views, answers if cut_out[0] is chosen[0] else None)
        lines += [f"{ph.path.name}: {ph.angle:+.0f}° round ({ph.how})" for ph in cut_out]
        lines += [f"{ph.path.name}: left out: {ph.note}" for ph in before if ph not in cut_out]
        step(f"carving the {noun} from {len(cut_out)} cut-outs")
        parts, grid = _carve(cut_out, (wide, deep, tall))
        shape = "carved"
        if found["holes"] is True and found["shape"] in MADE_SHAPES and not flat_from_above:
            fitted, extra = _fit_parts(meta, front, name, found, (wide, deep, tall), parts[0], grid, out,
                                          getattr(parts, "shots", None))
            lines += extra
            if fitted:
                parts, shape = fitted, "parts"
        every = np.concatenate([p.positions for p in parts])
        span = every.max(axis=0) - every.min(axis=0)
        size = (float(span[0]), float(span[2]), float(span[1]))

    # The check: what was built against what the research said such a thing measures.
    biggest, expected = max(size), max(wide, deep, tall)
    if expected and not (1 / SIZE_OFF <= biggest / expected <= SIZE_OFF):
        warnings.append(f"it came out {biggest:.2f} m at its biggest, but things like it are about {expected:.2f} m: "
                        "the photos may not show it well")
    if found["holes"] is True and shape in ("carved", "imagined"):
        lines.append("it has openings you can see through: carving fills them in")
    if found["made_of"]:
        lines.append(f"research: {found['made_of']}")

    step("saving the model")
    from .build import write_glb

    model = write_glb(out / "prop.glb", parts, name=name)
    # Where it stands, for big things that stay put (a tank, a statue). A thing you could pick up goes where you put
    # it: a phone's GPS is metres out, so on the map it would land somewhere odd (a racing wheel at the house, 1 Nov).
    location = _where(front, size[2], view.distance) if max(size) >= CARRY else None
    if location is None and max(size) < CARRY:
        lines.append(f"small enough to carry ({max(size):.2f} m): in your things to put where you like, not on the map")
    lines += [f"{n}: not needed (close to another photo's view)" for n in left_out]
    made = Made(name, out, model, [round(v, 2) for v in size], shape, location, [ph.path.name for ph in cut_out],
                lines, warnings)
    (out / "object.json").write_text(json.dumps(made.as_dict(), indent=1), encoding="utf-8")
    _place_file(made, out)
    return made


def _camera(path: Path):
    """The photo's GPS, compass and lens; a photo with no readable EXIF still works (no GPS or compass)."""
    from .photo import DEFAULT_FOV, Camera, read_camera

    try:
        return read_camera(path)
    except Exception:
        return Camera(None, None, None, DEFAULT_FOV, "default", 0, 0)


RESEARCH_PROMPT = """Look at the main free-standing object in this photo: the thing it was taken of, not a building, \
the ground or the sky. Before anyone builds a 3D model of it, research it: what it is, how things like it are made, \
and how big they really are. Use what you know about such objects (a product's real dimensions, if you recognise \
it), then check that against what the photo shows.

Answer in JSON only:
{"name": "what it is, a few words", \
"segment": "a short noun phrase that picks out just this object, for a segmentation model (e.g. "water tank", \
"bench", "potted plant")", \
"other_names": three other short noun phrases a segmentation model might know it by, the most general last \
(e.g. for a garden gnome ["figurine", "statue", "ornament"]), \
"made_of": "how such things are made, in a sentence: main parts and materials", \
"shape": "round" (the same all round an upright axis: a tank, pot, bottle), "flat" (a panel, sign, board, \
something thin), "boxy" or "organic" (an animal, a toy, a plant), \
"holes": true if it has openings you can see through (a wheel's spokes, a chair's legs), \
"pose": "upright", "lying flat" or "on its side": how it sits in this photo, \
"camera": "level" or "from above": where the photo was taken from, relative to it, \
"width_m": its real size across the photo in metres, \
"depth_m": its real size front to back (along the ground, away from the camera) in metres, \
"height_m": its real height as it sits here, up from what it stands or lies on, in metres, \
"distance_m": distance from the camera to it in metres, \
"symmetric": true if its left and right are mirror images, \
"logos": true if a brand name, logo or trademark is visible on it, \
"fully_visible": true if the whole object is in the photo and nothing hides part of it, \
"hidden": "what hides or cuts off part of it, if anything", \
"confidence": 0.0 to 1.0}"""


def _research(answers: list[str]) -> dict:
    """The research's own fields (beyond what parse_prop reads): the most common answer to each."""
    from collections import Counter

    seen: dict[str, list] = {k: [] for k in ("shape", "pose", "camera", "holes", "made_of", "logos")}
    for a in answers:
        m = re.search(r"\{.*\}", a, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            continue
        for k in seen:
            if isinstance(d.get(k), (str, bool)):
                seen[k].append(d[k].strip().lower() if isinstance(d[k], str) else d[k])
    return {k: (Counter(v).most_common(1)[0][0] if v else None) for k, v in seen.items()}


UNBRAND_PROMPT = """This object was described as "{name}". Say what it is in a few plain words with no brand, maker, \
product line or model name: what anyone would call it (e.g. "Nike Air Max running shoe" -> "running shoe"). Answer \
with the words only."""


def _unbranded_name(meta, name: str, photo: Path, out: Path) -> str:
    """`name` without the brand in it, from Muse (free on Spark Open); `name` as it was if Muse gives nothing usable."""
    from .meta import OPEN
    from .muse import ask

    prompt = UNBRAND_PROMPT.format(name=name)
    try:
        said = ask(meta, prompt, photo, out / f"{photo.stem}_unbranded_{_tag(prompt)}.txt", OPEN, 1, effort="low")
    except Exception:  # (a nicety: never fail the make over the name)
        return name
    plain = re.sub(r"[\"'.\n]+", " ", said[0] if said else "").strip().lower()
    return plain if 0 < len(plain) <= 60 and len(plain.split()) <= 6 else name


SEEN_AS_PROMPT = """This photo shows a {name}, maybe from its back or side, where it may not look like one. In a \
few plain words, what does the object in it look like from here (e.g. "round green pot", "black box with a handle")? \
Answer with the words only."""


def _seen_as(meta, photo: Path, name: str, out: Path) -> str | None:
    """What the object looks like in this photo, in a few words, from Muse (free on Spark Open): for SAM, when the
    object's own names find nothing. None if Muse gives nothing usable."""
    from .meta import OPEN
    from .muse import ask

    prompt = SEEN_AS_PROMPT.format(name=name)
    try:
        said = ask(meta, prompt, photo, out / f"{photo.stem}_seen_as_{_tag(prompt)}.txt", OPEN, 1, effort="low")
    except Exception:
        return None
    words = re.sub(r"[\"'.\n]+", " ", said[0] if said else "").strip().lower()
    return words if 0 < len(words) <= 80 and len(words.split()) <= 12 else None  # (_shorter cuts it down for SAM)


def _shorter(words: str | None) -> list[str]:
    """A description, then shorter forms of it for SAM (which finds short names best): "small round white vase with
    a white candle", "small round white vase", "white vase", "vase"."""
    if not words:
        return []
    main = re.split(r"\b(?:with|on|holding|and)\b", words)[0].strip()
    head = main.split()
    out = [words, main, " ".join(head[-2:]), head[-1] if head else ""]
    return [w for i, w in enumerate(out) if w and w not in out[:i]]


def _names(noun: str, name: str | None, answers: list[str]) -> list[str]:
    """What SAM is asked for, in turn: Muse's noun, the other names it gave, then shorter forms of the noun."""
    out = [noun]
    for a in answers:
        m = re.search(r"\{.*\}", a, re.S)
        try:
            others = json.loads(m.group(0)).get("other_names", []) if m else []
        except (json.JSONDecodeError, AttributeError):
            others = []
        out += [str(o) for o in others if isinstance(o, str)]
    words = noun.split()
    out += [" ".join(words[-2:]), words[-1]] if len(words) > 1 else []
    if name:
        out.append(name)
    seen, kept = set(), []
    for n in (o.strip().lower() for o in out):
        if n and n not in seen:
            seen.add(n)
            kept.append(n)
    return kept[:MAX_NAMES]


def _tag(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def _segment(meta, small: Path, noun: str, out: Path) -> np.ndarray | None:
    """SAM's biggest match for `noun` as a mask over the small photo (saved, so asked once)."""
    from .meta import segments_from_file

    slug = re.sub(r"[^a-z0-9]+", "-", noun.lower()).strip("-")
    reply = out / f"{small.stem}_sam_{slug}.txt"
    if reply.exists():
        segments = segments_from_file(reply)
    else:
        segments, raw = meta.segment(small, noun)
        reply.write_text(raw, encoding="utf-8")
    if not segments:
        return None
    with Image.open(small) as img:
        return segments[0].full_mask(img.size)


def _place(meta, photos: list[Photo], out: Path, views: dict[str, float | str] | None,
           answers: dict | None = None) -> None:
    """Each photo's angle round the object: as the person arranged them (the studio's circle), else worked out
    (_angles). An arranged photo the person left without a place is left out."""
    if not views:
        return _angles(meta, photos, out, answers)
    step("using where you placed each photo round it (no compass or Muse needed)")
    for i, ph in enumerate(photos):
        v = views.get(ph.path.name)
        if i == 0:
            ph.angle, ph.how = 0.0, "the front (you chose it)"
        elif v == "above":
            ph.overhead, ph.angle, ph.how = True, 0.0, "from above (you placed it); its turn found by carving"
        elif isinstance(v, (int, float)):
            ph.angle, ph.how = float((v + 180) % 360 - 180), "where you placed it"
        else:
            ph.mask, ph.note = None, "not placed round it"
    photos[:] = [ph for ph in photos if ph.mask is not None]


SIDE_SURE = 0.6  # how sure Muse must be of a side to overrule the compass
SIDE_OFF = 60.0  # degrees between the compass and Muse's side before Muse wins
MIRROR = 30.0  # degrees: Muse naming the mirror of the compass's side (left for right) is no reason to overrule
STOOD_STILL = 60.0  # the compass moved less than this over all the photos...
SAW_ROUND = 90.0  # ...while Muse saw sides at least this far apart: the object was turned, not walked round


def _side_answers(meta, first: Photo, rest: list[Photo], out: Path) -> dict[str, object]:
    """Muse's side for each photo against the front (`first`), all at once: photo file name -> merged side, or None
    (no answer). Needs no cut-outs, so it runs while SAM cuts them out. Saved per photo and per front: the same
    photo is a different side against a different front."""
    from .carve import SIDE_PROMPT
    from .meta import OPEN
    from .muse import ask, merge_sides
    from .sides import parse_side

    def side(ph: Photo):
        # A quick look, so low effort: at the default, Spark Open thought past its token cap on a chair's side view
        # and answered nothing, twice (2 Nov); at low it said "left" in 17 s. Asked again once if empty.
        for _ in range(2):
            try:
                return merge_sides([parse_side(a) for a in ask(
                    meta, SIDE_PROMPT, [first.small, ph.small],
                    out / f"{ph.path.stem}_side_vs_{first.path.stem}_{_tag(SIDE_PROMPT)}_low.txt", OPEN, 1,
                    effort="low")])
            except Exception:
                continue
        return None  # (no answer: the compass stands alone)

    with ThreadPoolExecutor(SAM_AT_ONCE) as pool:
        return dict(zip((ph.path.name for ph in rest), pool.map(side, rest), strict=True))


def _angles(meta, photos: list[Photo], out: Path, answers: dict | None = None) -> None:
    """Each photo's angle round the object from the first. Two ways of taking the photos, both handled:
    - walked round it: the compass (and where the object sits across the photo) gives the angle; Muse names which
      side each photo shows, and if Muse is sure and the two differ by more than SIDE_OFF, Muse wins (an indoor
      compass can be far out);
    - turned it in place (an office chair, 2 Nov): the compass hardly moves while Muse sees different sides, so
      every angle is Muse's line of sight.
    No compass: Muse's side alone. `answers`: Muse's sides already asked (_side_answers, while SAM was cutting
    out); any photo missing from them is asked now."""
    from .carve import WALK
    from .pick import pixel_to_offset

    def bearing(ph: Photo) -> float:
        cols = np.flatnonzero(ph.mask.any(axis=0))
        return ph.heading + pixel_to_offset((cols[0] + cols[-1]) / 2, ph.mask.shape[1], ph.fov)

    first = photos[0]
    first.how = "the front: every angle is measured from it"
    base = bearing(first) if first.heading is not None else None
    rest = photos[1:]
    if not rest:
        return
    answers = dict(answers or {})
    missing = [ph for ph in rest if ph.path.name not in answers]
    if missing:
        step(f"Muse checks which side of it each photo shows ({len(missing)} photo{'s' if len(missing) != 1 else ''})")
        answers.update(_side_answers(meta, first, missing, out))
    seen = [answers[ph.path.name] for ph in rest]
    # A photo looking down from over the top has no side, and a phone pointing down hardly knows its compass: it
    # carves as overhead, its turn found by fit (Muse said "above", sure, for a chair's top photo, and it was carved as
    # if from behind at eye level, 2 Nov).
    for ph, s in zip(rest, seen, strict=True):
        if s and s.side == "above" and s.confidence >= SIDE_SURE:
            ph.overhead, ph.angle, ph.how = True, 0.0, "from above (Muse); its turn found by carving"
    sides = [(ph, s) for ph, s in zip(rest, seen, strict=True) if not ph.overhead]
    rest = [ph for ph, _ in sides]
    seen = [s for _, s in sides]
    sure = {id(ph): (s, float(WALK[s.side])) for ph, s in zip(rest, seen, strict=True)
            if s and s.side in WALK and s.confidence >= SIDE_SURE}
    compass = {id(ph): -((bearing(ph) - base + 180) % 360 - 180)  # walking right turns your view left
               for ph in rest if ph.heading is not None and base is not None}

    def spread_of(angles: list[float]) -> float:
        """The widest gap between two of these angles (and 0 from the first photo)."""
        a = [0.0] + angles
        return max(abs((x - y + 180) % 360 - 180) for x in a for y in a)

    turned = len(compass) == len(rest) and len(sure) >= 1 \
        and spread_of(list(compass.values())) < STOOD_STILL and spread_of([m for _, m in sure.values()]) >= SAW_ROUND
    for ph, s in zip(rest, seen, strict=True):
        muse = sure.get(id(ph))
        if turned:
            if muse:
                ph.angle, ph.how = muse[1], f"turned in place; Muse: {muse[0].side}"
            else:
                ph.mask, ph.note = None, "the object was turned in place, and Muse couldn't tell which side this shows"
        elif id(ph) in compass:
            c = compass[id(ph)]
            off = abs((c - muse[1] + 180) % 360 - 180) if muse else 0.0
            # Left and right of a symmetric thing are mirror images: Muse can't tell them apart (it said "right" for a
            # chair's left side, 2 Nov), so an answer that is just the compass's mirror doesn't overrule it.
            mirrored = muse is not None and abs((c + muse[1] + 180) % 360 - 180) <= MIRROR
            if muse and off > SIDE_OFF and not mirrored:
                ph.angle, ph.how = muse[1], f"Muse: {muse[0].side} (compass said {c:+.0f}°, {off:.0f}° out)"
            else:
                ph.angle, ph.how = c, f"compass, Muse agrees ({muse[0].side})" if muse else "compass"
        elif s and s.side in WALK:
            ph.angle, ph.how = float(WALK[s.side]), f"no compass; Muse: {s.side}"
        else:
            ph.mask, ph.note = None, "no compass, and Muse couldn't tell the side"
    photos[:] = [ph for ph in photos if ph.mask is not None]


def _shots(photos: list[Photo]):
    from .carve import OVERHEAD, Shot

    shots = []
    for ph in photos:
        img = np.asarray(Image.open(ph.small).convert("RGB"))
        focal = img.shape[1] / 2 / math.tan(math.radians(ph.fov / 2))
        shots.append(Shot(ph.mask, img, focal, ph.angle, elevation=OVERHEAD if ph.overhead else 0.0,
                          overhead=ph.overhead))
    return shots


def _carve(photos: list[Photo], size: tuple[float, float, float]) -> list:
    """The visual hull of the cut-outs, the cameras nudged till the outlines agree, painted from the photos."""
    from .carve import carve, find_elevation, fit_cameras, place_shots, skin, turn_overhead

    shots = _shots(photos)
    place_shots(shots, size[2])
    find_elevation(shots, size)
    turn_overhead(shots, size)
    fit_cameras(shots, size)
    kept, corner, voxel = carve(shots, size, misses=0)
    if not kept.any():
        raise ObjectFailed("the photos' outlines didn't agree on a shape; walk round it with the phone level")
    made = _Carved([skin(kept, corner, voxel, shots, size[2])])
    made.grid, made.shots = (kept, corner, voxel), shots  # (the cameras: parts fitted to it are checked from them)
    return made, made.grid


class _Carved(list):
    """Model parts from a carving, carrying the carved grid (kept, corner, voxel) as `.grid`."""

    grid = None
    shots = None  # carve.Shot per photo (or drawing), its camera fitted


GOOD_PHOTOS = 0.5  # parts whose outline matches the photos' this well (intersection over union) are kept
SIGHT_MOVES = 3  # Muse questions to correct the parts by looking at them over the photos (0: none)
MADE_SHAPES = ("boxy", "flat")  # research shapes of hard, made things: built from parts fitted to the carving


def _fit_parts(meta, front: Photo, name: str, found: dict, size, carved, grid, out: Path, shots=None):
    """A hard, made thing with gaps (an office chair): Muse's parts fitted to the carving, coloured from it (fit.py).
    Returns the model parts (or None to keep the carving) and notes."""
    from .fit import GOOD_FILL, GOOD_INSIDE, colour_from, fit
    from .parts import describe_parts, model_parts

    about = f"{name}. {found['made_of'] or ''} Shape: {found['shape']}. It has openings you can see through."
    shapes, _ = describe_parts(meta, front.small, about, size, False, out, step)
    if not shapes:
        return None, ["Muse didn't describe it as parts: kept the carving"]
    step(f"fitting Muse's {len(shapes)} parts to the carving")
    from .sight import seen

    real = [s for s in shots or [] if not s.imagined and not s.overhead]  # (photos the model is checked against)
    shapes, inside_, fill = fit(shapes, *grid, seen=(lambda p: seen(p, real, size[2] / 2)) if len(real) > 1 else None)
    # (one photo can't say: a part anywhere along its line of sight falls inside the outline)
    note = f"{len(shapes)} parts fitted to the carving: {inside_:.0%} inside it, filling {fill:.0%}"
    # Filling the carving is one test; matching the photos' outlines is the other (the carving fills in a chair's
    # gaps, so parts that are right can't fill it: 2 Nov). Either will do, with the parts inside it.
    # Muse gets its moves first (it sees what it built over the photos and corrects it: sight.py), then it's judged.
    from .sight import refine
    from .sight import score as photo_score

    if inside_ < GOOD_INSIDE:
        return None, [note + ": not close enough, kept the carving"]
    notes = [note]
    if shots and SIGHT_MOVES:
        shapes, said = refine(meta, shapes, shots, size[2] / 2, name, out, SIGHT_MOVES, step)
        notes += said
    photos = photo_score(shapes, real, size[2] / 2) if len(real) > 1 and shapes else 0.0
    if fill < GOOD_FILL and photos < GOOD_PHOTOS:
        return None, notes + [f"matching the photos {photos:.0%}: not close enough, kept the carving"]
    colour_from(shapes, carved.positions, carved.colors)
    (out / "parts.json").write_text(__import__("json").dumps(shapes, indent=1), encoding="utf-8")
    return model_parts(shapes), notes + ["coloured from the photos"]


def _one_photo(meta, ph: Photo, noun: str, name: str, size, is_round: bool, imagine: bool, symmetric, notes,
               out: Path, flat_from_above: bool = False, others: list[Photo] = ()):
    """One photo: a round thing turned on a lathe; else the sides Muse Image draws carved with it; else a solid
    cut-out. Returns the parts, the shape's kind, its size (wide, deep, tall) and notes."""
    from .carve import place_shots
    from .prop import lathe, outline, solid

    wide, deep, tall = size
    mask = ph.mask
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    top, bottom, left, right = rows[0], rows[-1], cols[0], cols[-1]
    with Image.open(ph.small) as img:
        crop = img.convert("RGB").crop((left, top, right + 1, bottom + 1))
    box = mask[top:bottom + 1, left:right + 1]
    if flat_from_above or _looking_down(box, wide, tall):  # (the research said so, or the photo can't be a side view
        # of a thing that size: scale by its real width, and lay it as it lay)
        from .prop import cutout_png

        shape = outline(box)
        if shape is not None:
            long = wide * box.shape[0] / box.shape[1]  # metres the photo's height covers, along the ground
            parts = _lay_flat(solid(shape, (box.shape[1], box.shape[0]), long, max(tall, 0.01), cutout_png(crop, box)))
            return parts, "flat", (wide, long, tall), [
                f"{ph.path.name} looks down on it lying flat: a flat cut-out, {wide:.2f} m across (Muse's size)"]
    if is_round:
        step(f"turning the {noun} on a lathe")
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=90)
        cut = (left <= BORDER_PX, right >= mask.shape[1] - BORDER_PX)
        per_px = tall / max(bottom - top, 1)
        parts = lathe(box, buf.getvalue(), tall, cut, max_radius_px=wide / 2 / per_px if wide else None)
        across = float(2 * np.hypot(parts[0].positions[:, 0], parts[0].positions[:, 2]).max())
        return parts, "round", (across, across, tall), []
    if imagine:
        try:
            drawn, lines = _imagined(meta, ph, noun, name, bool(symmetric), notes, out, [o.angle for o in others])
        except Exception as e:  # (no paid calls, or Muse Image down: a solid cut-out instead)
            drawn, lines = [], [f"sides not drawn ({type(e).__name__}); a solid cut-out instead"]
        if drawn:
            from .carve import carve, skin

            shots = _shots([ph, *others]) + drawn
            place_shots(shots, tall)
            kept, corner, voxel = carve(shots, (wide, deep, tall), misses=0)
            if kept.any():
                parts = _Carved([skin(kept, corner, voxel, shots, tall)])
                parts.grid = (kept, corner, voxel)  # (for fitting parts to it: a hard, made thing)
                parts.shots = shots
                every = np.concatenate([p.positions for p in parts])
                span = every.max(axis=0) - every.min(axis=0)
                if span[1] >= SHRUNK * tall and span[0] >= SHRUNK * wide:
                    return parts, "imagined", (float(span[0]), float(span[2]), float(span[1])), lines
                # (the drawings and the photo disagree: a photo looking down on a thing lying flat, against level
                # drawings, carved a racing wheel to 9 x 4 cm, 1 Nov)
                lines.append(f"the drawn sides didn't fit the photo (carved to {span[0]:.2f} x {span[1]:.2f} m of "
                             f"{wide:.2f} x {tall:.2f}); a solid cut-out instead. Take it from the side, phone level")
    from .prop import cutout_png

    shape = outline(box)
    if shape is None:
        raise ObjectFailed("the cut-out has no usable outline")
    depth = deep or wide * DEPTH_SHARE
    parts = solid(shape, (right - left + 1, bottom - top + 1), tall, depth, cutout_png(crop, box))
    across = (shape.bounds[2] - shape.bounds[0]) * tall / max(shape.bounds[3] - shape.bounds[1], 1)
    return parts, "cutout", (across, depth, tall), ["one photo: a solid cut-out"]


def _unbrand(meta, ph: Photo, out: Path) -> bool:
    """Brand names and logos on the object painted over from what's round them (OpenCV inpainting, in this photo's
    copy, which the model's colours then come from). SAM 3.1 finds them. True if it found any."""
    import cv2
    from scipy import ndimage

    from .meta import segments_from_file

    found = np.zeros(ph.mask.shape, bool)
    for noun in LOGO_NOUNS:
        reply = out / f"{ph.small.stem}_sam_{noun.replace(' ', '-')}.txt"
        if reply.exists():
            segments = segments_from_file(reply)
        else:
            segments, raw = meta.segment(ph.small, noun)
            reply.write_text(raw, encoding="utf-8")
        for s in segments:
            found |= s.full_mask((ph.mask.shape[1], ph.mask.shape[0]))
    found &= ndimage.binary_dilation(ph.mask, iterations=4)  # (only what's on the object)
    if not found.any() or found.sum() > LOGO_MOST * ph.mask.sum():  # (none, or "logo" took the whole thing)
        return False
    holes = ndimage.binary_dilation(found, iterations=LOGO_MARGIN)
    image = cv2.imread(str(ph.small))
    clean = _clone_over(image, holes, ph.mask)
    seam = (ndimage.binary_dilation(holes, iterations=2) & ~ndimage.binary_erosion(holes, iterations=2))
    clean = cv2.inpaint(clean, seam.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)  # (the patches' edges blended)
    target = ph.small.with_name(f"{ph.small.stem}_unbranded.jpg")
    cv2.imwrite(str(target), clean, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ph.small = target
    return True


def _clone_over(image: np.ndarray, holes: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Each painted-over patch filled with the same surface from right beside it (as a clone stamp: a pattern such
    as carbon weave carries on, where a blur left a pale smudge): the nearest same-size patch above, below, left or
    right that's all on the object and clear of what's painted over. A patch with none nearby is blurred over."""
    import cv2
    from scipy import ndimage

    out = image.copy()
    labels, count = ndimage.label(holes)
    h, w = holes.shape
    for k, box in enumerate(ndimage.find_objects(labels), start=1):
        rows, cols = box
        bh, bw = rows.stop - rows.start, cols.stop - cols.start
        spot = labels[box] == k
        done = False
        for dy, dx in ((-bh - 2, 0), (bh + 2, 0), (0, -bw - 2), (0, bw + 2), (-2 * bh, 0), (2 * bh, 0)):
            r0, c0 = rows.start + dy, cols.start + dx
            if r0 < 0 or c0 < 0 or r0 + bh > h or c0 + bw > w:
                continue
            if on[r0:r0 + bh, c0:c0 + bw].all() and not holes[r0:r0 + bh, c0:c0 + bw].any():
                out[box][spot] = image[r0:r0 + bh, c0:c0 + bw][spot]
                done = True
                break
        if not done:
            one = (labels == k).astype(np.uint8) * 255
            out = cv2.inpaint(out, one, 9, cv2.INPAINT_TELEA)
    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _looking_down(box: np.ndarray, wide: float, tall: float) -> bool:
    """A photo looking down on a thing lying flat: Muse's height (as it lies) is far shorter, against its width, than
    the cut-out is tall against its width. Seen from the side, the two would roughly agree."""
    if not wide or not tall:
        return False
    seen = box.shape[0] / max(box.shape[1], 1)  # the cut-out's height over its width
    return seen > LOOKING_DOWN * (tall / wide)


def _lay_flat(parts: list) -> list:
    """Parts made standing up (front +Z) laid on their back: the front faces up, standing on Y = 0."""
    for p in parts:
        x, y, z = p.positions[:, 0].copy(), p.positions[:, 1].copy(), p.positions[:, 2].copy()
        p.positions = np.stack([x, z, -y], axis=1).astype(np.float32)
        nx, ny, nz = p.normals[:, 0].copy(), p.normals[:, 1].copy(), p.normals[:, 2].copy()
        p.normals = np.stack([nx, nz, -ny], axis=1).astype(np.float32)
        p.positions[:, 1] -= p.positions[:, 1].min()
        p.positions[:, 2] -= (p.positions[:, 2].min() + p.positions[:, 2].max()) / 2
    return parts


def _imagined(meta, ph: Photo, noun: str, name: str, symmetric: bool, notes, out: Path, seen: list[float] = ()):
    """Muse Image draws the back, left and right (at once), each but the sides a real photo shows (`seen`: their
    angles round); each drawing is cut out. A symmetric object's missing side is the other one mirrored. Returns
    carving shots and a line per side."""
    from .carve import WALK, Shot
    from .imagine import DRAWN, DRAWN_FOCAL, draw_prompt
    from .muse import noted

    step(f"Muse Image is drawing the {name}'s other sides")

    def draw(side: str):
        prompt = noted(draw_prompt(name, side), notes)
        picture = out / f"drawn_{side}_{_tag(prompt)}.jpg"
        if not picture.exists():
            drawn = Image.open(io.BytesIO(meta.draw(prompt, [ph.small]))).convert("RGB")
            drawn.thumbnail((1600, 1600))
            drawn.save(picture, quality=90)
        mask = _segment(meta, picture, noun, out)
        if mask is None:
            for said in _shorter(_seen_as(meta, picture, name, out)):
                mask = _segment(meta, picture, said, out) if said != noun else None
                if mask is not None:
                    break
        if mask is None:
            return side, None, f"{side}: drawn, but SAM found no {noun!r} in it; not used"
        img = np.asarray(Image.open(picture).convert("RGB"))
        return side, Shot(mask, img, DRAWN_FOCAL * img.shape[1], WALK[side], imagined=True), f"{side}: drawn"

    shown = [side for side in DRAWN if any(abs((WALK[side] - a + 180) % 360 - 180) < PHOTO_COVERS for a in seen)]
    wanted = [side for side in DRAWN if side not in shown]
    with ThreadPoolExecutor(max(1, len(wanted))) as pool:
        results = list(pool.map(draw, wanted))
    results += [(side, None, f"{side}: from a photo, not drawn") for side in shown]
    shots = [s for _, s, _ in results if s is not None]
    lines = [line for _, _, line in results]
    have = {s.angle: s for s in shots}
    for side, other in (("left", "right"), ("right", "left")):
        if symmetric and side not in shown and WALK[side] not in have and WALK[other] in have:
            twin = have[WALK[other]]
            shots.append(Shot(twin.mask[:, ::-1].copy(), twin.image[:, ::-1].copy(), twin.focal, WALK[side],
                              imagined=True))
            lines.append(f"{side}: the {other} drawing mirrored (it's symmetric)")
    return shots, lines


def _warnings(ph: Photo, view) -> list[str]:
    rows, cols = np.flatnonzero(ph.mask.any(axis=1)), np.flatnonzero(ph.mask.any(axis=0))
    h, w = ph.mask.shape
    out = []
    if cols[0] <= BORDER_PX or cols[-1] >= w - BORDER_PX or rows[0] <= BORDER_PX or rows[-1] >= h - BORDER_PX:
        out.append(f"{ph.path.name}: it runs off the edge of the photo; step back and take it whole")
    if view.fully_visible is False:
        out.append(f"part of it is hidden ({view.hidden or 'something in front'})")
    return out


def _where(ph: Photo, tall: float, distance: float | None) -> dict | None:
    """Where it stands: the photo's GPS and compass, where it sits across the photo, and how far away (Muse's
    distance, else how far a thing this tall must be to look this tall)."""
    from .pick import pixel_to_offset
    from .prop import place
    from .view import LocalFrame

    if ph.lat is None or ph.heading is None:
        return None
    rows, cols = np.flatnonzero(ph.mask.any(axis=1)), np.flatnonzero(ph.mask.any(axis=0))
    width = ph.mask.shape[1]
    focal = width / 2 / math.tan(math.radians(ph.fov / 2))
    by_size = focal * tall / max(rows[-1] - rows[0], 1)
    away = distance or by_size
    east, north, facing = place(ph.heading, pixel_to_offset((cols[0] + cols[-1]) / 2, width, ph.fov), away)
    lat, lon = LocalFrame(ph.lat, ph.lon).to_latlon(east, north)
    return {"lat": round(lat, 7), "lon": round(lon, 7), "facing_bearing": round(facing, 1),
            "distance_m": round(away, 1)}


def _place_file(made: Made, out: Path) -> None:
    """prop.geojson (what `pack` reads) and view.html (the model in a browser)."""
    from .viewer import write_viewer

    loc = made.location
    feature = {"type": "Feature",
               "geometry": {"type": "Point", "coordinates": [loc["lon"], loc["lat"]]} if loc else None,
               "properties": {"name": made.name, "model": made.model.name,
                              "model_axes": "metres, Y up, standing on Y = 0, front (first photo's side) +Z",
                              "size_m": made.size_m, "facing_bearing": loc["facing_bearing"] if loc else None,
                              "photos": made.photos_used}}
    (out / "prop.geojson").write_text(json.dumps(feature, indent=1), encoding="utf-8")
    write_viewer(out / "view.html", made.model, made.name.capitalize(),
                 {"Size": "{:.2f} m wide, {:.2f} m deep, {:.2f} m tall".format(*made.size_m),
                  "Photos": f"{len(made.photos_used)} used"}, "Made by PlaceLens from photos.", facing=0.0)
