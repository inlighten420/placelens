"""Read where a photo was taken, which way it faced, and how wide the lens saw."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

try:  # iPhones save HEIC by default
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

GPS_IFD = 0x8825
EXIF_IFD = 0x8769
ORIENTATION = 0x0112
FOCAL_35MM = 0xA405

# A typical phone main camera (26 mm equivalent) sees about 69 degrees across in landscape.
DEFAULT_FOV = 65.0
FULL_FRAME_DIAGONAL = math.hypot(36, 24)


@dataclass
class Camera:
    lat: float | None
    lon: float | None
    heading: float | None  # degrees clockwise from north
    fov: float  # horizontal field of view, degrees
    fov_source: str  # "exif" or "default"
    width: int
    height: int


def _to_degrees(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    value = d + m / 60 + s / 3600
    return -value if ref in ("S", "W") else value


def horizontal_fov(focal_35mm: float, width: int, height: int) -> float:
    """Horizontal field of view from a 35 mm-equivalent focal length, matched on the diagonal."""
    across = FULL_FRAME_DIAGONAL * width / math.hypot(width, height)
    return math.degrees(2 * math.atan(across / (2 * focal_35mm)))


def read_camera(path: str | Path) -> Camera:
    with Image.open(path) as img:
        exif = img.getexif()
        width, height = img.size
    gps = exif.get_ifd(GPS_IFD)
    details = exif.get_ifd(EXIF_IFD)

    if exif.get(ORIENTATION) in (5, 6, 7, 8):  # stored sideways
        width, height = height, width

    lat = lon = heading = None
    if 2 in gps and 4 in gps:
        lat = _to_degrees(gps[2], gps.get(1, "N"))
        lon = _to_degrees(gps[4], gps.get(3, "E"))
    if 17 in gps:
        heading = float(gps[17]) % 360

    focal = details.get(FOCAL_35MM)
    if focal:
        fov, source = horizontal_fov(float(focal), width, height), "exif"
    else:
        fov, source = DEFAULT_FOV, "default"

    return Camera(lat, lon, heading, fov, source, width, height)
