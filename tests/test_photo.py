from fractions import Fraction

from PIL import Image

from placelens.photo import DEFAULT_FOV, EXIF_IFD, FOCAL_35MM, GPS_IFD, horizontal_fov, read_camera


def dms(value):
    d = int(value)
    m = int((value - d) * 60)
    s = (value - d - m / 60) * 3600
    return (Fraction(d), Fraction(m), Fraction(s).limit_denominator(10000))


def make_photo(path, size=(400, 300), gps=True, heading=None, focal=None):
    img = Image.new("RGB", size, "grey")
    exif = Image.Exif()
    if gps:
        g = exif.get_ifd(GPS_IFD)
        g[1], g[2] = "S", dms(33.9255)
        g[3], g[4] = "E", dms(18.4239)
        if heading is not None:
            g[16], g[17] = "T", Fraction(heading).limit_denominator(100)
    if focal:
        exif.get_ifd(EXIF_IFD)[FOCAL_35MM] = focal
    img.save(path, exif=exif)
    return path


def test_reads_gps_and_compass(tmp_path):
    cam = read_camera(make_photo(tmp_path / "a.jpg", heading=123.5, focal=26))
    assert abs(cam.lat + 33.9255) < 1e-4
    assert abs(cam.lon - 18.4239) < 1e-4
    assert abs(cam.heading - 123.5) < 1e-6
    assert cam.fov_source == "exif"
    assert 60 < cam.fov < 75


def test_missing_data_is_none_and_default_lens(tmp_path):
    cam = read_camera(make_photo(tmp_path / "b.jpg", gps=False))
    assert cam.lat is None and cam.heading is None
    assert cam.fov == DEFAULT_FOV and cam.fov_source == "default"


def test_portrait_sees_narrower():
    assert horizontal_fov(26, 3000, 4000) < horizontal_fov(26, 4000, 3000)
