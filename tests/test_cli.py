import json

import pytest

from placelens import osm
from placelens.cli import main
from placelens.view import LocalFrame

from .test_photo import make_photo
from .test_view import square

LAT, LON = -33.9255, 18.4239


@pytest.fixture
def street(tmp_path, monkeypatch):
    """A fake OSM reply cached where `find` looks, so the tests never touch the network."""
    frame = LocalFrame(LAT, LON)
    def way(osm_id, ring, **tags):
        return {"type": "way", "id": osm_id, "tags": {"building": "yes", **tags},
                "geometry": [{"lat": a, "lon": b} for a, b in ring]}
    data = {"elements": [
        way(1, square(frame, 0, 40, 12), name="Town Hall", **{"building:levels": "3"}),
        way(2, square(frame, 0, 80, 8)),  # hidden behind the hall
    ]}
    monkeypatch.setattr(osm, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / f"osm_{LAT:.4f}_{LON:.4f}_200.json").write_text(json.dumps(data))
    return tmp_path


def run(capsys, *argv):
    with pytest.raises(SystemExit) as exit_:
        main(list(argv))
        raise SystemExit(0)
    return exit_.value.code, capsys.readouterr().out


def test_find_json(street, capsys):
    photo = make_photo(street / "p.jpg", heading=0, focal=26)
    code, out = run(capsys, "find", str(photo), "--json", "--out", str(street / "out"))
    assert code == 0
    result = json.loads(out)
    assert [c["name"] for c in result["candidates"]] == ["Town Hall"]
    assert result["candidates"][0]["levels"] == "3"
    assert "OpenStreetMap" in result["attribution"]
    assert (street / "out" / "p_map.png").exists()


def test_find_without_compass_says_how_to_fix(street, capsys):
    photo = make_photo(street / "p.jpg")  # GPS but no compass
    code, out = run(capsys, "find", str(photo), "--json", "--no-map")
    assert code == 2
    assert "--heading" in json.loads(out)["error"]


def test_heading_flag_fills_the_gap(street, capsys):
    photo = make_photo(street / "p.jpg")
    code, out = run(capsys, "find", str(photo), "--heading", "0", "--json", "--no-map")
    assert code == 0 and json.loads(out)["candidates"]


def test_info_json(street, capsys):
    photo = make_photo(street / "p.jpg", heading=90)
    code, out = run(capsys, "info", str(photo), "--json")
    info = json.loads(out)
    assert code == 0 and info["has_gps"] and info["heading"] == 90
