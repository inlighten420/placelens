from pathlib import Path

import pytest

from placelens.meta import MissingKeyError, data_url, load_key, parse_segments

# SAM 3.1's real reply for "building" on a 1280x891 photo of Cape Town City Hall (26 Oct 2026).
RAW = (Path(__file__).parent / "data" / "sam_city_hall.txt").read_text(encoding="utf-8")


def test_parses_real_sam_reply_biggest_first():
    segments = parse_segments(RAW)
    assert [s.object_id for s in segments] == ["1", "0"]
    hall = segments[0]
    assert hall.box == (0, 57, 1251, 891)
    # SAM sent this mask scaled down; it comes back full size over its box.
    assert hall.mask.shape == (891 - 57, 1251)
    assert 0.4 < hall.mask.mean() < 0.7  # a building, not the whole box: the sky is cut away


def test_full_mask_places_the_segment_on_the_photo():
    small = parse_segments(RAW)[1]  # the building at the right edge
    canvas = small.full_mask((1280, 891))
    assert canvas.shape == (891, 1280)
    assert canvas.sum() == small.area
    assert not canvas[:, :1161].any()


def test_empty_reply_means_no_matches():
    assert parse_segments("") == []


def test_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('OTHER=1\nMODEL_API_KEY="abc123"\n', encoding="utf-8")
    assert load_key(env) == "abc123"
    with pytest.raises(MissingKeyError):
        load_key(tmp_path / "missing.env")


def test_data_url_mime(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG")
    assert data_url(p).startswith("data:image/png;base64,")
