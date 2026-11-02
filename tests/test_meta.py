from pathlib import Path

import pytest

from placelens.meta import MissingKeyError, data_url, final_answer, load_key, parse_segments

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


def test_named_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("MODEL_API_KEY=abc\nBASETEN_API_KEY=xyz\n", encoding="utf-8")
    assert load_key(env, "BASETEN_API_KEY") == "xyz"


def test_final_answer_drops_reasoning():
    assert final_answer('<think>front, so...</think>\n{"side": "back"}') == '{"side": "back"}'
    assert final_answer("<think>cut off mid-thought") == ""
    assert final_answer("plain answer") == "plain answer"


def test_paid_calls_can_be_blocked(monkeypatch):
    from placelens.meta import SPARK, MetaClient, PaidCallBlocked

    monkeypatch.setenv("PLACELENS_META_OFF", "1")
    meta = MetaClient("not-a-key")
    with pytest.raises(PaidCallBlocked):
        meta.ask("hi", model=SPARK)
    with pytest.raises(PaidCallBlocked):
        meta.segment("x.jpg")


def test_usage_since_totals_both_logs(tmp_path, monkeypatch):
    import json

    from placelens import meta

    monkeypatch.setattr(meta, "META_USAGE", tmp_path / "meta.jsonl")
    monkeypatch.setattr(meta, "OPEN_USAGE", tmp_path / "open.jsonl")
    (tmp_path / "meta.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"time": 50, "what": "sam-3.1", "usd": 1.0},  # before: not counted
        {"time": 100, "what": "sam-3.1", "usd": 0.0025}, {"time": 101, "what": "muse-image-1.0", "usd": 0.01}]))
    (tmp_path / "open.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"time": 100, "in": 1000, "out": 200, "secs": 4.2}, {"time": 102, "in": 500, "out": 50}]) + "\nnot json\n")
    assert meta.usage_since(100) == {"meta_usd": 0.0125, "meta_calls": 2, "open_calls": 2, "open_tokens": 1750,
                                     "open_slowest_s": 4.2}


def test_paid_calls_are_logged(tmp_path, monkeypatch):
    from placelens import meta

    monkeypatch.setattr(meta, "META_USAGE", tmp_path / "meta.jsonl")
    client = meta.MetaClient("not-a-key")
    client._add(meta.SAM_PER_IMAGE, what=meta.SAM)
    client._add(tokens=500)  # Spark Open tokens: free, not in the paid log
    lines = (tmp_path / "meta.jsonl").read_text().splitlines()
    assert len(lines) == 1 and '"sam-3.1"' in lines[0]


def test_progress_steps_go_to_the_file_named_in_the_environment(tmp_path, monkeypatch):
    import json

    from placelens.progress import ENV, Counter, step

    monkeypatch.delenv(ENV, raising=False)
    step("nowhere")  # no file asked for: nothing happens
    monkeypatch.setenv(ENV, str(tmp_path / "p.jsonl"))
    step("cutting out the bear in photo 3 of 9", 2, 9)
    Counter("finding the building in photo", 2).tick()
    events = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["detail"].startswith("cutting") and (events[0]["count"], events[0]["total"]) == (2, 9)
    assert "done" not in events[0]  # (a game reads "done": true as the end of the make)
    assert [e["detail"] for e in events[1:]] == ["finding the building in photo 0 of 2",
                                                 "finding the building in photo 1 of 2"]
