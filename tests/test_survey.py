import json
from pathlib import Path

from placelens.survey import Survey, _brief_build, groups


def _call(i: int, tool: str, **args) -> dict:
    return {"id": f"c{i}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}


def test_groups_split_by_place_and_time():
    photos = [
        {"photo": "a1", "lat": -34.0531, "lon": 23.3461, "time": 0},
        {"photo": "a2", "lat": -34.0532, "lon": 23.3461, "time": 60},
        {"photo": "b1", "lat": -34.0600, "lon": 23.3461, "time": 120},  # ~770 m away
        {"photo": "a3", "lat": -34.0531, "lon": 23.3461, "time": 7200},  # same spot, two hours later
    ]
    assert groups(photos) == [["a1", "a2"], ["b1"], ["a3"]]


def test_the_agent_runs_tools_and_keeps_what_it_built(tmp_path):
    script = [
        {"role": "assistant", "content": "", "tool_calls": [_call(1, "describe", photo="p1.jpg")]},
        {"role": "assistant", "content": "", "tool_calls": [_call(2, "build", photos=["p1.jpg", "p2.jpg"],
                                                                  refine=True),
                                                            _call(3, "ask_person", question="Which house is p3?")]},
        {"role": "assistant", "content": "", "tool_calls": [_call(4, "pack", name="street",
                                                                  folders=["out/build/way_1"])]},
        {"role": "assistant", "content": "Built the house; p3 needs you."},
    ]
    seen_tools, ran = [], []

    def chat(messages, tools):
        seen_tools.append(len(tools))
        return script.pop(0)

    def run(argv):
        ran.append(argv)
        if argv[0] == "build":
            return {"building": {"osm_id": "way/1"}, "model": "out/build/way_1/building.glb", "size_m": [10, 8],
                    "height_m": 6, "roof": {"kind": "gable"},
                    "photos": [{"photo": "p1.jpg", "used": True, "why": "ok"},
                               {"photo": "p2.jpg", "used": False, "why": "compass off"}],
                    "refined": [{"error": 0.08, "kept": []}, {"error": 0.07, "kept": [{"edit": "eaves up 0.5 m"}]}]}
        return {"pack": "out/pack/street", "items": 1}

    survey = Survey(tmp_path, chat, run, lambda p: '{"kind": "building"}', tmp_path / "log", log=lambda *_: None)
    assert survey.go() == "Built the house; p3 needs you."
    work = str(tmp_path / "log" / "work")
    assert ran[0] == ["build", "p1.jpg", "p2.jpg", "--json", "--out", work, "--replies", str(Path("out/pick")),
                      "--refine"]
    assert ran[1] == ["pack", "street", "out/build/way_1", "--json", "--out", work]
    assert survey.questions == ["Which house is p3?"]
    assert survey.built[0][:2] == ("building", str(Path("out/build/way_1")))
    tool_results = [json.loads(m["content"]) for m in survey.messages if m["role"] == "tool"]
    assert tool_results[1]["photos_not_used"] == {"p2.jpg": "compass off"}
    assert tool_results[1]["refined"]["kept"] == ["eaves up 0.5 m"]
    summary = (tmp_path / "log" / "summary.md").read_text(encoding="utf-8")
    assert "Which house is p3?" in summary and "Built the house" in summary
    assert (tmp_path / "log" / "transcript.json").exists()


def test_a_failing_tool_is_reported_to_the_agent(tmp_path):
    script = [{"role": "assistant", "content": "", "tool_calls": [_call(1, "pick", photo="x.jpg"),
                                                                  _call(2, "fly", height=3)]},
              {"role": "assistant", "content": "done"}]

    def run(argv):
        raise RuntimeError("boom")

    survey = Survey(tmp_path, lambda m, t: script.pop(0), run, lambda p: "", tmp_path / "log", log=lambda *_: None)
    survey.go()
    results = [json.loads(m["content"]) for m in survey.messages if m["role"] == "tool"]
    assert results == [{"error": "RuntimeError: boom"}, {"error": "no tool 'fly'"}]


def test_the_agent_is_stopped_after_its_heavy_steps(tmp_path):
    def chat(messages, tools):
        if not tools:
            return {"role": "assistant", "content": "stopped"}
        n = len(messages)  # a cheap look and a heavy pick each turn: only the picks count
        return {"role": "assistant", "content": "", "tool_calls": [_call(n, "describe", photo="a.jpg"),
                                                                   _call(n + 1, "pick", photo="a.jpg")]}

    ran = []
    survey = Survey(tmp_path, chat, lambda a: ran.append(a) or {}, lambda p: "", tmp_path / "log",
                    log=lambda *_: None)
    assert survey.go(max_steps=3) == "stopped"
    assert len(ran) == 3


def test_brief_build_passes_errors_through():
    assert _brief_build({"error": "not sure which building"}) == {"error": "not sure which building"}


def test_what_was_built_is_packed_even_if_the_agent_forgets(tmp_path):
    script = [{"role": "assistant", "content": "", "tool_calls": [_call(1, "prop", photos=["t.jpg"], what="tank")]},
              {"role": "assistant", "content": "Built a tank."}]
    ran = []

    def run(argv):
        ran.append(argv)
        return {"name": "tank", "model": "w/props/t/prop.glb"} if argv[0] == "prop" else {"items": 1}

    survey = Survey(tmp_path / "photos", lambda m, t: script.pop(0), run, lambda p: "", tmp_path / "log",
                    log=lambda *_: None)
    survey.go()
    assert ran[-1][:3] == ["pack", "photos", str(Path("w/props/t"))]
    assert survey.pack == "photos"


def test_looks_are_limited(tmp_path):
    from placelens.survey import LOOKS

    survey = Survey(tmp_path, None, None, lambda p: "a house", tmp_path / "log", log=lambda *_: None)
    answers = [survey.call("describe", {"photo": "a.jpg"}) for _ in range(LOOKS + 1)]
    assert "error" not in answers[LOOKS - 1] and "error" in answers[LOOKS]


def test_the_persons_notes_brief_the_agent_and_become_settings(tmp_path):
    script = [
        {"role": "assistant", "content": "", "tool_calls": [
            _call(1, "build", photos=["h.jpg"], symmetric=True, floors=2, notes="back is like the front"),
            _call(2, "prop", photos=["t.jpg"], what="water tank", round=True, height=1.8, notes="1.8 m tall")]},
        {"role": "assistant", "content": "Done."},
    ]
    briefs, ran, events = [], [], []

    def chat(messages, tools):
        briefs.append(messages[1]["content"])
        return script.pop(0)

    def run(argv):
        ran.append(argv)
        return {"error": "no"} if argv[0] == "build" else {"name": "tank", "model": "w/props/t/prop.glb"}

    survey = Survey(tmp_path, chat, run, lambda p: "", tmp_path / "log", log=lambda *_: None,
                    notes="The house is symmetrical; the tank is 1.8 m tall.", progress=events.append)
    survey.go()
    assert "symmetrical" in briefs[0]
    assert ran[0][-5:] == ["--floors", "2", "--notes", "back is like the front", "--symmetric"]
    assert ran[1][ran[1].index("--what"):] == ["--what", "water tank", "--height", "1.8", "--notes", "1.8 m tall",
                                               "--shape", "round"]
    assert events[0] == {"tool": "build", "about": "1 photo(s), symmetric=True, floors=2", "heavy": True}
    assert events[1] == {"tool": "build", "ok": False, "error": "no"}
    assert events[3]["ok"] is True
    assert "symmetrical" in (tmp_path / "log" / "summary.md").read_text(encoding="utf-8")


def test_the_agent_hears_whether_mirroring_happened():
    told = _brief_build({"model": "w/b/building.glb", "photos": [], "symmetric": "asked, but nothing was mirrored"})
    assert told["symmetric"] == "asked, but nothing was mirrored"
