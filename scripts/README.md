# scripts

Probes and evaluations written while building PlaceLens, to answer one question each before writing the real code.
None of them are needed to use PlaceLens. Each one's docstring says how to run it.

| Script | Question it answers | API calls |
|---|---|---|
| [`api_smoke.py`](api_smoke.py) | Does the key work, and what does each Meta model we depend on return and cost? One call each to Muse Spark, SAM 3.1 (with a mask overlay picture), Muse Wonder and Muse Spark Open, timed and priced. `--redraw` re-draws a saved SAM reply for free | Meta API, Baseten |
| [`open_probe.py`](open_probe.py) | What does Muse Spark Open on Baseten support: JSON mode, tool calls, reasoning effort, several photos, parallel calls? Its answers decided how `meta.py` and the survey agent talk to it | Baseten |
| [`model_eval.py`](model_eval.py) | Muse Spark against Muse Spark Open on the same photos: building picked, confidence, floors, distance, side, roof. Reads only saved answers | none |
| [`match_probe.py`](match_probe.py) | Can SIFT matches between walk-round photos (essential matrix, RANSAC) correct the compass? It led to `solve.py`; see [ARCHITECTURE.md §12](../docs/ARCHITECTURE.md#12-what-we-tried-and-dropped) for why it isn't the default | none |
| [`open_usage.py`](open_usage.py) | How much of the Baseten credits has Spark Open used? Reads `out/open_usage.jsonl` | none |
| [`cockpit_shot.gd`](cockpit_shot.gd) | A picture of a car's cockpit view as Open-Planet draws it, to check a steering wheel made by PlaceLens sits right in the car (run with Godot) | none |
