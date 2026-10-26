# Handoff

Where we are (26 Oct 2026, ~19:00 SAST, day 1 of the hackathon):

- Idea locked in: photo to real place (see PLAN.md). The user's idea.
- Repo: `D:\Hackathon\placelens`, public on GitHub (github.com/inlighten420/placelens), MIT licence. Name is provisional.
- Day 1 code done: `placelens info` / `placelens find` (GPS, compass, lens width; OSM buildings; view cone with
  hidden buildings removed; map picture; `--json` for agents). Tested on Cape Town City Hall: picked it first.
- No more idea-hunting: PlaceLens it is (26 Oct). Rules and judging are in PLAN.md. No Meta API calls yet: the $150 credits weren't on Billing (card is on the account: calls before the
  credits land get charged).

Next:

1. User: one photo of their own house from the street, location on, into `private/` (gitignored, never public).
2. Run `placelens info` on that photo: does it have a compass direction? (Many Androids don't: `--heading`.)
   Then `placelens find`. Overture fallback for unmapped areas still to do.
   Setup: `python -m venv .venv`, `.venv/Scripts/pip install -e .[dev]`. Overpass is slow at times: be patient.
3. User to-dos: claim the $150 when the link works and check Billing; claim Modal; ask @CV whether SAM 3D Objects
   counts as a Meta open-weight model; pick the demo place (Beacon Isle: own photos, or a Cape Town landmark).
