# Handoff

Where we are (26 Oct 2026, ~19:00 SAST, day 1 of the hackathon):

- Idea locked in: photo to real place (see PLAN.md). The user's idea.
- Repo: `D:\Hackathon\placelens`, local git, nothing committed, **not public yet**. Name is provisional.
- No code yet. No Meta API calls yet: the $150 credits weren't on Billing (card is on the account: calls before the
  credits land get charged).

Next:

1. User: one photo of their own house from the street, location on, into `private/` (gitignored, never public).
2. Day 1 code (no AI): read the photo's GPS/compass/focal length, fetch OSM buildings (Overture as backup), the
   view cone, the candidates, a small map picture to see it. Python CLI, `placelens --help`.
3. User to-dos: claim the $150 when the link works and check Billing; claim Modal; ask @CV whether SAM 3D Objects
   counts as a Meta open-weight model; pick the demo place (Beacon Isle: own photos, or a Cape Town landmark).
