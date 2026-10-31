"""How much of the Baseten credits Muse Spark Open has used, from out/open_usage.jsonl (written by every call).

    .venv/Scripts/python scripts/open_usage.py
"""

from __future__ import annotations

import json
from collections import Counter

from placelens.meta import OPEN_USAGE


def main():
    if not OPEN_USAGE.exists():
        print(f"no calls logged yet ({OPEN_USAGE})")
        return
    calls = [json.loads(line) for line in OPEN_USAGE.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokens_in, tokens_out = sum(c["in"] for c in calls), sum(c["out"] for c in calls)
    print(f"{len(calls)} calls: {tokens_in:,} tokens in, {tokens_out:,} out (reasoning included)")
    for effort, n in Counter(c["effort"] or "default" for c in calls).most_common():
        print(f"  effort {effort}: {n} calls")
    print("Prices: see the Baseten dashboard (Usage); the hackathon gave $100 in credits.")


if __name__ == "__main__":
    main()
