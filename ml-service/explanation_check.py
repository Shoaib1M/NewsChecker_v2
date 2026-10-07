"""
FILE PURPOSE:
Run a handful of varied claims through the real /api/check and report what the
explanation layer did with each: how many sentences the LLM wrote, how many
the NLI faithfulness filter kept and dropped, and why.

    python explanation_check.py                       # the built-in 10 claims
    python explanation_check.py --save out.json
    python explanation_check.py "claim one" "claim two"

WHY THIS EXISTS:
The unit tests prove the filter's LOGIC with a fake NLI model and a fake LLM.
They cannot say how often a real LLM writes sentences a real NLI model rejects,
or whether the sentences it keeps are actually supported — and those two
numbers are the whole case for the feature. This measures the first and prints
every sentence beside its source for a human to judge the second.

REQUIRES live network, a GOOGLE_API_KEY, and the NLI model. Explanations and
dense passage ranking are switched ON here regardless of .env, because
measuring them is the point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

# Varied on purpose: established facts, debunked myths, viral fabrications and
# recent events, so the explainer is exercised on supported, contradicted and
# abstaining verdicts alike.
DEFAULT_CLAIMS = [
    "Elon Musk completed his acquisition of Twitter",
    "India landed a spacecraft near the Moon's south pole",
    "The Great Wall of China is visible from space with the naked eye",
    "5G mobile networks spread the coronavirus",
    "Barack Obama was born in Kenya",
    "Bill Gates put microchips in COVID-19 vaccines",
    "The United States banned Google across all its cities",
    "The Federal Reserve cut interest rates",
    "NASA launched the Artemis II mission around the Moon",
    "Russia invaded Ukraine in February 2022",
]


def run(claims: list[str]) -> list[dict]:
    os.environ["EXPLANATIONS_ENABLED"] = "true"
    os.environ["SEMANTIC_PASSAGES"] = "true"
    from fastapi.testclient import TestClient
    import main

    rows = []
    with TestClient(main.app) as client:
        for index, claim in enumerate(claims, 1):
            started = time.monotonic()
            body = client.post("/api/check", json={"statement": claim}).json()
            seconds = round(time.monotonic() - started, 1)
            explanation = body.get("explanation", {})
            sentences = explanation.get("sentences", [])
            evidence = body.get("top_evidence", [])
            row = {
                "claim": claim,
                "status": body.get("verification", {}).get("status"),
                "seconds": seconds,
                "available": explanation.get("available"),
                "reason": explanation.get("reason"),
                "model": explanation.get("model"),
                "written": len(sentences),
                "kept": sum(1 for s in sentences if s.get("kept")),
                "dropped": sum(1 for s in sentences if not s.get("kept")),
                "sentences": [
                    {
                        **s,
                        "sources": [
                            evidence[n - 1].get("best_sentence", "")
                            for n in s.get("citations", [])
                            if 0 < n <= len(evidence)
                        ],
                    }
                    for s in sentences
                ],
            }
            rows.append(row)
            print(f"\n[{index}/{len(claims)}] {claim}")
            print(f"  status={row['status']}  explanation={'yes' if row['available'] else 'no'}"
                  f"  kept {row['kept']}/{row['written']}  ({seconds}s, {row['model']})")
            if not row["available"]:
                print(f"  reason: {row['reason']}")
            for s in row["sentences"]:
                mark = "KEEP" if s.get("kept") else "DROP"
                print(f"  {mark} e={s.get('entailment', 0):.2f} {s.get('text')}"
                      + (f"   <- {s.get('drop_reason')}" if not s.get("kept") else ""))
                for source in s["sources"]:
                    print(f"         source: {source[:150]}")
    return rows


def summarise(rows: list[dict]) -> dict:
    written = sum(r["written"] for r in rows)
    kept = sum(r["kept"] for r in rows)
    return {
        "claims": len(rows),
        "explained": sum(1 for r in rows if r["available"]),
        "sentences_written": written,
        "sentences_kept": kept,
        "sentences_dropped": written - kept,
        "keep_rate": kept / written if written else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("claims", nargs="*", help="claims to check (default: built-in 10)")
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    rows = run(args.claims or DEFAULT_CLAIMS)
    summary = summarise(rows)
    print("\n" + "=" * 60)
    print(f"  explained {summary['explained']}/{summary['claims']} claims")
    print(f"  sentences kept {summary['sentences_kept']}/{summary['sentences_written']}"
          f"  ({100 * summary['keep_rate']:.0f}%), dropped {summary['sentences_dropped']}")
    if args.save:
        Path(args.save).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"  saved to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
