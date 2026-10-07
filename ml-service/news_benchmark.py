"""
FILE PURPOSE:
Produce an accuracy number for TODAY's news, against today's real headlines,
and state plainly what that number does and does not mean.

    python news_benchmark.py                     # pull today's headlines and score
    python news_benchmark.py --limit 20
    python news_benchmark.py --save results.json
    python news_benchmark.py --from-file run.json   # re-score a saved set

WHY THIS EXISTS:
"How accurate is it?" had no answer. The only number the project had was a
LIAR-dataset accuracy for a wording-only classifier (since removed), which
said nothing about the evidence pipeline that actually decides verdicts.
Nothing measured the thing that matters, so every claim about quality was
intuition.

────────────────────────────────────────────────────────────────────────────
WHERE THE GROUND TRUTH COMES FROM — READ THIS BEFORE QUOTING A NUMBER
────────────────────────────────────────────────────────────────────────────
There is no labelled corpus of today's news, and there cannot be: labelling it
is the task itself. So this uses the one signal available for free — a headline
published today by a mainstream outlet asserts something that outlet believes
happened — and builds the negative class by CORRUPTING those headlines into
statements that are false given the original.

That construction has three honest weaknesses, and the report prints them:

1. A headline is not truth. It is "reported by one outlet". Outlets are wrong,
   and headlines are compressed to the point of distortion. Triage is used to
   drop questions, opinion pieces and non-assertions, but what remains is
   still "reported", not "true".

2. TRUE cases are partly self-confirming. The system retrieves from the same
   news indexes the headlines came from, so confirming one is closer to
   "can it find the article it came from" than "can it establish a fact".
   The confirmation rate is therefore an UPPER bound and is reported as
   recall, never as accuracy.

3. Corrupted headlines are not real misinformation. Real false claims are
   engineered to be plausible and often carry supporting coverage from poor
   sources; a negated headline carries none. Rejecting these is necessary,
   not sufficient.

Because of (2) and (3), the two directions are NEVER averaged into one
"accuracy". They measure different things and fail differently.

THE NUMBER WORTH QUOTING is the wrong-answer rate: how often the system stated
something confidently false — confirmed a corrupted claim, or contradicted a
real headline. Missing a true claim is a shortfall; asserting a false one is a
failure, and only the second is what a fact-checker must not do.

REQUIRES LIVE NETWORK. Like check_providers.py, this cannot run where the news
hosts are unreachable, and it says so rather than reporting a number.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from claim_triage import triage_claim  # noqa: E402
from event_vocabulary import (
    ANTONYMS,
    EVENT_VERBS,
    SURFACE_TO_EVENT,
    surface_forms_in,
)  # noqa: E402
from numeric_consistency import quantities_in  # noqa: E402

GREEN, RED, YELLOW, GREY, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[90m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = GREY = RESET = ""


# ── Building the corpus ──────────────────────────────────────────────

@dataclass
class Case:
    """One claim to check, and what the answer should be."""
    claim: str
    truth: str            # "reported" | "corrupted"
    origin: str           # the headline it came from
    corruption: str = ""  # which corruption produced it
    # Filled in by the run.
    status: str = ""
    verdict: str = ""
    outcome: str = ""     # correct | missed | wrong
    seconds: float = 0.0
    # retrieval.status from the response. Recorded because a provider outage
    # (rate limiting, mostly) turns every case into an abstention, and without
    # this the run reads as a cautious system rather than a broken network —
    # which is exactly how one hybrid run was nearly reported as "0% wrong".
    retrieval: str = ""


def usable_headline(title: str) -> bool:
    """Whether a headline is an assertion this system could check at all.

    Questions, opinion pieces, listicles and live-blog markers are not
    propositions, and scoring the system on them measures nothing. Triage
    already makes this judgement for user input; reusing it keeps the
    benchmark honest about what the system claims to handle.
    """
    if not title or len(title.split()) < 5:
        return False
    if title.rstrip().endswith("?"):
        return False
    if re.match(r"(?i)^(opinion|analysis|comment|editorial|live|watch|video)\b", title):
        return False
    if re.search(r"(?i)\b(here'?s what|what to know|everything we know|explained|"
                 r"best \d+|\d+ best|top \d+|\d+ things|how to|we tried|"
                 r"ranked|deals?)\b", title):
        return False
    triage = triage_claim(title)
    return triage.kind == "checkable" and triage.search_worthwhile


def _swap_surface(text: str, source: str, target: str) -> str | None:
    """Replace one event word with another, preserving the original casing."""
    pattern = re.compile(rf"(?<![a-z]){re.escape(source)}(?![a-z])", re.IGNORECASE)
    if not pattern.search(text):
        return None
    return pattern.sub(target, text, count=1)


def _matching_form(event: str, original: str) -> str:
    """The form of ``event`` that is inflected like ``original``.

    Taking the family's first entry produced "The central bank lower interest
    rates" and "Regulators repeal the merger" — corrupted claims that are
    ungrammatical rather than false. A fact-checker asked to rule on a broken
    sentence is being tested on its parser, and whatever it answers tells you
    nothing about its judgement.
    """
    # Prefer the family's canonical verb ("reject" over "repeal") so the
    # corrupted headline reads naturally: "Regulators repealed the merger" is
    # grammatical but odd, and oddness is a confound.
    forms = sorted(EVENT_VERBS[event], key=lambda f: not f.startswith(event))
    lowered = original.lower()
    if lowered.endswith("ing"):
        suffixes = ("ing",)
    elif lowered.endswith("ed"):
        suffixes = ("ed", "d")
    elif lowered.endswith("s") and not lowered.endswith("ss"):
        suffixes = ("s",)
    else:
        suffixes = ()
    for suffix in suffixes:
        for form in forms:
            if form.endswith(suffix) and " " not in form:
                return form
    return forms[0]


def _base_form(surface: str) -> str | None:
    """The bare infinitive of a past-tense event word, or None if unsure.

    Guessing is not allowed here: an invented stem produces "the minister did
    not collaps", and a benchmark that feeds broken English to the system is
    measuring its parser. Every candidate is therefore checked against the
    event vocabulary, which doubles as a dictionary of words this project
    knows — a stem that isn't in it is rejected rather than emitted.
    """
    lowered = surface.lower()
    if " " in lowered:
        return None
    if lowered in SURFACE_TO_EVENT and not lowered.endswith(("ed", "s")):
        return lowered
    candidates = []
    if lowered.endswith("ied"):
        candidates.append(lowered[:-3] + "y")
    if lowered.endswith("ed"):
        candidates += [lowered[:-2], lowered[:-1]]
    if lowered.endswith("s") and not lowered.endswith("ss"):
        candidates.append(lowered[:-1])
    for candidate in candidates:
        if candidate in SURFACE_TO_EVENT:
            return candidate
    return None


def corrupt(headline: str, rng: random.Random) -> tuple[str, str] | None:
    """Turn a headline into a claim that is FALSE given the headline.

    Returns (corrupted_claim, which_corruption), or None when no corruption
    applies — a headline that cannot be corrupted is skipped rather than
    mangled into something merely ungrammatical, which would test the parser
    rather than the fact-checker.
    """
    strategies = ["antonym", "number", "negate"]
    rng.shuffle(strategies)

    for strategy in strategies:
        if strategy == "antonym":
            # The strongest corruption: same subject, opposite event, so the
            # coverage that exists says the reverse of the claim.
            #
            # Only the headline's FIRST event is eligible. Iterating the whole
            # vocabulary picked an arbitrary one, and corrupting a subordinate
            # clause leaves the claim substantially true — "India's PM
            # resigned after coalition talks collapsed" became "…after
            # coalition talks launch", which a correct system SHOULD still
            # confirm, because the PM did resign. Scoring that as a wrong
            # answer would blame the pipeline for a bad label.
            surfaces = surface_forms_in(headline)
            if surfaces:
                main = surfaces[0]
                opposite_event = ANTONYMS.get(SURFACE_TO_EVENT[main])
                if opposite_event:
                    swapped = _swap_surface(
                        headline, main, _matching_form(opposite_event, main))
                    if swapped and swapped.lower() != headline.lower():
                        return swapped, "antonym"

        elif strategy == "number":
            quantities = quantities_in(headline)
            if quantities:
                original = quantities[0].text
                digits = re.search(r"[\d,.]+", original)
                if digits:
                    try:
                        value = float(digits.group(0).replace(",", ""))
                    except ValueError:
                        value = 0
                    if value:
                        # A large multiple, so the result is unambiguously a
                        # different figure rather than a rounding difference.
                        changed = original.replace(
                            digits.group(0),
                            f"{value * 7:,.0f}" if value >= 1 else "0.5")
                        swapped = headline.replace(original, changed, 1)
                        if swapped != headline:
                            return swapped, "number"

        elif strategy == "negate":
            # Weakest of the three — a negated headline has no coverage at
            # all, so rejecting it mostly tests absence reasoning rather than
            # the system's reading of evidence.
            match = re.search(
                r"(?i)\b(was|were|is|are|has|have|will)\b", headline)
            if match:
                return (headline[:match.end()] + " not"
                        + headline[match.end():]), "negate"
            # No auxiliary to hang "not" on, so rebuild around the main verb:
            # "the minister resigned" -> "the minister did not resign".
            surfaces = surface_forms_in(headline)
            # Only an INFLECTED surface was used as a verb here. Several
            # families are largely nouns — "earthquake", "flood", "merger" —
            # and negating one produced "a magnitude 7 did not earthquake
            # struck northern Japan". A headline that cannot be negated
            # cleanly is skipped, not mangled.
            if surfaces and re.search(r"(?<!s)s$|ed$", surfaces[0].lower()):
                base = _base_form(surfaces[0])
                if base:
                    swapped = _swap_surface(headline, surfaces[0], f"did not {base}")
                    if swapped and swapped.lower() != headline.lower():
                        return swapped, "negate"
    return None


def build_cases(headlines: list[str], rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    for headline in headlines:
        cases.append(Case(claim=headline, truth="reported", origin=headline))
        corrupted = corrupt(headline, rng)
        if corrupted:
            claim, strategy = corrupted
            cases.append(Case(claim=claim, truth="corrupted",
                              origin=headline, corruption=strategy))
    return cases


# ── Scoring ──────────────────────────────────────────────────────────

# What each verdict means for each class of claim. The distinction that
# matters is between being WRONG and being SHORT: "I could not establish this"
# is a shortfall, "this is confirmed" about a false claim is a failure.
CORRECT_FOR_REPORTED = {"supported"}
WRONG_FOR_REPORTED = {"contradicted", "unsupported_no_coverage"}
WRONG_FOR_CORRUPTED = {"supported"}
CORRECT_FOR_CORRUPTED = {"contradicted", "unsupported_no_coverage"}


def classify_outcome(case: Case) -> str:
    if case.truth == "reported":
        if case.status in CORRECT_FOR_REPORTED:
            return "correct"
        if case.status in WRONG_FOR_REPORTED:
            return "wrong"
        return "missed"
    if case.status in WRONG_FOR_CORRUPTED:
        return "wrong"
    if case.status in CORRECT_FOR_CORRUPTED:
        return "correct"
    return "missed"


# A verdict that commits to an answer about the claim. Everything else —
# insufficient evidence, mixed, a future event, not a claim — is the system
# declining to answer, which is what the abstention rate counts. Mixed counts
# as an abstention on purpose: it asserts neither direction.
COMMITTED_STATUSES = {"supported", "contradicted", "unsupported_no_coverage"}
SEARCH_FAILURES = {"SEARCH_FAILED", "NO_RESULTS"}


def summarise(cases: list[Case]) -> dict:
    def tally(subset):
        return {
            "n": len(subset),
            "correct": sum(1 for c in subset if c.outcome == "correct"),
            "missed": sum(1 for c in subset if c.outcome == "missed"),
            "wrong": sum(1 for c in subset if c.outcome == "wrong"),
        }

    reported = [c for c in cases if c.truth == "reported"]
    corrupted = [c for c in cases if c.truth == "corrupted"]
    wrong = sum(1 for c in cases if c.outcome == "wrong")
    abstained = sum(1 for c in cases if c.status not in COMMITTED_STATUSES)
    return {
        "reported": tally(reported),
        "corrupted": tally(corrupted),
        "total": len(cases),
        "wrong_answers": wrong,
        "wrong_answer_rate": wrong / len(cases) if cases else 0.0,
        # Recall over real headlines — an UPPER bound, see the module docstring.
        "confirmation_recall": (
            sum(1 for c in reported if c.outcome == "correct") / len(reported)
            if reported else 0.0
        ),
        # How often no committed verdict was given at all. Reported beside the
        # wrong-answer rate because either can be bought with the other: a
        # system that never answers is never wrong.
        "abstention_rate": abstained / len(cases) if cases else 0.0,
        "mean_seconds": sum(c.seconds for c in cases) / len(cases) if cases else 0.0,
        # Cases whose search never ran. Their verdicts measure the network,
        # not the pipeline; a run with many of these is not comparable.
        "search_failures": sum(1 for c in cases if c.retrieval in SEARCH_FAILURES),
    }


# ── Running ──────────────────────────────────────────────────────────

def fetch_headlines(limit: int, topic: str) -> list[str]:
    """Today's headlines, straight from the news index."""
    from providers import google_news

    seen: set[str] = set()
    headlines: list[str] = []
    # Several queries, because one topic's coverage is correlated: a batch of
    # twelve headlines about the same story would measure one retrieval, not
    # twelve. `when:1d` keeps this to today.
    for query in (topic, "world news", "business", "politics", "technology"):
        try:
            results = google_news.search(query, max_results=12, recent_days=1)
        except Exception as error:
            raise SystemExit(
                f"\nCould not reach Google News: {error}\n"
                "This benchmark needs live network access. Run "
                "`python check_providers.py` first — if that fails too, the "
                "problem is reachability, not this script.\n"
            )
        for result in results:
            title = result.title.strip()
            key = title.lower()
            if key in seen or not usable_headline(title):
                continue
            seen.add(key)
            headlines.append(title)
            if len(headlines) >= limit:
                return headlines
    return headlines


def run_cases(cases: list[Case], mode: str) -> None:
    """Check every case through the real API, in process."""
    from fastapi.testclient import TestClient
    import main

    with TestClient(main.app) as client:
        for index, case in enumerate(cases, 1):
            started = time.monotonic()
            try:
                response = client.post(
                    "/api/check", json={"statement": case.claim, "mode": mode})
                body = response.json()
                case.status = body.get("verification", {}).get("status", "error")
                case.verdict = body.get("verdict", "")
                case.retrieval = body.get("retrieval", {}).get("status", "")
            except Exception as error:  # noqa: BLE001 - a failed case is data
                case.status = "error"
                case.verdict = str(error)[:80]
            case.seconds = round(time.monotonic() - started, 1)
            case.outcome = classify_outcome(case)

            mark = {"correct": f"{GREEN}ok  {RESET}",
                    "missed": f"{YELLOW}miss{RESET}",
                    "wrong": f"{RED}WRONG{RESET}"}[case.outcome]
            print(f"  [{index:>3}/{len(cases)}] {mark} {case.truth:<9} "
                  f"{case.status:<24} ({case.seconds:>4.1f}s) {case.claim[:56]}")


def report(cases: list[Case], summary: dict) -> None:
    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)

    reported, corrupted = summary["reported"], summary["corrupted"]

    def pct(part, whole):
        return f"{100 * part / whole:5.1f}%" if whole else "    —"

    print(f"\n  Real headlines ({reported['n']})            "
          f"— can it confirm what was actually reported?")
    print(f"    confirmed          {reported['correct']:>3}   {pct(reported['correct'], reported['n'])}"
          f"   {GREY}recall, and an upper bound — see below{RESET}")
    print(f"    not established    {reported['missed']:>3}   {pct(reported['missed'], reported['n'])}"
          f"   {GREY}a shortfall, not a false statement{RESET}")
    print(f"    {RED}stated as false    {reported['wrong']:>3}   {pct(reported['wrong'], reported['n'])}"
          f"   a wrong answer{RESET}")

    print(f"\n  Corrupted headlines ({corrupted['n']})       "
          f"— does it refuse what the coverage contradicts?")
    print(f"    rejected           {corrupted['correct']:>3}   {pct(corrupted['correct'], corrupted['n'])}")
    print(f"    not established    {corrupted['missed']:>3}   {pct(corrupted['missed'], corrupted['n'])}"
          f"   {GREY}refused, but without finding the refutation{RESET}")
    print(f"    {RED}confirmed as true  {corrupted['wrong']:>3}   {pct(corrupted['wrong'], corrupted['n'])}"
          f"   a wrong answer{RESET}")

    rate = summary["wrong_answer_rate"]
    colour = GREEN if rate < 0.05 else YELLOW if rate < 0.15 else RED
    print(f"\n  {colour}WRONG-ANSWER RATE   {summary['wrong_answers']}/{summary['total']}"
          f"   {100 * rate:.1f}%{RESET}"
          f"   {GREY}confident statements that were false{RESET}")
    print(f"  ABSTENTION RATE     {100 * summary['abstention_rate']:.1f}%"
          f"   {GREY}no committed verdict either way{RESET}")
    print(f"  MEAN LATENCY        {summary['mean_seconds']:.1f}s per check")
    failures = summary.get("search_failures", 0)
    if failures:
        print(f"  {RED if failures > summary['total'] * 0.2 else YELLOW}"
              f"SEARCH FAILURES     {failures}/{summary['total']}{RESET}"
              f"   {GREY}cases where no provider answered — if this is more than"
              f" a few, the run measures the network, not the pipeline;"
              f" re-run it{RESET}")

    wrong = [c for c in cases if c.outcome == "wrong"]
    if wrong:
        print(f"\n  Every wrong answer, for inspection:")
        for case in wrong:
            print(f"    {RED}{case.truth:<9}{RESET} {case.status:<24} {case.claim[:60]}")
            if case.truth == "corrupted":
                print(f"      {GREY}corrupted by {case.corruption}, from: "
                      f"{case.origin[:56]}{RESET}")

    print(f"""
  {GREY}HOW TO READ THIS
  The two halves are not averaged, because they measure different things.

  The confirmation rate is an UPPER bound: these headlines came from the same
  news indexes the system searches, so confirming one is closer to "can it
  find the article it came from" than "can it establish a fact".

  Rejecting corrupted headlines is necessary but not sufficient. Real
  misinformation is built to be plausible and often carries supporting
  coverage from poor sources; a corrupted headline carries none.

  The wrong-answer rate is the number worth quoting. Missing a true claim is
  a shortfall; asserting a false one is the failure a fact-checker must not
  make.{RESET}
""")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the pipeline on today's news.")
    parser.add_argument("--limit", type=int, default=12,
                        help="headlines to pull (each yields up to 2 cases)")
    parser.add_argument("--topic", default="breaking news",
                        help="first search topic; others are added for spread")
    parser.add_argument("--mode", default="recent",
                        choices=["auto", "recent", "historical"])
    parser.add_argument("--seed", type=int, default=0,
                        help="corruption RNG seed, for reproducible runs")
    parser.add_argument("--save", metavar="PATH", help="write results as JSON")
    parser.add_argument("--from-file", metavar="PATH",
                        help="re-score a saved run instead of fetching")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.from_file:
        saved = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
        headlines = [c["origin"] for c in saved["cases"] if c["truth"] == "reported"]
        print(f"\nRe-scoring {len(headlines)} saved headlines.\n")
    else:
        print(f"\nPulling today's headlines…")
        headlines = fetch_headlines(args.limit, args.topic)
        if len(headlines) < 4:
            raise SystemExit(
                f"\nOnly {len(headlines)} usable headlines came back. Too few to "
                "measure anything. Check `python check_providers.py`.\n")
        print(f"{len(headlines)} usable headlines "
              f"(questions, opinion and listicles dropped by triage).\n")

    cases = build_cases(headlines, rng)
    corrupted = sum(1 for c in cases if c.truth == "corrupted")
    print(f"{len(cases)} cases: {len(headlines)} as reported, "
          f"{corrupted} corrupted into falsehoods.\n")

    run_cases(cases, args.mode)
    summary = summarise(cases)
    report(cases, summary)

    if args.save:
        # UTF-8 explicitly: the Windows default codepage cannot encode every
        # headline, and a crash here loses the whole run.
        Path(args.save).write_text(encoding="utf-8", data=json.dumps(
            {"summary": summary, "mode": args.mode, "seed": args.seed,
             "cases": [asdict(c) for c in cases]}, indent=2))
        print(f"  Saved to {args.save}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
