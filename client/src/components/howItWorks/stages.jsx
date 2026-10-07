/*
FILE PURPOSE:
The content of every stage on the How It Works page, in pipeline order.

Every number here is taken from the code, not from memory; the `code` field of
each stage names where to check it. If you change a threshold, change it here
too — this page is the explanation people read.
*/

import FunnelDiagram from "./FunnelDiagram";
import HybridDiagram from "./HybridDiagram";
import NliDiagram from "./NliDiagram";
import StanceDiagram from "./StanceDiagram";
import WeighDiagram from "./WeighDiagram";
import ExplanationDiagram from "./ExplanationDiagram";

export const STAGES = [
  {
    id: "normalise",
    title: "Clean up the submission",
    summary: "Find the claim inside whatever was pasted.",
    what: (
      <>
        <p>
          People paste what they saw, with the framing they saw it in: “Is it true that…?”,
          “BREAKING:”, a link in front of the headline, “— Reuters, March 2024” after it, emoji and
          hashtags. All of that is removed before anything else reads the text.
        </p>
        <p>
          Nothing that changes the meaning is touched — not “not”, “all”, “only” or “may”. Your
          original words are what the page and your history show; the cleaned-up version is only
          for the machinery.
        </p>
      </>
    ),
    why: "“Is it true that the prime minister of India resigned?” used to be read as a question and never searched. A pasted Twitter link once made “Twitter” the main subject of the search.",
    example: { input: "is it true that the prime minister of india resigned?", output: "the prime minister of india resigned" },
    code: ["claim_normalizer.py"],
  },
  {
    id: "triage",
    title: "Triage: is this even checkable?",
    summary: "Decide what kind of statement it is before any searching.",
    what: (
      <>
        <p>Every submission is sorted into one of these before a single search runs:</p>
        <ul>
          <li><strong>Checkable</strong> — a statement about the world. Goes on to the evidence pipeline.</li>
          <li><strong>Prospective</strong> — about the future (“will”, “plans to”). It is searched, but can only ever be <em>reported as planned</em> or <em>not yet verifiable</em>, never “true”.</li>
          <li><strong>Opinion</strong> — a value judgement (“best”, “should”). Answered as <em>not objectively verifiable</em>, without searching.</li>
          <li><strong>Not a claim</strong> — a question, a fragment, a bare link, keyboard mash. Answered without searching.</li>
          <li><strong>Not English</strong> — reported as out of scope, rather than as nonsense.</li>
        </ul>
        <p>
          Triage also notes two facts used later: whether the claim is <strong>negated</strong>, and its{" "}
          <strong>salience</strong> — would a true version of it necessarily have been reported
          everywhere (a head of state resigning, a country banning a company)?
        </p>
      </>
    ),
    why: "Running “Pineapple is the best pizza topping” through a news search wastes the time budget and then presents the empty result as if the claim had failed a check.",
    example: { input: "Pineapple is the best pizza topping", output: "opinion → not objectively verifiable (nothing searched)" },
    code: ["claim_triage.py"],
  },
  {
    id: "deterministic",
    title: "Known-fact check",
    summary: "A small table of textbook facts is answered instantly.",
    what: (
      <>
        <p>
          Arithmetic and a short list of well-known facts (water freezes at 0 °C, World War II
          ended in 1945, the Great Wall is <em>not</em> visible from the Moon with the naked eye)
          are answered directly, with no search and no model. That is why those claims come back
          in under a second.
        </p>
        <p>
          It only fires on plain statements. “It is false that a triangle has four sides” or
          “Nobody claims WWII ended in 1945” are handed to the evidence pipeline instead, because
          a pattern match cannot read negation or quotation safely.
        </p>
      </>
    ),
    why: "This layer answers with very high confidence and skips all evidence, so a wrong match here would be the most confidently wrong thing the system could say.",
    example: { input: "The Great Wall of China is visible from the Moon with the naked eye", output: "false — instant, no search" },
    code: ["knowledge_verifier.py"],
  },
  {
    id: "understand",
    title: "Understand the claim",
    summary: "Split it into parts and find who did what.",
    what: (
      <>
        <p>
          A long submission can hold several claims; up to three are checked separately. For each,
          the system extracts the <strong>entities</strong> (who or what), the <strong>action</strong>{" "}
          (what is claimed to have happened), negation, attribution (“officials said” vs “officials
          denied”), modality (factual vs speculative) and any time words.
        </p>
        <p>
          A multi-claim statement is only as well supported as its weakest part: the overall verdict
          is conservative, and one failed search marks the whole statement's search as failed.
        </p>
      </>
    ),
    why: "Splitting naïvely on full stops turned “The U.S. government banned Google” into “government banned Google” — the subject deleted before anything was searched. Abbreviations are protected.",
    example: { input: "The EU fined Apple 1.8 billion euros last week. Google launched a new phone in California.", output: "two claims, each searched and judged on its own" },
    code: ["claim_verifier.py (extract_claims)", "claim_decomposer.py"],
  },
  {
    id: "time",
    title: "Pick the time window",
    summary: "Recent news, or no date limit — decided from the wording.",
    what: (
      <>
        <p>
          A claim about something happening now is searched in the <strong>last 30 days</strong>{" "}
          only; one that names a past year or month is searched without a date limit. The choice is
          made automatically from the claim's wording.
        </p>
        <p>
          For a claim about now, an article older than <strong>45 days</strong> cannot confirm it —
          “the prime minister resigned” in a 2014 article is a coincidence of wording, not
          confirmation. It is never turned into a contradiction either: an old article is not
          evidence that today's event did <em>not</em> happen.
        </p>
      </>
    ),
    why: "News feeds rank by recency but do not filter by it, so last year's coverage of the same subject used to confirm this week's claim whenever it matched the words better.",
    example: { input: "The prime minister resigned this morning", output: "recent mode — last 30 days only" },
    code: ["claim_recency.py"],
  },
  {
    id: "search",
    title: "Search six providers at once",
    summary: "Several targeted queries, run in parallel under a time budget.",
    what: (
      <>
        <p>
          Each claim produces up to <strong>four queries</strong>, in order of usefulness: the claim
          as written, the subject plus the action, a quoted version, a fact-check query, and so on.
          They are sent to every provider in parallel:
        </p>
        <ul>
          <li><strong>Google News RSS</strong> and <strong>Wikipedia</strong> — no key needed, on by default</li>
          <li><strong>The Guardian</strong>, <strong>GNews</strong>, <strong>NewsAPI</strong> — when an API key is configured</li>
          <li><strong>DuckDuckGo</strong> — fallback, often rate-limited</li>
        </ul>
        <p>
          The whole evidence phase has a <strong>45-second budget</strong>, shared by every claim in
          the statement, and searching may use at most half of it. A provider that stalls is cut off
          and reported as a timeout. Every provider's outcome is shown under “How this was checked”
          on the result page.
        </p>
      </>
    ),
    why: "A blocked provider used to stall a request for over two minutes. And a failed search must never look like “no evidence” — “we couldn't look” and “we looked and found nothing” are different answers.",
    example: { input: "The US banned Google across all its cities", output: "“The US banned Google across all its cities” · “Google banned” · “Google United States banned” · “\"Google\" \"banned\"”" },
    code: ["query_generator.py", "providers/registry.py"],
  },
  {
    id: "relevance",
    title: "Keep only what is on-topic",
    summary: "Being about the right names is not the same as being about the claim.",
    what: (
      <>
        <p>Every result gets a relevance score built from five signals:</p>
        <ul>
          <li><strong>Entities</strong> mentioned (32%) — are the claim's subjects there?</li>
          <li><strong>Action</strong> (23%) — does it discuss what the claim says happened? “Resigned” matches “steps down”.</li>
          <li><strong>Predicate</strong>, <strong>coherence</strong> and <strong>specificity</strong> (15% each).</li>
        </ul>
        <p>
          Only results scoring at least <strong>0.42</strong> survive. At most <strong>8</strong> are
          then read per claim — and <strong>3 of those seats are held for credible outlets</strong>{" "}
          (fact-checkers, wire services, official sources) if any were found, so a crowd of
          near-identical posts cannot push them out.
        </p>
      </>
    ),
    why: "“Google expands advertising tools in the United States” scored as relevant to “the US is going to ban Google” just because both names appeared. And for a viral false claim, posts repeating it word-for-word outranked the fact-check, which ranked ninth and was never read.",
    diagram: <FunnelDiagram />,
    code: ["relevance_filter.py", "evidence_pipeline.py (RESERVED_TIER_SLOTS)"],
  },
  {
    id: "passages",
    title: "Pick the passages worth reading",
    summary: "Hybrid retrieval: meaning and wording, fused.",
    what: (
      <>
        <p>
          Search snippets are usually too short to judge, so any result under 120 words has its full
          article fetched. Site furniture — paywall pitches, cookie banners, newsletter boxes — is
          stripped out.
        </p>
        <p>
          The NLI model then reads at most <strong>8 passages</strong> per article: the headline, the
          snippet, and the 6 best sentences. “Best” is decided two ways at once:
        </p>
        <ul>
          <li><strong>Word overlap</strong> with the claim — exact matches on names and numbers are the strongest signal there is.</li>
          <li><strong>Meaning</strong> — a small sentence-embedding model (all-MiniLM-L6-v2) scores how close each sentence is to the claim, keeping those above 0.30 similarity.</li>
        </ul>
        <p>
          The two rankings are merged with <strong>reciprocal rank fusion</strong>: each sentence scores
          Σ 1/(60 + rank) over the lists it appears in. No weight between the two has to be tuned, and
          the scores never need to be on the same scale. Embeddings are computed in memory for the
          request and thrown away — there is no vector database.
        </p>
      </>
    ),
    why: "Debunks are paraphrases. “Washington has not prohibited the search giant anywhere in the country” shares no word with “The United States banned Google across all its cities”, so word overlap alone never chose it.",
    diagram: <HybridDiagram />,
    code: ["article_extractor.py", "passage_retriever.py"],
  },
  {
    id: "nli",
    title: "Natural-language inference",
    summary: "A model reads each passage against the claim.",
    what: (
      <>
        <p>
          A DeBERTa-v3 cross-encoder reads each passage (the <em>premise</em>) together with the claim
          (the <em>hypothesis</em>) and outputs three probabilities: the passage <strong>entails</strong>{" "}
          the claim, <strong>contradicts</strong> it, or is <strong>neutral</strong>. This is the only
          model that decides anything about the claim.
        </p>
        <p>
          If the model is unavailable, nothing is classified and no verdict is given — an unavailable
          model is never treated as “neutral”. Its labels are matched by name, never guessed from
          their order, because different checkpoints order them differently.
        </p>
      </>
    ),
    why: "An LLM asked “is this true?” answers from memory and sounds equally sure when wrong. NLI answers a narrower question — does this passage say this? — that can be measured and shown to you.",
    diagram: <NliDiagram />,
    code: ["nli_service.py"],
  },
  {
    id: "stance",
    title: "Each source's position",
    summary: "From passage scores to supports / contradicts / unclear.",
    what: (
      <>
        <p>
          A document's strongest entailment and strongest contradiction are taken separately across
          its passages, and compared using a threshold of <strong>0.35</strong> and a dominance ratio
          of <strong>1.6×</strong>. Then three checks can withdraw the position — never flip it.
        </p>
      </>
    ),
    why: "A fact-check quotes the claim it refutes (“Posts claim the US banned Google…”), and NLI scores that quote as strong support. Reading both scores off the highest-scoring passage once filed PolitiFact debunks as supporting the claim.",
    diagram: <StanceDiagram />,
    code: ["evidence_pipeline.py (decide_stance)", "numeric_consistency.py", "claim_recency.py"],
  },
  {
    id: "weigh",
    title: "Weigh the sources",
    summary: "Credible sources count for more; copies count once.",
    what: (
      <>
        <p>
          Each direction is scored as a weighted average over the sources that take it, weighted by
          source tier. Neutral sources are left out entirely, so background coverage cannot dilute a
          real signal.
        </p>
        <p>
          A direction must have at least <strong>2×</strong> the weighted mass of the other to win
          outright; otherwise the honest answer is <strong>mixed</strong>. And publishers are counted,
          not articles: four reprints of one wire story under one masthead are one confirmation.
          Links via Google News are resolved to the real publisher first.
        </p>
      </>
    ),
    why: "One Reuters article entailing a claim gave “supported”; adding three on-topic articles that said nothing either way used to drag the average down and turn the same evidence into “insufficient evidence”. More evidence, none of it disagreeing, made it less certain.",
    diagram: <WeighDiagram />,
    code: ["evidence_aggregator.py", "claim_verifier.py (classify_source)"],
  },
  {
    id: "verdict",
    title: "Verdict and confidence",
    summary: "Nine distinct answers, each meaning something different.",
    what: (
      <>
        <p>
          The weighed evidence becomes a verdict (see the glossary below). Two special cases are
          decided here:
        </p>
        <p>
          <strong>Silence as evidence.</strong> No outlet writes articles denying things that never
          happened, so a fabricated but huge claim (“Elon Musk bought the Eiffel Tower”) has nothing
          contradicting it. “No credible source reports this” is returned only when <em>all</em> of
          these hold: the search worked; it returned at least 4 candidates; at least 2 providers
          actually answered; nothing supported or contradicted the claim; the claim is high-salience;
          it is not negated; and the NLI model was available.
        </p>
        <p>
          <strong>Future events</strong> are never “supported”: coverage of a plan becomes “reported as
          planned — not yet done”.
        </p>
        <p>
          <strong>Confidence</strong> is a word, not a percentage: high with 3+ independent publishers
          backing the verdict, medium with 2, low with 1. For “no credible source reports this” it
          scales with how many candidates were searched instead.
        </p>
      </>
    ),
    why: "“Insufficient evidence” used to cover everything from a broken search to a fabricated headline. A correct abstention was indistinguishable from a bug.",
    code: ["main.py", "evidence_aggregator.py (assess_coverage)"],
  },
  {
    id: "explain",
    title: "Explanation (display only)",
    summary: "An LLM explains the verdict; NLI checks every sentence.",
    what: (
      <>
        <p>
          After the verdict is final, Gemini Flash is given the verdict and the sources on its side,
          and asked for 2–4 sentences explaining it, each citing a source as [n]. Every sentence is
          then checked: it must cite a source the model was given, and NLI must find that source
          entails it. Only surviving sentences are shown, and each [n] links to its evidence card.
        </p>
        <p>
          The explanation <strong>cannot change the verdict</strong>: it runs last, writes only its
          own field, and nothing reads it back — a test runs the whole system with an LLM told to
          argue the opposite verdict and checks every verdict field is unchanged. If Gemini is down,
          a fallback model is tried; if NLI is down, no explanation is shown at all.
        </p>
      </>
    ),
    why: "A prompt saying “use only the evidence” reduces invented facts; it does not stop them. The filter turns “the model was asked to be faithful” into “every sentence you see was checked against its source”.",
    diagram: <ExplanationDiagram />,
    code: ["explainer.py", "components/ExplanationPanel.jsx"],
  },
];

export const VERDICTS = [
  ["evidence supports the claim", "Credible sources entail it, and outweigh any that contradict it by at least 2×."],
  ["evidence contradicts the claim", "Credible sources contradict it, and outweigh any support by at least 2×."],
  ["claims have mixed evidence", "Credible sources point both ways and neither side dominates."],
  ["reported as planned — not yet done", "A future event that sources report as announced. Confirms the announcement, not the event."],
  ["no credible source reports this", "A high-salience claim that a working search found no coverage of. A finding, under strict conditions — never “false”."],
  ["not yet verifiable", "A future event not reported as announced either."],
  ["subjective — not objectively verifiable", "An opinion. Nothing was searched."],
  ["no verifiable claim found", "A question, fragment, link or non-English text. Nothing was searched."],
  ["insufficient evidence", "A limitation on our side — search failed, or nothing relevant was found, or NLI was unavailable. Never a statement about the claim."],
];

export const PRINCIPLES = [
  ["A search result is not evidence.", "A source counts only after it is relevant and has been read by NLI. The page always separates found, on-topic and checked."],
  ["Unavailable is not neutral.", "If a model or search fails, the system says so and abstains, rather than reporting “no evidence”."],
  ["Checks only withdraw, never flip.", "Different numbers, stale dates and off-topic documents can remove a position; nothing converts support into contradiction."],
  ["Copies are not confirmation.", "Independent publishers are counted, not articles, and confidence scales with them."],
  ["Verdicts come from evidence, never wording.", "Nothing judges a claim from how it is phrased. An earlier wording-only classifier was removed for that reason."],
  ["An LLM explains; it never decides.", "The explanation runs after the verdict and is checked sentence by sentence."],
];

export const LIMITATIONS = [
  "Results depend on what free news sources return today; a rate-limited or blocked provider means thinner evidence (and the page says so).",
  "NLI handles dates poorly: “Russia gained ground in August” can read as contradicting “Russia invaded in February 2022”.",
  "Same kind of event is not the same event: “Crash closes Route 44” can read as contradicting a claim about Route 209.",
  "Short headlines sometimes produce weak search queries, letting loosely related articles in.",
  "English only. Claims in other languages are recognised and declined.",
  "“No credible source reports this” is an inference about coverage, not proof — a very fresh story can still produce it.",
];
