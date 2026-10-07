/*
FILE PURPOSE:
An interactive "How it Works" page that acts as a technical blog post or whitepaper.
It explains the hybrid pipeline from claim understanding to evidence-aware results.

FLOW:
1. Defines the pipeline steps in a large data array (`PIPELINE_STEPS`).
2. Renders the interactive step buttons (Accordion UI).
3. Conditionally renders the detail panel when a step is clicked.
4. Renders a custom SVG architecture diagram of the Neural Network.
5. Explains the final mathematical scoring formula.

WHY THIS EXISTS:
This serves as the "Documentation" for the project, directly integrated into the app.
It shows employers or users exactly how much thought went into the system's design.
*/

import { useState } from "react";
import {
  Package,
  FileText,
  Filter,
  Brain,
  CheckCircle2,
  Globe,
  Target,
  Layers,
  Calculator,
  Code2,
  MessageSquareCheck,
} from "lucide-react";

// Final architecture for the end-to-end fact-checking pipeline.
const PIPELINE_STEPS = [
  {
id: "reading-the-submission",
icon: <Package className="hiw-step-icon" />,
title: "Reading the submission",
short: "Find the proposition inside what was actually pasted",
detail: `People don't submit propositions. They submit what they saw, with the framing they saw it in: "is it true that…?", a headline in quotes with "- Reuters, March 2024" after it, a pasted link in front of the sentence, emoji and hashtags around it.

Every stage after this one reads the claim — including the entailment model, which uses it as its hypothesis — so that packaging used to reach all of them. A link made the site it pointed at into one of the claim's entities; a source credit did the same for the publisher's name; and "is it true that the prime minister of India resigned?" was classified as not a claim and never searched at all.

We strip the packaging and keep the proposition. What the claim asserts is never touched: negation, hedges and quantifiers all change the meaning, so "did not", "may", "all" and "only" survive exactly as written, and you always see your own words back rather than our cleaned-up version.`,
  },
  {
id: "claim-triage",
icon: <Filter className="hiw-step-icon" />,
title: "Claim triage",
short: "Decide what kind of question the claim even poses",
detail: `Before any searching, we work out whether there is something here that evidence could settle. An open question, an unparseable string, or a value judgment is reported as such and never searched — telling someone "we couldn't verify this" when there was no proposition to verify is misleading. ("Why did the prime minister resign?" is a real question and is refused; "is it true that the prime minister resigned?" is a claim in a question's clothing, and is checked.)

A claim about a future event is marked as such: nothing can make it true or false today, so the most that can be established is whether it has been announced. We also judge whether a true version of the claim would necessarily have been reported, which is what later licenses treating an absence of coverage as meaningful.`,
  },
  {
id: "claim-understanding",
icon: <Package className="hiw-step-icon" />,
title: "Claim understanding",
short: "Identify what the claim is actually saying",
detail: `We parse the claim into its subject, action, object, timeline, qualifiers, and any negation or attribution.

This matters because a claim like "government X banned platform Y" is not the same as a vague mention of X and Y in the same article. The system keeps the relationship between the actor, the action, and the target object so retrieval stays focused on the actual proposition.`,
  },
  {
id: "query-generation",
icon: <FileText className="hiw-step-icon" />,
title: "Targeted search",
short: "Search for the actual story, not generic keywords",
detail: `We generate multiple query variants for each claim: the exact headline, a normalized wording, subject + action + object wording, entity + event queries, contradiction searches, and date/location variants where relevant.

This is designed to find the original report, same-event coverage, primary sources, and contradiction checks without drifting into unrelated articles that merely share a few words.`,
  },
  {
id: "retrieval",
icon: <Globe className="hiw-step-icon" />,
title: "High-recall retrieval",
short: "Collect likely candidates from multiple providers",
detail: `The system queries live search providers and normalizes each result before filtering.

We track provider status, raw results, normalized results, and candidate-level diagnostics so a failed provider or empty provider is never silently treated as a substantive truth signal.`,
  },
  {
id: "relevance",
icon: <Target className="hiw-step-icon" />,
title: "Relevance filtering",
short: "Reject weak or unrelated matches",
detail: `Candidates are scored by entity match, action match, predicate overlap, coherence, and keyword specificity. Action match is what separates an article about the right subjects from one about the right event: for the claim "the US is going to ban Google", a story headlined "Google expands advertising tools in the United States" mentions both entities and reports nothing about a ban.

Matching accepts the claim's action or its opposite, because a source that refutes a claim describes the opposite outcome in its own words and would otherwise be filtered out before it could be read.

The thresholds are measured against a labelled set rather than chosen by feel, and are deliberately tuned for recall: a rejected document is gone for good, while one that passes still has to be classified before it counts as evidence.`,
  },
  {
id: "nli",
icon: <Brain className="hiw-step-icon" />,
title: "Evidence reading and NLI",
short: "Check the claim against the actual passage",
detail: `When article text is available, the system extracts the useful passages and compares the claim directly against them.

Which passages get read is decided by hybrid retrieval: sentence embeddings (all-MiniLM-L6-v2) find sentences that MEAN the same as the claim, word overlap finds exact matches on names and numbers, and the two rankings are merged with reciprocal rank fusion. Without the embeddings, a debunk written in its own words — "Washington has not prohibited the search giant" for a claim about banning Google — shares no words with the claim and was never read.

The key question is not "Does the article title mention the same words?" but "Does the evidence passage support, contradict, or remain neutral about the claim?" This is where NLI/stance classification matters.`,
  },
  {
id: "evidence-fusion",
icon: <CheckCircle2 className="hiw-step-icon" />,
title: "Source quality + evidence fusion",
short: "Weight direct, independent, recent reporting more heavily",
detail: `The verdict weighs each source by tier — primary, fact-check, reporting, reference — and counts distinct publishers rather than articles, so several copies of one wire story are one confirmation and not several.

Each direction is scored only over the sources that take it, so background coverage that says nothing either way cannot dilute a real signal. A direction has to clearly outweigh the other to win outright; otherwise the evidence is genuinely contested and we say so.

Where a search ran properly across several providers and found nothing supporting a claim that would certainly have been reported, we report that as a finding — "no credible source reports this" — rather than as a failure to check. That is deliberately narrow: it never applies to a negated claim, a thin candidate pool, a failed search, or an unavailable NLI model, because those tell us nothing about the world.`,
  },
  {
id: "explanation",
icon: <MessageSquareCheck className="hiw-step-icon" />,
title: "Grounded explanation",
short: "An LLM explains the verdict; NLI checks every sentence",
detail: `Once the verdict is final, Gemini Flash writes two to four sentences explaining why the evidence leads to it, citing each source as [n].

Every sentence is then checked by the same NLI model against the source it cites. A sentence with no citation, a citation to a source it was not given, or a claim its source does not entail is dropped before you see it. If the NLI model is unavailable, no explanation is shown at all.

The LLM never decides or changes the verdict — it runs after the verdict is computed, and nothing reads its output back.`,
  },
];

export default function HowItWorks() {
  const [expanded, setExpanded] = useState(null);

  // Expands or collapses a pipeline step
  const toggle = (id) => setExpanded(expanded === id ? null : id);

  return (
    <div className="hiw-page">
      <section className="intro" style={{ animationDelay: "0s" }}>
        <p className="intro-tag">Evidence-first verification pipeline</p>
        <h2 className="intro-heading">How It Works</h2>
        <p className="intro-desc">
          We start by understanding the claim, search for the real story, filter to the most relevant evidence,
          and compare that evidence against the proposition before giving a verdict.
        </p>
      </section>

      {/* Visual Pipeline (The clickable buttons) */}
      <div className="hiw-pipeline" id="pipeline-diagram">
        {PIPELINE_STEPS.map((step, i) => (
          <div
            key={step.id}
            className="hiw-step-wrapper"
            style={{ animationDelay: `${i * 0.08}s` }}
          >
            {/* Draw a connecting arrow between steps */}
            {i > 0 && (
              <div className="hiw-arrow">
                <svg viewBox="0 0 40 20" className="hiw-arrow-svg">
                  <line x1="0" y1="10" x2="32" y2="10" />
                  <polygon points="30,5 40,10 30,15" />
                </svg>
              </div>
            )}

            <button
              className={`hiw-step-card ${expanded === step.id ? "hiw-step-expanded" : ""}`}
              onClick={() => toggle(step.id)}
              id={`step-${step.id}`}
            >
              {step.icon}
              <span className="hiw-step-title">{step.title}</span>
              <span className="hiw-step-short">{step.short}</span>
              <span className="hiw-step-toggle">
                {expanded === step.id ? "−" : "+"}
              </span>
            </button>
          </div>
        ))}
      </div>

      {/* Expanded detail panel (Shows the long text when a button is clicked) */}
      {expanded && (
        <div className="hiw-detail-panel" id="detail-panel">
          <div className="hiw-detail-header">
            <span className="hiw-detail-icon">
              {PIPELINE_STEPS.find((s) => s.id === expanded)?.icon}
            </span>
            <h3>{PIPELINE_STEPS.find((s) => s.id === expanded)?.title}</h3>
          </div>
          <div className="hiw-detail-body">
            {/* Split the detail text by newlines and render proper HTML tags */}
            {PIPELINE_STEPS.find((s) => s.id === expanded)
              ?.detail.split("\n")
              .map((line, i) => {
                const trimmed = line.trim();
                if (!trimmed) return <br key={i} />;
                // Automatically turn bullet points into <li> tags
                if (trimmed.startsWith("•")) {
                  return <li key={i}>{trimmed.slice(1).trim()}</li>;
                }
                return <p key={i}>{trimmed}</p>;
              })}
          </div>
        </div>
      )}

      {/* Pipeline summary */}
      <div className="hiw-arch-card" id="architecture-diagram">
        <h3 className="eval-section-title">
          <Layers className="eval-icon" size={20} /> Final pipeline summary
        </h3>
        <div className="hiw-arch-visual">
          <EvidencePipelineDiagram />
        </div>
        <p className="hiw-arch-caption">
          Claim understanding → targeted retrieval → relevance filtering → hybrid passage selection → NLI/stance → source quality check → evidence fusion → verdict → NLI-checked explanation.
        </p>
      </div>

      {/* Verdict logic */}
      <div className="hiw-formula-card" id="scoring-formula">
        <h3 className="eval-section-title">
          <Calculator className="eval-icon" size={20} /> Verdict logic
        </h3>
        <div className="hiw-formula">
          <div className="hiw-formula-eq">
            <span className="hiw-f-label">Verdict</span>
            <span className="hiw-f-eq">=</span>
            <span className="hiw-f-term hiw-f-ev">
              <span className="hiw-f-weight">Evidence</span>
              <span className="hiw-f-name">Relevance</span>
            </span>
            <span className="hiw-f-op">+</span>
            <span className="hiw-f-term hiw-f-st">
              <span className="hiw-f-weight">Stance</span>
              <span className="hiw-f-name">Support</span>
            </span>
            <span className="hiw-f-op">+</span>
            <span className="hiw-f-term hiw-f-ml">
              <span className="hiw-f-weight">Source</span>
              <span className="hiw-f-name">Quality</span>
            </span>
          </div>
        </div>
        <div className="hiw-formula-legend">
          <div className="hiw-legend-item">
            <span
              className="hiw-legend-dot"
              style={{ background: "var(--purple)" }}
            />
            <span>
              <strong>Evidence</strong> — direct relevance to the same event or proposition
            </span>
          </div>
          <div className="hiw-legend-item">
            <span
              className="hiw-legend-dot"
              style={{ background: "var(--blue)" }}
            />
            <span>
              <strong>Stance</strong> — support, contradiction, or neutrality from the passage itself
            </span>
          </div>
          <div className="hiw-legend-item">
            <span
              className="hiw-legend-dot"
              style={{ background: "var(--green)" }}
            />
            <span>
              {/* Not "recency": the pipeline does not compare an article's
                  publish date against the claim's timeframe. The README
                  lists that as a known gap, and this page claiming it was a
                  direct contradiction anyone could catch. */}
              <strong>Source quality</strong> — independence and tier of the publisher
            </span>
          </div>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="hiw-tech-card" id="tech-stack">
        <h3 className="eval-section-title">
          <Code2 className="eval-icon" size={20} /> Tech Stack
        </h3>
        <div className="hiw-tech-grid">
          {[
            {
              label: "ML / NLP",
              items: [
                "PyTorch (CPU) + Hugging Face Transformers",
                "DeBERTa-v3 NLI cross-encoder",
                "Sentence-Transformers (all-MiniLM-L6-v2)",
                "Reciprocal rank fusion (hybrid retrieval)",
                "Gemini Flash via google-genai (explanations only)",
                "NumPy MLP + TF-IDF (auxiliary only)",
              ],
            },
            {
              label: "Backend",
              items: [
                "Python 3.12 · FastAPI · Uvicorn",
                "Node.js · Express 5",
                "MongoDB · Mongoose",
                "Google OAuth + JWT",
              ],
            },
            {
              label: "Frontend",
              items: ["React 19 (Vite)", "Vanilla CSS", "Lucide Icons"],
            },
            {
              label: "Data sources",
              items: [
                "Google News RSS (no key)",
                "Wikipedia (no key)",
                "GNews", "NewsAPI", "The Guardian",
                "DuckDuckGo (fallback)",
              ],
            },
            {
              label: "Testing",
              items: [
                "pytest — 517 offline tests",
                "Node test runner",
                "ESLint",
                "Live benchmark (news_benchmark.py)",
              ],
            },
          ].map((group) => (
            <div className="hiw-tech-group" key={group.label}>
              <h4>{group.label}</h4>
              <ul>
                {group.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Evidence pipeline diagram ───────────────────────────────────────

function EvidencePipelineDiagram() {
  const steps = [
    "Claim",
    "Triage",
    "Search",
    "Relevance",
    "Passages",
    "NLI",
    "Verdict",
    "Explain",
  ];
  // Evenly spaced from a single pitch so adding a stage can't reintroduce the
  // clipping bug: the last node's right edge must stay inside the viewBox,
  // and that is derived rather than hand-maintained. Sized so the longest
  // label ("Relevance", ~62px at 12px type) sits well inside its box.
  const nodeW = 84;
  const nodeH = 56;
  const gap = 28;
  const pitch = nodeW + gap;
  const pad = 8;
  const top = 12;
  const x = steps.map((_, i) => pad + i * pitch);
  const width = pad * 2 + (steps.length - 1) * pitch + nodeW;
  const height = top * 2 + nodeH;
  const mid = top + nodeH / 2;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="hiw-nn-svg" role="img"
         aria-label={`Pipeline: ${steps.join(", then ")}`}>
      {x.slice(0, -1).map((val, i) => (
        <g key={`arrow-${i}`}>
          <line
            x1={val + nodeW + 3}
            y1={mid}
            x2={x[i + 1] - 9}
            y2={mid}
            className="hiw-nn-conn"
          />
          <polygon
            points={`${x[i + 1] - 9},${mid - 5} ${x[i + 1] - 2},${mid} ${x[i + 1] - 9},${mid + 5}`}
            className="hiw-nn-arrowhead"
          />
        </g>
      ))}
      {steps.map((label, i) => (
        <g key={label}>
          <rect
            x={x[i]}
            y={top}
            width={nodeW}
            height={nodeH}
            rx={12}
            className={
              i === steps.length - 1
                ? "hiw-nn-node hiw-nn-output"
                : i % 2 === 0 ? "hiw-nn-node hiw-nn-input" : "hiw-nn-node hiw-nn-hidden"
            }
          />
          <text
            x={x[i] + nodeW / 2}
            y={mid}
            className="hiw-nn-label"
            style={{ fontSize: 12 }}
          >
            {label}
          </text>
        </g>
      ))}
    </svg>
  );
}
