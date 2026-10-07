/*
FILE PURPOSE:
The How It Works page: a complete, illustrated walkthrough of what happens
between pasting a claim and seeing a verdict.

STRUCTURE:
1. The one-minute version and the end-to-end flow diagram.
2. A table of contents, then one section per pipeline stage: what it does,
   a diagram where one helps, why it works that way (usually a real bug it
   fixed), a worked example and where it lives in the code.
3. The verdict glossary, design principles, limitations and tech stack.

The content lives in ./howItWorks/stages.jsx so this file stays about layout.
Every number on the page is taken from the code; stages.jsx says where.
*/

import {
  BookOpen,
  Code2,
  Compass,
  ListChecks,
  ShieldAlert,
  Scale,
} from "lucide-react";
import FlowDiagram from "./howItWorks/FlowDiagram";
import { STAGES, VERDICTS, PRINCIPLES, LIMITATIONS } from "./howItWorks/stages";
import { jumpTo } from "./howItWorks/jump";

const ONE_MINUTE = [
  "You paste a claim. It is cleaned up and sorted: questions, opinions and textbook facts are answered without searching.",
  "Everything else is searched for across six news and reference sources at once, under a 45-second budget.",
  "Only on-topic results are kept, and the best passages of each article are chosen by both meaning and wording.",
  "A natural-language-inference model reads each passage and decides whether it supports the claim, contradicts it, or neither.",
  "Sources are weighed by credibility and counted by independent publisher, giving a verdict — or an honest “we can't tell” — with the evidence shown. An AI-written explanation is added last, checked sentence by sentence.",
];

const TECH_STACK = [
  {
    label: "ML / NLP",
    items: [
      "PyTorch (CPU) + Hugging Face Transformers",
      "DeBERTa-v3 NLI cross-encoder",
      "Sentence-Transformers (all-MiniLM-L6-v2)",
      "Reciprocal rank fusion (hybrid retrieval)",
      "Gemini Flash via google-genai (explanations only)",
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
      "Google News",
      "Wikipedia",
      "GNews",
      "NewsAPI",
      "The Guardian",
      "DuckDuckGo (fallback)",
    ],
  },
  {
    label: "Testing",
    items: [
      "pytest — 480 offline tests",
      "Node test runner",
      "ESLint",
      "Live benchmark (news_benchmark.py)",
    ],
  },
];

function Stage({ stage, index }) {
  return (
    <section className="hiw-stage" id={`stage-${stage.id}`}>
      <header className="hiw-stage-header">
        <span className="hiw-stage-num">{index + 1}</span>
        <div>
          <h3 className="hiw-stage-title">{stage.title}</h3>
          <p className="hiw-stage-summary">{stage.summary}</p>
        </div>
      </header>

      <div className="hiw-stage-body">{stage.what}</div>

      {stage.diagram}

      <div className="hiw-stage-why">
        <span className="hiw-stage-label">Why it works this way</span>
        <p>{stage.why}</p>
      </div>

      {stage.example && (
        <div className="hiw-stage-example">
          <span className="hiw-stage-label">Example</span>
          <div className="hiw-example-row">
            <span className="hiw-example-in">{stage.example.input}</span>
            <span className="hiw-example-arrow" aria-hidden="true">
              →
            </span>
            <span className="hiw-example-out">{stage.example.output}</span>
          </div>
        </div>
      )}

      <p className="hiw-stage-code">
        <Code2 size={13} aria-hidden="true" /> {stage.code.join(" · ")}
      </p>
    </section>
  );
}

export default function HowItWorks() {
  return (
    <div className="hiw-page" id="how-it-works-page">
      <section className="intro">
        <p className="intro-tag">How it works</p>
        <h2 className="intro-heading">
          From a pasted claim to an evidence-backed verdict
        </h2>
        <p className="intro-desc">
          NewsChecker never guesses from how a claim is worded. It finds what
          independent sources actually say, reads them with a language model
          trained for exactly one question — does this passage say that? — and
          tells you when the evidence isn't there.
        </p>
      </section>

      <div className="hiw-card hiw-oneminute">
        <h3 className="eval-section-title">
          <BookOpen className="eval-icon" size={20} /> The one-minute version
        </h3>
        <ol>
          {ONE_MINUTE.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ol>
      </div>

      <div className="hiw-card">
        <h3 className="eval-section-title">
          <Compass className="eval-icon" size={20} /> The whole pipeline
        </h3>
        <FlowDiagram />
      </div>

      <nav className="hiw-card hiw-toc" aria-label="Stages">
        <h3 className="eval-section-title">
          <ListChecks className="eval-icon" size={20} /> Stage by stage
        </h3>
        <ol>
          {STAGES.map((stage) => (
            <li key={stage.id}>
              <a
                href={`#stage-${stage.id}`}
                onClick={(e) => jumpTo(e, `stage-${stage.id}`)}
              >
                {stage.title}
              </a>
            </li>
          ))}
        </ol>
      </nav>

      {STAGES.map((stage, i) => (
        <Stage stage={stage} index={i} key={stage.id} />
      ))}

      <div className="hiw-card" id="verdict-glossary">
        <h3 className="eval-section-title">
          <Scale className="eval-icon" size={20} /> What each verdict means
        </h3>
        <table className="hiw-glossary">
          <tbody>
            {VERDICTS.map(([verdict, meaning]) => (
              <tr key={verdict}>
                <th scope="row">{verdict}</th>
                <td>{meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="hiw-card">
        <h3 className="eval-section-title">
          <ShieldAlert className="eval-icon" size={20} /> Rules the system never
          breaks
        </h3>
        <div className="hiw-principles">
          {PRINCIPLES.map(([title, body]) => (
            <div className="hiw-principle" key={title}>
              <strong>{title}</strong>
              <p>{body}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="hiw-card">
        <h3 className="eval-section-title">
          <ShieldAlert className="eval-icon" size={20} /> Known limitations
        </h3>
        <ul className="hiw-limitations">
          {LIMITATIONS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      <div className="hiw-tech-card" id="tech-stack">
        <h3 className="eval-section-title">
          <Code2 className="eval-icon" size={20} /> Tech stack
        </h3>
        <div className="hiw-tech-grid">
          {TECH_STACK.map((group) => (
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
