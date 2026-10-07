/*
FILE PURPOSE:
The whole pipeline on one picture, including the two early exits that never
search the web and the explanation branch that runs after the verdict.

Built from HTML boxes rather than SVG so the text wraps naturally and stays
selectable; each node links to its section further down the page.
*/

import Figure from "./Figure";
import { jumpTo } from "./jump";

function Node({ href, children, tone = "default" }) {
  return (
    <a
      className={`hiw-flow-node hiw-flow-${tone}`}
      href={href}
      onClick={(e) => jumpTo(e, href.slice(1))}
    >
      {children}
    </a>
  );
}

const Arrow = () => <span className="hiw-flow-arrow" aria-hidden="true">→</span>;
const Down = () => <span className="hiw-flow-down" aria-hidden="true">↓</span>;

export default function FlowDiagram() {
  return (
    <Figure
      minWidth={760}
      caption="Every claim enters at the top left. Questions, opinions and textbook facts leave early without any web search; everything else goes through retrieval, NLI and weighing to a verdict. The explanation is generated afterwards and cannot change it."
    >
      <div className="hiw-flow">
        <div className="hiw-flow-row">
          <Node href="#stage-normalise">1 · Clean up the submission</Node>
          <Arrow />
          <Node href="#stage-triage">2 · Triage</Node>
          <Arrow />
          <Node href="#stage-deterministic">3 · Known-fact check</Node>
          <Arrow />
          <Node href="#stage-understand">4 · Understand the claim</Node>
        </div>

        <div className="hiw-flow-exits">
          <div className="hiw-flow-branch">
            <Down />
            <Node href="#stage-triage" tone="exit">
              Question, opinion, future event or not English → answered without searching
            </Node>
          </div>
          <div className="hiw-flow-branch">
            <Down />
            <Node href="#stage-deterministic" tone="exit">
              Textbook fact → instant, deterministic verdict
            </Node>
          </div>
        </div>

        <div className="hiw-flow-row">
          <Node href="#stage-time">5 · Pick the time window</Node>
          <Arrow />
          <Node href="#stage-search">6 · Search six providers</Node>
          <Arrow />
          <Node href="#stage-relevance">7 · Keep what is on-topic</Node>
          <Arrow />
          <Node href="#stage-passages">8 · Pick the best passages</Node>
        </div>

        <div className="hiw-flow-row">
          <Node href="#stage-nli" tone="model">9 · NLI reads each passage</Node>
          <Arrow />
          <Node href="#stage-stance">10 · Each source's position</Node>
          <Arrow />
          <Node href="#stage-weigh">11 · Weigh the sources</Node>
          <Arrow />
          <Node href="#stage-verdict" tone="verdict">12 · Verdict + confidence</Node>
        </div>

        <div className="hiw-flow-after">
          <Down />
          <Node href="#stage-explain" tone="display">
            13 · Explanation — written by an LLM, every sentence checked by NLI. Display only.
          </Node>
        </div>
      </div>
    </Figure>
  );
}
