/*
FILE PURPOSE:
The faithfulness filter for LLM explanations: what happens to each sentence
the model writes before anything reaches the screen.
*/

import Figure from "./Figure";

const SENTENCES = [
  {
    text: "Regulators confirmed that no nationwide prohibition on Google is in force [2].",
    outcome: "kept",
    reason: "cites source 2, and source 2 entails it",
  },
  {
    text: "The rumour started on a Russian forum in March [2].",
    outcome: "dropped",
    reason: "source 2 says nothing of the kind — not entailed",
  },
  {
    text: "This is clearly misinformation.",
    outcome: "dropped",
    reason: "no citation",
  },
  {
    text: "Fact-checkers rated the claim false [5].",
    outcome: "dropped",
    reason: "source 5 was not given to the model",
  },
];

export default function ExplanationDiagram() {
  return (
    <Figure
      minWidth={680}
      caption="Illustrative. Each sentence the model writes must cite a source it was given, and that source must entail it under the same rule used for evidence. Only kept sentences are shown. If the NLI model is unavailable, no explanation is shown at all."
    >
      <div className="hiw-explain">
        <div className="hiw-explain-flow">
          {[
            "Final verdict + verdict-side sources",
            "Gemini writes 2–4 cited sentences",
            "Citation check",
            "NLI: does the cited source entail it?",
          ].map((step, i) => (
            <span key={step} className="hiw-explain-flow-item">
              {i > 0 && <span className="hiw-explain-sep" aria-hidden="true">→ </span>}
              <span className="hiw-explain-step">{step}</span>
            </span>
          ))}
        </div>
        <ul className="hiw-explain-list">
          {SENTENCES.map((s) => (
            <li key={s.text} className={`hiw-explain-${s.outcome}`}>
              <span className="hiw-explain-badge">{s.outcome}</span>
              <span className="hiw-explain-text">{s.text}</span>
              <span className="hiw-explain-reason">{s.reason}</span>
            </li>
          ))}
        </ul>
      </div>
    </Figure>
  );
}
