/*
FILE PURPOSE:
What natural-language inference actually computes, on three real-shaped
examples: one passage that entails the claim, one that contradicts it, and
one that is merely on the same topic.
*/

import Figure from "./Figure";

const CLAIM = "The Federal Reserve raised interest rates by 0.25%";

const EXAMPLES = [
  {
    passage: "The Fed lifted its benchmark rate by a quarter percentage point on Wednesday.",
    scores: { entailment: 0.94, neutral: 0.05, contradiction: 0.01 },
  },
  {
    passage: "The Federal Reserve left interest rates unchanged at its latest meeting.",
    scores: { entailment: 0.01, neutral: 0.04, contradiction: 0.95 },
  },
  {
    passage: "Markets were volatile ahead of the central bank's announcement.",
    scores: { entailment: 0.03, neutral: 0.94, contradiction: 0.03 },
  },
];

function Bar({ scores }) {
  return (
    <div className="hiw-nli-bar" role="img"
         aria-label={`entailment ${scores.entailment}, neutral ${scores.neutral}, contradiction ${scores.contradiction}`}>
      <span className="hiw-nli-ent" style={{ flexGrow: scores.entailment }} />
      <span className="hiw-nli-neu" style={{ flexGrow: scores.neutral }} />
      <span className="hiw-nli-con" style={{ flexGrow: scores.contradiction }} />
    </div>
  );
}

export default function NliDiagram() {
  return (
    <Figure
      minWidth={640}
      caption="Illustrative scores. The model reads a passage (the premise) and the claim (the hypothesis) together and outputs three probabilities that sum to 1. Only the passage is evidence; the claim is what is being tested."
    >
      <div className="hiw-nli-claim">
        <span className="hiw-nli-tag">Hypothesis (the claim)</span>
        “{CLAIM}”
      </div>
      <div className="hiw-nli-rows">
        {EXAMPLES.map((example) => (
          <div className="hiw-nli-row" key={example.passage}>
            <div className="hiw-nli-passage">
              <span className="hiw-nli-tag">Premise (a passage)</span>
              {example.passage}
            </div>
            <div className="hiw-nli-scores">
              <Bar scores={example.scores} />
              <div className="hiw-nli-legend">
                <span>entails {example.scores.entailment.toFixed(2)}</span>
                <span>neutral {example.scores.neutral.toFixed(2)}</span>
                <span>contradicts {example.scores.contradiction.toFixed(2)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Figure>
  );
}
