/*
FILE PURPOSE:
How a pile of search results narrows to the few sources a verdict rests on.

The numbers are an illustration, labelled as one: they are typical of a news
claim, and every stage's real count is shown on each result page under
"Retrieval: N candidates · N on-topic · N checked".
*/

import Figure from "./Figure";

const STAGES = [
  { count: 42, label: "Search results", note: "raw hits from all providers, duplicates removed" },
  { count: 14, label: "On-topic", note: "passed relevance filtering (≥ 0.42)" },
  { count: 8, label: "Read by NLI", note: "at most 8 per claim; 3 seats held for credible outlets" },
  { count: 3, label: "Take a side", note: "support or contradict after the checks" },
];

export default function FunnelDiagram() {
  const max = STAGES[0].count;
  return (
    <Figure
      minWidth={520}
      caption="An illustrative example. Most search results never become evidence: being found is not the same as being relevant, and being relevant is not the same as taking a position on the claim."
    >
      <div className="hiw-funnel">
        {STAGES.map((stage) => (
          <div className="hiw-funnel-row" key={stage.label}>
            <div className="hiw-funnel-bar-wrap">
              <div
                className="hiw-funnel-bar"
                style={{ width: `${Math.max(14, (stage.count / max) * 100)}%` }}
              >
                {stage.count}
              </div>
            </div>
            <div className="hiw-funnel-text">
              <strong>{stage.label}</strong>
              <span>{stage.note}</span>
            </div>
          </div>
        ))}
      </div>
    </Figure>
  );
}
