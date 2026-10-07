/*
FILE PURPOSE:
Hybrid passage ranking: two ranked lists, merged by reciprocal rank fusion.

The example is the one that motivated the feature: a debunking sentence that
paraphrases the claim shares no words with it, so word overlap ranks it last,
while the embedding model ranks it first — and fusion lets it through.
*/

import Figure from "./Figure";

const CLAIM = "The United States banned Google across all its cities";

const LEXICAL = [
  { id: "a", text: "Google shares rose on Monday in early trading." },
  { id: "b", text: "Analysts in the United States expect strong sales." },
  { id: "c", text: "Several cities hosted technology fairs." },
];

const DENSE = [
  { id: "d", text: "Washington has not prohibited the search giant anywhere in the country." },
  { id: "b", text: "Analysts in the United States expect strong sales." },
];

const FUSED = [
  { id: "b", text: "Analysts in the United States expect strong sales.", score: "1/62 + 1/62" },
  { id: "d", text: "Washington has not prohibited the search giant anywhere…", score: "1/61" },
  { id: "a", text: "Google shares rose on Monday…", score: "1/61" },
  { id: "c", text: "Several cities hosted technology fairs.", score: "1/63" },
];

function List({ title, subtitle, items, highlight }) {
  return (
    <div className="hiw-rank-list">
      <div className="hiw-rank-title">{title}</div>
      <div className="hiw-rank-sub">{subtitle}</div>
      <ol>
        {items.map((item) => (
          <li key={item.id} className={item.id === highlight ? "hiw-rank-hit" : ""}>
            <span>{item.text}</span>
            {item.score && <code>{item.score}</code>}
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function HybridDiagram() {
  return (
    <Figure
      minWidth={780}
      caption="Word overlap never ranks the paraphrased denial (purple) — it shares no words with the claim. The embedding model ranks it first. Fusion scores every sentence by Σ 1/(60 + rank) across the lists it appears in, so it reaches NLI. It ties with the Google-shares sentence at 1/61; ties keep article order, and here the denial comes first."
    >
      <div className="hiw-rank-claim">Claim: “{CLAIM}”</div>
      <div className="hiw-rank">
        <List title="Word overlap" subtitle="shares the claim's words" items={LEXICAL} highlight="d" />
        <div className="hiw-rank-plus" aria-hidden="true">+</div>
        <List title="Meaning (MiniLM)" subtitle="cosine similarity ≥ 0.30" items={DENSE} highlight="d" />
        <div className="hiw-rank-plus" aria-hidden="true">=</div>
        <List title="Fused (RRF, k = 60)" subtitle="what NLI reads first" items={FUSED} highlight="d" />
      </div>
    </Figure>
  );
}
