/*
FILE PURPOSE:
How classified sources become a direction: source-tier weights, independent
publishers, and the 2× rule. Uses the viral-false-claim scenario from the
README, where six low-quality posts "support" a claim and two credible
sources contradict it — and the verdict is still "contradicted".
*/

import Figure from "./Figure";

const TIERS = [
  { tier: "Primary", weight: "1.0", example: ".gov sites, WHO, UN, World Bank, SEC" },
  { tier: "Fact-check", weight: "0.95", example: "PolitiFact, FactCheck.org, Full Fact, Snopes, AFP" },
  { tier: "Reporting", weight: "0.8", example: "Reuters, AP, BBC, NPR, The Guardian, Bloomberg" },
  { tier: "Reference", weight: "0.5", example: "Wikipedia, Britannica" },
  { tier: "Unclassified", weight: "0.1", example: "everything else (counted at a 0.1 floor)" },
];

const SUPPORT = Array.from({ length: 6 }, (_, i) => ({ who: `rumour blog ${i + 1}`, w: 0.1 }));
const CONTRA = [
  { who: "politifact.com", w: 0.95 },
  { who: "reuters.com", w: 0.8 },
];

function Pile({ title, items, tone }) {
  const mass = items.reduce((sum, item) => sum + item.w, 0);
  return (
    <div className={`hiw-weigh-pile hiw-weigh-${tone}`}>
      <div className="hiw-weigh-title">{title}</div>
      <div className="hiw-weigh-blocks">
        {items.map((item) => (
          <div
            key={item.who}
            className="hiw-weigh-block"
            style={{ height: `${Math.max(10, item.w * 70)}px` }}
            title={`${item.who} — weight ${item.w}`}
          />
        ))}
      </div>
      <div className="hiw-weigh-mass">weight ≈ {mass.toFixed(2)}</div>
      <div className="hiw-weigh-pubs">{items.length} independent publisher{items.length === 1 ? "" : "s"}</div>
    </div>
  );
}

export default function WeighDiagram() {
  return (
    <Figure
      minWidth={680}
      caption="Six anonymous posts “support” a false claim; a fact-check and a wire report contradict it. Weighted by tier, the contradicting side outweighs the supporting side by more than the 2× the rule requires, so the verdict is “contradicted”. Below 2× either way, it would be “mixed”. (Scores are taken as 1.0 here to show the weights alone.)"
    >
      <div className="hiw-weigh">
        <table className="hiw-weigh-table">
          <thead>
            <tr><th>Source tier</th><th>Weight</th><th>Examples</th></tr>
          </thead>
          <tbody>
            {TIERS.map((t) => (
              <tr key={t.tier}><td>{t.tier}</td><td>{t.weight}</td><td>{t.example}</td></tr>
            ))}
          </tbody>
        </table>
        <div className="hiw-weigh-scale">
          <Pile title="Supports" items={SUPPORT} tone="support" />
          <div className="hiw-weigh-vs">vs</div>
          <Pile title="Contradicts" items={CONTRA} tone="contra" />
        </div>
      </div>
    </Figure>
  );
}
