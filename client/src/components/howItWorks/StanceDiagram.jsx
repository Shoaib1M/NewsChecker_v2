/*
FILE PURPOSE:
How one document's passage scores become that document's position, and the
three checks that can withdraw a position afterwards. Mirrors
evidence_pipeline.decide_stance and the checks that follow it.
*/

import Figure from "./Figure";

function Step({ title, children, tone = "default" }) {
  return (
    <div className={`hiw-tree-step hiw-tree-${tone}`}>
      <div className="hiw-tree-title">{title}</div>
      <div className="hiw-tree-body">{children}</div>
    </div>
  );
}

const Arrow = () => <div className="hiw-tree-arrow" aria-hidden="true">↓</div>;

export default function StanceDiagram() {
  return (
    <Figure
      minWidth={640}
      caption="A document's strongest entailment and strongest contradiction are found independently across its passages, then compared. Three checks run after that; each can only withdraw a position to 'unclear' — none can turn support into contradiction or the reverse."
    >
      <div className="hiw-tree">
        <Step title="Best support · best contradiction">
          Highest entailment and highest contradiction across the document's passages, found
          separately. A passage that merely <em>reports</em> the claim (“posts claim…”, “viral”,
          “fact check”) cannot supply the support score.
        </Step>
        <Arrow />
        <Step title="Does either side clear 0.35?">
          Neither → <strong>unclear</strong>. Only one → that side. Both → one must be at least{" "}
          <strong>1.6×</strong> the other, otherwise the document argues both ways →{" "}
          <strong>unclear</strong>.
        </Step>
        <Arrow />
        <div className="hiw-tree-checks">
          <Step title="Different number?" tone="check">
            States 0.5% where the claim says 0.25% → support withdrawn.
          </Step>
          <Step title="Too old?" tone="check">
            For a claim about now, an article older than 45 days cannot confirm it.
          </Step>
          <Step title="About something else?" tone="check">
            Title, snippet and decisive passage all below 0.30 similarity to the claim → no position.
          </Step>
        </div>
        <Arrow />
        <Step title="Position" tone="result">
          <strong>supports</strong> · <strong>contradicts</strong> · <strong>unclear</strong>{" "}
          (shown under “Related coverage”, never counted as evidence)
        </Step>
      </div>
    </Figure>
  );
}
