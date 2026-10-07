/*
FILE PURPOSE:
This component displays two horizontal progress bars that break down the
evidence behind the verdict: how strong the NLI-classified evidence is, and
which way it points.

FLOW:
1. Receives the evidence scores as props (`evidenceScore`, `stanceNet`).
2. Converts the raw decimal scores (e.g., 0.85) into percentages (85%).
3. Maps over the items array to render identical progress bar UI blocks.

WHY THIS EXISTS:
Transparency is crucial in AI. A single number out of 100 isn't enough; the user needs 
to know *why* the AI gave that score.
*/

export default function ScoreBreakdown({
  evidenceScore,
  stanceNet,
  hasClassifiedEvidence,
  hasDirectionalEvidence,
}) {
  const evidencePct = Math.round(evidenceScore * 100);
  const directionPct = Math.round(((stanceNet + 1) / 2) * 100);

  const items = [
    {
      label: "Evidence strength",
      // Without any NLI-classified evidence, a percentage here would be fake
      // precision — there is nothing to measure yet, not a low score.
      value: hasClassifiedEvidence ? evidencePct : 0,
      display: hasClassifiedEvidence ? `${evidencePct}%` : "—",
      tooltip: "Strength and coverage of NLI-classified evidence",
    },
    {
      // Gated on *directional* evidence, not merely classified evidence.
      // Sources the model classified as neutral leave stanceNet at 0, which
      // rendered as a confident-looking half-filled bar at "50%" — a made-up
      // midpoint for a claim nothing had taken a position on.
      label: "Evidence direction",
      value: hasDirectionalEvidence ? directionPct : 0,
      display: hasDirectionalEvidence ? `${directionPct}%` : "—",
      tooltip: "Whether NLI-checked evidence supports or contradicts the claim",
    },
  ];

  return (
    <div className="breakdown-list" id="score-breakdown">
      {items.map((item) => (
        <div
          className="breakdown-item"
          key={item.label}
          title={item.tooltip}
        >
          <span className="breakdown-label">{item.label}</span>
          <div className="breakdown-bar-track">
            {/* The actual filled portion of the bar */}
            <div
              className="breakdown-bar-fill"
              style={{ width: `${item.value}%` }}
            />
          </div>
          <span className="breakdown-value">{item.display}</span>
        </div>
      ))}
      <p className="breakdown-caption">
        The verdict is computed from NLI-classified evidence only.
      </p>
    </div>
  );
}
