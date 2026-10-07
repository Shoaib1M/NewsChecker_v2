/*
FILE PURPOSE:
This component displays a single piece of evidence (a news article) returned by the scraper.
It shows the article title, domain, source tier, and a relevant passage extracted by NLI.

CHANGES FROM LEGACY:
- Removed misleading "Match" percentage (was lexical overlap, not relevance)
- Added source tier indicator (primary, reporting, fact-check, unclassified)
- Shows best_sentence from NLI-verified passage, not arbitrary snippet
- Explains why this source matters for the claim
- Shows support/contradiction scores instead of generic similarity
- Says plainly when a source does NOT address the claim. Previously any
  NLI-classified card claimed the source "supports" or "contradicts" the
  claim, chosen by whichever score was larger — so an article the model had
  explicitly judged neutral (0.04 vs 0.03) was described as contradicting it.
  That single sentence is what made on-topic-but-unrelated results look like
  the system misunderstanding the claim.
*/

function getFaviconUrl(url) {
  try {
    const domain = new URL(url).hostname;
    // We use Google's free favicon service to grab the site's logo automatically.
    return `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
  } catch {
    return null;
  }
}

function getDomain(url) {
  try {
    return new URL(url).hostname.replace("www.", "");
  } catch {
    return url;
  }
}

function getSourceTierBadge(tier) {
  const tierLabels = {
    'primary': 'Primary Source',
    'fact-check': 'Fact Check',
    'reporting': 'News Reporting',
    'reference': 'Reference Work',
    'unclassified': 'General Source',
  };
  return tierLabels[tier] || 'Source';
}

// `position` is this source's 1-based place in top_evidence — the number the
// explanation cites it by. The id is built from it (not from `index`, which
// restarts at 0 in each section) so a citation finds exactly one card.
export default function EvidenceCard({ evidence, index, position }) {
  const {
    title,
    url,
    stance,
    source,
    best_sentence,
    support_score,
    contradiction_score,
    source_tier,
    nli_available,
    stance_note,
    published,
    publisher,
  } = evidence;

  const favicon = getFaviconUrl(url);
  // Prefer the resolved publisher: for a source reached through a news
  // aggregator the URL's host names the aggregator, so every card would
  // otherwise read "news.google.com".
  const domain = publisher || source || getDomain(url);
  // Providers differ on whether they give a date at all, so this is often
  // absent. When it is present it is the fastest way for a reader to see why
  // a source did or did not count towards a claim about today.
  const publishedLabel = published
    ? new Date(published).toLocaleDateString(undefined, {
        year: "numeric", month: "short", day: "numeric",
      })
    : "";
  const tierLabel = getSourceTierBadge(source_tier);

  // Determine if this evidence supports or contradicts
  const isSupportive = support_score > contradiction_score;
  const strength = Math.max(support_score, contradiction_score);
  const strengthLabel = strength > 0.7 ? 'Strong' : strength > 0.4 ? 'Moderate' : 'Weak';

  // "unclear" is the NLI model's considered answer that this passage neither
  // entails nor contradicts the claim — a real result, not a missing one.
  const addressesClaim = stance === 'supports' || stance === 'contradicts';
  const stanceLabel = addressesClaim ? stance : "doesn't address claim";

  return (
    <div
      className="evidence-card"
      style={{ animationDelay: `${index * 0.08}s` }}
      id={position ? `evidence-${position}` : undefined}
    >
      <div className="evidence-card-header">
        <div className="evidence-source">
          {favicon && (
            <img
              src={favicon}
              alt=""
              className="evidence-favicon"
              loading="lazy"
              onError={(e) => { e.target.style.display = "none"; }}
            />
          )}
          <div className="evidence-source-info">
            <span className="evidence-domain">
              {position ? <span className="evidence-number">[{position}]</span> : null}
              {domain}
            </span>
            <span className="evidence-tier">{tierLabel}</span>
          </div>
        </div>
        <div className="evidence-badges">
          <span className={`verified-badge ${nli_available ? "verified" : "unverified"}`}>
            {nli_available ? "Verified" : "Unverified"}
          </span>
          <span className={`stance-badge ${stance} ${strengthLabel.toLowerCase()}`}>
            {stanceLabel}
          </span>
        </div>
      </div>

      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        className="evidence-link"
      >
        {title || "Untitled"}
      </a>

      {/* Evidence strength indicator — only meaningful when the source
          actually takes a position on the claim. */}
      {nli_available && addressesClaim && (
        <div className="evidence-strength">
          <span className="strength-label">{strengthLabel} Evidence</span>
          <div className="strength-bar-track">
            <div
              className={`strength-bar-fill ${isSupportive ? 'support' : 'contradiction'}`}
              style={{ width: `${Math.round(strength * 100)}%` }}
            />
          </div>
          <span className="strength-value">{Math.round(strength * 100)}%</span>
        </div>
      )}

      {/* Most relevant passage extracted by NLI */}
      {best_sentence && (
        <div className="evidence-passage">
          <p className="passage-label">Key passage:</p>
          <p className="passage-text">"{best_sentence}"</p>
        </div>
      )}

      {publishedLabel && (
        <p className="evidence-date">{publishedLabel}</p>
      )}

      {/* Why this source is not counted the way its scores read. Shown only
          when the pipeline overrode them, so a source that reads as relevant
          but counts as neutral does not look like a bug. */}
      {stance_note && (
        <p className="stance-note">This article {stance_note}.</p>
      )}

      {/* Explanation of why this source matters */}
      <div className="evidence-explanation">
        <p className="explanation-text">
          {!nli_available ? (
            'This source was retrieved but not checked against the claim — the NLI model was unavailable, so it counts as a candidate, not evidence.'
          ) : addressesClaim ? (
            <>
              This {tierLabel.toLowerCase()} {isSupportive ? 'supports' : 'contradicts'} the claim.
              {source_tier === 'primary' && ' Primary sources carry significant weight.'}
              {source_tier === 'fact-check' && ' Fact-checking organizations provide expert analysis.'}
              {source_tier === 'reference' && ' Reference works are useful background but are tertiary sources.'}
            </>
          ) : (
            'This source covers the same subject but states neither that the claim is true nor that it is false, so it does not count as evidence either way.'
          )}
        </p>
      </div>
    </div>
  );
}
