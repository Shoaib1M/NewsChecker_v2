/*
FILE PURPOSE:
Shows the LLM-written explanation of a verdict, with each [n] citation linked
to the evidence card it refers to.

WHY IT LOOKS LIKE THIS:
- Only sentences the NLI model confirmed are rendered. The backend already
  puts nothing else in `text`; this component reads the per-sentence `kept`
  flags as well, so a future backend bug cannot surface an unchecked sentence.
- Nothing at all is rendered when `available` is false. An explanation that
  failed, was skipped, or had every sentence rejected is not replaced by a
  placeholder: the verdict and evidence above it stand on their own, and a
  "no explanation" message would read as a problem with the verdict.
- [n] is the source's 1-based position in `top_evidence`, matching the
  `evidence-{n}` id each EvidenceCard carries, so a citation can scroll to and
  highlight the exact card it relies on. Being able to check a sentence
  against its source in one click is the point of citing it.
*/

const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function scrollToEvidence(number) {
  const card = document.getElementById(`evidence-${number}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  // Restart the highlight animation even if the same card is clicked twice.
  card.classList.remove("evidence-highlight");
  void card.offsetWidth;
  card.classList.add("evidence-highlight");
  window.setTimeout(() => card.classList.remove("evidence-highlight"), 2200);
}

function renderWithCitations(text) {
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    const numbers = match[1].split(",").map((n) => Number(n.trim()));
    parts.push(
      <span key={`c${match.index}`} className="citation-group">
        [
        {numbers.map((n, i) => (
          <span key={n}>
            {i > 0 && ", "}
            <button
              type="button"
              className="citation-link"
              onClick={() => scrollToEvidence(n)}
              title={`Show source ${n}`}
              aria-label={`Show source ${n}`}
            >
              {n}
            </button>
          </span>
        ))}
        ]
      </span>,
    );
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export default function ExplanationPanel({ explanation }) {
  if (!explanation?.available || !explanation.text) return null;

  const kept = (explanation.sentences || []).filter((s) => s.kept);
  if (kept.length === 0) return null;
  const dropped = explanation.dropped_count || 0;

  return (
    <div className="explanation-panel">
      <p className="explanation-label">Why this verdict</p>
      <p className="explanation-body">
        {kept.map((sentence, i) => (
          <span key={i}>
            {i > 0 && " "}
            {renderWithCitations(sentence.text)}
          </span>
        ))}
      </p>
      <p className="explanation-note">
        Each sentence checked against its source by the NLI model.
        {dropped > 0 &&
          ` ${dropped} sentence${dropped === 1 ? " was" : "s were"} removed because the cited source did not support ${dropped === 1 ? "it" : "them"}.`}
        {explanation.model && ` Written by ${explanation.model}; the verdict above was decided without it.`}
      </p>
    </div>
  );
}
