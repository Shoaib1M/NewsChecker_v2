/*
FILE PURPOSE:
This is the root component of the entire React Frontend.
It holds the main application state (user, token, the statement being checked, results, and history).
It also acts as the central router, deciding which page or component to show.

FLOW:
1. useHashRouter: Manages simple navigation without needing a complex library like React Router.
2. User Session: Checks if the user is logged in via Google OAuth.
3. API Calls: Handles sending statements to the backend and fetching history.
4. Render: Displays the Header, the main content (Home, Evaluation, Comparison, etc.), and the Footer.

WHY THIS EXISTS:
We need a central "brain" for the frontend to manage the state that is shared across multiple components
(e.g., the Header needs to know if the user is logged in, and the Home page needs to know the check results).
*/

import { useState, useEffect, useCallback, useRef } from "react";
import { Info } from "lucide-react";
import "./App.css";

// Import all the modular UI pieces (components)
import Header from "./components/Header";
import ScoreGauge from "./components/ScoreGauge";
import ScoreBreakdown from "./components/ScoreBreakdown";
import EvidenceCard from "./components/EvidenceCard";
import LoadingSkeleton from "./components/LoadingSkeleton";
import HistoryPanel from "./components/HistoryPanel";
import ModelEvaluation from "./components/ModelEvaluation";
import ModelComparison from "./components/ModelComparison";
import HowItWorks from "./components/HowItWorks";

// The backend API URL (e.g., http://localhost:3000)
const API_BASE = import.meta.env.VITE_API_URL || "";

/*
PURPOSE: Turn an HTTP failure into something a person can act on.

WHY THIS EXISTS: The raw strings are backend-speak — "ML service error.",
"ML service timed out." — and read as if the system had judged the claim.
Every message here says what failed and what to do, and none of them can be
mistaken for a verdict. The technical detail is kept, because the most common
cause is a misconfigured FASTAPI_URL and hiding that wastes an afternoon.
*/
/*
PURPOSE: How many independent publishers actually back the verdict shown.

WHY THIS EXISTS: The two sides can disagree, and source tiering means the
smaller side can win — two credible publishers outweigh six anonymous ones.
Reporting the larger count then credits the verdict with sources that argued
against it. Mirrors `_independent_backing` in ml-service/main.py.
*/
function publishersBackingVerdict(result) {
  const supporting = result.evidence?.independent_supporting || 0;
  const contradicting = result.evidence?.independent_contradicting || 0;
  const status = result.verification?.status || result.assessment_status;
  if (status === "supported" || status === "reported_plan") return supporting;
  if (status === "contradicted") return contradicting;
  return Math.max(supporting, contradicting);
}

function describeRequestFailure(status, payload) {
  const detail = payload?.upstream ? ` (tried ${payload.upstream})` : "";
  if (status === 400) {
    return payload?.error || "That input can't be checked — try rephrasing it.";
  }
  if (status === 504) {
    return (
      "The analysis service took too long to respond, so this claim wasn't checked. " +
      "This is a problem on our side, not a finding about the claim. " +
      "The first check of a session is slowest — try again."
    );
  }
  if (status === 502 || status === 503) {
    return (
      "Couldn't reach the analysis service, so this claim wasn't checked. " +
      `This is a problem on our side, not a finding about the claim${detail}.`
    );
  }
  return payload?.error || `The request failed (${status}).`;
}

// ─── Hash Router ────────────────────────────────────────────────────
/*
PURPOSE: A custom, lightweight router using the URL hash (e.g., #/how-it-works).
WHY THIS EXISTS: Keeps the URL updated and allows users to use the Back button, 
without needing a heavy third-party library like `react-router-dom` for a simple app.
*/
function useHashRouter() {
  const getPage = () => {
    const hash = window.location.hash.replace("#/", "").replace("#", "");
    return hash || "";
  };

  const [page, setPage] = useState(getPage);

  useEffect(() => {
    const handler = () => setPage(getPage());
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  const navigate = (p) => {
    window.location.hash = `#/${p}`;
    setPage(p);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return [page, navigate];
}

function App() {
  // ─── STATE MANAGEMENT ────────────────────────────────────────────────

  // Router state
  const [page, navigate] = useHashRouter();
  
  // Auth state
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem("nc_token"));
  
  // Fact-check state
  const [statement, setStatement] = useState("");
  // "auto" | "recent" | "historical" — see the picker below.
  const [mode, setMode] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  
  // History state
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState([]);
  
  const textareaRef = useRef(null);

  // ─── Restore user session ───────────────────────────────────────────
  /*
  PURPOSE: Automatically logs the user back in if they refresh the page.
  It sends their saved token to the backend to verify it's still valid.
  */
  useEffect(() => {
    if (!token) return;
    fetch(`${API_BASE}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Session expired");
        return res.json();
      })
      .then(setUser)
      .catch(() => {
        // If the token is invalid (expired/tampered), clear it out
        localStorage.removeItem("nc_token");
        setToken(null);
        setUser(null);
      });
  }, [token]);

  // ─── Initialize Google Sign-In ──────────────────────────────────────
  /*
  PURPOSE: Loads the official Google Sign-In button.
  WHY THIS EXISTS: We don't want to manage passwords ourselves. 
  Google handles the security and just gives us a verified token.
  */
  useEffect(() => {
    if (user) return;
    const initGsi = () => {
      if (!window.google?.accounts?.id) return;
      const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
      if (!clientId) return;
      
      // Tell Google what to do when the user successfully logs in
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleCredential,
      });
      
      // Draw the actual button inside our HTML container
      const btnContainer = document.getElementById("google-signin-btn");
      if (btnContainer) {
        window.google.accounts.id.renderButton(btnContainer, {
          theme: "outline",
          size: "medium",
          shape: "pill",
          text: "signin_with",
        });
      }
    };
    
    // Wait until Google's external script has finished loading
    if (window.google?.accounts?.id) {
      initGsi();
    } else {
      const timer = setInterval(() => {
        if (window.google?.accounts?.id) {
          clearInterval(timer);
          initGsi();
        }
      }, 200);
      return () => clearInterval(timer);
    }
  }, [user]);

  /*
  PURPOSE: Takes the Google token and exchanges it for OUR backend token.
  */
  const handleGoogleCredential = async (response) => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ credential: response.credential }),
      });
      if (!res.ok) throw new Error("Auth failed");
      const data = await res.json();
      
      // Save our custom JWT to localStorage so they stay logged in
      localStorage.setItem("nc_token", data.token);
      setToken(data.token);
      setUser(data.user);
    } catch (err) {
      console.error("Google sign-in failed:", err);
      setError("Sign-in failed. Please try again.");
    }
  };

  const handleSignOut = () => {
    localStorage.removeItem("nc_token");
    setToken(null);
    setUser(null);
    setHistory([]);
    if (window.google?.accounts?.id) {
      window.google.accounts.id.disableAutoSelect();
    }
  };

  // ─── Fetch history ──────────────────────────────────────────────────
  /*
  PURPOSE: Grabs the user's past fact-checks from the database.
  */
  const fetchHistory = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/api/history`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      setHistory(data.checks || []);
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  }, [token]);

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  // ─── Fact-check submission ──────────────────────────────────────────
  /*
  PURPOSE: Sends the user's typed statement to our backend to be checked.
  */
  const handleSubmit = async (e) => {
    e.preventDefault(); // Stop the page from refreshing on form submit
    
    const trimmed = statement.trim();
    if (trimmed.length < 5) {
      setError("Enter at least 5 characters.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const headers = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`; // Send token if logged in
      
      const res = await fetch(`${API_BASE}/api/check`, {
        method: "POST",
        headers,
        body: JSON.stringify({ statement: trimmed, mode }),
      });
      
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(describeRequestFailure(res.status, errData));
      }
      
      const data = await res.json();
      setResult(data);
      
      if (token) fetchHistory(); // Refresh history sidebar if logged in
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  /*
  PURPOSE: When a user clicks a past item in their history sidebar, load it into the main view.
  */
  const handleHistorySelect = async (item) => {
    setHistoryOpen(false);
    try {
      const res = await fetch(`${API_BASE}/api/history/${item._id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error("Failed to load");
      
      const full = await res.json();
      setStatement(full.statement);
      setResult({
        statement: full.statement,
        claim_type: full.claimType,
        verdict: full.verdict,
        confidence: full.confidence,
        reasoning: full.reasoning,
        external_evidence_available: full.externalEvidenceAvailable,
        external_evidence_checked: full.externalEvidenceChecked,
        ml_score: full.mlScore,
        ml_verdict: full.mlVerdict,
        evidence_score: full.evidenceScore,
        evidence_stance: full.evidenceStance,
        combined_score: full.combinedScore,
        combined_verdict: full.combinedVerdict,
        assessment_status: full.assessmentStatus,
        claim_assessments: full.claimAssessments,
        top_evidence: full.topEvidence,
        processing_time_seconds: full.processingTime,
        // New structured schema — same shape the live /api/check response
        // uses, so history playback renders identically to a live check.
        verification: full.verification && {
          status: full.verification.status,
          reasoning: full.verification.reasoning,
          claim_kind: full.verification.claimKind,
          salience: full.verification.salience,
        },
        ml: full.ml && {
          available: full.ml.available,
          auxiliary_only: full.ml.auxiliaryOnly,
          score: full.ml.score,
          verdict: full.ml.verdict,
          threshold: full.ml.threshold,
        },
        coverage: full.coverage && {
          mode: full.coverage.mode,
          requested: full.coverage.requested,
          window_days: full.coverage.windowDays,
          reason: full.coverage.reason,
          stale_evidence_count: full.coverage.staleEvidenceCount,
        },
        retrieval: full.retrieval && {
          status: full.retrieval.status,
          candidate_count: full.retrieval.candidateCount,
          relevant_count: full.retrieval.relevantCount,
          diagnostics: full.retrieval.diagnostics,
          passage_ranking: full.retrieval.passageRanking,
        },
        explanation: full.explanation && {
          available: full.explanation.available,
          reason: full.explanation.reason,
          text: full.explanation.text,
          sentences: full.explanation.sentences,
          dropped_count: full.explanation.droppedCount,
          model: full.explanation.model,
        },
        nli: full.nli && {
          available: full.nli.available,
          status: full.nli.status,
          classified_count: full.nli.classifiedCount,
        },
        evidence: full.evidenceSummary && {
          supporting_count: full.evidenceSummary.supportingCount,
          contradicting_count: full.evidenceSummary.contradictingCount,
          neutral_count: full.evidenceSummary.neutralCount,
          independent_groups: full.evidenceSummary.independentGroups,
          independent_supporting: full.evidenceSummary.independentSupporting,
          independent_contradicting: full.evidenceSummary.independentContradicting,
        },
      });
    } catch {
      setStatement(item.statement);
    }
  };

  const handleHistoryDelete = async (id) => {
    try {
      await fetch(`${API_BASE}/api/history/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      setHistory((prev) => prev.filter((h) => h._id !== id));
    } catch (err) {
      console.error("Failed to delete:", err);
    }
  };

  // ─── Route-based rendering ──────────────────────────────────────────
  /*
  PURPOSE: Switches which page component is shown based on the URL hash.
  */
  const renderPage = () => {
    switch (page) {
      case "evaluation":
        return <ModelEvaluation />;
      case "comparison":
        return <ModelComparison />;
      case "how-it-works":
        return <HowItWorks />;
      default:
        return renderHome();
    }
  };

  const renderHome = () => (
    <>
      {/* Intro */}
      <section className="intro">
        <p className="intro-tag">Hybrid claim verification · reasoning + targeted evidence</p>
        <h2 className="intro-heading">
          Paste a claim. I'll tell you if it holds up.
        </h2>
        <p className="intro-desc">
          I identify the claim type, apply reliable checks where possible, and use
          external evidence when the claim needs fresh or independent verification.
        </p>
      </section>

      {/* Input */}
      <form className="input-card" onSubmit={handleSubmit} id="check-form">
        <textarea
          id="statement-input"
          ref={textareaRef}
          className="input-textarea"
          placeholder="e.g. The United States spends more on its military than the next 10 countries combined."
          value={statement}
          onChange={(e) => setStatement(e.target.value)}
          maxLength={2000}
          disabled={loading}
        />
        <div className="input-footer">
          {/* Coverage mode. "Auto" reads the claim's own wording, which only
              works when the claim says "today" — someone who just saw a story
              on television types it without a time word at all, and no parser
              recovers that. The choice belongs to the person checking. */}
          <div className="mode-picker" role="group" aria-label="Coverage to search">
            {[
              ["auto", "Auto", "Decide from the wording of the claim"],
              ["recent", "Recent", "Only coverage from the last month"],
              ["historical", "Historical", "Search without a date limit"],
            ].map(([value, label, title]) => (
              <button
                key={value}
                type="button"
                title={title}
                className={`mode-option${mode === value ? " active" : ""}`}
                onClick={() => setMode(value)}
                disabled={loading}
                aria-pressed={mode === value}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="char-count">{statement.length}/2000</span>
          <button
            type="submit"
            className="btn-check"
            disabled={loading || statement.trim().length < 5}
            id="btn-check"
          >
            {loading ? (
              <>
                <span className="spinner" />
                Checking…
              </>
            ) : (
              "Check this"
            )}
          </button>
        </div>
      </form>

      {/* What to expect note — evidence retrieval depends on live web search,
          which is imperfect (especially for very fresh headlines with no
          news-API keys configured). This is here so a reviewer reads a
          borderline/irrelevant source as a known retrieval limitation, not
          as the system being broken. */}
      <aside className="reviewer-note" aria-label="What to expect from results">
        <Info size={15} className="reviewer-note-icon" />
        <p>
          <strong>What to expect:</strong> this checks claims against live sources, so
          it answers with what the evidence shows rather than a guess. It distinguishes{" "}
          <strong>supported</strong>, <strong>contradicted</strong>,{" "}
          <strong>no credible source reports this</strong> (a real finding for a claim
          that would have been widely covered), <strong>not yet verifiable</strong> (the
          claim is about a future event), and <strong>could not verify</strong> (our
          search or model failed — a limitation on our side, never a statement about the
          claim). Articles that cover the topic without addressing the claim are listed
          separately under <em>Related coverage</em> and are never counted as evidence.
        </p>
      </aside>

      <section className="info-panel" aria-label="How it works">
        <h3 className="info-panel-title">How it works</h3>
        <div className="info-panel-grid">
          <div>
            <strong>1.</strong> Enter a claim or headline
          </div>
          <div>
            <strong>2.</strong> We work out what kind of claim it is
          </div>
          <div>
            <strong>3.</strong> We search for relevant evidence
          </div>
          <div>
            <strong>4.</strong> We compare the claim against the evidence
          </div>
          <div>
            <strong>5.</strong> We check supporting and contradicting sources
          </div>
          <div>
            <strong>6.</strong> We give a verdict, or say why we can't
          </div>
        </div>
        <div className="info-panel-list">
          <span>Claim meaning</span>
          <span>Relevant sources</span>
          <span>Supporting evidence</span>
          <span>Contradicting evidence</span>
          <span>Source credibility</span>
          <span>Independent reporting</span>
        </div>
      </section>

      <aside className="claim-examples" aria-label="Claim examples">
        <p className="claim-examples-title">Few examples</p>
        <p><span>"Water freezes at 0°C at sea level."</span> — true</p>
        <p><span>"A four-day workweek improves productivity."</span> — middle (50%) ish</p>
        <p><span>"The Great Wall of China is visible from the Moon with the naked eye."</span> — false</p>
      </aside>

      {/* Error */}
      {error && (
        <div className="error-banner" id="error-message">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && <LoadingSkeleton />}

      {/* Results */}
      {result && !loading && (
        <section className="results" id="results-section">
          <div className="results-label">
            <span>Result</span>
            {result.processing_time_seconds && (
              <span className="results-time">{result.processing_time_seconds}s</span>
            )}
          </div>

          {/* Core Score UI */}
          <div className="score-card">
            <ScoreGauge
              score={result.combined_score}
              assessmentStatus={result.assessment_status}
            />
            <div className="score-details">
              <p className="verdict-text" data-score={result.combined_score}>
                {result.combined_verdict}
              </p>
              <p className="verdict-statement">
                "{result.statement}"
              </p>
              {result.reasoning && (
                <p className="assessment-note">
                  {result.reasoning}
                </p>
              )}
              <p className="assessment-note">
                {result.claim_type} · {result.confidence} confidence
                {result.external_evidence_available
                  ? " · evidence used"
                  : result.external_evidence_checked
                    ? " · no qualifying external evidence"
                    : " · deterministic check"}
              </p>
              {result.external_evidence_checked && (() => {
                const retrievalStatus = result.retrieval?.status || "NO_RESULTS";
                const candidates = result.retrieval?.candidate_count ?? 0;
                const relevant = result.retrieval?.relevant_count ?? 0;
                return (
                  <>
                    {/* One line of retrieval provenance, in words rather
                        than an enum name. The reasoning line above already
                        explains the verdict; this says how wide the search
                        that produced it actually was. */}
                    <p className="assessment-note">
                      Retrieval: {candidates} candidate{candidates === 1 ? "" : "s"} ·{" "}
                      {relevant} on-topic · {result.nli?.classified_count ?? 0} checked against the claim
                    </p>
                    {/* Which slice of coverage was searched. Worth stating even
                        when the user picked the mode, and necessary when they
                        left it on Auto: a thin result reads very differently
                        once you know only the last 30 days was searched. */}
                    {result.coverage && (
                      <p className="assessment-note">
                        {result.coverage.window_days
                          ? `Searched the last ${result.coverage.window_days} days`
                          : "Searched without a date limit"}
                        {result.coverage.reason ? ` — ${result.coverage.reason}` : ""}
                        {result.coverage.stale_evidence_count > 0 && (
                          `. ${result.coverage.stale_evidence_count} source` +
                          `${result.coverage.stale_evidence_count === 1 ? " was" : "s were"}` +
                          ` found but published too long ago to be reporting this.`
                        )}
                      </p>
                    )}
                    {retrievalStatus === "SEARCH_FAILED" && (
                      <p className="assessment-note">
                        Search providers could not be reached, so nothing here reflects on the claim itself.
                      </p>
                    )}
                    {result.nli && !result.nli.available && (
                      <p className="assessment-note">
                        Evidence classification is currently unavailable (NLI model: {result.nli.status}).
                      </p>
                    )}
                  </>
                );
              })()}
              {/* What was actually searched. The response has carried
                  per-provider diagnostics all along and nothing displayed
                  them, so a thin result was indistinguishable from a
                  misconfigured one — the single most common reason results
                  look wrong is a provider that never ran. Collapsed by
                  default so it doesn't compete with the verdict. */}
              {result.external_evidence_checked &&
                result.retrieval?.diagnostics?.length > 0 && (
                <details className="retrieval-details">
                  <summary>How this was checked</summary>
                  <ul className="retrieval-provider-list">
                    {Object.entries(
                      result.retrieval.diagnostics.reduce((acc, d) => {
                        const name = d.provider || "unknown";
                        if (!acc[name]) acc[name] = { queries: 0, results: 0, statuses: {} };
                        acc[name].queries += 1;
                        acc[name].results += d.normalized_result_count || 0;
                        acc[name].statuses[d.status] = (acc[name].statuses[d.status] || 0) + 1;
                        return acc;
                      }, {}),
                    ).map(([name, info]) => {
                      // Report the least healthy outcome that provider had:
                      // "3 of 4 queries succeeded" hides the one that didn't.
                      const worst =
                        ["failed", "timeout", "disabled", "no_results", "success"].find(
                          (s) => info.statuses[s],
                        ) || "unknown";
                      return (
                        <li key={name}>
                          <span className="retrieval-provider">{name}</span>
                          <span className={`retrieval-status ${worst}`}>{worst}</span>
                          <span className="retrieval-count">
                            {info.results} result{info.results === 1 ? "" : "s"} from{" "}
                            {info.queries} quer{info.queries === 1 ? "y" : "ies"}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                  <p className="assessment-note">
                    Providers marked <em>disabled</em> have no API key configured. A thin
                    result usually means fewer providers ran, not that nothing exists.
                  </p>
                </details>
              )}

              <ScoreBreakdown
                mlScore={result.ml_score}
                evidenceScore={result.evidence_score}
                stanceNet={result.evidence_stance?.net || 0}
                hasClassifiedEvidence={(result.nli?.classified_count || 0) > 0}
                hasDirectionalEvidence={
                  (result.evidence?.supporting_count || 0) +
                    (result.evidence?.contradicting_count || 0) >
                  0
                }
              />
            </div>
          </div>

          {/* Evidence Articles.

              Split deliberately. Every source the NLI model classified as
              neither supporting nor contradicting the claim used to sit in
              the same grid under a heading counting it as evidence, which is
              how an on-topic-but-unrelated article came across as the system
              claiming it as proof. Sources that take a position on the claim
              are evidence; the rest is context, and is labelled as context. */}
          {result.top_evidence && result.top_evidence.length > 0 && (() => {
            const addresses = (ev) =>
              ev.nli_available && (ev.stance === "supports" || ev.stance === "contradicts");
            const evidence = result.top_evidence.filter(addresses);
            const context = result.top_evidence.filter((ev) => !addresses(ev));
            const candidateCount = result.retrieval?.candidate_count;

            return (
              <div className="evidence-section" id="evidence-section">
                {evidence.length > 0 && (() => {
                  // Independent publishers backing THIS verdict's direction,
                  // mirroring the backend's _independent_backing. Four
                  // reprints of one wire story are one confirmation, and this
                  // is the only number on screen that says so.
                  //
                  // Taking the larger of the two sides was wrong whenever the
                  // verdict went against the more numerous one: on a viral
                  // false claim, six anonymous blogs "supporting" it and two
                  // credible publishers refuting it yields a `contradicted`
                  // verdict — and this line reported "6 independent
                  // publishers", counting the sources the verdict rejected.
                  const publishers = publishersBackingVerdict(result);
                  return (
                  <>
                    <p className="section-label">
                      Evidence used — {evidence.length} source{evidence.length === 1 ? "" : "s"}
                      {publishers > 0
                        ? ` · ${publishers} independent publisher${publishers === 1 ? "" : "s"}`
                        : ""}
                      {candidateCount ? ` · ${candidateCount} candidates searched` : ""}
                    </p>
                    <div className="evidence-grid">
                      {evidence.map((ev, i) => (
                        <EvidenceCard key={ev.url || i} evidence={ev} index={i} />
                      ))}
                    </div>
                  </>
                  );
                })()}

                {context.length > 0 && (
                  <>
                    <p className="section-label">
                      {evidence.length > 0 ? "Related coverage" : "Related coverage — no evidence used"}
                      {" — "}{context.length} source{context.length === 1 ? "" : "s"}
                    </p>
                    <p className="assessment-note evidence-empty">
                      {result.nli?.status === "ready"
                        ? "These articles cover the same subject but state neither that the claim is true nor that it is false. They are shown so you can see what was searched — they are not being counted as evidence."
                        : "Evidence classification is currently unavailable, so these are unchecked candidates rather than evidence."}
                    </p>
                    <div className="evidence-grid">
                      {context.map((ev, i) => (
                        <EvidenceCard key={ev.url || i} evidence={ev} index={i} />
                      ))}
                    </div>
                  </>
                )}
              </div>
            );
          })()}
          {result.external_evidence_checked &&
            (!result.top_evidence || result.top_evidence.length === 0) &&
            !result.external_evidence_available && (
            <p className="assessment-note evidence-empty">
              External sources were checked, but none were relevant and reliable enough to show as evidence.
            </p>
          )}
        </section>
      )}
    </>
  );

  return (
    <div className="app-layout">
      <Header
        user={user}
        onSignOut={handleSignOut}
        onHistoryToggle={() => setHistoryOpen(!historyOpen)}
        historyCount={history.length}
        currentPage={page}
        onNavigate={navigate}
      />

      <main className="app-main">
        {renderPage()}
      </main>

      <footer className="app-footer">
        <p>
          Built by Shoaib — deterministic checks for clear facts, with external evidence
          for claims that require it
        </p>
      </footer>

      {historyOpen && (
        <HistoryPanel
          history={history}
          onClose={() => setHistoryOpen(false)}
          onSelect={handleHistorySelect}
          onDelete={handleHistoryDelete}
        />
      )}
    </div>
  );
}

export default App;
