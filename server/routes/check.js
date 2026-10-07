/*
FILE PURPOSE:
This route acts as a "Proxy" (a middleman) between the React Frontend and the Python ML Service.

FLOW:
1. Receives a statement from the frontend.
2. Forwards (proxies) that statement to the Python FastAPI service.
3. Receives the verdict and classified evidence from Python.
4. If the user is logged in, saves the result to MongoDB.
5. Sends the final result back to the frontend.

USED BY:
- The main input form on the React frontend when a user clicks "Check Statement".
*/

import { Router } from "express";
// fetch is built into Node 18+, so there is no package to import. Using the
// global also makes this route testable: an imported binding cannot be
// stubbed, which is why the test asserting "no upstream request for invalid
// input" was passing without ever checking anything.
import Check from "../models/Check.js";
import { optionalAuth } from "../middleware/auth.js";

const router = Router();

// URL of the Python ML Service
// In development, this is usually http://localhost:8000
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";

// Must match ml-service's CheckRequest.mode pattern. An unknown value
// falls back to "auto" rather than 400ing: the mode is a search hint, and
// refusing the whole check over it would be worse than ignoring it.
const VALID_MODES = new Set(["auto", "recent", "historical"]);

// The evidence pipeline can involve several sequential network calls
// (multiple providers x multiple queries, then article fetches), so this
// needs real headroom — but it must still have a ceiling so a hung
// ml-service can't hang every Express request indefinitely.
//
// 180s (not something tighter) because this project runs locally for demos:
// the NLI model downloads/loads on the first evidence-requiring request of
// the process's lifetime, and without GNEWS_API_KEY/GUARDIAN_API_KEY/
// NEWSAPI_KEY configured, retrieval falls back entirely to DuckDuckGo across
// up to 4 queries, each with its own retries — comfortably able to exceed a
// minute on that first request alone. Subsequent requests are much faster
// once the model is warm in memory.
const ML_SERVICE_TIMEOUT_MS = Number(process.env.ML_SERVICE_TIMEOUT_MS) || 180_000;

// Mirrors CheckRequest.max_length in ml-service/main.py. Kept in sync by hand;
// if they drift, the looser side simply produces a worse error message.
const MAX_STATEMENT_LENGTH = 2000;

/*
PURPOSE:
Process a fact-check request.

INPUT:
req.body.statement (The string the user typed)

OUTPUT:
The verdict, evidence and explanation from the ML service.

WHY THIS EXISTS:
Why not have the frontend talk directly to Python? 
Because we need to save the result in our MongoDB database to show the user's history. 
Our Node server handles the database, so the request must pass through here first.
*/
router.post("/", optionalAuth, async (req, res) => {
  try {
    const { statement, mode } = req.body;

    // Validated here as well as in ml-service. This proxy forwards an explicit
    // allowlist of fields rather than req.body, so anything not named here
    // silently never reaches FastAPI — which is how a new request field looks
    // like a backend that ignores it.
    const coverageMode = VALID_MODES.has(mode) ? mode : "auto";

    // Step 1: Basic validation.
    // Both bounds must match ml-service's CheckRequest (min_length=5,
    // max_length=2000). Without the upper bound, an over-long statement was
    // forwarded, rejected by Pydantic with a 422, and relayed to the user as
    // a 502 "ML service error" carrying a raw validation payload — a broken
    // backend, apparently, rather than "your text is too long".
    if (!statement || typeof statement !== "string") {
      return res.status(400).json({ error: "Statement must be text." });
    }
    const trimmed = statement.trim();
    if (trimmed.length < 5) {
      return res.status(400).json({ error: "Statement must be at least 5 characters." });
    }
    if (trimmed.length > MAX_STATEMENT_LENGTH) {
      return res.status(400).json({
        error: `Statement must be at most ${MAX_STATEMENT_LENGTH} characters.`,
      });
    }

    // Step 2: Forward the request to the Python FastAPI service
    // We send a POST request to FastAPI, passing along the statement.
    // AbortController bounds how long a hung ml-service can block this
    // request — without it, a stuck upstream call would hang forever.
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), ML_SERVICE_TIMEOUT_MS);

    let mlResponse;
    try {
      mlResponse = await fetch(`${FASTAPI_URL}/api/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ statement: trimmed, mode: coverageMode }),
        signal: controller.signal,
      });
    } catch (fetchError) {
      if (fetchError.name === "AbortError") {
        console.error("FastAPI request timed out after", ML_SERVICE_TIMEOUT_MS, "ms");
        return res.status(504).json({ error: "ML service timed out." });
      }
      throw fetchError;
    } finally {
      clearTimeout(timeout);
    }

    // Step 3: Handle potential errors from the Python service
    if (!mlResponse.ok) {
      const errorText = await mlResponse.text();
      // Log the URL, not just the status. A bare "FastAPI error: 502" gives
      // no way to tell a broken local service apart from FASTAPI_URL still
      // pointing at a decommissioned remote one — which returns exactly
      // that, while the local ml-service sits idle and logs nothing.
      console.error(
        `FastAPI error: ${mlResponse.status} from ${FASTAPI_URL}/api/check`,
        errorText ? `— ${errorText.slice(0, 500)}` : "(empty body)",
      );
      // 502 Bad Gateway means our server (Node) couldn't get a valid response from the upstream server (Python).
      // Outside production, name the upstream that failed: "ML service
      // error." alone sends you debugging the wrong process.
      return res.status(502).json({
        error: "ML service error.",
        details: errorText,
        ...(process.env.NODE_ENV === "production"
          ? {}
          : { upstream: `${FASTAPI_URL}/api/check`, upstreamStatus: mlResponse.status }),
      });
    }

    // Step 4: Parse the JSON response from Python
    const result = await mlResponse.json();

    // Step 5: Save to MongoDB (ONLY if the user is logged in)
    let savedCheck = null;
    
    // req.user is populated by the `optionalAuth` middleware if a valid token was sent
    if (req.user) {
      try {
        // Create a new record in the 'checks' collection
        savedCheck = await Check.create({
          userId: req.user.userId,
          statement: result.statement,
          evidenceScore: result.evidence_score,
          evidenceStance: result.evidence_stance,
          combinedScore: result.combined_score,
          combinedVerdict: result.combined_verdict,
          assessmentStatus: result.assessment_status,
          claimAssessments: (result.claim_assessments || []).map((assessment) => ({
            claim: assessment.claim,
            status: assessment.status,
            verdict: assessment.verdict,
            support: assessment.support,
            contradiction: assessment.contradiction,
            evidenceCount: assessment.evidence_count,
          })),
          topEvidence: result.top_evidence,
          processingTime: result.processing_time_seconds,

          claimType: result.claim_type,
          verdict: result.verdict,
          confidence: result.confidence,
          reasoning: result.reasoning,
          externalEvidenceAvailable: result.external_evidence_available,
          externalEvidenceChecked: result.external_evidence_checked,
          verification: result.verification && {
            status: result.verification.status,
            reasoning: result.verification.reasoning,
            claimKind: result.verification.claim_kind,
            salience: result.verification.salience,
          },
          coverage: result.coverage && {
            mode: result.coverage.mode,
            requested: result.coverage.requested,
            windowDays: result.coverage.window_days,
            reason: result.coverage.reason,
            staleEvidenceCount: result.coverage.stale_evidence_count,
          },
          retrieval: result.retrieval && {
            status: result.retrieval.status,
            candidateCount: result.retrieval.candidate_count,
            relevantCount: result.retrieval.relevant_count,
            diagnostics: result.retrieval.diagnostics,
            passageRanking: result.retrieval.passage_ranking,
          },
          // Stored as generated: regenerating on replay could produce
          // different, unchecked text.
          explanation: result.explanation && {
            available: result.explanation.available,
            reason: result.explanation.reason,
            text: result.explanation.text,
            sentences: result.explanation.sentences,
            droppedCount: result.explanation.dropped_count,
            model: result.explanation.model,
          },
          nli: result.nli && {
            available: result.nli.available,
            status: result.nli.status,
            classifiedCount: result.nli.classified_count,
          },
          evidenceSummary: result.evidence && {
            supportingCount: result.evidence.supporting_count,
            contradictingCount: result.evidence.contradicting_count,
            neutralCount: result.evidence.neutral_count,
            independentGroups: result.evidence.independent_groups,
            independentSupporting: result.evidence.independent_supporting,
            independentContradicting: result.evidence.independent_contradicting,
          },
        });
      } catch (dbError) {
        // If saving fails, we log it, but we DON'T crash the request.
        // We still want to show the user their result even if history failed to save.
        console.error("Failed to save check to DB:", dbError.message);
      }
    }

    // Step 6: Send the data back to the React frontend
    res.json({
      ...result, // Spread all properties from the Python result
      _id: savedCheck?._id || null, // Attach the MongoDB ID if we saved it (useful for UI updates)
    });
    
  } catch (error) {
    console.error("Check proxy error:", error.message);
    res.status(500).json({ error: "Failed to process fact-check request." });
  }
});

export default router;
