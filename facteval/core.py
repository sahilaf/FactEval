"""
Core – The main check() and verify() functions that wire the FactEval pipeline.

Usage:
    from facteval import check, verify

    # Full pipeline (extract + retrieve + verify)
    result = check(answer, contexts)

    # Lightweight mode (skip extraction, bring your own claims)
    result = verify(claims=["claim 1", "claim 2"], contexts=docs)
"""

import re
import logging
import time
from pathlib import Path

import numpy as np

from facteval.calibrator import Calibrator
from facteval.claim_extractor import ClaimExtractor
from facteval.retriever import EvidenceRetriever
from facteval.verifier import Verifier, FactLabel
from facteval.models import Claim

logger = logging.getLogger(__name__)

# Module-level singletons (lazy-loaded)
_extractor: ClaimExtractor | None = None
_retriever: EvidenceRetriever | None = None
_verifier: Verifier | None = None
_calibrator: Calibrator | None = None
_calibrator_path: str | None = None


def _get_extractor() -> ClaimExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ClaimExtractor()
    return _extractor


def _get_retriever() -> EvidenceRetriever:
    global _retriever
    if _retriever is None:
        _retriever = EvidenceRetriever()
    return _retriever


def _get_verifier() -> Verifier:
    global _verifier
    if _verifier is None:
        _verifier = Verifier()
    return _verifier


def _get_calibrator(path: str | None = None) -> Calibrator:
    global _calibrator, _calibrator_path
    if _calibrator is None or path != _calibrator_path:
        _calibrator = Calibrator(calibrator_path=path)
        _calibrator_path = path
    return _calibrator


# ── Full pipeline ────────────────────────────────────────────────────────────

def check(
    answer: str,
    contexts: list[str],
    top_k: int = 3,
    max_claims: int = 10,
    calibrator_path: str | Path | None = None,
) -> dict:
    """
    Run the full FactEval pipeline on an answer + contexts.

    Stages: extract claims → retrieve evidence → NLI verify → calibrate.

    Args:
        answer:          The LLM-generated text to evaluate.
        contexts:        List of reference passages (ground truth).
        top_k:           Number of evidence sentences to retrieve per claim.
        max_claims:      Maximum claims to extract.
        calibrator_path: Path to a pre-fitted calibrator pickle file.

    Returns:
        A dict with claims, summary, highlighted_answer, and pipeline_time.
    """
    t0 = time.perf_counter()

    # 1. Extract claims
    extractor = _get_extractor()
    claims = extractor.extract(answer, max_claims=max_claims)
    logger.info("Extracted %d claims.", len(claims))

    if not claims:
        return {
            "claims": [],
            "summary": _build_summary([]),
            "highlighted_answer": answer,
            "calibrated": False,
            "pipeline_time_seconds": round(time.perf_counter() - t0, 3),
        }

    # 2–5. Shared pipeline
    return _run_pipeline(claims, contexts, answer, top_k, calibrator_path, t0)


# ── Lightweight mode ─────────────────────────────────────────────────────────

def verify(
    claims: list[str],
    contexts: list[str],
    top_k: int = 3,
    calibrator_path: str | Path | None = None,
) -> dict:
    """
    Verify pre-extracted claims against contexts. Skips claim extraction.

    Use this when you already have claims and want faster results
    (avoids the ~1s extraction step and the Qwen model entirely).

    Args:
        claims:          List of claim strings to verify.
        contexts:        List of reference passages (ground truth).
        top_k:           Number of evidence sentences to retrieve per claim.
        calibrator_path: Path to a pre-fitted calibrator pickle file.

    Returns:
        Same output format as check().
    """
    t0 = time.perf_counter()

    claim_objs = [Claim(text=c) for c in claims if c.strip()]

    if not claim_objs:
        return {
            "claims": [],
            "summary": _build_summary([]),
            "highlighted_answer": "",
            "calibrated": False,
            "pipeline_time_seconds": round(time.perf_counter() - t0, 3),
        }

    answer = " ".join(claims)  # reconstruct for highlighting
    return _run_pipeline(claim_objs, contexts, answer, top_k, calibrator_path, t0)


# ── Shared pipeline ──────────────────────────────────────────────────────────

def _run_pipeline(
    claims: list[Claim],
    contexts: list[str],
    answer: str,
    top_k: int,
    calibrator_path: str | Path | None,
    t0: float,
) -> dict:
    """Shared pipeline: retrieve → verify → calibrate → diagnose → highlight."""

    # 2. Retrieve evidence
    retriever = _get_retriever()
    retriever.index(contexts)
    claims_with_evidence = retriever.retrieve_for_claims(claims, top_k=top_k)

    # 3. Verify (batch NLI)
    verifier = _get_verifier()
    results = verifier.verify_batch(claims_with_evidence)

    # 4. Calibrate
    calibrator = _get_calibrator(str(calibrator_path) if calibrator_path else None)
    for r in results:
        if r.raw_scores:
            cal_conf, cal_err = calibrator.calibrate(r.raw_scores)
            r.calibrated_confidence = cal_conf
            r.calibration_error = cal_err

    # 5. Build output with diagnostics
    elapsed = time.perf_counter() - t0
    claim_dicts = [r.to_dict() for r in results]

    # Add diagnostics to each claim
    for cd in claim_dicts:
        cd["diagnostics"] = _diagnose(cd)

    return {
        "claims": claim_dicts,
        "summary": _build_summary(results),
        "highlighted_answer": _highlight_answer_semantic(
            answer, claim_dicts, retriever.embedder
        ),
        "calibrated": calibrator.is_calibrated,
        "pipeline_time_seconds": round(elapsed, 3),
    }


# ── Diagnostics ──────────────────────────────────────────────────────────────

def _diagnose(claim_dict: dict) -> dict:
    """
    Generate pipeline diagnostics for a claim.

    Tells the developer *why* a claim got its label —
    was it a retrieval failure or a genuine hallucination?
    """
    label = claim_dict["label"]
    ev_score = claim_dict.get("evidence_score")
    confidence = claim_dict.get("confidence", 0)

    # Retrieval quality assessment
    if ev_score is None:
        retrieval_quality = "none"
    elif ev_score >= 0.7:
        retrieval_quality = "strong"
    elif ev_score >= 0.4:
        retrieval_quality = "moderate"
    else:
        retrieval_quality = "weak"

    # Failure type classification
    if label == "supported":
        failure_type = "verified"
        suggestion = None
    elif label == "contradicted":
        if retrieval_quality in ("strong", "moderate"):
            failure_type = "hallucination"
            suggestion = "Claim directly contradicts the evidence. This is a factual error in the LLM output."
        else:
            failure_type = "possible_hallucination"
            suggestion = "Claim contradicts weak evidence. Consider adding more specific context for reliable verification."
    elif ev_score is None:
        failure_type = "no_evidence"
        suggestion = "No relevant context was provided. Add reference passages covering this topic."
    elif retrieval_quality == "weak":
        failure_type = "retrieval_gap"
        suggestion = "Evidence was found but too dissimilar to trust. The context may not cover this claim."
    else:
        failure_type = "inconclusive"
        suggestion = "Evidence exists but is neutral — neither confirms nor denies the claim."

    d = {
        "failure_type": failure_type,
        "retrieval_quality": retrieval_quality,
    }
    if suggestion:
        d["suggestion"] = suggestion
    return d


# ── Summary ──────────────────────────────────────────────────────────────────

def _build_summary(results: list) -> dict:
    """Build summary statistics from verification results."""
    total = len(results)
    supported = sum(1 for r in results if r.label == FactLabel.SUPPORTED)
    contradicted = sum(1 for r in results if r.label == FactLabel.CONTRADICTED)
    unverifiable = total - supported - contradicted

    return {
        "total_claims": total,
        "supported": supported,
        "contradicted": contradicted,
        "unverifiable": unverifiable,
        "hallucination_rate": round(contradicted / max(total, 1), 4),
    }


# ── Semantic Highlighting ────────────────────────────────────────────────────

_LABEL_EMOJI = {"supported": "✅", "contradicted": "❌", "unverifiable": "❓"}
_LABEL_COLOR = {"supported": "#22c55e", "contradicted": "#ef4444", "unverifiable": "#f59e0b"}


def _highlight_answer_semantic(answer: str, claim_dicts: list[dict], embedder) -> str:
    """
    Map claims to source sentences using embedding similarity (not Jaccard).

    Uses the retriever's SentenceTransformer to compute cosine similarity
    between each claim and each sentence in the original answer. This handles
    paraphrasing, reordering, and partial overlaps much better than token overlap.
    """
    if not answer.strip() or not claim_dicts:
        return answer

    # Split answer into sentences with positions
    sentences = []
    for m in re.finditer(r'[^.!?]+[.!?]*', answer):
        text = m.group().strip()
        if text:
            sentences.append(text)

    if not sentences:
        return answer

    # Compute embedding similarity
    claim_texts = [c["claim"] for c in claim_dicts]
    claim_labels = [c["label"] for c in claim_dicts]

    sent_embeddings = embedder.encode(sentences, normalize_embeddings=True)
    claim_embeddings = embedder.encode(claim_texts, normalize_embeddings=True)

    # Similarity matrix: sentences × claims
    sim_matrix = np.dot(sent_embeddings, claim_embeddings.T)

    # For each sentence, find best matching claim
    sentence_labels: dict[str, str] = {}
    for i, sent_text in enumerate(sentences):
        best_j = int(sim_matrix[i].argmax())
        best_sim = float(sim_matrix[i, best_j])

        if best_sim > 0.35:  # Semantic similarity threshold
            sentence_labels[sent_text] = claim_labels[best_j]

    # Build highlighted text (longest matches first to avoid partial replacements)
    highlighted = answer
    for sent_text in sorted(sentence_labels, key=len, reverse=True):
        label = sentence_labels[sent_text]
        color = _LABEL_COLOR.get(label, "#94a3b8")
        emoji = _LABEL_EMOJI.get(label, "")
        highlighted = highlighted.replace(
            sent_text,
            f'<mark style="background:{color}30;padding:2px 4px;border-radius:3px">'
            f'{sent_text} {emoji}</mark>',
            1,
        )

    return highlighted
