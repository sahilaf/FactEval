"""
Verifier – NLI-based factual verification of claims against evidence.

Uses DeBERTa-v3 fine-tuned on MNLI+FEVER+ANLI to classify each
claim/evidence pair as entailment, contradiction, or neutral.
Maps NLI labels to FactEval labels: supported, contradicted, unverifiable.
"""

import logging
from enum import Enum

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from facteval import suppress_stdout
from facteval.config import NLI_MODEL, MIN_EVIDENCE_SCORE
from facteval.models import Claim, Evidence, ClaimWithEvidence

logger = logging.getLogger(__name__)


class FactLabel(str, Enum):
    """FactEval verdict labels."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


# Map DeBERTa NLI labels → FactEval labels
_NLI_TO_FACT = {
    "entailment": FactLabel.SUPPORTED,
    "contradiction": FactLabel.CONTRADICTED,
    "neutral": FactLabel.UNVERIFIABLE,
}


class VerificationResult:
    """Result of verifying a single claim."""

    def __init__(
        self,
        claim: str,
        label: FactLabel,
        confidence: float,
        evidence: str | None,
        evidence_score: float | None,
        raw_scores: dict[str, float],
        reason: str = "",
        calibrated_confidence: float | None = None,
        calibration_error: float | None = None,
    ):
        self.claim = claim
        self.label = label
        self.confidence = confidence
        self.evidence = evidence
        self.evidence_score = evidence_score
        self.raw_scores = raw_scores
        self.reason = reason
        self.calibrated_confidence = calibrated_confidence
        self.calibration_error = calibration_error

    def to_dict(self) -> dict:
        d = {
            "claim": self.claim,
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "evidence": self.evidence,
            "evidence_score": round(self.evidence_score, 4) if self.evidence_score else None,
            "raw_nli_scores": {k: round(v, 4) for k, v in self.raw_scores.items()},
        }
        if self.calibrated_confidence is not None:
            d["calibrated_confidence"] = round(self.calibrated_confidence, 4)
        if self.calibration_error is not None:
            d["calibration_error"] = round(self.calibration_error, 4)
        return d


class Verifier:
    """Verify claims against evidence using NLI."""

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading NLI model: %s on %s", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        with suppress_stdout():
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        logger.info("Verifier ready. Labels: %s", self.id2label)

    def verify(
        self,
        claim_with_evidence: ClaimWithEvidence,
        min_evidence_score: float = MIN_EVIDENCE_SCORE,
    ) -> VerificationResult:
        """
        Verify a single claim against its retrieved evidence.

        If no evidence meets the min_score threshold, returns 'unverifiable'
        with zero confidence.
        """
        claim_text = claim_with_evidence.claim.text
        best = claim_with_evidence.best_evidence

        # Fallback: no usable evidence
        if best is None or best.score < min_evidence_score:
            logger.debug("No evidence for claim: %s", claim_text)
            return VerificationResult(
                claim=claim_text,
                label=FactLabel.UNVERIFIABLE,
                confidence=0.0,
                evidence=None,
                evidence_score=None,
                raw_scores={},
                reason="No relevant evidence found in the provided context.",
            )

        # Run NLI: premise=evidence, hypothesis=claim
        return self._run_nli(claim_text, best.sentence, best.score)

    def verify_batch(
        self,
        claims_with_evidence: list[ClaimWithEvidence],
        min_evidence_score: float = MIN_EVIDENCE_SCORE,
    ) -> list[VerificationResult]:
        """
        Verify a batch of claims using batched NLI inference.

        Claims without evidence are immediately marked unverifiable.
        Remaining claims are processed in a single forward pass for speed.
        """
        results: list[VerificationResult | None] = [None] * len(claims_with_evidence)
        nli_pairs: list[tuple[int, str, str, float]] = []

        for i, cwe in enumerate(claims_with_evidence):
            claim_text = cwe.claim.text
            best = cwe.best_evidence

            if best is None or best.score < min_evidence_score:
                results[i] = VerificationResult(
                    claim=claim_text,
                    label=FactLabel.UNVERIFIABLE,
                    confidence=0.0,
                    evidence=None,
                    evidence_score=None,
                    raw_scores={},
                    reason="No relevant evidence found in the provided context.",
                )
            else:
                nli_pairs.append((i, claim_text, best.sentence, best.score))

        # Batch NLI inference for all claims with evidence
        if nli_pairs:
            indices, claims, evidences, scores = zip(*nli_pairs)
            inputs = self.tokenizer(
                list(evidences), list(claims),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            all_probs = torch.softmax(logits, dim=-1).cpu()

            for idx, probs_t, claim, evidence, ev_score in zip(
                indices, all_probs, claims, evidences, scores
            ):
                probs = probs_t.tolist()
                label_probs = {self.id2label[i]: float(p) for i, p in enumerate(probs)}
                predicted_nli = self.id2label[probs_t.argmax().item()]
                fact_label = _NLI_TO_FACT.get(predicted_nli, FactLabel.UNVERIFIABLE)

                results[idx] = VerificationResult(
                    claim=claim,
                    label=fact_label,
                    confidence=max(probs),
                    evidence=evidence,
                    evidence_score=ev_score,
                    raw_scores=label_probs,
                    reason=self._make_reason(fact_label, evidence),
                )

        return results

    def _run_nli(
        self, claim: str, evidence: str, evidence_score: float
    ) -> VerificationResult:
        """Run NLI inference on a single claim/evidence pair."""
        inputs = self.tokenizer(
            evidence, claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        label_probs = {self.id2label[i]: float(p) for i, p in enumerate(probs)}
        predicted_nli = self.id2label[logits.argmax().item()]
        fact_label = _NLI_TO_FACT.get(predicted_nli, FactLabel.UNVERIFIABLE)

        return VerificationResult(
            claim=claim,
            label=fact_label,
            confidence=max(probs),
            evidence=evidence,
            evidence_score=evidence_score,
            raw_scores=label_probs,
            reason=self._make_reason(fact_label, evidence),
        )

    @staticmethod
    def _make_reason(label: FactLabel, evidence: str) -> str:
        """Generate a human-readable reason for the verdict."""
        ev_short = evidence[:80] + "..." if len(evidence) > 80 else evidence
        if label == FactLabel.SUPPORTED:
            return f"Matched evidence: \"{ev_short}\""
        elif label == FactLabel.CONTRADICTED:
            return f"Contradicted by: \"{ev_short}\""
        else:
            return f"No strong match — evidence is neutral: \"{ev_short}\""
