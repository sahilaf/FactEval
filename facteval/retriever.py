"""
Evidence Retriever – FAISS-based semantic search over user-provided contexts.

Encodes context sentences with all-MiniLM-L6-v2 and retrieves the top-k
most similar evidence sentences for each claim.
"""

import re
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from facteval import suppress_stdout
from facteval.config import DEFAULT_TOP_K, EMBEDDING_MODEL, MIN_EVIDENCE_SCORE
from facteval.models import Claim, Evidence, ClaimWithEvidence

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Build a FAISS index over context sentences and retrieve evidence for claims."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        logger.info("Loading embedding model: %s", model_name)
        with suppress_stdout():
            self.embedder = SentenceTransformer(model_name, device=self.device)

        # Populated by .index()
        self._sentences: list[str] = []
        self._sentence_to_context: dict[int, str] = {}
        self._index: faiss.IndexFlatIP | None = None

    def index(self, contexts: list[str]) -> "EvidenceRetriever":
        """
        Build a FAISS index from a list of context passages.

        Each context is split into individual sentences before indexing.

        Args:
            contexts: List of context passages (strings).

        Returns:
            self (for chaining: `retriever.index(ctx).retrieve(claim)`).
        """
        if not contexts:
            logger.warning("No contexts provided; retriever will return empty results.")
            self._sentences = []
            self._index = None
            return self

        self._sentences = []
        self._sentence_to_context = {}

        for ctx in contexts:
            for sent in self._split_sentences(ctx):
                idx = len(self._sentences)
                self._sentences.append(sent)
                self._sentence_to_context[idx] = ctx

        if not self._sentences:
            logger.warning("No sentences extracted from contexts.")
            self._index = None
            return self

        logger.info("Indexing %d evidence sentences.", len(self._sentences))
        embeddings = self.embedder.encode(
            self._sentences, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Cosine similarity (normalized)
        self._index.add(embeddings)

        return self

    def retrieve(
        self,
        claim: Claim | str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_EVIDENCE_SCORE,
    ) -> list[Evidence]:
        """
        Retrieve the top-k most relevant evidence sentences for a claim.

        Args:
            claim: A Claim object or plain string.
            top_k: Number of evidence sentences to return.
            min_score: Minimum cosine similarity to include.

        Returns:
            List of Evidence objects, sorted by score descending.
        """
        if self._index is None or not self._sentences:
            return []

        query_text = claim.text if isinstance(claim, Claim) else claim
        q_emb = self.embedder.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self._index.search(q_emb, top_k)
        results: list[Evidence] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._sentences):
                continue
            clamped_score = float(min(max(score, 0.0), 1.0))
            if clamped_score < min_score:
                continue
            results.append(
                Evidence(
                    sentence=self._sentences[idx],
                    score=clamped_score,
                    source_context=self._sentence_to_context.get(idx, ""),
                )
            )

        return results

    def retrieve_for_claims(
        self,
        claims: list[Claim],
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_EVIDENCE_SCORE,
    ) -> list[ClaimWithEvidence]:
        """
        Batch-retrieve evidence for a list of claims.

        Returns:
            List of ClaimWithEvidence objects.
        """
        return [
            ClaimWithEvidence(
                claim=claim,
                evidence=self.retrieve(claim, top_k=top_k, min_score=min_score),
            )
            for claim in claims
        ]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on sentence-ending punctuation."""
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip() and len(s.strip()) > 3]
