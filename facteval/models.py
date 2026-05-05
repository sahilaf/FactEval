"""
Pydantic data models for FactEval's pipeline objects.
"""

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A single atomic, verifiable claim extracted from text."""

    text: str = Field(..., description="The claim statement.")
    source_text: str = Field(
        default="",
        description="The original text this claim was extracted from.",
    )

    def __str__(self) -> str:
        return self.text


class Evidence(BaseModel):
    """A single piece of evidence retrieved for a claim."""

    sentence: str = Field(..., description="The evidence sentence.")
    score: float = Field(
        ..., ge=0.0, description="Cosine similarity score (may slightly exceed 1.0 due to float precision)."
    )
    source_context: str = Field(
        default="",
        description="The full context passage this sentence came from.",
    )

    def __str__(self) -> str:
        return f"[{self.score:.3f}] {self.sentence}"


class ClaimWithEvidence(BaseModel):
    """A claim paired with its retrieved evidence."""

    claim: Claim
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def best_evidence(self) -> Evidence | None:
        """Return the highest-scoring evidence, or None."""
        return self.evidence[0] if self.evidence else None
