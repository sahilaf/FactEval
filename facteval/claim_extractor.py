"""
Claim Extractor – Breaks text into atomic, verifiable claims.

Uses Qwen2.5-1.5B-Instruct (chosen in Week 0 for speed and output quality)
with the model's chat template to produce clean numbered lists.
"""

import re
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from facteval import suppress_stdout

from facteval.config import (
    CLAIM_MODEL,
    CLAIM_SYSTEM_PROMPT,
    CLAIM_USER_PROMPT,
    MAX_CLAIMS,
    MAX_NEW_TOKENS,
)
from facteval.models import Claim

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extract atomic claims from text using a causal LM with chat prompting."""

    def __init__(
        self,
        model_name: str = CLAIM_MODEL,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        logger.info("Loading claim extractor: %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        with suppress_stdout():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()

        # Clear sampling params from generation_config to avoid
        # "generation flags are not valid" warnings with do_sample=False
        gen_cfg = self.model.generation_config
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(gen_cfg, attr):
                setattr(gen_cfg, attr, None)

        logger.info("Claim extractor ready.")

    def extract(
        self,
        text: str,
        max_claims: int = MAX_CLAIMS,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> list[Claim]:
        """
        Extract atomic claims from *text*.

        Args:
            text: The text to decompose into claims.
            max_claims: Maximum number of claims to return.
            max_new_tokens: Generation length cap (prevents rambling).

        Returns:
            A deduplicated list of Claim objects.
        """
        if not text or not text.strip():
            return []

        raw_output = self._generate(text, max_new_tokens)
        claims = self._parse_claims(raw_output, text, max_claims)
        logger.info("Extracted %d claims from %d-char text.", len(claims), len(text))
        return claims

    # ── Private helpers ──────────────────────────────────────────────────────

    def _generate(self, text: str, max_new_tokens: int) -> str:
        """Run the LLM to generate claim text."""
        messages = [
            {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
            {"role": "user", "content": CLAIM_USER_PROMPT.format(text=text)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_claims(
        raw: str, source_text: str, max_claims: int
    ) -> list[Claim]:
        """Parse numbered/bulleted list into deduplicated Claim objects."""
        seen: set[str] = set()
        claims: list[Claim] = []

        for line in raw.split("\n"):
            # Strip numbering (e.g. "1.", "1)", "- ", "• ")
            cleaned = re.sub(r"^[\d.\)\-•\s]+", "", line).strip()
            if len(cleaned) <= 5:
                continue

            # Normalize for dedup (lowercase, collapse whitespace)
            key = re.sub(r"\s+", " ", cleaned.lower())
            if key in seen:
                continue
            seen.add(key)

            claims.append(Claim(text=cleaned, source_text=source_text))
            if len(claims) >= max_claims:
                break

        return claims
