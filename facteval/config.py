"""
Default configuration for FactEval models and parameters.
"""

# ── Model IDs (Hugging Face Hub) ─────────────────────────────────────────────
# Claim extraction – chosen in Week 0: 1.5B was 3.5x faster with cleaner output
CLAIM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Sentence embeddings for evidence retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NLI verification (used in Week 2)
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# ── Retrieval defaults ───────────────────────────────────────────────────────
DEFAULT_TOP_K = 3
MIN_EVIDENCE_SCORE = 0.3  # Below this, evidence is too weak to use

# ── Claim extraction defaults ────────────────────────────────────────────────
MAX_NEW_TOKENS = 200
MAX_CLAIMS = 10

CLAIM_SYSTEM_PROMPT = (
    "You are a claim extraction engine. Given a text, break it into atomic, "
    "independently verifiable claims. Each claim states exactly ONE fact. "
    "Return ONLY a numbered list. No explanations, no commentary."
)

CLAIM_USER_PROMPT = "Break this into atomic claims:\n\n{text}"
