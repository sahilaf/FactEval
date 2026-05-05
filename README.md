# 🔍 FactEval

**Find exactly which parts of your LLM output are hallucinated.**

Debug hallucinations in RAG and LLM pipelines with claim-level verification.

```
Input:  "Paris is the capital of Germany."
Output:  Paris is the capital of Germany ❌
         → Contradicts evidence: "Paris is the capital of France."
```

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

---

## What is FactEval?

FactEval verifies claims against provided reference context (e.g., retrieved documents in a RAG system). It helps developers debug LLM outputs by:
- **Breaking** answers into atomic claims
- **Checking** each claim against reference evidence
- **Highlighting** hallucinated parts with explanations and diagnostics

---

## ⚡ Try It Instantly

> **[Live Demo on Hugging Face Spaces →](https://huggingface.co/spaces/sahilfarib/FactEval)**
>
> No setup needed — paste your text and see hallucinations highlighted in seconds.

---

## 📦 Install

```bash
pip install facteval
```

For development:
```bash
git clone https://github.com/sahilaf/facteval.git
cd facteval && pip install -e ".[dev]"
```

---

## 🚀 Quick Start

```python
from facteval import check

result = check(
    answer="Paris is the capital of Germany and has 5 million people.",
    contexts=[
        "Paris is the capital of France. Paris has approximately 2.2 million inhabitants.",
        "Germany's capital is Berlin.",
    ],
)

for claim in result["claims"]:
    print(f'{claim["label"]:15s} {claim["claim"]}')
    print(f'                → {claim["reason"]}')
```

```
contradicted    Paris is the capital of Germany.
                → Contradicts evidence: "Paris is the capital of France."
contradicted    Paris has 5 million people.
                → Contradicts evidence: "Paris has approximately 2.2 million inhabitants."
```

### Lightweight Mode (up to 10x faster)

Already have claims? Skip extraction entirely:

```python
from facteval import verify

result = verify(
    claims=["Paris is the capital of Germany.", "Paris has 5 million people."],
    contexts=["Paris is the capital of France. Population: 2.2M."],
)
# No Qwen model loaded → ~0.5 GB VRAM, ~0.3s latency
```

### CLI (Command Line Interface)

```bash
# Quick check
facteval check --answer "The earth is flat." --context "The earth is an oblate spheroid."

# From file
facteval check input.json --output results.json

# With calibrator
facteval check input.json --calibrator calibrator.pkl
```

---

## ✨ Features

- **Claim-level verdicts** — each sentence gets ✅ supported, ❌ contradicted, or ❓ unverifiable
- **Highlighted output** — color-coded HTML showing exactly which parts are wrong
- **Human-readable reasons** — every verdict explains *why*
- **Pipeline diagnostics** — hallucination vs. retrieval gap vs. missing context
- **Calibrated confidence** — isotonic regression for trustworthy probability scores
- **Lightweight mode** — `verify()` skips extraction for up to 10x faster results
- **Batch NLI** — all claims in a single forward pass
- **CLI** — `facteval check` for scripting and CI/CD

---

## 📋 Output Format

**Short version:**

```json
{
  "claims": [
    {
      "claim": "Paris is the capital of Germany.",
      "label": "contradicted",
      "confidence": 0.9971,
      "reason": "Contradicts evidence: \"Paris is the capital of France.\"",
      "diagnostics": {
        "failure_type": "hallucination",
        "retrieval_quality": "strong",
        "suggestion": "Claim directly contradicts the evidence."
      }
    }
  ],
  "summary": {
    "total_claims": 2, "supported": 0, "contradicted": 2,
    "unverifiable": 0, "hallucination_rate": 1.0
  },
  "highlighted_answer": "<mark>Paris is the capital of Germany ❌</mark>..."
}
```

<details>
<summary><strong>Full output includes</strong></summary>

Each claim also contains:
- `evidence` — the matched reference sentence
- `evidence_score` — retrieval similarity (0–1)
- `raw_nli_scores` — per-label NLI probabilities (`entailment`, `neutral`, `contradiction`)
- `calibrated_confidence` — post-calibration confidence (if calibrator provided)
- `calibration_error` — estimated calibration error

Top-level fields:
- `calibrated` — whether a fitted calibrator was used
- `pipeline_time_seconds` — total processing time

</details>

### Labels

| Label | Meaning |
|-------|---------|
| ✅ `supported` | Claim is entailed by the evidence |
| ❌ `contradicted` | Claim contradicts the evidence |
| ❓ `unverifiable` | No relevant evidence, or evidence is neutral |

### Diagnostics

| `failure_type` | What happened | What to do |
|----------------|---------------|------------|
| `verified` | Supported by strong evidence | Nothing — it's correct |
| `hallucination` | Contradicts strong evidence | Factual error in LLM output |
| `possible_hallucination` | Contradicts weak evidence | Add better context to confirm |
| `no_evidence` | No context for this topic | Add reference passages |
| `retrieval_gap` | Evidence too dissimilar | Context may not cover this claim |
| `inconclusive` | Evidence is neutral | Cannot confirm or deny |

---

## 🤔 When Should You Use FactEval?

**Use FactEval if you are:**
- Building a RAG system and need to verify answers against retrieved documents
- Debugging hallucinations in LLM outputs
- Evaluating whether generated answers are grounded in context
- Building CI/CD checks for LLM-powered features

**Not intended for:**
- General fact-checking without reference context (FactEval needs ground truth documents)
- Real-time inference on user-facing APIs (model loading adds latency)

---

## ⚡ Performance

| Mode | First run | Subsequent runs | VRAM |
|------|-----------|-----------------|------|
| `check()` (full) | ~60s (loads 3 models) | ~1.3s | ~3.4 GB |
| `verify()` (lightweight) | ~15s (loads 2 models) | ~0.3s | ~0.5 GB |

> First run downloads and loads models from Hugging Face. After that, models are cached in memory. Use `verify()` when you already have claims and need low-latency evaluation.

---

## 🧪 Built For

- **RAG pipelines** — verify that generated answers are grounded in retrieved documents
- **LLM evaluation workflows** — measure hallucination rates across test sets
- **AI product debugging** — find exactly where your model is making things up

---

## 🏗️ Architecture

```
Answer Text ──→ Claim Extractor ──→ Evidence Retriever ──→ NLI Verifier ──→ Calibrator ──→ Output
                (Qwen 1.5B)         (MiniLM + FAISS)      (DeBERTa)        (Isotonic)
                                          │                                      │
                                          └── Semantic Highlighting ─────────────┘
                                               (reuses MiniLM embeddings)
```

| Stage | Model | Size | Latency |
|-------|-------|------|---------|
| Claim Extraction | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | ~1s |
| Evidence Retrieval | `all-MiniLM-L6-v2` + FAISS | ~90 MB | <10ms |
| NLI Verification | `DeBERTa-v3-base-mnli-fever-anli` | ~370 MB | <10ms/batch |
| Calibration | Isotonic Regression (sklearn) | ~1 KB | <1ms |

---

## 🖥️ Try It Locally

```bash
pip install -e ".[demo]"
python demo/app.py
```

Features:
- Highlighted answer text with ✅❌❓ annotations
- Per-claim cards with reasons, diagnostic badges, and suggestions
- Summary dashboard with hallucination rate
- 4 built-in examples

### Deploy to Hugging Face Spaces

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/facteval
git push hf main
```

---

## 📁 Project Structure

```
FactEval/
├── README.md
├── ANALYSIS.md                # Development analysis & lessons learned
├── LICENSE                    # MIT
├── pyproject.toml             # Package config + CLI entry point
├── app.py                     # HF Spaces entry point
├── requirements.txt           # HF Spaces dependencies
├── examples/
│   ├── basic.py               # Minimal usage example
│   └── rag_debug.py           # RAG pipeline debugging example
├── demo/
│   └── app.py                 # Gradio interactive demo
└── facteval/
    ├── __init__.py            # Public API: check(), verify()
    ├── config.py              # Model names, prompts, defaults
    ├── models.py              # Claim, Evidence, ClaimWithEvidence
    ├── claim_extractor.py     # Qwen2.5-1.5B claim decomposition
    ├── retriever.py           # FAISS + MiniLM evidence retrieval
    ├── verifier.py            # DeBERTa batch NLI + reasons
    ├── calibrator.py          # Isotonic regression calibration
    ├── core.py                # Pipeline orchestration + diagnostics
    └── cli.py                 # facteval check CLI
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).
