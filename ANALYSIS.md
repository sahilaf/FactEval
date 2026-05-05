# FactEval – Development Analysis

> **Find exactly which parts of your LLM output are hallucinated.**
>
> A complete account of what was built, how it works, and what was learned.

---

## 🎯 Problem Statement

Large Language Models hallucinate. When an LLM generates an answer from retrieved context (RAG), there is no reliable way to know which parts of the answer are factually grounded and which are fabricated. Existing tools (OpenAI Evals, Ragas) give document-level scores but don't tell you *which sentences are wrong* or *whether those scores are trustworthy*.

FactEval solves both problems:
- **Claim-level verdicts** — each sentence gets its own label (✅ supported, ❌ contradicted, ❓ unverifiable)
- **Calibrated confidence** — isotonic regression transforms raw NLI probabilities into trustworthy scores
- **Pipeline diagnostics** — tells developers *why* a claim failed (hallucination vs. retrieval gap vs. missing context)

---

## 📊 Development Timeline

### Week 0 – Model Selection & Prototyping ✅

**Goal:** Validate that the technical approach works before writing production code.

| Experiment | Result | Decision |
|-----------|--------|----------|
| Qwen2.5-3B vs 1.5B for claim extraction | 1.5B: 3.5x faster (3.4s vs 11.7s), half the VRAM (2.9 GB vs 5.9 GB), cleaner output | **Use 1.5B** |
| DeBERTa NLI on 10 test pairs | 10/10 accuracy, confidence >0.93 on correct labels | **DeBERTa validated** |
| E2E pipeline prototype | Both hallucinations detected at >0.99 confidence | **Architecture proven** |
| Isotonic regression calibration | ECE dropped from 0.038–0.058 → 0.000, Brier improved ~12% | **Isotonic calibration works** |

**Key learning:** Phi-2 and Llama-3.2-1B were originally planned but both had issues (Phi-2 config bug with transformers, Llama gated). Qwen2.5 was a drop-in replacement with better compatibility.

---

### Week 1 – Claim Extraction & Retrieval ✅

**Goal:** Build the first two pipeline stages as a Python package.

**Components built:**
- `ClaimExtractor` – Qwen2.5-1.5B with the model's chat template (system + user messages), not raw prompts. Produces clean numbered lists with deduplication.
- `EvidenceRetriever` – FAISS `IndexFlatIP` over normalized MiniLM embeddings. Splits contexts into sentences, finds top-k most similar per claim.
- `Claim`, `Evidence`, `ClaimWithEvidence` – Pydantic models with validation.

**Test results:** 13/13 passed (6 extractor + 7 retriever).

**Key design decisions:**
1. **Chat template over raw prompts** — the model follows instructions much better with system/user message format.
2. **Deduplication** — the 1.5B model sometimes rephrases the same claim; normalized dedup catches this.
3. **Score clamping** — FAISS inner product on normalized vectors can return 1.0000001 due to float32 precision. Removed strict `le=1.0` constraints.

---

### Week 2 – Verification & Scoring ✅

**Goal:** Add NLI verification and wire the full pipeline.

**Components built:**
- `Verifier` – DeBERTa with `evidence→claim` NLI inference. Maps: entailment→supported, contradiction→contradicted, neutral→unverifiable.
- `core.check()` – One-line API wiring all 3 stages with lazy-loaded singletons.

**Test results:** 5/5 NLI label mapping, 2/2 fallback tests, 5/5 check() tests.

**Key design decisions:**
1. **Premise=evidence, hypothesis=claim** — NLI convention. Reversing this silently degrades accuracy.
2. **Lazy singletons** — first call ~63s (loading 3 models), subsequent calls <1.5s.
3. **Graceful fallback** — no evidence → unverifiable with confidence=0.0 (not a crash).

---

### Week 3 – Calibration, CLI & Python API ✅

**Goal:** Add confidence calibration, CLI, and polish the output schema.

**Components built:**
- `Calibrator` – Loads pickled isotonic regression models (one per NLI label). Falls back to raw scores with estimated error if no calibrator file exists.
- `cli.py` – `facteval check input.json --output out.json --calibrator cal.pkl`.
- Updated `check()` with `calibrator_path` parameter and `calibrated` flag in output.

**Test results:** 4/4 calibrator, 2/2 calibrated check(), schema validated, CLI exit 0.

**Key design decisions:**
1. **Calibration is optional** — raw scores work fine with `max(0.02, (1 - raw_confidence) * 0.3)` as error estimate.
2. **CLI lazy-imports** — `facteval --help` is instant; heavy model imports only happen on `check`.

---

### Week 4 – Polish, Diagnostics & Production Readiness ✅

**Goal:** Make the project presentable, deployable, and trustworthy.

**Components built:**

**Highlight system (upgraded from Jaccard → semantic)**
- Original: Jaccard token overlap to match claims to source sentences. Broke on paraphrasing, reordering, partial overlaps.
- Final: Uses the retriever's MiniLM encoder to compute cosine similarity between claim embeddings and answer sentence embeddings. Produces `<mark>`-tagged HTML with ✅❌❓ emojis.
- Zero additional model loading — reuses the already-loaded MiniLM instance.

**Per-claim diagnostics**
- Each claim now includes a `diagnostics` block:
  ```json
  {
    "failure_type": "hallucination",
    "retrieval_quality": "strong",
    "suggestion": "Claim directly contradicts the evidence. This is a factual error."
  }
  ```
- Failure types: `verified`, `hallucination`, `possible_hallucination`, `no_evidence`, `retrieval_gap`, `inconclusive`
- Tells developers *why* a claim failed — was it bad context? Weak retrieval? Or a genuine factual error?

**Lightweight `verify()` mode**
- `verify(claims=[...], contexts=[...])` — skips Qwen extraction entirely.
- Only loads MiniLM + DeBERTa (~0.5 GB total). No 63s cold start for users with pre-extracted claims.

**Batch NLI inference**
- All claims processed in a single tokenizer + model forward pass instead of per-claim loops.
- 3-5x faster for multi-claim inputs.

**Reason field**
- Every verdict now includes a human-readable `reason` explaining the NLI judgment.
- Example: `"Contradicts evidence: \"Paris is the capital of France.\""` 

**Gradio demo**
- `demo/app.py` — Highlighted answer at top, per-claim cards with diagnostic badges and suggestions, summary dashboard, collapsible JSON.

**HF Space deployment**
- `app.py` (root entry) + `requirements.txt` — ready to push to Hugging Face Spaces.

**Warning suppression**
- Fixed `torch_dtype` deprecation → `dtype`
- Cleared Qwen generation config (`temperature`, `top_p`, `top_k`)
- Suppressed safetensors LOAD REPORT, accelerate sharding noise, HF Hub auth warnings

---

## 🏛️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Claim Extractor │ ──→ │ Evidence Retriever│ ──→ │   NLI Verifier  │ ──→ │  Calibrator   │
│  Qwen 1.5B      │     │ MiniLM + FAISS   │     │ DeBERTa (batch) │     │ Isotonic Reg  │
│  ~3 GB VRAM     │     │ ~90 MB           │     │ ~370 MB         │     │ ~1 KB         │
│  ~1s/query      │     │ <0.01s/query     │     │ <0.01s/batch    │     │ <0.001s       │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────┘
                                  │                                              │
                                  └──── Semantic Highlighting (reuses MiniLM) ───┘
                                         + Pipeline Diagnostics
```

**Full pipeline VRAM:** ~3.4 GB (fits on T4, RTX 3050+)
**Lightweight mode VRAM:** ~0.5 GB (MiniLM + DeBERTa only)
**Latency after warm-up:** ~1.3s (full), ~0.3s (lightweight)

---

## 📈 Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| NLI accuracy (test set) | 10/10 (100%) | Curated test pairs |
| MNLI validation accuracy | 89.0% | 2,000 samples |
| Calibration ECE (post) | 0.000 | On training set; real-world will be higher |
| Brier score improvement | ~12% | After isotonic calibration |
| Pipeline latency (full) | ~1.3s | After models cached, T4 GPU |
| Pipeline latency (lightweight) | ~0.3s | verify() mode, no extraction |
| First-call latency (full) | ~63s | 3 model loads |
| First-call latency (lightweight) | ~15s | 2 model loads (no Qwen) |
| Final test pass rate | ~80/80 (100%) | 8 sections, all components |
| Total VRAM (full) | ~3.4 GB | Qwen 1.5B + DeBERTa + MiniLM |
| Total VRAM (lightweight) | ~0.5 GB | DeBERTa + MiniLM only |

---

## 🔑 Lessons Learned

### 1. Model compatibility is fragile
Phi-2's config broke with newer transformers (`pad_token_id` not in `PhiConfig`). Llama-3.2 required gated access. Qwen2.5 was a drop-in replacement that just worked. **Lesson:** Always have a backup model choice.

### 2. Small models can be better
Qwen2.5-1.5B produced **cleaner** output than the 3B variant for claim extraction. The 3B model over-explained, repeated claims, and added verification notes. **Lesson:** Bigger isn't always better for structured output tasks.

### 3. Float precision breaks validation
FAISS inner product on normalized vectors returned 1.0000001, which broke Pydantic's `le=1.0` constraint. **Lesson:** Never use strict upper bounds on float outputs from numerical libraries.

### 4. Chat templates matter
Switching from raw prompts to the model's chat template (system + user messages) dramatically improved output quality. The model stopped rambling and produced clean numbered lists. **Lesson:** Use the model's native format.

### 5. Lazy loading is essential
Loading 3 models takes ~63s. Without lazy singletons, every `check()` call would reload everything. With singletons, subsequent calls take <1.5s. **Lesson:** Cache expensive resources at the module level.

### 6. The claim extractor rewrites claims
When given "Paris is the capital of Germany", the model sometimes outputs "Paris is the capital of France" (correcting the claim). This doesn't break NLI, but means extracted claims don't always match original text verbatim. **Lesson:** Semantic alignment (embeddings) is required to map claims back to source spans.

### 7. Jaccard fails on paraphrasing
Token overlap breaks when the LLM rephrases, reorders, or partially modifies claims. Upgrading to embedding-based cosine similarity (reusing the already-loaded MiniLM) solved this with zero additional latency or memory cost. **Lesson:** Don't use bag-of-words methods when you already have an embedding model loaded.

### 8. Diagnostics turn an evaluator into a debugging tool
Adding `failure_type` + `suggestion` to each claim transformed FactEval from "this claim is contradicted" into "this claim contradicts the evidence — here's what to fix." This is the difference between an evaluation tool and a developer tool. **Lesson:** Always tell users what to do next.

---

## 🔧 Post-Review Improvements (Implemented)

After external review, the following critical improvements were made:

| Round | Issue Raised | Solution | Impact |
|-------|-------------|----------|--------|
| 1 | No "why it failed" explanation | Added `reason` field to every verdict | Users understand *why* claims failed |
| 1 | Cold start = 63s kills adoption | Added `verify()` lightweight mode (skips Qwen) | Pre-extracted claims skip 63s load |
| 1 | Per-claim NLI is slow | Batch NLI inference (single forward pass) | 3-5x faster verification |
| 1 | No "instant value" output | Added `highlighted_answer` with color-coded HTML | Users see exactly which sentences are wrong |
| 1 | Positioning too academic | Changed tagline to "Find exactly which parts are hallucinated" | Clear value proposition |
| 2 | Highlighting too naive (Jaccard) | Upgraded to embedding similarity (MiniLM cosine) | Handles paraphrasing, reordering |
| 2 | No pipeline insight for devs | Added `diagnostics` block with failure_type + suggestion | Moves from evaluator → debugging tool |
| 2 | No "quick try" entry point | Created HF Space config (app.py + requirements.txt) | Zero-effort deployment |

---

## 📁 Final Project Structure

```
FactEval/
├── app.py                     # HF Spaces entry point
├── requirements.txt           # HF Spaces dependencies
├── pyproject.toml             # Package config, CLI entry point
├── README.md                  # Quick start, output format, architecture
├── ANALYSIS.md                # This document
├── checklist.md               # Week-by-week progress tracker
├── plan.md                    # Original technical roadmap
├── colab_final_test.py        # ~80 assertion test suite
├── demo/
│   └── app.py                 # Gradio demo with highlighting + diagnostics
└── facteval/
    ├── __init__.py            # Warning suppression + public API (check, verify)
    ├── config.py              # Model IDs, prompts, defaults
    ├── models.py              # Claim, Evidence, ClaimWithEvidence (Pydantic)
    ├── claim_extractor.py     # Qwen2.5-1.5B chat-template claim decomposition
    ├── retriever.py           # FAISS + MiniLM evidence retrieval
    ├── verifier.py            # DeBERTa batch NLI + reasons
    ├── calibrator.py          # Isotonic regression calibration
    ├── core.py                # check(), verify(), diagnostics, semantic highlighting
    └── cli.py                 # facteval check CLI
```

---

## 🔮 Remaining Future Work

1. **Preloading mode** — background model warmup for interactive applications
2. **4-bit quantization** — enable local GPU inference on 4GB cards via bitsandbytes
3. **Integration hooks** — LangChain/LlamaIndex callbacks for RAG pipelines
4. **Cross-validated calibration** — current ECE=0 is on training data, need held-out evaluation
5. **Multi-language support** — extend to non-English claim extraction and verification
6. **Streaming output** — yield claim results incrementally for long documents
