"""
FactEval – RAG Pipeline Debugging Example

Simulates a typical RAG workflow:
1. You have retrieved documents (contexts)
2. An LLM generated an answer
3. FactEval checks which parts of the answer are grounded

Run:  python examples/rag_debug.py
"""

from facteval import check, verify

# ── Scenario: LLM answers a question using retrieved docs ────────────────────

retrieved_docs = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python is a high-level, general-purpose programming language.",
    "As of 2024, Python is the most popular programming language on the TIOBE index.",
]

llm_answer = (
    "Python was created by Guido van Rossum and first released in 2005. "
    "It is a compiled, low-level language. "
    "Python is the most popular programming language on the TIOBE index."
)

print("=" * 60)
print("🔍 RAG Debug: Full Pipeline")
print("=" * 60)
print(f"\nLLM Answer: {llm_answer}\n")

result = check(answer=llm_answer, contexts=retrieved_docs)

# Analyze each claim
for claim in result["claims"]:
    diag = claim["diagnostics"]
    emoji = {"supported": "✅", "contradicted": "❌", "unverifiable": "❓"}

    print(f'{emoji.get(claim["label"], "?")} {claim["claim"]}')
    print(f'   → {claim["reason"]}')

    # Actionable debugging info
    if diag["failure_type"] == "hallucination":
        print(f'   🚨 HALLUCINATION — {diag["suggestion"]}')
    elif diag["failure_type"] == "no_evidence":
        print(f'   ⚠️  NO EVIDENCE — {diag["suggestion"]}')
    elif diag["failure_type"] == "retrieval_gap":
        print(f'   🔍 RETRIEVAL GAP — {diag["suggestion"]}')
    print()

s = result["summary"]
print(f'📊 Summary: {s["hallucination_rate"]:.0%} hallucination rate '
      f'({s["contradicted"]}/{s["total_claims"]} claims contradicted)')
print(f'⏱  Pipeline time: {result["pipeline_time_seconds"]:.1f}s')

# ── Scenario 2: Lightweight mode (you already have claims) ───────────────────

print(f'\n{"=" * 60}')
print("⚡ Lightweight Mode: verify() (no Qwen, much faster)")
print("=" * 60)

result2 = verify(
    claims=[
        "Python was first released in 2005.",
        "Python is a compiled, low-level language.",
        "Python is the most popular language on TIOBE.",
    ],
    contexts=retrieved_docs,
)

for claim in result2["claims"]:
    emoji = {"supported": "✅", "contradicted": "❌", "unverifiable": "❓"}
    print(f'{emoji.get(claim["label"], "?")} {claim["claim"]}')

print(f'\n⏱  verify() time: {result2["pipeline_time_seconds"]:.1f}s (no extraction)')
