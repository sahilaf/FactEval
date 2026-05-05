"""
FactEval – Basic Usage Example

Run:  python examples/basic.py
"""

from facteval import check

# Check an answer against reference contexts
result = check(
    answer="Paris is the capital of Germany and has 5 million people.",
    contexts=[
        "Paris is the capital of France. Paris has approximately 2.2 million inhabitants.",
        "Germany's capital is Berlin.",
    ],
)

# Print claim-level verdicts
print("=" * 60)
print("📋 Claim Verdicts")
print("=" * 60)
for claim in result["claims"]:
    emoji = {"supported": "✅", "contradicted": "❌", "unverifiable": "❓"}
    print(f'\n{emoji.get(claim["label"], "?")} {claim["claim"]}')
    print(f'   Label:       {claim["label"]}')
    print(f'   Confidence:  {claim["confidence"]:.1%}')
    print(f'   Reason:      {claim["reason"]}')
    print(f'   Diagnostic:  {claim["diagnostics"]["failure_type"]}')

# Print summary
print(f'\n{"=" * 60}')
s = result["summary"]
print(f'📊 {s["total_claims"]} claims: '
      f'{s["supported"]} supported, '
      f'{s["contradicted"]} contradicted, '
      f'{s["unverifiable"]} unverifiable')
print(f'   Hallucination rate: {s["hallucination_rate"]:.0%}')

# Print highlighted answer (HTML)
print(f'\n📝 Highlighted:\n{result["highlighted_answer"]}')
