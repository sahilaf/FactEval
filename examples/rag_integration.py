"""
Example: Using FactEval in a RAG pipeline.

This shows how FactEval acts as a "drop-in evaluator" for Retrieval-Augmented Generation.
"""

from facteval import fast_check

def mock_rag_pipeline(query: str):
    """A simulated RAG pipeline."""
    # 1. Retrieve documents (mock)
    retrieved_docs = [
        "In 2021, the global electric vehicle market was valued at $163 billion.",
        "The market is projected to reach $823 billion by 2030, growing at a CAGR of 18.2%.",
        "Tesla remained the top-selling EV manufacturer globally in 2021."
    ]
    
    # 2. Generate response (mock)
    llm_response = (
        "The global EV market was valued at $163 billion in 2021. "
        "It is expected to hit $1 trillion by 2030. "
        "Toyota was the top-selling EV manufacturer in 2021."
    )
    
    return llm_response, retrieved_docs

if __name__ == "__main__":
    print("--- 1. Running RAG Pipeline ---")
    query = "Tell me about the EV market in 2021 and its future."
    response, docs = mock_rag_pipeline(query)
    
    print("\nLLM Response:")
    print(response)
    print("\nRetrieved Contexts:")
    for doc in docs:
        print(f" - {doc}")
        
    print("\n--- 2. Evaluating with FactEval ---")
    # In a real pipeline, you can split the response into sentences for the claims.
    claims = [s.strip() for s in response.split(".") if s.strip()]
    
    # We use `fast_check` because we already split the response into claims.
    # This skips the Qwen model and runs instantly (once models are loaded).
    result = fast_check(
        claims=claims,
        contexts=docs
    )
    
    print("\n--- 3. Results ---")
    for claim in result["claims"]:
        print(f"[{claim['label'].upper()}] {claim['claim']}")
        print(f"      ↳ {claim['reason']}")
        if claim['label'] == 'contradicted':
            print(f"      🚨 Diagnostics: {claim['diagnostics']['suggestion']}")
