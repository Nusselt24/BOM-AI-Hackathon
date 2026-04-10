def infer_intent(query: str):
    q = query.lower()

    # TODO: use an LLM to infer intent instead of simple keyword matching

    scope = "population" if any(w in q for w in ["trend", "impact", "population", "netherlands"]) else "individual"

    purpose = "evidence" if any(w in q for w in ["evidence", "study", "guideline"]) else "understanding"

    return {
        "scope": scope,
        "purpose": purpose
    }

if __name__ == "__main__":
    test_queries = [
        "What is the impact of cancer on the population in the Netherlands?",
        "What are the guidelines for treating lung cancer?",
        "How does cancer affect an individual's quality of life?",
        "What evidence is there for the effectiveness of immunotherapy in cancer treatment?"
    ]

    for query in test_queries:
        intent = infer_intent(query)
        print(f"Query: {query}")
        print(f"Inferred Intent: {intent}\n")