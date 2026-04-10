def weighted_retrieval(vectorstore, query, strategy, k=8):
    results = vectorstore.similarity_search(query, k=20)

    weighted = []
    for doc in results:
        source = doc.metadata.get("source", "")
        weight = strategy["weights"].get(source, 0.0)
        weighted.append((weight, doc))

    weighted.sort(key=lambda x: x[0], reverse=True)

    return [doc for w, doc in weighted[:k]]

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from retrieval.loader import load_all_documents
    from retrieval.vector import build_vectorstore
    from strategy.strategy import select_strategy
    from strategy.intent import infer_intent

    documents = load_all_documents()
    vectorstore = build_vectorstore(documents)

    test_query = "What is the impact of cancer on the population in the Netherlands?"
    intent = infer_intent(test_query)
    strategy = select_strategy(intent)

    retrieved_docs = weighted_retrieval(vectorstore, test_query, strategy)
    print(f"Retrieved {len(retrieved_docs)} documents for query: '{test_query}'")