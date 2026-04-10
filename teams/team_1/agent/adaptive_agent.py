from strategy.intent import infer_intent
from strategy.strategy import select_strategy
from retrieval.retriever import weighted_retrieval
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(temperature=0)

SYSTEM_PROMPT = """
You are an adaptive cancer information assistant.
Rules:
- Use ONLY the provided sources
- Do NOT add medical information not present in sources
- If evidence is weak or incomplete, say so explicitly
- Keep the tone aligned with the requested information focus
"""

def answer_query(query, vectorstore, mode=None):
    intent = infer_intent(query)
    strategy = select_strategy(intent, mode)

    docs = weighted_retrieval(vectorstore, query, strategy)

    context = "\n\n".join(
        f"[{d.metadata.get('source')}] {d.page_content}"
        for d in docs
    )

    prompt = f"""
Question: {query}

Context:
{context}

Answer the question.
"""

    response = llm([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return response.content, docs, strategy

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from retrieval.loader import load_all_documents
    from retrieval.vector import build_vectorstore

    documents = load_all_documents()
    vectorstore = build_vectorstore(documents)

    test_query = "What is the impact of cancer on the population in the Netherlands?"
    answer, docs, strategy = answer_query(test_query, vectorstore)
    print("Answer:", answer)
    print("Used Strategy:", strategy)
    print(f"Used {len(docs)} documents:")
    for d in docs:
        print(f"- {d.metadata.get('source')}: {d.page_content[:100]}...")