import nest_asyncio
nest_asyncio.apply()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embeddings)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from loader import load_all_documents
    documents = load_all_documents()
    vectorstore = build_vectorstore(documents)
    print("Vectorstore built with", len(vectorstore.index), "vectors.")