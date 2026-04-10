import os
import ssl

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Patch ssl.create_default_context so httpx (used by huggingface_hub) also skips verification
_orig_create_default_context = ssl.create_default_context
def _unverified_context(*args, **kwargs):
    ctx = _orig_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _unverified_context
ssl._create_default_https_context = ssl._create_unverified_context

import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_PATH = Path(__file__).parent.parent / "vectorstore"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(documents):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(VECTORSTORE_PATH))
    return vectorstore

def load_vectorstore():
    """Load from disk if available, otherwise build and save."""
    embeddings = get_embeddings()
    if VECTORSTORE_PATH.exists():
        return FAISS.load_local(str(VECTORSTORE_PATH), embeddings, allow_dangerous_deserialization=True)
    from loader import load_all_documents
    documents = load_all_documents()
    return build_vectorstore(documents)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from loader import load_all_documents
    documents = load_all_documents()
    vectorstore = build_vectorstore(documents)
    print("Vectorstore built with", vectorstore.index.ntotal, "vectors.")