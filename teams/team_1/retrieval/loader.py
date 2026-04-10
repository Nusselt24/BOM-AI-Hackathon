from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")

def load_pdfs(folder: Path):
    documents = []
    for pdf in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = folder.name
            d.metadata["file"] = pdf.name
        documents.extend(docs)
    return documents

def load_kanker_nl(json_path: Path):
    loader = JSONLoader(
        file_path=str(json_path),
        jq_schema=".[] | {text: .text, title: .title, url: .url}",
        text_content=False,
    )
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = "kanker.nl"
    return docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def load_all_documents():
    docs = []
    docs += load_pdfs(DATA_DIR / "reports")
    docs += load_pdfs(DATA_DIR / "scientific_publications")
    docs += load_kanker_nl(DATA_DIR / "kanker_nl_pages_all.json")
    return chunk_documents(docs)

if __name__ == "__main__":
    # write some code to try out the function and print the number of documents loaded and a sample document
    documents = load_all_documents()
    print(f"Loaded {len(documents)} documents.")
    if documents:
        print("Sample document:")
        print(documents[0].page_content)
        print("Metadata:", documents[0].metadata)
