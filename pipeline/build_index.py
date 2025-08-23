import os
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from .scrape import fetch_corpus
from app.embeddings_e5 import E5Embeddings

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_poc")
DOC_URLS = [u.strip() for u in os.getenv("DOC_URLS","").split(",") if u.strip()]

# ---- Helpers: 20% overlap windowing at ~N tokens (rough via words) ----
def overlap_windows(text: str, target_words: int = 300, overlap_ratio: float = 0.2) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, int(target_words * (1 - overlap_ratio)))
    size = max(2, target_words)
    out = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        out.append(" ".join(chunk))
        i += step
    return out

def semantic_chunk(text: str, embedder: E5Embeddings,
                   window_words: int = 300, overlap_ratio: float = 0.2) -> List[Document]:
    # Use semantic splitter to find natural boundaries
    splitter = SemanticChunker(embedder, breakpoint_threshold_type="percentile")
    docs = splitter.create_documents([text])  # list[Document] of big, semantically-coherent chunks

    # Now apply 20% overlap within each semantic segment for retrieval robustness
    final_docs: List[Document] = []
    for d in docs:
        windows = overlap_windows(d.page_content, target_words=window_words, overlap_ratio=overlap_ratio)
        for w in windows:
            final_docs.append(
                Document(
                    page_content=w,
                    metadata={**(d.metadata or {}), "source": "web_corpus"}
                )
            )
    return final_docs

def ensure_collection(client: QdrantClient, collection: str, vector_size: int = 768):
    # Safer: try/except around get_collection
    try:
        client.get_collection(collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def run_build_index():
    # 1) Build corpus & chunks as you already do
    corpus = fetch_corpus(DOC_URLS)
    embeddings = E5Embeddings()
    docs = semantic_chunk(corpus, embeddings, window_words=300, overlap_ratio=0.2)
    print(f"[INGEST] Created {len(docs)} chunks")

    # 2) Create client & ensure collection
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    ensure_collection(client, COLLECTION, vector_size=768)

    # 3) Use an instance + add_documents (avoid from_documents path)
    vs = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    vs.add_documents(docs)

    print("[INGEST] Upsert complete.")

if __name__ == "__main__":
    run_build_index()
