from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class E5Embeddings(Embeddings):
    """
    E5 wrapper that prefixes inputs with 'query:' / 'passage:' and normalizes vectors.
    Uses intfloat/e5-base-v2 (768-dim) by default.
    """
    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(prefixed, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(f"query: {text}", normalize_embeddings=True)
        return vec.tolist()
