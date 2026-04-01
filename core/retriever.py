import faiss
import numpy as np
from core.config import EMBEDDING_DIMENSION, TOP_K


class Retriever:
    def __init__(self, dimension=None):
        dim = dimension or EMBEDDING_DIMENSION
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity (inner product)
        self.chunks = []  # List of (text, source)

    def add_chunks(self, chunks, embeddings, source):
        """Add document chunks and their embeddings to the FAISS index."""
        normalized = self._normalize(embeddings)
        self.index.add(normalized)
        for chunk in chunks:
            self.chunks.append((chunk, source))

    def search(self, query_embedding, k=None):
        """Search for the top-k most similar chunks."""
        k = k or TOP_K
        normalized = self._normalize(query_embedding)
        distances, indices = self.index.search(normalized, k)
        results = []
        for idx in indices[0]:
            if idx != -1:
                chunk_text, source = self.chunks[idx]
                results.append((chunk_text, source))
        return results

    @staticmethod
    def _normalize(embeddings):
        """L2-normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
