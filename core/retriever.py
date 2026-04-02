import faiss
import numpy as np
from core.config import EMBEDDING_DIMENSION, TOP_K, SIMILARITY_THRESHOLD


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

    def search(self, query_embedding, k=None, threshold=None):
        """Search for the top-k most similar chunks, filtered by similarity threshold."""
        k = k or TOP_K
        threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
        normalized = self._normalize(query_embedding)
        distances, indices = self.index.search(normalized, k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and score >= threshold:
                chunk_text, source = self.chunks[idx]
                results.append((chunk_text, source))
        return results

    @staticmethod
    def _normalize(embeddings):
        """L2-normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
