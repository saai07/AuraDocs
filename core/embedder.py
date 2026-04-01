from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
from core.config import EMBEDDING_MODEL


@st.cache_resource(show_spinner="🧠 Loading embedding model... (one-time only)")
def _load_model(model_name):
    """Load and cache the embedding model across all sessions."""
    return SentenceTransformer(model_name)


class Embedder:
    def __init__(self, model_name=None):
        self.model = _load_model(model_name or EMBEDDING_MODEL)

    def embed(self, texts):
        """Embed a list of texts (or a single string) into float32 numpy array."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts)
        return np.array(embeddings).astype("float32")
