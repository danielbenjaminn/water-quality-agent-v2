from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Embeddings locais e multilíngues; não consomem tokens do LLM."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers não está instalado. Execute: "
                "pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


@lru_cache(maxsize=2)
def get_local_embeddings(model_name: str = DEFAULT_MODEL) -> Embeddings:
    return LocalSentenceTransformerEmbeddings(model_name=model_name)
