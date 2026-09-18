from __future__ import annotations

from pathlib import Path

from water_quality_agent.rag.chunker import split_documents
from water_quality_agent.rag.embeddings import DEFAULT_MODEL, get_local_embeddings
from water_quality_agent.rag.loader import load_pdf
from water_quality_agent.rag.metadata import enrich_metadata, source_id
from water_quality_agent.rag.retriever import search_references
from water_quality_agent.rag.vectorstore import (
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    open_vectorstore,
    replace_source_documents,
)


class KnowledgeBase:
    """Fachada do RAG técnico local usado pelas tools do agente."""

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.model_name = model_name
        self._embeddings = None
        self.vectorstore = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_local_embeddings(self.model_name)
        return self._embeddings

    def open(self) -> "KnowledgeBase":
        if self.vectorstore is None:
            self.vectorstore = open_vectorstore(
                embeddings=self.embeddings,
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
            )
        return self

    def ingest_pdf(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        author: str | None = None,
        reference_type: str = "book",
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
    ) -> int:
        pages = load_pdf(path)
        pages = enrich_metadata(
            pages,
            path=path,
            title=title,
            author=author,
            reference_type=reference_type,
        )
        chunks = split_documents(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.open()
        return replace_source_documents(
            self.vectorstore,
            chunks,
            source_id(path),
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        self.open()
        return [item.as_dict() for item in search_references(self.vectorstore, query, k=k)]
