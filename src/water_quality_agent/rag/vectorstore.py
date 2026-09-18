from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_PERSIST_DIR = ".vectorstore"
DEFAULT_COLLECTION = "water_quality_references"


def open_vectorstore(
    embeddings: Embeddings,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(path),
        embedding_function=embeddings,
    )


def _chunk_id(doc: Document) -> str:
    basis = "|".join(
        [
            str(doc.metadata.get("source_id", "")),
            str(doc.metadata.get("page_number", doc.metadata.get("page", ""))),
            str(doc.metadata.get("chunk_index", "")),
            doc.page_content[:200],
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def replace_source_documents(
    store: Chroma,
    documents: list[Document],
    source_id: str,
) -> int:
    """Substitui os chunks de uma referência, evitando duplicação em reingestões."""
    try:
        store.delete(where={"source_id": source_id})
    except Exception:
        # Coleção nova/sem registros da fonte: não há nada a remover.
        pass

    if not documents:
        return 0

    store.add_documents(
        documents,
        ids=[_chunk_id(doc) for doc in documents],
    )
    return len(documents)
