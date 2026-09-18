from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievedReference:
    content: str
    source: str | None
    title: str | None
    author: str | None
    page: int | None
    score: float | None
    source_id: str | None

    def as_dict(self) -> dict:
        return {
            "content": self.content,
            "source": self.source,
            "title": self.title,
            "author": self.author,
            "page": self.page,
            "score": self.score,
            "source_id": self.source_id,
        }


def search_references(store, query: str, k: int = 5) -> list[RetrievedReference]:
    """Recupera chunks com score; menor distância indica maior proximidade no Chroma."""
    if not query.strip():
        return []
    if k < 1:
        raise ValueError("k deve ser >= 1.")

    results: list[tuple[Document, float]] = store.similarity_search_with_score(
        query,
        k=k,
    )

    retrieved: list[RetrievedReference] = []
    for doc, score in results:
        page = doc.metadata.get("page_number")
        if page is None:
            raw_page = doc.metadata.get("page")
            page = raw_page + 1 if isinstance(raw_page, int) else None

        retrieved.append(
            RetrievedReference(
                content=doc.page_content,
                source=doc.metadata.get("source"),
                title=doc.metadata.get("title"),
                author=doc.metadata.get("author"),
                page=page,
                score=float(score) if score is not None else None,
                source_id=doc.metadata.get("source_id"),
            )
        )
    return retrieved
