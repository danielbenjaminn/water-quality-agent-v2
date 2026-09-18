from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document


def source_id(path: str | Path) -> str:
    resolved = str(Path(path).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def enrich_metadata(
    documents: list[Document],
    *,
    path: str | Path,
    title: str | None = None,
    author: str | None = None,
    reference_type: str = "book",
) -> list[Document]:
    """Acrescenta metadata estável usada na recuperação e na resposta da tool."""
    pdf_path = Path(path).expanduser().resolve()
    sid = source_id(pdf_path)
    resolved_title = title or pdf_path.stem

    for doc in documents:
        doc.metadata.update(
            {
                "source_id": sid,
                "source": pdf_path.name,
                "source_path": str(pdf_path),
                "title": resolved_title,
                "reference_type": reference_type,
            }
        )
        if author:
            doc.metadata["author"] = author
    return documents
