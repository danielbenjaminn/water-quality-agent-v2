from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> list[Document]:
    """Carrega um PDF preservando a página de origem em metadata."""
    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"A referência deve ser um PDF: {pdf_path}")

    docs = PyPDFLoader(str(pdf_path)).load()
    for doc in docs:
        doc.metadata["source_path"] = str(pdf_path)
        doc.metadata["source"] = pdf_path.name
        # PyPDFLoader usa página zero-based; mantemos o valor bruto e
        # expomos page_number (1-based) para citações humanas.
        raw_page = doc.metadata.get("page")
        if isinstance(raw_page, int):
            doc.metadata["page_number"] = raw_page + 1
    return docs
