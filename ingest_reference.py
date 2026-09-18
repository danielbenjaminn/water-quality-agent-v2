from __future__ import annotations

import argparse
from pathlib import Path

from water_quality_agent.rag.knowledge_base import KnowledgeBase


def build_parser() -> argparse.ArgumentParser:
    """
    Cria a CLI para indexação de referências técnicas no RAG.
    """
    parser = argparse.ArgumentParser(
        description="Indexa uma referência técnica PDF no RAG local."
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Caminho para o arquivo PDF.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Título da referência.",
    )

    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="Autor ou autores da referência.",
    )

    parser.add_argument(
        "--reference-type",
        type=str,
        default="book",
        help="Tipo da referência: book, article, legislation etc.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Tamanho dos chunks. Padrão: 1200.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=180,
        help="Sobreposição entre chunks. Padrão: 180.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pdf_path: Path = args.pdf

    # ---------------------------------------------------------
    # Validação
    # ---------------------------------------------------------

    if not pdf_path.exists():
        parser.error(f"Arquivo não encontrado: {pdf_path}")

    if not pdf_path.is_file():
        parser.error(f"O caminho não é um arquivo: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        parser.error(f"O arquivo informado não é PDF: {pdf_path}")

    # ---------------------------------------------------------
    # Knowledge Base
    #
    # O KnowledgeBase já possui defaults para:
    # - persist_dir
    # - collection_name
    # - model_name
    #
    # Portanto não precisamos informar nada aqui.
    # ---------------------------------------------------------

    knowledge_base = KnowledgeBase()

    # ---------------------------------------------------------
    # Indexação
    # ---------------------------------------------------------

    n_chunks = knowledge_base.ingest_pdf(
        path=pdf_path,
        title=args.title,
        author=args.author,
        reference_type=args.reference_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # ---------------------------------------------------------
    # Resultado
    # ---------------------------------------------------------

    print()
    print("REFERÊNCIA INDEXADA")
    print("-" * 60)
    print(f"Arquivo       : {pdf_path}")
    print(f"Título        : {args.title or pdf_path.stem}")
    print(f"Autor         : {args.author or 'não informado'}")
    print(f"Tipo          : {args.reference_type}")
    print(f"Chunk size    : {args.chunk_size}")
    print(f"Chunk overlap : {args.chunk_overlap}")
    print(f"Chunks        : {n_chunks}")
    print("-" * 60)


if __name__ == "__main__":
    main()