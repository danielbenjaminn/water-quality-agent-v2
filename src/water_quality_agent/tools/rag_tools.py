from __future__ import annotations

from langchain_core.tools import tool

from water_quality_agent.rag.knowledge_base import KnowledgeBase


# KnowledgeBase compartilhada pelo processo.
# A abertura do vector store e o carregamento dos embeddings
# continuam lazy dentro da própria KnowledgeBase.
_KNOWLEDGE_BASE = KnowledgeBase()


@tool
def search_technical_references(
    query: str,
    k: int = 5,
) -> list[dict]:
    """
    Busca trechos relevantes na literatura técnica indexada no RAG.

    Use esta tool quando for necessária fundamentação técnica ou
    metodológica para interpretar dados de qualidade da água,
    selecionar ou justificar métodos estatísticos, discutir dados
    censurados, interpretar parâmetros ambientais ou consultar
    literatura técnica relacionada.

    A busca é semântica e pode consultar diferentes referências
    indexadas, como livros de qualidade da água e métodos
    estatísticos.

    Parameters
    ----------
    query:
        Consulta técnica em linguagem natural. Deve descrever
        claramente a informação ou metodologia procurada.

    k:
        Número máximo de trechos recuperados. O padrão é 5.

    Returns
    -------
    list[dict]
        Trechos recuperados da base técnica. Cada resultado pode
        conter conteúdo, título, autor, página, fonte e score.
    """

    query = query.strip()

    if not query:
        raise ValueError(
            "A consulta à literatura técnica não pode ser vazia."
        )

    if k < 1:
        raise ValueError(
            "k deve ser maior ou igual a 1."
        )

    if k > 10:
        k = 10

    return _KNOWLEDGE_BASE.search(
        query=query,
        k=k,
    )