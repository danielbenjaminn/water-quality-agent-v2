from langchain_core.tools import tool
from water_quality_agent.rag.knowledge_base import KnowledgeBase
KB = KnowledgeBase()

@tool
def search_technical_references(query: str, k: int = 5) -> list[dict]:
    """Consulta a base vetorial de referências técnicas para fundamentar interpretação ambiental/metodológica."""
    docs=KB.search(query,k=k)
    return [{"content":d.page_content,"source":d.metadata.get("source"),"page":d.metadata.get("page")} for d in docs]
