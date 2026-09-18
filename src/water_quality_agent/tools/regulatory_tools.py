from __future__ import annotations

from langchain_core.tools import tool

from water_quality_agent.core.domain import legal_limits_for


@tool
def get_conama_class2_limits(parameter: str) -> list[dict]:
    """
    Retorna os limites cadastrados da CONAMA 357/2005 para água doce
    Classe 2 de um parâmetro canônico.

    `parameter` deve ser o nome canônico previamente resolvido.

    Esta tool apenas consulta a referência regulatória. Não realiza
    resolução de parâmetros nem modifica os dados observacionais.
    """
    return legal_limits_for(parameter)