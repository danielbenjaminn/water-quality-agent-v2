from __future__ import annotations

from langchain_core.tools import tool

from water_quality_agent.core.domain import legal_limits_for
from water_quality_agent.core.regulatory import (
    calculate_regulatory_compliance,
)
from water_quality_agent.tools.data_tools import select_series


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


@tool
def regulatory_compliance(
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Avalia a conformidade das observações com os limites cadastrados
    da CONAMA 357/2005 para água doce Classe 2.

    A série é obtida diretamente do dataset canônico no backend.

    Observações censuradas não são tratadas como valores exatos nesta
    versão e são reportadas separadamente.

    `parameter` deve ser o nome canônico previamente resolvido.
    """

    series = select_series(
        parameter=parameter,
        point=point,
        start_date=start_date,
        end_date=end_date,
    )

    limits = legal_limits_for(parameter)

    if not limits:
        return {
            "point": point,
            "parameter": parameter,
            "valid": False,
            "reason": "no_regulatory_limit",
        }

    limit = limits[0]

    minimum = limit.get("Min")
    maximum = limit.get("Max")

    result = calculate_regulatory_compliance(
        df=series,
        minimum=minimum,
        maximum=maximum,
    )

    return {
        "point": point,
        "parameter": parameter,
        "unit": limit.get("Unidade"),
        "minimum": minimum,
        "maximum": maximum,
        **result,
    }