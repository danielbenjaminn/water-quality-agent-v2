from langchain_core.tools import tool
from water_quality_agent.core.domain import legal_limits_for, resolve_parameter

@tool
def get_conama_class2_limits(parameter: str) -> list[dict]:
    """Retorna limites cadastrados da CONAMA 357/2005 para água doce Classe 2 do parâmetro informado."""
    canonical = resolve_parameter(parameter).get("canonical") or parameter
    return legal_limits_for(canonical)
