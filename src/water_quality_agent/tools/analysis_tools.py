from __future__ import annotations
import pandas as pd
from langchain_core.tools import tool
from water_quality_agent.core.descriptive import descriptive_analysis
from water_quality_agent.core.trend import mann_kendall_analysis
from water_quality_agent.core.correlation import correlation_analysis
from water_quality_agent.core.outliers import flag_iqr_outliers


def _records_df(
    observations: list[dict],
) -> pd.DataFrame:
    """
    Adapta o schema canônico atual para os módulos estatísticos
    legados.

    Não executa limpeza, resolução de parâmetro, extração de
    qualifier ou conversão de unidade.
    """

    df = pd.DataFrame(observations)

    rename = {
        "parameter": "Parâmetro B.D.",
        "result": "Resultado Num",
        "point": "Ponto",
        "date": "Data",
        "qualifier": "Qualifier",
        "unit": "Unidade",
    }

    df = df.rename(
        columns=rename
    )

    if "Resultado Num" in df.columns:
        df["Resultado Num"] = pd.to_numeric(
            df["Resultado Num"],
            errors="coerce",
        )

    return df

@tool
def descriptive_statistics(observations: list[dict]) -> list[dict]:
    """Calcula estatísticas descritivas/censura somente sobre as observações fornecidas pelo agente."""
    if not observations: return []
    df = _records_df(observations)
    return descriptive_analysis(df=df, parameter_col="Parâmetro B.D.", value_col="Resultado Num", point_col="Ponto" if "Ponto" in df else None)

@tool
def mann_kendall_trend(observations: list[dict]) -> list[dict]:
    """Executa análise de tendência Mann-Kendall somente na série selecionada pelo agente."""
    if not observations: return []
    return mann_kendall_analysis(_records_df(observations), parameter_col="Parâmetro B.D.", value_col="Resultado Num")

@tool
def kendall_correlations(observations: list[dict]) -> list[dict]:
    """Calcula correlações Kendall sobre o subconjunto selecionado pelo agente."""
    if not observations: return []
    return correlation_analysis(_records_df(observations))

@tool
def iqr_outliers(observations: list[dict]) -> list[dict]:
    """Marca outliers IQR no subconjunto fornecido; não remove observações."""
    if not observations: return []
    df = flag_iqr_outliers(_records_df(observations))
    return df.where(pd.notna(df), None).to_dict("records")
