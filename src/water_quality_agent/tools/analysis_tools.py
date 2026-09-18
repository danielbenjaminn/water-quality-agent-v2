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
    Adapta observações do schema canônico para o schema esperado
    pelos módulos estatísticos legados.

    Não executa limpeza, resolução de parâmetros, extração de
    qualifiers ou conversão de unidades.
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

    df = df.rename(columns=rename)

    if "Resultado Num" in df.columns:
        df["Resultado Num"] = pd.to_numeric(
            df["Resultado Num"],
            errors="coerce",
        )

    return df


@tool
def descriptive_statistics(
    observations: list[dict],
) -> list[dict]:
    """
    Calcula estatísticas descritivas sobre as observações fornecidas.

    Os dados devem ter sido previamente harmonizados.
    """
    if not observations:
        return []

    df = _records_df(observations)

    return descriptive_analysis(
        df=df,
        parameter_col="Parâmetro B.D.",
        value_col="Resultado Num",
        point_col="Ponto" if "Ponto" in df.columns else None,
    )


@tool
def mann_kendall_trend(
    observations: list[dict],
) -> list[dict]:
    """
    Executa análise de tendência Mann-Kendall sobre a série
    fornecida.
    """
    if not observations:
        return []

    df = _records_df(observations)

    return mann_kendall_analysis(
        df,
        parameter_col="Parâmetro B.D.",
        value_col="Resultado Num",
    )


@tool
def kendall_correlations(
    observations: list[dict],
) -> list[dict]:
    """
    Calcula correlações de Kendall sobre o subconjunto fornecido.
    """
    if not observations:
        return []

    df = _records_df(observations)

    return correlation_analysis(df)


@tool
def iqr_outliers(
    observations: list[dict],
) -> list[dict]:
    """
    Identifica observações pelo critério IQR.

    As observações são marcadas, não removidas.
    """
    if not observations:
        return []

    df = _records_df(observations)

    result = flag_iqr_outliers(df)

    return (
        result
        .astype(object)
        .where(pd.notna(result), None)
        .to_dict("records")
    )