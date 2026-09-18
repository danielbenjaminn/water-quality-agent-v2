from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from water_quality_agent.core.descriptive import descriptive_analysis
from water_quality_agent.core.trend import mann_kendall_analysis
from water_quality_agent.core.correlation import correlation_analysis
from water_quality_agent.core.outliers import flag_iqr_outliers
from water_quality_agent.tools.data_tools import select_series


def _analysis_df(
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Recupera uma série diretamente do dataset harmonizado no backend
    e adapta somente o schema para os módulos estatísticos legados.

    Não executa:
    - resolução de parâmetro;
    - extração de qualifier;
    - conversão de unidade;
    - limpeza semântica.

    `parameter` deve ser o nome canônico previamente resolvido.
    """

    df = select_series(
        parameter=parameter,
        point=point,
        start_date=start_date,
        end_date=end_date,
    )

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
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    Calcula estatísticas descritivas diretamente sobre uma série
    armazenada no backend.

    `parameter` deve ser o nome canônico previamente resolvido.
    """

    df = _analysis_df(
        parameter=parameter,
        point=point,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        return []

    return descriptive_analysis(
        df=df,
        parameter_col="Parâmetro B.D.",
        value_col="Resultado Num",
        point_col=(
            "Ponto"
            if "Ponto" in df.columns
            else None
        ),
    )


@tool
def mann_kendall_trend(
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    Executa análise de tendência Mann-Kendall diretamente sobre
    uma série armazenada no backend.

    `parameter` deve ser o nome canônico previamente resolvido.
    """

    df = _analysis_df(
        parameter=parameter,
        point=point,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        return []

    return mann_kendall_analysis(
        df,
        parameter_col="Parâmetro B.D.",
        value_col="Resultado Num",
    )


@tool
def iqr_outliers(
    parameter: str,
    point: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """
    Identifica observações pelo critério IQR diretamente sobre
    uma série armazenada no backend.

    As observações são marcadas, não removidas.

    `parameter` deve ser o nome canônico previamente resolvido.
    """

    df = _analysis_df(
        parameter=parameter,
        point=point,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        return []

    result = flag_iqr_outliers(df)

    return (
        result
        .astype(object)
        .where(pd.notna(result), None)
        .to_dict("records")
    )


@tool
def kendall_correlations(
    observations: list[dict],
) -> list[dict]:
    """
    Calcula correlações de Kendall sobre o subconjunto fornecido.

    ATENÇÃO:
    Esta tool ainda mantém temporariamente o contrato antigo.
    Será refatorada separadamente porque correlação envolve
    múltiplos parâmetros.
    """

    if not observations:
        return []

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

    return correlation_analysis(df)